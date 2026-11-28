"""Turn raw sensor measurements into a few stable, explainable channels.

Every channel is 0..1 where 0.5 means "your normal" (after calibration),
or a plain physical unit (words per minute, seconds) when that is clearer.
The pattern rules and the LLM only ever read these channels, so any sensor
can be swapped or removed without rewriting them.
"""
import math
import time
from dataclasses import dataclass, field

from .config import Config
from .signals import Signals

# Population guesses used until the student's own baseline is ready: (mean, std)
PRIORS = {
    "voice_arousal": (0.5, 0.1),
    "voice_dominance": (0.5, 0.1),
    "voice_valence": (0.5, 0.1),
    "face_arousal": (0.0, 0.3),
    "face_valence": (0.0, 0.3),
    "pitch_st": (8.0, 5.0),        # semitones re 100 Hz; voices differ a lot, so calibration matters
    "pitch_var": (2.5, 1.0),
    "volume_db": (-30.0, 6.0),
    "wpm": (140.0, 25.0),
}
MIN_STD = {"voice_arousal": 0.03, "voice_dominance": 0.03, "voice_valence": 0.03, "face_arousal": 0.08,
           "face_valence": 0.08, "pitch_st": 1.0, "pitch_var": 0.4, "volume_db": 2.0, "wpm": 12.0}


def squash(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))


@dataclass
class Baseline:
    target_seconds: float
    samples: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    speech_seconds: float = 0.0

    @property
    def ready(self) -> bool:
        return bool(self.stats)

    def update(self, raw: dict, dt: float):
        if self.ready:
            return
        self.speech_seconds += dt
        for k, v in raw.items():
            if k in PRIORS and v is not None:
                self.samples.setdefault(k, []).append(v)
        if self.speech_seconds >= self.target_seconds:
            for k, (m0, s0) in PRIORS.items():
                vals = self.samples.get(k, [])
                if len(vals) >= 5:
                    m = sum(vals) / len(vals)
                    s = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))
                    self.stats[k] = (m, max(s, MIN_STD[k]))
                else:
                    self.stats[k] = (m0, s0)

    def z(self, key: str, value):
        if value is None:
            return None
        m, s = self.stats.get(key) or PRIORS[key]
        return (value - m) / s


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


class ChannelBuilder:
    SMOOTH = 3.0      # seconds of averaging for the noisy signals
    PACE_NOW = 10.0   # seconds of words behind the short "pace right now" (used by the panic fast lane)
    MIN_FRESH = 6.0   # after a restart, a rolling measure looks back at least this far (less is too noisy)

    def __init__(self, cfg: Config, signals: Signals):
        self.cfg, self.sig = cfg, signals
        self.baseline = Baseline(cfg.calibration_seconds)
        self.started = time.time()
        self.last_tick = None
        self.fresh_from = {"fillers": -math.inf, "pace": -math.inf}

    def restart(self, kinds, t: float):
        """Start the slow rolling measures ("fillers", "pace") again from time t (a tip was just shown).

        Fillers per minute and pace are counted over the last 30 s. Without a restart, the fillers you said
        *before* a filler tip keep the number high for 30 s after you stop, so the coach thinks the tip did
        not help and escalates. After a restart they only count what you said since the tip (from 6 s on, the same
        moment the coach starts judging the tip).
        """
        for k in kinds or ():
            if k in self.fresh_from:
                self.fresh_from[k] = t

    def _heard(self, t0: float, t1: float) -> float:
        """Seconds of [t0, t1] the mic was listening (not muted while a spoken tip played)."""
        spans = list(self.sig.values("muted", 120.0, t1 + 5))
        if getattr(self.sig, "mute_since", None) is not None:
            spans.append((self.sig.mute_since, t1))
        deaf = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in spans)
        return max(1.0, (t1 - t0) - deaf)

    def _span(self, kind: str, window: float, now: float) -> float:
        return max(1.0, min(window, now - self.started, max(self.MIN_FRESH, now - self.fresh_from[kind])))

    def raw(self, now: float) -> dict:
        s, c, w = self.sig, self.cfg, self.SMOOTH
        elapsed = max(1.0, now - self.started)
        pace_win = self._span("pace", c.pace_window, now)
        filler_win = self._span("fillers", c.filler_window, now)
        n_words = len([x for x in s.values("word", pace_win, now) if x not in c.fillers])
        speaking_share = s.mean("speaking", pace_win, now)
        # pace right now: the last 10 s of words, ending at the newest word. Words arrive a few seconds late
        # (speech-to-text runs after each phrase), so ending "now" would undercount; no recent word = no value.
        recent = [t for t, x in s.window("word", 30.0, now) if x not in c.fillers]
        wpm_now = None
        if recent and now - recent[-1] <= 8.0:
            wpm_now = (sum(1 for t in recent if t > recent[-1] - self.PACE_NOW)
                       / (self._heard(recent[-1] - self.PACE_NOW, recent[-1]) / 60.0))
        emos = s.values("face_emotions", w, now)
        face_emotions = {k: _mean([e[k] for e in emos]) for k in emos[0]} if emos else None
        return {
            # voice
            "voice_arousal": s.mean("voice_arousal", 4.0, now),
            "voice_dominance": s.mean("voice_dominance", 4.0, now),
            "voice_valence": s.mean("voice_valence", 4.0, now),
            "pitch_st": s.mean("pitch_st", w, now),
            "pitch_var": s.mean("pitch_var", 6.0, now),
            "volume_db": s.mean("volume_db", w, now),
            "wpm": (n_words / (self._heard(now - pace_win, now) / 60.0)) if speaking_share else None,
            "wpm_now": wpm_now,
            "talk_ratio": s.mean("speaking", 10.0, now),
            "silence_now": s.latest("silence_now", 1.0, now),
            "fillers_per_min": len(s.values("filler", filler_win, now)) / (self._heard(now - filler_win, now) / 60.0),
            "soft_fillers_30s": len(s.values("soft_filler", c.filler_window, now)),
            "pauses_30s": len(s.values("pause", 30.0, now)),
            "last_transcript": s.latest("transcript", 15.0, now),
            # face
            "face_present": s.mean("face_present", w, now),
            "face_arousal": s.mean("face_arousal", w, now),
            "face_valence": s.mean("face_valence", w, now),
            "face_emotions": face_emotions,
            "looking": s.mean("looking", c.eye_contact_window, now),
            "looking_down": s.mean("looking_down", c.eye_contact_window, now),
            "smile": s.mean("smile", w, now),
            "brow_down": s.mean("brow_down", w, now),
            "blinks_per_min": len(s.values("blink", min(30.0, elapsed), now)) / (min(30.0, elapsed) / 60.0),
        }

    def tick(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        dt = 0.0 if self.last_tick is None else now - self.last_tick
        self.last_tick = now
        r = self.raw(now)
        if r["talk_ratio"] and r["talk_ratio"] > 0.5:
            self.baseline.update(r, dt)
        b = self.baseline

        def up(key):     # higher than your normal -> towards 1
            z = b.z(key, r[key])
            return None if z is None else squash(z)

        def down(key):   # lower than your normal -> towards 1
            z = b.z(key, r[key])
            return None if z is None else squash(-z)

        components = {
            "stress": {"voice tense (arousal up)": up("voice_arousal"),
                       "voice negative (valence down)": down("voice_valence"),
                       "face tense (arousal up)": up("face_arousal"),
                       "face negative (valence down)": down("face_valence"),
                       "pitch higher than usual": up("pitch_st")},
            "confidence": {"voice dominant": up("voice_dominance"),
                           "louder than usual": up("volume_db"),
                           "eye contact": r["looking"],
                           "pitch not raised": down("pitch_st")},
            "positivity": {"face positive": up("face_valence"),
                           "smiling": None if r["smile"] is None else min(1.0, r["smile"] * 2),
                           "voice positive": up("voice_valence")},
            "energy": {"voice aroused": up("voice_arousal"),
                       "pitch varied": up("pitch_var"),
                       "louder than usual": up("volume_db")},
        }
        channels = {name: _mean(parts.values()) for name, parts in components.items()}
        channels.update({
            "eye_contact": r["looking"],
            "expressiveness": up("pitch_var"),      # low = monotone
            "pace": up("wpm"),                       # 0.5 = your normal pace
            "pace_wpm": r["wpm"],
            "pace_now": None if r["wpm_now"] is None else squash(b.z("wpm", r["wpm_now"])),  # 10 s, same scale
            "pace_now_wpm": r["wpm_now"],
            "fillers_per_min": r["fillers_per_min"],
            "silence_now_s": r["silence_now"],
            "talk_ratio": r["talk_ratio"],
        })
        top_emotion = max(r["face_emotions"].items(), key=lambda kv: kv[1]) if r["face_emotions"] else None
        return {
            "type": "tick",
            "t": round(now, 2),
            "session_s": round(now - self.started, 1),
            "calibrated": b.ready,
            "calibration_progress": min(1.0, b.speech_seconds / b.target_seconds),
            "channels": {k: (None if v is None else round(v, 3)) for k, v in channels.items()},
            "why": {name: {k: round(v, 2) for k, v in parts.items() if v is not None}
                    for name, parts in components.items()},
            "face_top_emotion": top_emotion and [top_emotion[0], round(top_emotion[1], 2)],
            "transcript": r["last_transcript"],
            "raw": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                    if k not in ("face_emotions", "last_transcript")},
        }
