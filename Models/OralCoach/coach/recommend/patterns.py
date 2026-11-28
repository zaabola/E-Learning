"""Temporal pattern detector: turns the 1-second snapshots into "what is going on" events.

A pattern is not a single bad value, it is a value that *lasts* or *changes*:
"stressed for 20 s", "was confident, then dropped", "voice tense while the face looks calm".
Each pattern also defines a `badness` number, so the coach can later check whether a tip
made it go down (the "did it help?" check behind IT WORKED / escalation).

All thresholds are at the top of this file. Channels are 0..1 with 0.5 = your normal.
"""
from collections import deque
from dataclasses import dataclass, field

# --- thresholds ------------------------------------------------------------------------------
STRESS_HIGH = 0.66
STRESS_HOLD = 15          # seconds
CONF_DROP = 0.15          # confidence drop that counts as "dropped"
PACE_FAST = 0.75
PACE_FAST_NERVOUS = 0.65
VOICE_TENSE = 0.68        # cross-modal: voice side of stress
FACE_CALM = 0.58        # 0.5 = your normal face
FACE_TENSE = 0.70
VOICE_CALM = 0.58
EYE_LOW = 0.35
FILLERS_HIGH = 6.0        # per minute
FLAT_VOICE = 0.25
ENERGY_LOW = 0.32
FREEZE_SECONDS = 4.0
BREAK_SECONDS = 30.0      # silent this long = taking a break, not a freeze (no more freeze tips)
MIN_FRACTION = 0.8        # share of the window in which the condition must hold

VOICE_STRESS_PARTS = ("voice tense (arousal up)", "voice negative (valence down)", "pitch higher than usual")
FACE_STRESS_PARTS = ("face tense (arousal up)", "face negative (valence down)")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _f(v, spec=".2f"):
    return "?" if v is None else format(v, spec)


def voice_stress(tick):
    return _mean([tick["why"].get("stress", {}).get(k) for k in VOICE_STRESS_PARTS])


def face_stress(tick):
    return _mean([tick["why"].get("stress", {}).get(k) for k in FACE_STRESS_PARTS])


def ch(key):
    return lambda tick: tick["channels"].get(key)


def inv(key):
    def f(tick):
        v = tick["channels"].get(key)
        return None if v is None else 1.0 - v
    return f


def fillers(tick):
    v = tick["channels"].get("fillers_per_min")
    return None if v is None else min(1.0, v / 10.0)


def silence(tick):
    v = tick["channels"].get("silence_now_s")
    return 0.0 if v is None else min(1.0, v / 6.0)


@dataclass
class PatternDef:
    id: str
    label: str            # shown to the student / teacher
    query: str            # words used to retrieve advice
    badness: callable     # tick -> 0..1, higher = worse; the "did it help?" check reads this
    min_gain: float       # how much badness must drop for a tip to count as "worked"
    priority: int         # higher wins when several patterns are active
    modality: str         # face / voice / face+voice / words
    family: str = ""      # fast-lane patterns: the normal pattern they are an early warning of (urgent.py)
    urgent: bool = False  # answer right away, skip the normal cooldown
    check_s: float = 0.0  # "did it help?" window in seconds (0 = the normal one from config)
    min_level: int = 1    # start at least at this escalation level
    restarts: tuple = ()  # slow rolling measures to restart when a tip on this is shown (ChannelBuilder.restart)


PATTERNS = {p.id: p for p in [
    PatternDef("freeze", "You froze", "freeze blank mind silence lost restart recover breathe",
               silence, 0.3, 10, "voice"),
    PatternDef("sustained_stress", "Stressed for a while", "anxiety nervous stress calm breathing arousal",
               ch("stress"), 0.06, 8, "face+voice"),
    PatternDef("rushing_nervous", "Speeding up while nervous", "fast rate rushing nervous slow down pause breathe",
               ch("pace"), 0.06, 8, "face+voice", restarts=("pace",)),
    PatternDef("masked_tension", "Voice tense, face calm", "voice tense pitch tight breath relax throat calm",
               voice_stress, 0.06, 8, "face+voice"),
    PatternDef("confidence_drop", "Confidence dropped", "confidence drop doubt posture eye contact volume reframe",
               inv("confidence"), 0.06, 7, "face+voice"),
    PatternDef("eye_contact_low", "Not looking at the audience", "eye contact audience notes look up",
               inv("eye_contact"), 0.15, 6, "face"),
    PatternDef("filler_burst", "Lots of filler words", "filler um uh vocalized pause silent pause",
               fillers, 0.08, 5, "words", restarts=("fillers",)),
    PatternDef("rushing", "Talking fast", "fast rate rushing slow down pause",
               ch("pace"), 0.06, 5, "voice", restarts=("pace",)),
    PatternDef("face_tense", "Face tense, voice fine", "facial expression tense jaw smile relax face",
               face_stress, 0.06, 5, "face+voice"),
    PatternDef("monotone", "Flat, monotone voice", "monotone vocal variety pitch emphasis expressive",
               inv("expressiveness"), 0.08, 4, "voice"),
    PatternDef("low_energy", "Low energy", "energy volume enthusiasm projection gesture",
               inv("energy"), 0.06, 4, "voice"),
]}


EXTRA_CHECKS = {}  # pattern id -> fn(detector, tick); filled by urgent.py


@dataclass
class Active:
    id: str
    since: float
    evidence: list = field(default_factory=list)
    off_ticks: int = 0


class PatternDetector:
    """Feed one snapshot per second with update(); read .active; get start/end events back."""

    def __init__(self, history_seconds: float = 120.0):
        self.ticks = deque()
        self.keep = history_seconds
        self.active: dict[str, Active] = {}

    # --- window helpers -----------------------------------------------------------------------
    def window(self, seconds: float, end_ago: float = 0.0):
        now = self.ticks[-1]["t"]
        return [x for x in self.ticks if now - seconds - end_ago < x["t"] <= now - end_ago]

    def held(self, fn, seconds: float, frac: float = MIN_FRACTION):
        """True if fn(tick) is True in at least `frac` of the last `seconds` (with enough data)."""
        w = self.window(seconds)
        vals = [fn(x) for x in w]
        vals = [v for v in vals if v is not None]
        if len(vals) < 0.7 * seconds or not fn(self.ticks[-1]):  # must also be true right now
            return False
        return sum(1 for v in vals if v) / len(vals) >= frac

    def held_for(self, fn, max_seconds: float = 90.0) -> int:
        """How many seconds back the condition has been (mostly) true, for the evidence text."""
        n = 0
        for x in reversed(self.ticks):
            v = fn(x)
            if v is None or not v:
                break
            n += 1
            if n >= max_seconds:
                break
        return n

    def mean(self, fn, seconds: float, end_ago: float = 0.0):
        return _mean([fn(x) for x in self.window(seconds, end_ago)])

    @staticmethod
    def top_parts(tick, channel: str, n: int = 2, high: bool = True) -> str:
        parts = tick["why"].get(channel, {})
        items = sorted(parts.items(), key=lambda kv: kv[1], reverse=high)[:n]
        return ", ".join(f"{k} {v:.2f}" for k, v in items)

    # --- the rules ----------------------------------------------------------------------------
    def check(self, pid: str, t: dict):
        """Return evidence (list of short strings) if pattern `pid` is happening now, else None."""
        c = t["channels"]
        if pid in EXTRA_CHECKS:
            return EXTRA_CHECKS[pid](self, t)

        def chv(k):
            return lambda x: (x["channels"].get(k))

        if pid == "freeze":
            sil = c.get("silence_now_s") or 0.0
            talked_before = self.mean(chv("talk_ratio"), 15, end_ago=sil)
            if FREEZE_SECONDS <= sil < BREAK_SECONDS and talked_before is not None and talked_before > 0.4:
                return [f"silent for {sil:.0f} s right after speaking"]
            return None

        if pid == "sustained_stress":
            f = lambda x: None if x["channels"].get("stress") is None else x["channels"]["stress"] >= STRESS_HIGH  # noqa: E731
            masked_now = (voice_stress(t) or 0) >= VOICE_TENSE and (face_stress(t) if face_stress(t) is not None else 1) <= FACE_CALM
            if self.held(f, STRESS_HOLD) and not masked_now:  # voice-only stress is the more specific masked_tension
                return [f"stress {_f(c.get('stress'))} for {self.held_for(f)} s (0.5 = your normal)",
                        f"mostly from: {self.top_parts(t, 'stress')}"]
            return None

        if pid == "rushing_nervous":
            def f(x):
                p, s = x["channels"].get("pace"), x["channels"].get("stress")
                return None if p is None or s is None else (p >= PACE_FAST_NERVOUS and s >= 0.6)
            if self.held(f, 10):
                return [f"pace {c.get('pace_wpm') or 0:.0f} wpm (faster than your normal) while stress is {_f(c.get('stress'))}",
                        f"for {self.held_for(f)} s"]
            return None

        if pid == "masked_tension":
            def f(x):
                v, fa = voice_stress(x), face_stress(x)
                return None if v is None or fa is None else (v >= VOICE_TENSE and fa <= FACE_CALM)
            if self.held(f, 10):
                return [f"voice tense {_f(voice_stress(t))} but face calm {_f(face_stress(t))}",
                        f"voice: {self.top_parts(t, 'stress', 3)}"]
            return None

        if pid == "face_tense":
            def f(x):
                v, fa = voice_stress(x), face_stress(x)
                return None if v is None or fa is None else (fa >= FACE_TENSE and v <= VOICE_CALM)
            if self.held(f, 12):
                return [f"face tense {_f(face_stress(t))} but voice relaxed {_f(voice_stress(t))}"]
            return None

        if pid == "confidence_drop":
            before = self.mean(chv("confidence"), 25, end_ago=20)
            recent = self.mean(chv("confidence"), 8)
            if before is not None and recent is not None and before >= 0.45 and recent <= before - CONF_DROP:
                old = self.window(25, end_ago=20)
                drops = []
                if old:
                    for k, v in t["why"].get("confidence", {}).items():
                        b = _mean([x["why"].get("confidence", {}).get(k) for x in old])
                        if b is not None:
                            drops.append((b - v, k, b, v))
                drops.sort(reverse=True)
                ev = [f"confidence fell from {before:.2f} to {recent:.2f} in about 20 s"]
                if drops and drops[0][0] > 0.05:
                    _, k, b, v = drops[0]
                    ev.append(f"biggest drop: {k} {b:.2f} -> {v:.2f}")
                return ev
            return None

        if pid == "eye_contact_low":
            f = lambda x: None if x["channels"].get("eye_contact") is None else x["channels"]["eye_contact"] <= EYE_LOW  # noqa: E731
            if self.held(f, 12):
                ev = [f"looking at the camera only {_f(c.get('eye_contact'), '.0%')} of the time, for {self.held_for(f)} s"]
                down = t.get("raw", {}).get("looking_down")
                if down is not None and down > 0.4:
                    ev.append("mostly looking down (notes?)")
                return ev
            return None

        if pid == "filler_burst":
            fpm = c.get("fillers_per_min")
            if fpm is not None and fpm >= FILLERS_HIGH and self.held(
                    lambda x: (x["channels"].get("fillers_per_min") or 0) >= FILLERS_HIGH, 8):
                return [f"{fpm:.1f} filler words per minute (um, uh...)"]
            return None

        if pid == "rushing":
            f = lambda x: None if x["channels"].get("pace") is None else x["channels"]["pace"] >= PACE_FAST  # noqa: E731
            if "rushing_nervous" not in self.active and self.held(f, 15):
                return [f"pace {c.get('pace_wpm') or 0:.0f} wpm, faster than your normal for {self.held_for(f)} s"]
            return None

        if pid == "monotone":
            f = lambda x: None if x["channels"].get("expressiveness") is None else x["channels"]["expressiveness"] <= FLAT_VOICE  # noqa: E731
            if self.held(f, 20):
                return [f"voice variety {_f(c.get('expressiveness'))} (pitch barely moves) for {self.held_for(f)} s"]
            return None

        if pid == "low_energy":
            def f(x):
                e, s = x["channels"].get("energy"), x["channels"].get("stress")
                return None if e is None else (e <= ENERGY_LOW and (s is None or s < 0.6))
            if self.held(f, 20):
                return [f"energy {_f(c.get('energy'))} for {self.held_for(f)} s",
                        f"from: {self.top_parts(t, 'energy', 2, high=False)}"]
            return None
        return None

    def update(self, tick: dict) -> list[dict]:
        self.ticks.append(tick)
        while self.ticks and self.ticks[0]["t"] < tick["t"] - self.keep:
            self.ticks.popleft()
        events = []
        if not tick.get("calibrated"):
            return events  # no coaching while we learn your normal
        for pid, pdef in PATTERNS.items():
            ev = self.check(pid, tick)
            a = self.active.get(pid)
            if ev:
                if a is None:
                    self.active[pid] = Active(pid, tick["t"], ev)
                    events.append({"type": "pattern_start", "t": tick["t"], "pattern": pid,
                                   "label": pdef.label, "modality": pdef.modality, "evidence": ev})
                else:
                    a.evidence, a.off_ticks = ev, 0
            elif a is not None:
                a.off_ticks += 1
                if a.off_ticks >= 4:  # a few quiet seconds before we call it over
                    del self.active[pid]
                    events.append({"type": "pattern_end", "t": tick["t"], "pattern": pid,
                                   "label": pdef.label, "duration_s": round(tick["t"] - a.since, 1)})
        return events
