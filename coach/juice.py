"""Game feel: turns the coach's events into sounds, XP, pop-ups and flashes for the window.

It only *listens* to the same snapshots and events that are logged / printed as JSON, and never changes them.
XP and levels are a display-only reward for the window (they are not logged), so the future game is free to
use its own scoring on the same events.

    feel = GameFeel(Sfx())
    feel.update(snap, events, coach_view, now)   # once per tick
    feel.typed(tip_id, n_chars)                  # every frame while a tip is being "typed" (text blips)
"""
import math
import time

# channel, label, high is bad (True), low is bad (False), both extremes bad (None)
ROWS = [
    ("stress", "Stress", True),
    ("confidence", "Confidence", False),
    ("eye_contact", "Eye contact", False),
    ("pace", "Pace", None),
    ("fillers", "Fillers", True),
    ("energy", "Energy", False),
    ("positivity", "Positivity", False),
    ("expressiveness", "Voice variety", False),
]
FILLERS_FULL = 10.0  # fillers/min that fill the bar
SHORT = {"stress": "STRESS", "confidence": "CONFIDENCE", "eye_contact": "EYE CONTACT", "pace": "PACE",
         "fillers": "FILLERS", "energy": "ENERGY", "positivity": "POSITIVITY", "expressiveness": "INTONATION"}

# XP for the window (display only)
XP_TIP_WORKED = 50
XP_PRAISE = 30
XP_CALIBRATED = 20
XP_STEADY_PER_10S = 5


def zone(v: float, high_bad: bool | None) -> str:
    """'good' / 'warn' / 'bad' for a bar value, same rule as the bar colours."""
    if high_bad is None:  # pace: 0.5 = your normal
        d = abs(v - 0.5)
        return "bad" if d > 0.3 else "warn" if d > 0.2 else "good"
    if high_bad:
        return "bad" if v > 0.7 else "warn" if v > 0.5 else "good"
    return "bad" if v < 0.2 else "warn" if v < 0.35 else "good"


def row_value(key: str, ch: dict) -> tuple[float | None, str]:
    if key == "fillers":
        f = ch.get("fillers_per_min")
        return (None, "") if f is None else (min(1.0, f / FILLERS_FULL), f"{f:.1f}/m")
    v = ch.get(key)
    if v is None:
        return None, ""
    if key == "pace" and ch.get("pace_wpm") is not None:
        return v, f"{ch['pace_wpm']:.0f}wpm"
    if key == "eye_contact":
        return v, f"{v:.0%}"
    return v, f"{v:.2f}"


def xp_needed(level: int) -> int:
    return 60 + 40 * level


class GameFeel:
    def __init__(self, sfx=None):
        self.sfx = sfx
        self.xp, self.level = 0, 1
        self.popups: list[dict] = []     # {"text", "kind", "t"}  floating "+50 XP" etc.
        self.flash = None                # {"kind", "t"}  short full-screen tint
        self.levelup_t = -math.inf
        self.shake_t = -math.inf
        self.result_banner = None        # {"worked", "text", "t"}
        self._zones: dict[str, str] = {}
        self._bad_since: dict[str, float] = {}
        self._last_warn = -math.inf
        self._was_calibrated = None
        self._steady_s = 0.0
        self._typed = {}                 # tip id -> chars already blipped
        self.started = False

    # --- sounds ------------------------------------------------------------------------------
    def _play(self, name, delay=0.0, gain=1.0):
        if self.sfx is not None:
            self.sfx.play(name, delay, gain)

    def start(self):
        if not self.started:
            self.started = True
            self._play("boot")

    def stop(self):
        self._play("quit")

    def key(self):
        self._play("menu")

    def typed(self, tip_id: str, n_chars: int, text: str):
        """Called every frame with how many characters of the tip are visible: one blip every 2 letters."""
        done = self._typed.get(tip_id, 0)
        if n_chars >= done + 2 or (n_chars == len(text) and n_chars > done):
            if text[max(0, n_chars - 1)] != " " and self.sfx is not None:
                self.sfx.blip()
            self._typed[tip_id] = n_chars

    # --- XP ----------------------------------------------------------------------------------
    def gain(self, amount: int, now: float, why: str = "", sound: bool = True):
        self.xp += amount
        if sound:
            self.popups.append({"text": f"+{amount} XP" + (f" {why}" if why else ""), "kind": "xp", "t": now})
            self._play("xp", delay=0.15)
        while self.xp >= xp_needed(self.level):
            self.xp -= xp_needed(self.level)
            self.level += 1
            self.levelup_t = now
            self.flash = {"kind": "level", "t": now}
            self._play("levelup", delay=0.55)

    def xp_frac(self) -> float:
        return self.xp / xp_needed(self.level)

    # --- once per tick -----------------------------------------------------------------------
    def update(self, snap: dict | None, events: list[dict], coach: dict | None = None, now: float | None = None):
        now = time.time() if now is None else now
        self.popups = [p for p in self.popups if now - p["t"] < 2.5]
        for ev in events:
            kind = ev.get("type")
            if kind == "tip":
                lvl = ev.get("level", 1)
                self._play("freeze" if ev.get("urgent") else f"tip{min(3, max(1, lvl))}")
                if ev.get("urgent"):
                    self._play("tip2", delay=0.45)
            elif kind == "tip_result":
                self.result_banner = {"worked": ev["worked"], "text": ev.get("message", ""), "t": now}
                if ev["worked"]:
                    self._play("worked")
                    self.flash = {"kind": "good", "t": now}
                    self.gain(XP_TIP_WORKED + 10 * (ev.get("level", 1) - 1), now, "tip worked")
                else:
                    self._play("not_yet")
                    self.shake_t = now
            elif kind == "praise":
                self._play("praise")
                self.flash = {"kind": "praise", "t": now}
                self.gain(XP_PRAISE, now, "steady")
            elif kind == "progress":  # recovered after a tip / improving over a minute
                self._play("recover")
                self.flash = {"kind": "good", "t": now}
                self.gain(XP_PRAISE, now, "better")
        if snap is None:
            return
        cal = bool(snap.get("calibrated"))
        if self._was_calibrated is False and cal:
            self._play("calibrated")
            self.popups.append({"text": "CALIBRATED!", "kind": "info", "t": now})
            self.gain(XP_CALIBRATED, now, sound=False)
        self._was_calibrated = cal
        if not cal:
            return
        # quiet XP trickle while nothing is wrong
        talking = (snap.get("channels", {}).get("talk_ratio") or 0) >= 0.5
        if coach is not None and not coach.get("active") and talking:  # no XP for a silent break
            self._steady_s += 1
            if self._steady_s % 10 == 0:
                self.gain(XP_STEADY_PER_10S, now, sound=False)
        # a bar going into the red: one soft warning (at most every 8 s, and only if it stays red 2 s)
        ch = snap.get("channels", {})
        for key, _, high_bad in ROWS:
            v, _ = row_value(key, ch)
            if v is None:
                continue
            z = zone(v, high_bad)
            prev = self._zones.get(key)
            if z == "bad":
                self._bad_since.setdefault(key, now)
                if prev != "bad_warned" and now - self._bad_since[key] >= 2 and now - self._last_warn >= 8:
                    self._play("warn")
                    self._last_warn = now
                    z = "bad_warned"
                elif prev == "bad_warned":
                    z = "bad_warned"
            else:
                self._bad_since.pop(key, None)
                if prev == "bad_warned" and z == "good":
                    self._play("recover", gain=0.8)
            self._zones[key] = z

    def red_rows(self) -> set[str]:
        return {k for k, z in self._zones.items() if z.startswith("bad")}
