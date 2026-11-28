"""More patterns: face+voice combinations and slow changes over the session.

The first patterns (patterns.py) mostly look at one thing at a time for 10-20 s. These look at
two signals together, or at how a signal moves over a minute:

  face + voice
    reading_notes     looking down AND a flat voice          -> reading from your notes
    flat_affect       flat voice AND a blank face            -> you look and sound neutral
    shrinking         stressed AND quieter than usual        -> nerves are making you small

  over time
    fading_energy     energy clearly lower than in your first minute ("strong start, then it fades")
    confidence_slide  confident first minute, then a slow slide down (confidence_drop catches fast drops)
    speeding_up       pace creeping up block after block, before it is "talking fast"
    stress_creep      stress rising block after block, before it is "stressed for a while"

The "over time" ones belong to a family (the older pattern they are an early or slow version of), so
they share cards and escalation with it, exactly like the fast lane does.

Cheap on purpose: every check is a few averages over the last <= 120 one-second snapshots that the
detector already keeps (no models, no extra threads). All thresholds are at the top.
"""
from .patterns import EXTRA_CHECKS, PATTERNS, PatternDef, _f, _mean, ch, inv

# --- thresholds ------------------------------------------------------------------------------
START_S = 60            # "your start" = your first 60 s of talking after calibration
DOWN_LOOK = 0.40        # share of time looking down ...
EYE_LOW_NOTES = 0.40    # ... with little eye contact ...
FLAT_NOTES = 0.40       # ... and a flat voice = reading
FLAT_VOICE = 0.30       # flat voice (pitch variety, 0.5 = your normal) ...
BLANK_FACE = 0.38       # ... and a blank face (positivity: smile + positive face + voice)
SHRINK_STRESS = 0.62    # stressed ...
SHRINK_QUIET = 0.35     # ... and quieter than usual (0.5 = your normal volume)
FADE_DROP = 0.15        # energy this much below your start ...
FADE_START_MIN = 0.45   # ... (only if your start had some energy)
SLIDE_DROP = 0.12       # confidence this much below your start
SLIDE_START_MIN = 0.50
TREND_BLOCK = 15        # trend = 3 blocks of 15 s, each one higher than the one before
TREND_PACE = 0.12       # total rise over the 45 s
TREND_FROM = 0.40       # pace must start near your normal (not a restart after a silence)
TREND_AFTER_S = 75      # pace is averaged over up to 30 s, so it ramps up at the start: no pace trend before this
TREND_STRESS = 0.10
CREEP_MAX = 0.66        # above this, "stressed for a while" takes over


def _chv(k):
    return lambda x: x["channels"].get(k)


def _loud(x):
    return x["why"].get("confidence", {}).get("louder than usual")


def _talking(x):
    return (x["channels"].get("talk_ratio") or 0) > 0.5


# --- "your start": averages of your first minute of talking, kept for the whole session --------
def session_start(det, t):
    """Running means of your first START_S seconds of talking (once per tick, stored on the detector)."""
    ref = det.__dict__.setdefault("_start", {"n": 0, "sum": {}, "cnt": {}, "last_t": None})
    if ref["last_t"] != t["t"] and ref["n"] < START_S and t.get("calibrated") and _talking(t):
        ref["last_t"] = t["t"]
        ref["n"] += 1
        for k in ("energy", "confidence", "stress", "pace"):
            v = t["channels"].get(k)
            if v is not None:
                ref["sum"][k] = ref["sum"].get(k, 0.0) + v
                ref["cnt"][k] = ref["cnt"].get(k, 0) + 1
    if ref["n"] < START_S:
        return None  # still learning your start
    return {k: ref["sum"][k] / ref["cnt"][k] for k in ref["sum"] if ref["cnt"][k] >= START_S // 2}


def blocks(det, fn, n=3, size=TREND_BLOCK):
    """Means of the last n blocks of `size` seconds, oldest first (None if a block has too little data)."""
    out = []
    for i in reversed(range(n)):
        vals = [v for v in (fn(x) for x in det.window(size, end_ago=i * size)) if v is not None]
        out.append(sum(vals) / len(vals) if len(vals) >= 0.6 * size else None)
    return out


def rising(bs, min_total):
    return None not in bs and all(b > a for a, b in zip(bs, bs[1:])) and bs[-1] - bs[0] >= min_total


# --- face + voice ----------------------------------------------------------------------------
def check_reading_notes(det, t):
    def f(x):
        c, down = x["channels"], x.get("raw", {}).get("looking_down")
        if c.get("eye_contact") is None or c.get("expressiveness") is None or down is None:
            return None
        return (_talking(x) and down >= DOWN_LOOK and c["eye_contact"] <= EYE_LOW_NOTES
                and c["expressiveness"] <= FLAT_NOTES)
    if det.held(f, 10):
        c = t["channels"]
        return [f"looking down {t['raw']['looking_down']:.0%} of the time while your voice is flat "
                f"(variety {_f(c.get('expressiveness'))}) for {max(10, det.held_for(f))} s",
                "eyes down + flat voice usually means reading from notes"]
    return None


def check_flat_affect(det, t):
    def f(x):
        c = x["channels"]
        if c.get("expressiveness") is None or c.get("positivity") is None:
            return None
        return (_talking(x) and c["expressiveness"] <= FLAT_VOICE and c["positivity"] <= BLANK_FACE
                and (c.get("stress") or 0) < 0.6)
    if det.held(f, 15):
        c = t["channels"]
        return [f"flat voice (variety {_f(c.get('expressiveness'))}) and a neutral face "
                f"(positivity {_f(c.get('positivity'))}) for {max(15, det.held_for(f))} s",
                f"face: {det.top_parts(t, 'positivity', 2, high=False)}"]
    return None


def check_shrinking(det, t):
    def f(x):
        s, loud = x["channels"].get("stress"), _loud(x)
        return None if s is None or loud is None else (_talking(x) and s >= SHRINK_STRESS and loud <= SHRINK_QUIET)
    if det.held(f, 10):
        return [f"stress {_f(t['channels'].get('stress'))} while your voice got quieter "
                f"(volume {_f(_loud(t))}, 0.5 = your normal) for {max(10, det.held_for(f))} s",
                f"stress mostly from: {det.top_parts(t, 'stress')}"]
    return None


# --- over time --------------------------------------------------------------------------------
def check_fading_energy(det, t):
    start = session_start(det, t)
    if not start or "low_energy" in det.active or start.get("energy", 0) < FADE_START_MIN:
        return None
    last, now = det.mean(_chv("energy"), 25), det.mean(_chv("energy"), 8)
    if last is None or now is None:
        return None
    if last <= start["energy"] - FADE_DROP and now <= start["energy"] - FADE_DROP * 0.8:
        return [f"energy {start['energy']:.2f} in your first minute, {last:.2f} in the last 25 s",
                f"from: {det.top_parts(t, 'energy', 2, high=False)}"]
    return None


def check_confidence_slide(det, t):
    start = session_start(det, t)
    if not start or "confidence_drop" in det.active or start.get("confidence", 0) < SLIDE_START_MIN:
        return None
    last, now = det.mean(_chv("confidence"), 30), det.mean(_chv("confidence"), 10)
    if last is None or now is None:
        return None
    if last <= start["confidence"] - SLIDE_DROP and now <= start["confidence"] - SLIDE_DROP:
        ev = [f"confidence {start['confidence']:.2f} in your first minute, {last:.2f} in the last 30 s "
              "(a slow slide, not a sudden drop)"]
        parts = det.top_parts(t, "confidence", 2, high=False)
        if parts:
            ev.append(f"lowest now: {parts}")
        return ev
    return None


def check_speeding_up(det, t):
    if "rushing" in det.active or "rushing_nervous" in det.active or "panic" in det.active:
        return None
    if (t.get("session_s") or 0) < TREND_AFTER_S:
        return None
    bs = blocks(det, _chv("pace"))
    if rising(bs, TREND_PACE) and bs[0] >= TREND_FROM and (t["channels"].get("pace") or 0) >= 0.6:
        wpm = blocks(det, _chv("pace_wpm"))
        ev = [f"pace {' -> '.join(f'{b:.2f}' for b in bs)} over the last {3 * TREND_BLOCK} s"]
        if None not in wpm:
            ev.append(f"{wpm[0]:.0f} -> {wpm[-1]:.0f} words per minute, a little faster every 15 s")
        return ev
    return None


def check_stress_creep(det, t):
    if "sustained_stress" in det.active or "stress_spike" in det.active:
        return None
    bs = blocks(det, _chv("stress"))
    now = t["channels"].get("stress")
    if now is not None and 0.55 <= now < CREEP_MAX and rising(bs, TREND_STRESS):
        return [f"stress {' -> '.join(f'{b:.2f}' for b in bs)} over the last {3 * TREND_BLOCK} s "
                "(creeping up before it gets big)",
                f"mostly from: {det.top_parts(t, 'stress')}"]
    return None


def _mean_of(*fns):
    return lambda x: _mean([f(x) for f in fns])


MORE = {p.id: p for p in [
    PatternDef("reading_notes", "Reading from your notes", "notes reading look up keywords eye contact outline extemporaneous",
               inv("eye_contact"), 0.12, 7, "face+voice"),
    PatternDef("shrinking", "Stressed and getting quieter", "stress quiet volume project breath voice confident",
               lambda x: None if _loud(x) is None else 1.0 - _loud(x), 0.08, 7, "face+voice"),
    PatternDef("flat_affect", "Flat voice, blank face", "flat expressive face voice gesture smile pitch variety",
               _mean_of(inv("expressiveness"), inv("positivity")), 0.08, 5, "face+voice"),
    PatternDef("fading_energy", "Energy fading since your start", "energy fading end finish strong volume project",
               inv("energy"), 0.06, 5, "voice", family="low_energy"),
    PatternDef("confidence_slide", "Confident start, then a slow slide", "confidence slide volume posture message",
               inv("confidence"), 0.06, 6, "face+voice", family="confidence_drop"),
    PatternDef("speeding_up", "Speeding up little by little", "fast rate speeding slow pause breathe",
               ch("pace"), 0.06, 4, "voice", family="rushing"),
    PatternDef("stress_creep", "Stress creeping up", "stress rising breathing calm exhale early",
               ch("stress"), 0.05, 5, "face+voice", family="sustained_stress"),
]}

# how the coach shows each one's "did it help?" number: (name, show 1 - badness?)
METRIC = {
    "reading_notes": ("eye contact", True), "shrinking": ("volume", True), "flat_affect": ("liveliness", True),
    "fading_energy": ("energy", True), "confidence_slide": ("confidence", True), "speeding_up": ("pace", False),
    "stress_creep": ("stress", False),
}

PATTERNS.update(MORE)
EXTRA_CHECKS.update({
    "reading_notes": check_reading_notes,
    "shrinking": check_shrinking,
    "flat_affect": check_flat_affect,
    "fading_energy": check_fading_energy,
    "confidence_slide": check_confidence_slide,
    "speeding_up": check_speeding_up,
    "stress_creep": check_stress_creep,
})
