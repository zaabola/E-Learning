"""Fast lane: sharp moments the coach answers right away instead of waiting for its normal pace.

The normal patterns (patterns.py) wait until something has *lasted* 10-20 s, and tips are at least
25 s apart. That calm pace is right most of the time, but some moments cannot wait:

  stress_spike   stress jumps up within a few seconds         (normally: "stressed for 15 s")
  panic          very stressed AND suddenly speeding up       (normally: "rushing while nervous" for 10 s)
  lost_audience  good eye contact, then suddenly looking away (normally: "low eye contact" for 12 s)
  still_frozen   still silent several seconds after the freeze tip -> go straight to "pause" level

Each fast pattern belongs to a *family* (the normal pattern it is an early warning of). The family shares
cards and escalation: if the fast tip on a stress spike did not help, the normal
"Stressed for a while" tip that comes later is escalated to the next level, and says why.

The existing "You froze" pattern (priority 10) was already on this fast lane; the priorities below put the
new ones around it (still frozen > freeze = panic > stress spike = suddenly looking away).

Fast tips skip the 25 s cooldown (they only need URGENT_GAP s since the last urgent tip), skip the LLM
(card wording, instantly), and are checked sooner (`check_s` instead of 20 s).
All the numbers are here, at the top.
"""
from .patterns import EXTRA_CHECKS, PATTERNS, PatternDef, ch, inv, silence

# --- thresholds ------------------------------------------------------------------------------
NOW_S = 3              # "right now" = the last 3 s ...
BEFORE_S = 15          # ... compared with the 15 s before that
SPIKE_LEVEL = 0.70     # stress must be at least this high now (0.5 = your normal) ...
SPIKE_JUMP = 0.18      # ... and have jumped by at least this much
PANIC_STRESS = 0.80
PANIC_PACE = 0.70
PANIC_PACE_JUMP = 0.15
EYE_WAS_GOOD = 0.60    # eye contact before ...
EYE_NOW_GONE = 0.20    # ... and now (share of the last 5 s looking at the camera)
STILL_FROZEN_S = 8.0   # silence that turns a freeze into "still frozen"
URGENT_GAP = 8.0       # min seconds between two urgent tips (only a freeze may interrupt sooner)


def _chv(k):
    return lambda x: x["channels"].get(k)


def _now_and_before(det, key):
    return det.mean(_chv(key), NOW_S), det.mean(_chv(key), BEFORE_S, end_ago=NOW_S + 1)


def check_stress_spike(det, t):
    now, before = _now_and_before(det, "stress")
    if now is None or before is None or len(det.window(BEFORE_S + NOW_S)) < BEFORE_S:
        return None
    if now >= SPIKE_LEVEL and now - before >= SPIKE_JUMP:
        return [f"stress jumped from {before:.2f} to {now:.2f} in a few seconds",
                f"mostly from: {det.top_parts(t, 'stress')}"]
    return None


def check_panic(det, t):
    s_now, _ = _now_and_before(det, "stress")
    p_now, p_before = _now_and_before(det, "pace_now")  # short 10 s pace: the 30 s one hides a sudden jump
    if None in (s_now, p_now, p_before):
        return None
    if s_now >= PANIC_STRESS and p_now >= PANIC_PACE and p_now - p_before >= PANIC_PACE_JUMP:
        wpm = t["channels"].get("pace_now_wpm") or 0
        return [f"very stressed ({s_now:.2f}) and suddenly faster (pace {p_before:.2f} -> {p_now:.2f}, {wpm:.0f} wpm)"]
    return None


def check_lost_audience(det, t):
    if (t["channels"].get("silence_now_s") or 0) >= 2:
        return None  # looking away while silent is part of a freeze, not a separate problem
    now, before = _now_and_before(det, "eye_contact")
    if now is None or before is None:
        return None
    if before >= EYE_WAS_GOOD and now <= EYE_NOW_GONE:
        ev = [f"eye contact fell from {before:.0%} to {now:.0%} within seconds"]
        down = t.get("raw", {}).get("looking_down")
        if down is not None and down > 0.4:
            ev.append("looking down (notes?)")
        return ev
    return None


def check_still_frozen(det, t):
    sil = t["channels"].get("silence_now_s") or 0.0
    if sil >= STILL_FROZEN_S and "freeze" in det.active:
        return [f"still silent after {sil:.0f} s"]
    return None


URGENT = {p.id: p for p in [
    PatternDef("stress_spike", "Stress spike", "anxiety nervous stress calm breathing arousal sudden",
               ch("stress"), 0.06, 9, "face+voice",
               family="sustained_stress", urgent=True, check_s=10),
    PatternDef("panic", "Panic: stressed and speeding up", "fast rate rushing nervous slow down pause breathe",
               ch("pace_now"), 0.06, 10, "face+voice",
               family="rushing_nervous", urgent=True, check_s=10),
    PatternDef("lost_audience", "Suddenly looking away", "eye contact audience notes look up",
               inv("eye_contact"), 0.15, 9, "face",
               family="eye_contact_low", urgent=True, check_s=8),
    PatternDef("still_frozen", "Still frozen", "freeze blank mind silence lost restart recover breathe",
               silence, 0.3, 12, "voice",
               family="freeze", urgent=True, check_s=10, min_level=3),
]}

PATTERNS.update(URGENT)
EXTRA_CHECKS.update({
    "stress_spike": check_stress_spike,
    "panic": check_panic,
    "lost_audience": check_lost_audience,
    "still_frozen": check_still_frozen,
})
