"""Good news: notices when things get better, not only when they go wrong.

Two kinds of `progress` events (no tip, just a green pop-up, a sound and some XP):

  recovered   a tip worked, and 30 s later you are *still* better than before the tip
              -> "Still better 30 s after slow breathing: stress 0.74 -> 0.52"
  improving   one of your signals got better block after block over the last minute,
              with no tip involved -> "Getting better: eye contact 41% -> 58% -> 77% over 60 s"

Both say what was measured, so the "why" stays visible. Cheap: averages over snapshots the
detector already keeps. Thresholds at the top.
"""
from .knowledge import STRATEGY_NAMES
from .patterns import PATTERNS

HOLD_S = 30             # "still better" this long after a tip that worked
IMPROVE_BLOCK = 20      # improving = 3 blocks of 20 s, each better than the one before ...
IMPROVE_MIN = 0.10      # ... and better by at least this much in total
IMPROVE_EVERY = 75      # at most one "improving" message per 75 s
# signal, how to say it, higher is better?, show as percent?
WATCH = [("stress", "stress", False, False), ("eye_contact", "eye contact", True, True),
         ("confidence", "confidence", True, False), ("energy", "energy", True, False),
         ("expressiveness", "voice variety", True, False)]


class ProgressWatcher:
    def __init__(self, detector, metric: dict):
        self.det = detector
        self.metric = metric
        self.watching = []          # tips that worked, waiting for the HOLD_S check
        self.last_improve_t = -1e9

    def tip_result(self, res: dict, before_badness: float):
        if res.get("worked") and res["pattern"] in PATTERNS:
            self.watching.append({"pattern": res["pattern"], "strategy": res["strategy"], "t": res["t"],
                                  "before": before_badness})

    def _badness(self, pid, seconds):
        f = PATTERNS[pid].badness
        return self.det.mean(f, seconds)

    def update(self, tick: dict) -> list[dict]:
        if not tick.get("calibrated") or not self.det.ticks:
            return []
        now, out, still = tick["t"], [], []
        for w in self.watching:
            if now - w["t"] < HOLD_S:
                still.append(w)
                continue
            after = self._badness(w["pattern"], 10)
            pdef = PATTERNS[w["pattern"]]
            if after is not None and w["before"] - after >= pdef.min_gain / 2:
                name, flip = self.metric.get(w["pattern"], (pdef.label, False))
                b, a = (1 - w["before"], 1 - after) if flip else (w["before"], after)
                strat = STRATEGY_NAMES.get(w["strategy"], w["strategy"])
                if PATTERNS[w["pattern"]].family == "freeze" or w["pattern"] == "freeze":
                    text = f"Back in flow: talking steadily {HOLD_S} s after {strat}"
                else:
                    text = f"Still better {HOLD_S} s after {strat}: {name} {b:.2f} -> {a:.2f}"
                out.append({"type": "progress", "kind": "recovered", "t": now, "pattern": w["pattern"],
                            "text": text, "strategy": w["strategy"],
                            "why": f"{name} stayed better for {HOLD_S} s after the tip, not just for a moment"})
        self.watching = still
        talks = [(x["channels"].get("talk_ratio") or 0) for x in self.det.window(3 * IMPROVE_BLOCK)]
        talked = bool(talks) and sum(t >= 0.5 for t in talks) >= 0.8 * len(talks)  # a break ending is not progress
        if now - self.last_improve_t >= IMPROVE_EVERY and not self.det.active and talked:
            for key, name, higher_good, pct in WATCH:
                bs = []
                for i in reversed(range(3)):
                    vals = [x["channels"].get(key) for x in self.det.window(IMPROVE_BLOCK, end_ago=i * IMPROVE_BLOCK)]
                    vals = [v for v in vals if v is not None]
                    bs.append(sum(vals) / len(vals) if len(vals) >= 0.6 * IMPROVE_BLOCK else None)
                if None in bs:
                    continue
                g = bs if higher_good else [-b for b in bs]
                if all(b > a for a, b in zip(g, g[1:])) and g[-1] - g[0] >= IMPROVE_MIN:
                    fmt = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.2f}")
                    span = 3 * IMPROVE_BLOCK
                    out.append({"type": "progress", "kind": "improving", "t": now, "signal": key,
                                "text": f"Getting better: {name} {' -> '.join(fmt(b) for b in bs)} over {span} s",
                                "why": f"{name} improved in each of the last three {IMPROVE_BLOCK} s blocks"})
                    self.last_improve_t = now
                    break
        return out
