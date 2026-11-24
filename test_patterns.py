"""Quick check of the new face+voice and over-time patterns with made-up snapshots (no sensors, < 1 s).

  python test_patterns.py

Each scenario feeds 3 minutes of one-second snapshots to the pattern detector and checks which
patterns start. A calm talk must start none of them.
"""
from coach.recommend import urgent  # noqa: F401  (registers the fast-lane patterns)
from coach.recommend import more_patterns  # noqa: F401
from coach.recommend.patterns import PatternDetector

NEW = set(more_patterns.MORE)


def tick(s, stress=0.5, conf=0.6, energy=0.55, eye=0.8, expr=0.5, pos=0.5, pace=0.5, down=0.0, loud=0.5):
    return {
        "t": 1000.0 + s, "session_s": s, "calibrated": s >= 30,
        "channels": {"stress": stress, "confidence": conf, "energy": energy, "positivity": pos, "eye_contact": eye,
                     "expressiveness": expr, "pace": pace, "pace_wpm": 140 + 100 * (pace - 0.5),
                     "fillers_per_min": 1.0, "silence_now_s": 0.0, "talk_ratio": 0.9},
        "why": {"stress": {"voice tense (arousal up)": stress, "face tense (arousal up)": stress},
                "confidence": {"voice dominant": conf, "louder than usual": loud},
                "positivity": {"face positive": pos, "smiling": pos},
                "energy": {"voice aroused": energy, "pitch varied": expr, "louder than usual": loud}},
        "raw": {"looking_down": down},
    }


def ramp(s, a, b, s0, s1):
    k = min(1.0, max(0.0, (s - s0) / (s1 - s0)))
    return a + (b - a) * k


SCENARIOS = {
    "calm talk": (lambda s: tick(s), set()),
    "reading_notes": (lambda s: tick(s, eye=0.2, down=0.7, expr=0.3) if 100 <= s < 130 else tick(s), {"reading_notes"}),
    "flat_affect": (lambda s: tick(s, expr=0.2, pos=0.25) if 100 <= s < 140 else tick(s), {"flat_affect"}),
    "shrinking": (lambda s: tick(s, stress=0.66, loud=0.25) if 100 <= s < 125 else tick(s), {"shrinking"}),
    "fading_energy": (lambda s: tick(s, energy=ramp(s, 0.6, 0.36, 110, 150)), {"fading_energy"}),
    "confidence_slide": (lambda s: tick(s, conf=ramp(s, 0.65, 0.45, 100, 160)), {"confidence_slide"}),
    "speeding_up": (lambda s: tick(s, pace=ramp(s, 0.45, 0.70, 100, 150)), {"speeding_up"}),
    "stress_creep": (lambda s: tick(s, stress=ramp(s, 0.45, 0.64, 100, 150)), {"stress_creep"}),
}


def main():
    ok = True
    for name, (make, want) in SCENARIOS.items():
        det = PatternDetector()
        started, first = set(), {}
        for s in range(180):
            for e in det.update(make(s)):
                if e["type"] == "pattern_start" and e["pattern"] in NEW:
                    started.add(e["pattern"])
                    first.setdefault(e["pattern"], (s, e["evidence"][0]))
        good = started == want
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {name:<17} started: {sorted(started) or '-'}")
        for pid, (s, ev) in first.items():
            print(f"      {s:3d}s {pid}: {ev}")
    print("all good" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
