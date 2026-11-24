"""End-to-end check of the coach with made-up snapshots (no sensors, no LLM, a few seconds).

  python test_coach.py

Feeds one-second snapshots straight into the Coach (card wording, no LLM) and checks the adaptive loop:
  - a calm talk gets no tips
  - stress that goes away after the tip      -> tip, then "IT WORKED" (tip_result worked=True)
  - stress that stays high                   -> tip, "not yet", then an escalated tip with a different strategy
  - filler words that stop after the tip     -> IT WORKED, no second filler tip (the 30 s filler count restarts
                                                at the tip, through the real ChannelBuilder)
  - filler words that go on                  -> "not yet", then an escalated tip
  - very stressed and suddenly much faster   -> the urgent "panic" tip within a few seconds (short 10 s pace)
Timing / interaction bugs found in the audit of 2026-11-21 (each has a check below):
  - a freeze while the LLM is still writing a normal tip -> the freeze tip comes at once, the other is dropped
  - the problem stops while the LLM writes            -> the stale tip is dropped, not shown
  - a 2-minute silent break                           -> one freeze tip, then quiet: no nagging, no eye
                                                          contact tips, no "steady" praise or XP for silence
  - a tip that got no verdict (e.g. face lost)        -> the next tip is not "escalated because it did not help"
  - the coach's spoken tip mutes the mic              -> that time is not counted as "no fillers / slow pace"
Every tip must come with its "why", its source and the reason for the chosen strategy.
"""
import concurrent.futures as cf
import os

from coach.channels import ChannelBuilder
from coach.config import Config
from coach.recommend import Coach, Knowledge, TipWriter
from coach.signals import Signals
from test_patterns import tick

HERE = os.path.dirname(os.path.abspath(__file__))


def run(stress_at, seconds=240):
    cfg = Config()
    coach = Coach(cfg, Knowledge(os.path.join(HERE, "knowledge")), TipWriter(mode="template"))
    events, first_tip = [], None
    for s in range(seconds):
        tip_age = None if first_tip is None else s - first_tip
        for e in coach.step(tick(s, stress=stress_at(s, tip_age))):
            e["_s"] = s
            events.append(e)
            if e["type"] == "tip" and first_tip is None:
                first_tip = s
    return events


def run_fillers(keep_going_after_tip, restart=True, seconds=240):
    """Fillers every 5 s (12/min) from 60 s on; after the tip they stop unless keep_going_after_tip.
    fillers_per_min comes from the real ChannelBuilder, which restarts its count when a tip is shown."""
    cfg = Config()
    sig = Signals(keep_seconds=600)
    builder = ChannelBuilder(cfg, sig)
    builder.started = 1000.0
    on_tip = (lambda tip: builder.restart(tip["restart"], tip["t"])) if restart else None
    coach = Coach(cfg, Knowledge(os.path.join(HERE, "knowledge")), TipWriter(mode="template"), on_tip=on_tip)
    events, first_tip = [], None
    for s in range(seconds):
        now = 1000.0 + s
        if s >= 60 and s % 5 == 0 and (first_tip is None or keep_going_after_tip):
            sig.push("filler", "um", now - 0.5)
        t = tick(s)
        t["channels"]["fillers_per_min"] = builder.raw(now)["fillers_per_min"]
        for e in coach.step(t):
            e["_s"] = s
            events.append(e)
            if e["type"] == "tip" and first_tip is None:
                first_tip = s
    return events


class SlowWriter(TipWriter):
    """An LLM that answers `delay` ticks later (call advance() once per tick)."""

    def __init__(self, delay):
        super().__init__(mode="template")
        self.delay, self.jobs = delay, []

    def submit(self, req):
        f = cf.Future()
        self.jobs.append([self.delay, f, req])
        return f

    def advance(self):
        for job in self.jobs:
            job[0] -= 1
            if job[0] <= 0 and not job[1].done():
                job[1].set_result(self.template(job[2], "slow test LLM"))


def run_with(make, seconds, writer=None):
    writer = writer or TipWriter(mode="template")
    coach = Coach(Config(), Knowledge(os.path.join(HERE, "knowledge")), writer)
    events = []
    for s in range(seconds):
        for e in coach.step(make(s)):
            e["_s"] = s
            events.append(e)
        if isinstance(writer, SlowWriter):
            writer.advance()
    return events


def silent_from(start, end, eye=None):
    def make(s):
        x = tick(s)
        if start <= s < end:
            x["channels"]["silence_now_s"] = float(s - start)
            x["channels"]["talk_ratio"] = 0.5 if s - start < 10 else 0.0
            if eye is not None:
                x["channels"]["eye_contact"] = eye
        return x
    return make


def audit_checks(check):
    # 1. freeze while a normal stress tip is being written (slow LLM, 6 s)
    def m(s):
        x = tick(s, stress=0.67 if s >= 40 else 0.5)
        if 51 <= s < 70:
            x["channels"]["silence_now_s"] = float(s - 51)
        return x
    ev = run_with(m, 70, SlowWriter(6))
    tips = [e for e in ev if e["type"] == "tip"]
    check("freeze cuts in while the LLM writes a stress tip (freeze tip at 55 s, no stale stress tip)",
          tips and tips[0]["pattern"] == "freeze" and tips[0]["_s"] == 55
          and not [t for t in tips if t["pattern"] == "sustained_stress"])

    # 2. the stress stops while the LLM is writing (9 s): the tip is dropped
    ev = run_with(lambda s: tick(s, stress=0.67 if 40 <= s < 53 else 0.5), 120, SlowWriter(9))
    check("a tip about a problem that already ended is dropped", not [e for e in ev if e["type"] == "tip"])

    # 3. a 2-minute silent break, looking at notes
    ev = run_with(silent_from(60, 200, eye=0.1), 240)
    tips = [e for e in ev if e["type"] == "tip"]
    check("silent break: only the freeze help at its start, no tips after 30 s of silence",
          tips and all(t["family"] == "freeze" for t in tips) and all(t["_s"] < 90 for t in tips))
    check("silent break: no 'steady' praise or progress for not talking",
          not [e for e in ev if e["type"] in ("praise", "progress") and 60 <= e["_s"] < 200])

    # 4. tip 1 fails, tip 2 gets no verdict (face lost during its check): tip 3 must not say "did not help"
    def m(s):
        x = tick(s, eye=0.2 if s >= 60 else 0.8)
        if 95 <= s < 115:
            x["channels"]["eye_contact"] = None
        return x
    ev = run_with(m, 150)
    tips = [e for e in ev if e["type"] == "tip" and e["family"] == "eye_contact_low"]
    check("a tip without a verdict does not make the next one 'escalated because it did not help'",
          len(tips) >= 3 and tips[1]["escalated_because"] and not tips[2]["escalated_because"]
          and tips[2]["level"] == 1)

    # 5. a spoken tip mutes the mic for 4 s: fillers per minute must not drop because of it
    sig = Signals(keep_seconds=600)
    builder = ChannelBuilder(Config(), sig)
    builder.started = 1000.0
    for i in range(0, 60, 3):                  # 20 fillers per minute ...
        if not 50 <= i < 54:                   # ... but the mic was deaf 1050-1054 (a spoken tip)
            sig.push("filler", "um", 1000.0 + i)
    sig.push("muted", (1050.0, 1054.0))
    fpm = builder.raw(1060.0)["fillers_per_min"]
    check(f"muted time is left out of the rate: {fpm:.1f} fillers/min for a true 20 (counting it: 18.0)",
          fpm >= 19.5)


def calm(s, tip_age):
    return 0.5


def stress_then_calm(s, tip_age):
    if tip_age is not None and tip_age >= 4:
        return 0.45  # the tip worked: back to normal
    return 0.8 if s >= 60 else 0.5


def stress_stays(s, tip_age):
    return 0.8 if s >= 60 else 0.5


def show(events):
    for e in events:
        if e["type"] == "tip":
            print(f"      {e['_s']:3d}s TIP L{e['level']} [{e['strategy']}] {e['text']}")
            print(f"           why: {e['why']}  |  {e['choice']}"
                  + (f"  |  escalated: {e['escalated_because']}" if e.get("escalated_because") else ""))
        elif e["type"] == "tip_result":
            print(f"      {e['_s']:3d}s RESULT {'IT WORKED' if e['worked'] else 'not yet'}: {e['message']}")


def main():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'OK ' if cond else 'BAD'} {name}")

    ev = run(calm)
    check("calm talk: no tips", not [e for e in ev if e["type"] == "tip"])

    ev = run(stress_then_calm)
    tips = [e for e in ev if e["type"] == "tip"]
    res = [e for e in ev if e["type"] == "tip_result"]
    check("stress goes away: a tip, then IT WORKED", tips and res and res[0]["worked"])
    check("every tip explains itself (why, source, choice)", all(t["why"] and t["source"] and t["choice"] for t in tips))
    show(ev)

    ev = run(stress_stays)
    tips = [e for e in ev if e["type"] == "tip"]
    res = [e for e in ev if e["type"] == "tip_result"]
    esc = [t for t in tips if t.get("escalated_because")]
    check("stress stays: first result is 'not yet'", res and not res[0]["worked"])
    check("stress stays: next tip is escalated to a higher level", esc and esc[0]["level"] > tips[0]["level"])
    check("escalated tip switches strategy", esc and esc[0]["strategy"] != tips[0]["strategy"])
    show(ev)

    ev = run_fillers(keep_going_after_tip=False)
    tips = [e for e in ev if e["type"] == "tip"]
    res = [e for e in ev if e["type"] == "tip_result"]
    check("fillers stop after the tip: tip asks to restart the filler count",
          tips and tips[0]["pattern"] == "filler_burst" and tips[0]["restart"] == ["fillers"])
    check("fillers stop after the tip: IT WORKED", res and res[0]["worked"])
    check("fillers stop after the tip: no second filler tip", len(tips) == 1)
    show(ev)

    ev = run_fillers(keep_going_after_tip=False, restart=False)
    old = [e for e in ev if e["type"] in ("tip", "tip_result")]
    print("      (the old 30 s count, without the restart, gave: "
          + ", ".join(f"{e['_s']}s " + (f"tip L{e['level']}" if e["type"] == "tip" else
                                         ("worked" if e["worked"] else "not yet")) for e in old) + ")")

    # pace: 180 wpm for a minute, a pace tip at 60 s, then 120 wpm. 10 s later the restarted count reads 120.
    sig, builder = Signals(keep_seconds=600), None
    builder = ChannelBuilder(Config(), sig)
    builder.started = 1000.0
    for i in range(60 * 3):
        sig.push("word", "word", 1000.0 + i / 3)
    for i in range(10 * 2):
        sig.push("word", "word", 1060.0 + i / 2)
    sig.push("speaking", 1.0, 1069.0)
    builder.restart(["pace"], 1060.0)
    wpm = builder.raw(1070.0)["wpm"]
    check(f"pace restarts at a pace tip: {wpm:.0f} wpm 10 s later (old 30 s count: 160)", abs(wpm - 120) < 1)

    # panic: the short pace follows a sudden speed-up within seconds, the 30 s one barely moves
    sig = Signals(keep_seconds=600)
    builder = ChannelBuilder(Config(), sig)
    builder.started = 1000.0
    t = 1000.0
    while t < 1065.0:                       # 140 wpm for 65 s ...
        sig.push("word", "word", t)
        t += 60 / 140
    while t < 1070.0:                       # ... then 220 wpm for 5 s
        sig.push("word", "word", t)
        t += 60 / 220
    sig.push("speaking", 1.0, 1069.0)
    c = builder.tick(1070.0)["channels"]
    check(f"short pace sees a sudden speed-up: pace now {c['pace_now']:.2f} vs 30 s pace {c['pace']:.2f}",
          c["pace_now"] - c["pace"] >= 0.1)

    cfg = Config()
    coach = Coach(cfg, Knowledge(os.path.join(HERE, "knowledge")), TipWriter(mode="template"))
    panic_at = None
    for s_ in range(140):
        x = tick(s_, stress=0.85 if s_ >= 60 else 0.5, pace=0.5)
        x["channels"]["pace_now"] = 0.85 if s_ >= 100 else 0.5
        x["channels"]["pace_now_wpm"] = 210 if s_ >= 100 else 140
        for e in coach.step(x):
            if e["type"] == "tip" and e["pattern"] == "panic" and panic_at is None:
                panic_at = s_
    check(f"panic tip fires {'-' if panic_at is None else panic_at - 100} s after a sudden speed-up while stressed",
          panic_at is not None and panic_at - 100 <= 4)

    audit_checks(check)

    ev = run_fillers(keep_going_after_tip=True)
    tips = [e for e in ev if e["type"] == "tip"]
    res = [e for e in ev if e["type"] == "tip_result"]
    check("fillers go on: 'not yet', then an escalated tip",
          res and not res[0]["worked"] and len(tips) >= 2 and tips[1]["level"] > tips[0]["level"])
    show(ev)

    print("all good" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
