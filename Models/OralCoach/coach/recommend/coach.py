"""The live coach: decides *when* to speak, *what* to say, and checks *whether it helped*.

Once per second:
  1. the pattern detector updates (pattern_start / pattern_end events)
  2. tips whose 20-second check window is over get scored -> tip_result event (worked -> reset, else escalate)
  3. if a tip is allowed now (cooldowns, nothing being checked), pick the most important active pattern,
     pick the escalation level, retrieve cards, pick the best-matching strategy (a different one if the
     last one just failed), and ask the writer for a tip
  4. a finished tip is emitted as a `tip` event (with the full "why")

Sharp moments (urgent.py: stress spike, panic, suddenly looking away, freeze, still frozen) take a fast lane:
no 25 s cooldown, card wording instead of waiting for the LLM, and a shorter "did it help?" check.

Every event is a plain dict with a "type", ready to log, print as JSON, or feed to a game.
"""
import concurrent.futures as cf
import itertools
import math
from dataclasses import dataclass, replace

from .knowledge import STRATEGY_NAMES, Knowledge
from . import urgent as _urgent  # noqa: F401  (registers the fast-lane patterns)
from . import more_patterns as _more  # face+voice and over-time patterns
from .patterns import PATTERNS, PatternDetector
from .progress import ProgressWatcher
from .writer import LEVEL_NAMES, TipWriter

SILENT_NO_TIPS = 8.0  # silent at least this long: only freeze tips (looking at notes in a pause is fine)

# how a pattern's badness is shown to the student: (name, show 1 - badness?)
METRIC = {
    "freeze": ("silence", False), "sustained_stress": ("stress", False), "rushing_nervous": ("pace", False),
    "masked_tension": ("voice tension", False), "confidence_drop": ("confidence", True),
    "eye_contact_low": ("eye contact", True), "filler_burst": ("fillers", False), "rushing": ("pace", False),
    "face_tense": ("face tension", False), "monotone": ("voice variety", True), "low_energy": ("energy", True),
    "stress_spike": ("stress", False), "panic": ("pace", False), "lost_audience": ("eye contact", True),
    "still_frozen": ("silence", False),
}
METRIC.update(_more.METRIC)


def family(pid: str) -> str:
    """Fast-lane patterns share cards and escalation with the normal pattern they warn about."""
    return PATTERNS[pid].family or pid


def is_urgent(pid: str) -> bool:
    return PATTERNS[pid].urgent or PATTERNS[pid].priority >= 10


@dataclass
class PatternState:
    level: int = 0
    last_tip_t: float = -math.inf
    last_result: bool | None = None
    last_strategy: str | None = None
    rest_until: float = -math.inf
    check_s: float = 20.0           # length of the last tip's "did it help?" check


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


class Coach:
    def __init__(self, cfg, knowledge: Knowledge, writer: TipWriter, on_tip=None):
        self.cfg, self.kb, self.writer = cfg, knowledge, writer
        self.on_tip = on_tip
        self.detector = PatternDetector()
        self.progress = ProgressWatcher(self.detector, METRIC)
        self.state = {pid: PatternState() for pid in PATTERNS}
        self.pending = None          # tip being written by the LLM
        self.open_checks = []        # tips waiting for their "did it help?" result
        self.last_tip_t = -math.inf
        self.last_urgent_t = -math.inf
        self.last_praise_t = -math.inf
        self.quiet_since = None
        self.current_tip = None
        self.toast = None
        self.now = 0.0
        self._ids = itertools.count(1)

    # --- main loop -----------------------------------------------------------------------------
    def step(self, tick: dict) -> list[dict]:
        self.now = now = tick["t"]
        events = self.detector.update(tick)
        events += self._check_results(now)
        events += self._deliver_pending(now)
        if self.pending is None or not self.pending["tip"]["urgent"]:  # an urgent moment may cut in (see below)
            events += self._maybe_start_tip(tick)
        events += self._maybe_praise(tick)
        for e in self.progress.update(tick):  # good news: recovered after a tip, improving over a minute
            self.toast = {"text": e["text"], "kind": "good", "t": now}
            events.append(e)
        return events

    # --- 1. did the last tip help? -------------------------------------------------------------
    def _badness(self, pid: str, t0: float, t1: float):
        f = PATTERNS[pid].badness
        vals = [f(x) for x in self.detector.ticks if t0 < x["t"] <= t1]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _check_results(self, now: float) -> list[dict]:
        out, still = [], []
        for tip in self.open_checks:
            if now < tip["t"] + tip["check_after_s"]:
                still.append(tip)
                continue
            pid, pdef = tip["pattern"], PATTERNS[tip["pattern"]]
            fam = family(pid)
            before = tip["_before"]
            after = self._badness(pid, tip["t"] + tip["_check_delay"], now)
            if before is None or after is None:
                continue  # no data (e.g. face lost): no verdict
            gain = before - after
            still_active = pid in self.detector.active or fam in self.detector.active
            worked = gain >= pdef.min_gain or (not still_active and gain >= pdef.min_gain / 2)
            st = self.state[fam]
            st.last_result = worked
            if worked:
                st.level = 0
                st.rest_until = now + self.cfg.coach_pattern_rest
            elif tip["level"] >= 3:   # tried everything up to a pause: stop nagging about this for a while
                st.rest_until = now + self.cfg.coach_give_up_rest
            name, flip = METRIC[pid]
            b, a = (1 - before, 1 - after) if flip else (before, after)
            if fam == "freeze":
                msg = "Back on track, nice recovery!" if worked else "Take your time, then pick up your last point"
            elif fam == "filler_burst":
                msg = (f"That helped: fewer fillers ({b * 10:.0f} -> {a * 10:.0f}/min)" if worked
                       else f"Not yet: fillers {b * 10:.0f} -> {a * 10:.0f}/min")
            else:
                msg = f"That helped: {name} {b:.2f} -> {a:.2f}" if worked else f"Not yet: {name} {b:.2f} -> {a:.2f}"
            res = {"type": "tip_result", "t": now, "tip_id": tip["id"], "pattern": pid, "strategy": tip["strategy"],
                   "level": tip["level"], "worked": worked, "metric": name, "before": round(b, 3),
                   "after": round(a, 3), "message": msg, "urgent": tip["urgent"]}
            self.toast = {"text": msg, "kind": "good" if worked else "meh", "t": now}
            self.progress.tip_result(res, before)
            out.append(res)
        self.open_checks = still
        return out

    # --- 2. decide whether to speak, and what ---------------------------------------------------
    def _maybe_start_tip(self, tick: dict) -> list[dict]:
        now = tick["t"]
        cands = []
        silent = tick["channels"].get("silence_now_s") or 0.0
        for pid, act in self.detector.active.items():
            pdef = PATTERNS[pid]
            st = self.state[family(pid)]
            if now < st.rest_until or act.off_ticks:  # only patterns happening right now
                continue
            if silent >= SILENT_NO_TIPS and family(pid) != "freeze":
                continue  # not talking (thinking, notes, a break): no eye contact / face tips, only freeze help
            stronger = pdef.min_level > max(1, st.level)  # e.g. "still frozen" after a freeze tip
            if now - st.last_tip_t < st.check_s + 2 and not stronger:  # still judging the last tip on it
                continue
            cands.append((pdef.priority, -act.since, pid))
        if not cands:
            return []
        _, _, pid = max(cands)
        fam = family(pid)
        st, act, pdef = self.state[fam], self.detector.active[pid], PATTERNS[pid]
        urgent = is_urgent(pid)
        if self.pending is not None:
            if not urgent:
                return []
            # a normal tip is still being written by the LLM (up to 6 s): a freeze or a panic can't wait for it.
            # Drop it as if it never started, and answer the urgent moment now.
            self._cancel_pending()
            st = self.state[fam]
        since_last = now - self.last_tip_t
        if not urgent:
            if self.open_checks or since_last < self.cfg.coach_tip_cooldown:
                return []
        else:
            same_tip_up = (self.current_tip and self.current_tip["pattern"] == pid
                           and now - self.current_tip["t"] < self.current_tip["check_after_s"])
            # being stuck (freeze) may interrupt another urgent tip; anything else waits URGENT_GAP s
            more_serious = (pdef.min_level > max(1, st.level) or not self.current_tip
                            or (fam == "freeze" and pdef.priority > PATTERNS[self.current_tip["pattern"]].priority))
            if since_last < 3 or same_tip_up or (now - self.last_urgent_t < _urgent.URGENT_GAP and not more_serious):
                return []

        escalated_because = None
        if st.last_result is False and now - st.last_tip_t <= self.cfg.coach_escalate_within and st.level >= 1:
            level = min(3, st.level + 1)
            escalated_because = f"the last tip on this {now - st.last_tip_t:.0f} s ago did not help enough"
        elif pdef.min_level > st.level >= 1 and now - st.last_tip_t <= 30:
            level = pdef.min_level
            escalated_because = f"still stuck {now - st.last_tip_t:.0f} s after the last tip"
        else:
            level = 1
        if fam == "freeze":
            level = max(level, 2)  # no time for subtle hints when you are stuck
        level = max(level, pdef.min_level)
        urgent_because = None
        if urgent:
            urgent_because = (f"{act.evidence[0]}: a sharp moment, so the coach speaks now "
                              f"instead of waiting {self.cfg.coach_tip_cooldown:.0f} s between tips")

        query = pdef.query + " " + " ".join(act.evidence)
        retrieved = self.kb.search(fam, query, k=6)
        if not retrieved:
            return []
        # same advice for everyone: the card that best matches this moment (BM25), but never the strategy
        # that just failed on this problem when escalating (try another approach instead)
        scored = []
        for card, rel in retrieved:
            score = rel * (0.3 if escalated_because and card.strategy == st.last_strategy else 1.0)
            scored.append((score, card, rel))
        scored.sort(key=lambda x: x[0], reverse=True)
        _, card, rel = scored[0]
        name = STRATEGY_NAMES.get(card.strategy, card.strategy)
        if escalated_because and st.last_strategy and card.strategy != st.last_strategy:
            choice = f"switching to {name}: {STRATEGY_NAMES.get(st.last_strategy, st.last_strategy)} did not help"
        else:
            choice = f"{name}: best match for this moment"
        books = self.kb.search_books(query, k=2)

        template_why = f"{_cap(act.evidence[0])}. {card.why}"
        req = {
            "pattern": pid, "label": pdef.label, "evidence": act.evidence, "level": level,
            "strategy": STRATEGY_NAMES.get(card.strategy, card.strategy), "choice": choice,
            "previous": self.current_tip["text"] if escalated_because and self.current_tip else None,
            "transcript": tick.get("transcript"), "passage": card.passage, "source": card.source,
            "template_tip": card.text_for(level), "template_why": template_why,
            "extra": [{"citation": p.citation, "text": p.text} for p, _ in books],
        }
        tip = {
            "type": "tip", "id": f"tip-{next(self._ids)}", "pattern": pid, "label": pdef.label,
            "modality": pdef.modality, "level": level, "level_name": LEVEL_NAMES[level],
            "strategy": card.strategy, "card": card.id, "evidence": act.evidence, "choice": choice,
            "escalated_because": escalated_because, "source": card.source,
            "urgent": urgent, "urgent_because": urgent_because, "family": fam,
            "source_full": self.kb.sources.get(card.source, card.source),
            "retrieved": [{"card": c.id, "strategy": c.strategy, "relevance": round(r, 2), "score": round(s, 2)}
                          for s, c, r in scored],
            "books": [{"citation": p.citation, "id": p.id, "score": round(s, 2)} for p, s in books],
            "detected_at": now,
        }
        saved = (fam, replace(st), self.last_tip_t)  # to undo this tip if it is dropped before it is shown
        st.level, st.last_strategy, st.last_tip_t = level, card.strategy, now
        self.last_tip_t = now
        if urgent:  # no time to wait for the LLM: the card's own wording, right now
            self.last_urgent_t = now
            future = cf.Future()
            future.set_result(self.writer.template(req, "urgent, no LLM wait"))
        else:
            future = self.writer.submit(req)
        self.pending = {"future": future, "tip": tip, "req": req, "saved": saved,
                        "deadline": now + self.writer.timeout + 1}
        return self._deliver_pending(now)

    # --- 3. hand the written tip out -----------------------------------------------------------
    def _deliver_pending(self, now: float) -> list[dict]:
        p = self.pending
        if p is None:
            return []
        f = p["future"]
        if f.done():
            try:
                written = f.result()
            except Exception:  # noqa: BLE001
                written = self.writer.template(p["req"], "LLM error")
        elif now >= p["deadline"]:
            written = self.writer.template(p["req"], "LLM too slow")
        else:
            return []
        pid0 = p["tip"]["pattern"]
        if pid0 not in self.detector.active and family(pid0) not in self.detector.active:
            # the problem stopped while the LLM was writing: a tip about it now would be stale and confusing
            self._cancel_pending()
            return []
        self.pending = None
        check = PATTERNS[p["tip"]["pattern"]].check_s or self.cfg.coach_check_window
        tip = dict(p["tip"], t=now, text=written["text"], why=written["why"], backend=written["backend"],
                   check_after_s=check)
        # short version to read out loud (--speak): the tip if it is short, else the card's own wording
        # fillers / pace are 30 s rolling numbers: start them again from now, so "did it help?" and the
        # next decision only see what you did after reading the tip (run.py wires this to the ChannelBuilder)
        tip["restart"] = list(PATTERNS[tip["family"]].restarts or PATTERNS[tip["pattern"]].restarts)
        tip["say"] = tip["text"] if len(tip["text"].split()) <= 12 else p["req"]["template_tip"]
        tip["_check_delay"] = min(self.cfg.coach_check_delay, check / 3)
        tip["_before"] = self._badness(tip["pattern"], tip["detected_at"] - (3 if tip["urgent"] else 8), now)
        if tip["urgent"]:  # the moment changed: older checks it interrupts give no verdict
            self.open_checks = [c for c in self.open_checks if c["urgent"] and c["family"] != tip["family"]]
        self.current_tip = tip
        self.last_tip_t = now
        st = self.state[tip["family"]]
        st.last_tip_t, st.check_s = now, check
        st.last_result = None  # this tip's own verdict decides the next escalation, not the one before it
        self.open_checks.append(tip)
        if self.on_tip:
            self.on_tip(tip)
        return [{k: v for k, v in tip.items() if not k.startswith("_")}]

    def _cancel_pending(self):
        """Forget the tip being written, and undo what starting it changed (level, cooldowns)."""
        p, self.pending = self.pending, None
        if p is not None:
            fam, st, last_tip_t = p["saved"]
            self.state[fam] = st
            self.last_tip_t = last_tip_t

    # --- 4. say something nice when things go well ---------------------------------------------
    def _maybe_praise(self, tick: dict) -> list[dict]:
        now = tick["t"]
        talking = (tick["channels"].get("talk_ratio") or 0) >= 0.5
        if not tick.get("calibrated") or self.detector.active or self.open_checks or self.pending or not talking:
            self.quiet_since = None  # no "steady for 45 s" praise for 45 s of not talking at all
            return []
        if self.quiet_since is None:
            self.quiet_since = now
            return []
        if now - self.quiet_since < self.cfg.coach_praise_after or now - self.last_praise_t < 90:
            return []
        c = tick["channels"]
        good = []
        if (c.get("eye_contact") or 0) >= 0.7:
            good.append("good eye contact")
        if c.get("stress") is not None and c["stress"] < 0.6:
            good.append("calm voice")
        if c.get("pace") is not None and 0.3 <= c["pace"] <= 0.7:
            good.append("steady pace")
        if (c.get("fillers_per_min") or 0) < 3:
            good.append("few fillers")
        text = f"Steady for {now - self.quiet_since:.0f} s" + (": " + ", ".join(good[:3]) if good else "")
        self.last_praise_t = now
        self.quiet_since = now
        self.toast = {"text": text, "kind": "good", "t": now}
        return [{"type": "praise", "t": now, "text": text, "reasons": good}]

    # --- for the window -------------------------------------------------------------------------
    def view(self, now: float | None = None) -> dict:
        now = self.now if now is None else now
        tip = self.current_tip
        checking, check_total = None, self.cfg.coach_check_window
        for c in self.open_checks:
            checking = max(0.0, c["t"] + c["check_after_s"] - now)
            check_total = c["check_after_s"]
        return {
            "tip": tip if tip and (now - tip["t"] < self.cfg.coach_tip_show or checking is not None) else None,
            "tip_age": None if tip is None else now - tip["t"],
            "tip_show": self.cfg.coach_tip_show,
            "checking": checking,
            "check_total": check_total,
            "writing": self.pending is not None,
            "toast": self.toast if self.toast and now - self.toast["t"] < 5 else None,
            "active": [PATTERNS[p].label for p in self.detector.active],
            "backend": self.writer.status,
        }
