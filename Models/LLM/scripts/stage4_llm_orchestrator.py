"""
Stage 4 Distributed Multi-Agent LLM Orchestrator (`stage4_llm_orchestrator.py`)
===============================================================================
Coordinates Model #6 (`DeepSeek-R1-Distill-Qwen-7B` Pedagogical Reasoner) and
Model #7 (`Meta-Llama-3.1-8B-Instruct` + QLoRA Retro-GBA Persona Stylizer) with:
  - Serialized GPU generation lock (prevents concurrent KV-cache OOM on Kaggle 2x T4)
  - VLM-busy coordination flag (pauses background prefetching during PDF ingestion
    so `Qwen3-VL-8B` gets uninterrupted VRAM headroom)
  - Asynchronous Background Pre-Fetching Queue for upcoming battle encounters and
    Professor ADA's live DKT-driven overworld briefings (making battles instant!)
  - 100% graceful offline/fallback mode on local 4 GB PCs.
"""

import collections
import queue
import re
import threading
import time
import torch

from pedagogical_reasoner import (
    PRIMARY_REASONER_ID,
    PedagogicalReasonerLLM,
    detect_course_language,
)
from persona_stylizer import (
    DEFAULT_ADAPTER_DIR,
    PRIMARY_STYLIZER_ID,
    PersonaStylizerLLM,
)


class Stage4LLMOrchestrator:
    def __init__(
        self,
        llm_mode: str = "fallback",
        reasoner_model_id: str = PRIMARY_REASONER_ID,
        stylizer_model_id: str = PRIMARY_STYLIZER_ID,
        stylizer_adapter_dir: str = DEFAULT_ADAPTER_DIR,
    ):
        self.mode = llm_mode  # "live" (Kaggle 2x T4) or "fallback" (local 4GB PC)
        self.reasoner = PedagogicalReasonerLLM(
            model_id=reasoner_model_id,
            mode=llm_mode,
            device_index=0,
        )
        self.stylizer = PersonaStylizerLLM(
            model_id=stylizer_model_id,
            adapter_dir=stylizer_adapter_dir,
            mode=llm_mode,
        )

        self._gpu_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._priority_lock = threading.Lock()
        self._interactive_pending = 0
        self._vlm_busy = threading.Event()

        # Key -> deque of pre-generated encounter dicts (Reasoner + Stylizer)
        self._enc_cache = collections.defaultdict(collections.deque)
        self._ada_cache = {}
        self._story_cache = {}
        self._inflight = set()
        self._prefetch_q = queue.Queue(maxsize=24)

        if self.mode == "live":
            self.reasoner.load_live()
            self.stylizer.load_live()
            if not self.reasoner.is_live() and not self.stylizer.is_live():
                self.mode = "fallback"

        self._worker = threading.Thread(
            target=self._prefetch_loop,
            name="Stage4LLMPrefetch",
            daemon=True,
        )
        self._worker.start()

    def is_live(self) -> bool:
        return self.reasoner.is_live() or self.stylizer.is_live()

    def get_info(self) -> dict:
        return {
            "mode": "live" if self.is_live() else "fallback",
            "reasoner": {
                "name": self.reasoner.short_name,
                "model_id": self.reasoner.model_id,
                "live": self.reasoner.is_live(),
                "params": self.reasoner.num_params,
            },
            "stylizer": {
                "name": self.stylizer.short_name,
                "model_id": self.stylizer.model_id,
                "live": self.stylizer.is_live(),
                "qlora_active": self.stylizer.qlora_active,
                "params": self.stylizer.num_params,
            },
        }

    def set_vlm_busy(self, busy: bool):
        """Called before/after live VLM PDF ingestion to pause background LLM prefetching."""
        if busy:
            self._vlm_busy.set()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        else:
            self._vlm_busy.clear()

    def _begin_interactive(self):
        with self._priority_lock:
            self._interactive_pending += 1

    def _end_interactive(self):
        with self._priority_lock:
            self._interactive_pending = max(0, self._interactive_pending - 1)

    def clear_caches(self):
        with self._cache_lock:
            self._enc_cache.clear()
            self._ada_cache.clear()
            self._story_cache.clear()

    def _build_encounter_bundle(self, spec: dict) -> dict:
        """Runs Model #6 (Reasoner) -> Model #7 (Stylizer) sequentially under GPU lock."""
        concept_data = spec["concept_data"]
        rl_action = int(spec["rl_action"])
        rl_action_name = spec["rl_action_name"]
        mastery = float(spec["mastery"])
        affect = spec["affect"]
        fallback_item = spec["fallback_item"]
        persona_key = spec.get("persona_key") or "wild"
        course_title = spec.get("course_title") or ""
        asked_history = spec.get("asked_history") or []
        win_streak = int(spec.get("win_streak", 0))

        with self._gpu_lock:
            r_out = self.reasoner.generate_encounter_item(
                concept_data=concept_data,
                rl_action=rl_action,
                mastery=mastery,
                affect=affect,
                fallback_item=fallback_item,
                asked_history=asked_history,
                course_title=course_title,
            )
            lang = detect_course_language(
                f"{concept_data.get('title', '')} {r_out['question']} {r_out['reference_answer']}"
            )
            s_out = self.stylizer.generate_battle_banter(
                persona_key=persona_key,
                topic=concept_data.get("short_title") or concept_data.get("title") or "Concept",
                verified_fact=r_out["reference_answer"],
                rl_action=rl_action,
                rl_action_name=rl_action_name,
                mastery=mastery,
                affect=affect,
                win_streak=win_streak,
                language=lang,
            )

        return {
            "reasoner": r_out,
            "stylizer": s_out,
            "generated_at": time.time(),
        }

    def get_or_generate_encounter(self, spec: dict, allow_blocking: bool = True) -> dict:
        """
        Returns a Stage 4 (Reasoner + Stylizer) bundle for an active battle turn:
          1. Instant (<2ms) pop from background prefetch cache if ready.
          2. Otherwise runs live on-demand (unless VLM PDF ingestion is currently busy).
          3. Always guarantees a valid verified pedagogical question + stylized banter.
        """
        cache_key = (
            spec["course_id"],
            int(spec["concept_id"]),
            int(spec["rl_action"]),
            spec.get("persona_key") or "wild",
        )
        # Try exact match first, then same (course, concept, rl_action) with quick persona swap
        with self._cache_lock:
            q = self._enc_cache.get(cache_key)
            if q:
                bundle = q.popleft()
                bundle["cache_hit"] = True
                return bundle

        if (not self.is_live()) or (not allow_blocking) or self._vlm_busy.is_set():
            # Instant zero-stall fallback
            fb = spec["fallback_item"]
            lang = detect_course_language(f"{spec['concept_data'].get('title', '')} {fb['question']}")
            return {
                "reasoner": {
                    "question": fb["question"],
                    "choices": fb.get("choices"),
                    "correct_idx": fb.get("correct_idx"),
                    "reference_answer": fb.get("reference") or fb.get("reference_answer", ""),
                    "socratic_hint": None,
                    "reasoning_trace": "",
                    "source": "curriculum_bank",
                    "model_name": f"{self.reasoner.short_name} (Instant)",
                    "latency_ms": 0.0,
                },
                "stylizer": self.stylizer._fallback_battle_banter(
                    spec.get("persona_key") or "wild",
                    spec["concept_data"].get("short_title") or spec["concept_data"].get("title", ""),
                    lang,
                ),
                "cache_hit": False,
            }

        self._begin_interactive()
        try:
            bundle = self._build_encounter_bundle(spec)
        finally:
            self._end_interactive()
        bundle["cache_hit"] = False
        return bundle

    def schedule_encounter_prefetch(self, spec: dict):
        """Enqueues a non-blocking background prefetch for the next predicted battle turn."""
        if not self.is_live():
            return
        cache_key = (
            spec["course_id"],
            int(spec["concept_id"]),
            int(spec["rl_action"]),
            spec.get("persona_key") or "wild",
        )
        with self._cache_lock:
            if len(self._enc_cache.get(cache_key, ())) >= 1 or ("enc", cache_key) in self._inflight:
                return
            self._inflight.add(("enc", cache_key))
        try:
            self._prefetch_q.put_nowait(("enc", cache_key, spec))
        except queue.Full:
            with self._cache_lock:
                self._inflight.discard(("enc", cache_key))

    def get_or_generate_prof_ada(self, ada_spec: dict, allow_blocking: bool = True) -> dict:
        cache_key = (
            ada_spec["course_id"],
            int(ada_spec["weakest_concept_id"]),
            int(ada_spec["badges_earned"]),
            int(round(float(ada_spec["overall_mastery"]) * 20)),  # 5% bucket
        )
        with self._cache_lock:
            cached = self._ada_cache.get(cache_key)
            if cached:
                out = dict(cached)
                out["cache_hit"] = True
                return out

        if (not self.stylizer.is_live()) or (not allow_blocking) or self._vlm_busy.is_set():
            out = self.stylizer.generate_prof_ada_briefing(
                course_title=ada_spec["course_title"],
                weakest_topic=ada_spec["weakest_topic"],
                weakest_mastery=ada_spec["weakest_mastery"],
                overall_mastery=ada_spec["overall_mastery"],
                badges_earned=ada_spec["badges_earned"],
                win_streak=ada_spec["win_streak"],
                frustration=ada_spec["frustration"],
                verified_fact=ada_spec["verified_fact"],
                starter_name=ada_spec.get("starter_name", "your partner"),
            ) if not self.stylizer.is_live() else {
                "npc": "prof_ada",
                "speaker": "ADA",
                "lines": [
                    f"ADA: Welcome back! Your mastery on {ada_spec['course_title'][:26]} is {round(ada_spec['overall_mastery']*100)}%.",
                    f"ADA: Focus next on '{ada_spec['weakest_topic'][:34]}' ({round(ada_spec['weakest_mastery']*100)}%).",
                    f"ADA: Key insight: {ada_spec['verified_fact'][:96]}",
                ],
                "source": "persona_template",
                "model_name": f"{self.stylizer.short_name} (Instant)",
                "latency_ms": 0.0,
            }
            out["cache_hit"] = False
            return out

        self._begin_interactive()
        try:
            with self._gpu_lock:
                out = self.stylizer.generate_prof_ada_briefing(
                    course_title=ada_spec["course_title"],
                    weakest_topic=ada_spec["weakest_topic"],
                    weakest_mastery=ada_spec["weakest_mastery"],
                    overall_mastery=ada_spec["overall_mastery"],
                    badges_earned=ada_spec["badges_earned"],
                    win_streak=ada_spec["win_streak"],
                    frustration=ada_spec["frustration"],
                    verified_fact=ada_spec["verified_fact"],
                    starter_name=ada_spec.get("starter_name", "your partner"),
                )
        finally:
            self._end_interactive()
        with self._cache_lock:
            self._ada_cache[cache_key] = out
        out["cache_hit"] = False
        return out

    def schedule_prof_ada_prefetch(self, ada_spec: dict):
        if not self.stylizer.is_live():
            return
        cache_key = (
            ada_spec["course_id"],
            int(ada_spec["weakest_concept_id"]),
            int(ada_spec["badges_earned"]),
            int(round(float(ada_spec["overall_mastery"]) * 20)),
        )
        with self._cache_lock:
            if cache_key in self._ada_cache or ("ada", cache_key) in self._inflight:
                return
            self._inflight.add(("ada", cache_key))
        try:
            self._prefetch_q.put_nowait(("ada", cache_key, ada_spec))
        except queue.Full:
            with self._cache_lock:
                self._inflight.discard(("ada", cache_key))

    def get_or_generate_course_story(
        self,
        course_id: str,
        course_title: str,
        concepts: list,
        allow_blocking: bool = True,
    ) -> dict:
        with self._cache_lock:
            cached = self._story_cache.get(course_id)
            if cached:
                out = dict(cached)
                out["cache_hit"] = True
                return out

        if (not self.stylizer.is_live()) or (not allow_blocking) or self._vlm_busy.is_set():
            out = self.stylizer.generate_course_story(course_title=course_title, concepts=concepts) if not self.stylizer.is_live() else None
            if out is None:
                # Non-blocking template while live story generates in background
                prev_mode = self.stylizer.mode
                self.stylizer.mode = "fallback"
                try:
                    out = self.stylizer.generate_course_story(course_title=course_title, concepts=concepts)
                finally:
                    self.stylizer.mode = prev_mode
            out["cache_hit"] = False
            return out

        self._begin_interactive()
        try:
            with self._gpu_lock:
                out = self.stylizer.generate_course_story(course_title=course_title, concepts=concepts)
        finally:
            self._end_interactive()
        with self._cache_lock:
            self._story_cache[course_id] = out
        out["cache_hit"] = False
        return out

    def schedule_course_story_prefetch(self, course_id: str, course_title: str, concepts: list):
        if not self.stylizer.is_live():
            return
        with self._cache_lock:
            if course_id in self._story_cache or ("story", course_id) in self._inflight:
                return
            self._inflight.add(("story", course_id))
        try:
            self._prefetch_q.put_nowait(("story", course_id, {"course_title": course_title, "concepts": concepts}))
        except queue.Full:
            with self._cache_lock:
                self._inflight.discard(("story", course_id))

    def _fast_match_course_concept(self, student_question: str, course_title: str, concepts: list) -> tuple:
        """Fast (<1 ms) lexical course-notes RAG lookup (3x title weight) for NPC Q&A."""
        q_tokens = set(re.findall(r"[0-9a-zA-ZÀ-ÿ$_]{3,}", str(student_question or "").lower()))
        best_c = None
        best_score = -1
        for idx, c in enumerate(concepts or []):
            cid = int(c.get("concept_id", idx))
            t_blob = f"{c.get('title', '')} {c.get('short_title', '')}".lower()
            b_blob = " ".join(
                [str(c.get("podcast_hint", ""))] + [str(b) for b in (c.get("summary_bullets") or [])]
            ).lower()
            t_tokens = set(re.findall(r"[0-9a-zA-ZÀ-ÿ$_]{3,}", t_blob))
            b_tokens = set(re.findall(r"[0-9a-zA-ZÀ-ÿ$_]{3,}", b_blob))
            score = 3 * len(q_tokens & t_tokens) + len(q_tokens & b_tokens)
            if score > best_score:
                best_score = score
                best_c = c
        if best_c is None:
            return course_title, f"Review the key concepts of {course_title} in your Field Journal."
        bullets = best_c.get("summary_bullets") or []
        fact = str(bullets[0] if bullets else (best_c.get("podcast_hint") or best_c.get("title") or course_title))
        matched_title = str(best_c.get("title") or best_c.get("short_title") or course_title)
        return matched_title, fact

    def chat_with_npc(
        self,
        npc_key: str,
        student_question: str,
        course_title: str,
        concepts: list,
        mastery_vec: list,
        chat_history: list = None,
    ) -> dict:
        """
        Live interactive NPC Q&A:
          1. Instant (<1 ms) lexical RAG extracts matched concept + verified fact from course/PDF notes.
          2. Base `Meta-Llama-3.1-8B-Instruct` (via `_base_model_ctx()`) converses naturally in the
             player's own language (English or French) with humor + all 5 course concepts!
        """
        overall_m = sum(float(x) for x in mastery_vec) / max(1, len(mastery_vec)) if mastery_vec else 0.5
        matched_concept, verified_fact = self._fast_match_course_concept(
            student_question=student_question,
            course_title=course_title,
            concepts=concepts,
        )

        if (not self.stylizer.is_live()) or self._vlm_busy.is_set():
            rep = self.stylizer.generate_npc_chat_reply(
                npc_key=npc_key,
                student_question=student_question,
                verified_fact=verified_fact,
                course_title=course_title,
                matched_concept=matched_concept,
                overall_mastery=overall_m,
                chat_history=chat_history,
                language=None,
                concepts=concepts,
                mastery_vec=mastery_vec,
            )
            rep["reasoner_source"] = "course_notes_rag"
            rep["reasoner_ms"] = 0.0
            return rep

        self._begin_interactive()
        try:
            with self._gpu_lock:
                rep = self.stylizer.generate_npc_chat_reply(
                    npc_key=npc_key,
                    student_question=student_question,
                    verified_fact=verified_fact,
                    course_title=course_title,
                    matched_concept=matched_concept,
                    overall_mastery=overall_m,
                    chat_history=chat_history,
                    language=None,
                    concepts=concepts,
                    mastery_vec=mastery_vec,
                )
        finally:
            self._end_interactive()
        rep["reasoner_source"] = "course_notes_rag"
        rep["reasoner_ms"] = 0.0
        return rep

    def _prefetch_loop(self):
        while True:
            kind, cache_key, spec = self._prefetch_q.get()
            try:
                # Yield cleanly if PDF ingestion OR any interactive player call is active
                while self._vlm_busy.is_set() or self._interactive_pending > 0:
                    time.sleep(0.25)
                if kind == "story":
                    with self._gpu_lock:
                        out = self.stylizer.generate_course_story(
                            course_title=spec["course_title"],
                            concepts=spec["concepts"],
                        )
                    with self._cache_lock:
                        self._story_cache[cache_key] = out
                    print(
                        f"  [STAGE-4 PREFETCH] Course Story '{out['quest_title']}' ready ({out['model_name']} {out['latency_ms']:.0f}ms)",
                        flush=True,
                    )
                elif kind == "enc":
                    bundle = self._build_encounter_bundle(spec)
                    with self._cache_lock:
                        q = self._enc_cache[cache_key]
                        if len(q) < 2:
                            q.append(bundle)
                    r = bundle["reasoner"]
                    s = bundle["stylizer"]
                    print(
                        f"  [STAGE-4 PREFETCH] C{spec['concept_id']} RL[{spec['rl_action']}]/{spec.get('persona_key','wild')} ready "
                        f"({r['model_name']} {r['latency_ms']:.0f}ms + {s['model_name']} {s['latency_ms']:.0f}ms)",
                        flush=True,
                    )
                elif kind == "ada":
                    with self._gpu_lock:
                        out = self.stylizer.generate_prof_ada_briefing(
                            course_title=spec["course_title"],
                            weakest_topic=spec["weakest_topic"],
                            weakest_mastery=spec["weakest_mastery"],
                            overall_mastery=spec["overall_mastery"],
                            badges_earned=spec["badges_earned"],
                            win_streak=spec["win_streak"],
                            frustration=spec["frustration"],
                            verified_fact=spec["verified_fact"],
                            starter_name=spec.get("starter_name", "your partner"),
                        )
                    with self._cache_lock:
                        self._ada_cache[cache_key] = out
                    print(
                        f"  [STAGE-4 PREFETCH] Prof. ADA briefing ready ({out['model_name']} {out['latency_ms']:.0f}ms)",
                        flush=True,
                    )
            except Exception as exc:
                print(f"  [STAGE-4 PREFETCH WARN] {kind}: {exc}", flush=True)
            finally:
                with self._cache_lock:
                    self._inflight.discard((kind, cache_key))
                self._prefetch_q.task_done()
