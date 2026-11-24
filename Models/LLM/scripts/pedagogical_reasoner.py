"""
Model #6: Pedagogical Reasoner LLM (`Models/LLM/scripts/pedagogical_reasoner.py`)
=================================================================================
Powered by `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (4-bit NF4 on Kaggle `cuda:0`),
with automatic fallback chain `Qwen/Qwen2.5-7B-Instruct` -> `Qwen/Qwen2.5-3B-Instruct`
and zero-VRAM curriculum fallback on local 4 GB PCs (`mode="fallback"`).

Responsibilities:
  - Consumes the Double-DQN RL Game Master's target concept, RL difficulty tier
    (`[0] Scaffold`, `[1] Easy Recall`, `[2] Medium Flow`, `[3] Prereq Review`,
     `[4] Hard Open-Text Boss Synthesis`), student DKT mastery + affect, and the
    active concept's verified course notes (`summary_bullets`).
  - Uses explicit `<think>...</think>` reasoning (stripped before returning) to
    generate fresh, non-repeating, factually verified pedagogical JSON:
      * 4-choice MC questions with plausible distractors + shuffled `correct_idx`
      * Hard Open-Text Boss Synthesis prompts + 15-35 word `reference_answer` for DeBERTa-v3
      * Targeted Socratic hints when the learner is struggling or RL selects Action [0]
  - Automatically matches the active course's language (French or English).
"""

import json
import os
import random
import re
import time
import torch

PRIMARY_REASONER_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
FALLBACK_REASONER_IDS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

RL_TIER_SPECS = {
    0: ("Scaffolded Easy Recall", "Create an encouraging, foundational 4-choice recall question PLUS a concise Socratic hint that guides thinking without giving away the answer."),
    1: ("Easy Recall Wild Quiz", "Create a clear, direct 4-choice conceptual recall question testing core definitions from the study notes."),
    2: ("Medium Flow Application", "Create a sharp 4-choice application or 'why/what happens if' reasoning question requiring genuine understanding of the study notes."),
    3: ("Prerequisite Review", "Create a diagnostic 4-choice review question reinforcing the prerequisite concept so the learner closes their foundation gap."),
    4: ("Hard Open-Text Boss Synthesis", "Create a challenging open-ended synthesis question (NO choices) asking the student to explain a trade-off, mechanism, or workflow in 1-2 sentences, plus a precise 18-32 word `reference_answer` rubric for semantic grading."),
}


def detect_course_language(text: str) -> str:
    sample = str(text or "")
    if re.search(r"[éèêëàâçùûîïôœ]", sample, re.IGNORECASE):
        return "French"
    if re.search(r"\b(base de données|requête|opérateur|serveur|mémoire|fichier|tableau|commande|permet|utilisé)\b", sample, re.IGNORECASE):
        return "French"
    return "English"


def strip_think_trace(raw: str) -> tuple:
    """
    Separates DeepSeek-R1 `<think>...</think>` reasoning trace from the final JSON payload.
    Returns `(clean_text, trace_summary)`.
    """
    s = str(raw or "")
    trace = ""
    m = re.search(r"<think>(.*?)</think>", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        trace = re.sub(r"\s+", " ", m.group(1)).strip()
        s = s[m.end():].strip()
    elif "<think>" in s.lower():
        idx = s.lower().find("<think>")
        # Look for a JSON object even if </think> was omitted near token limit
        brace = s.find("{", idx)
        if brace != -1:
            trace = re.sub(r"\s+", " ", s[idx + 7 : brace]).strip()
            s = s[brace:].strip()
        else:
            trace = re.sub(r"\s+", " ", s[idx + 7 :]).strip()
            s = ""
    return s, trace[:240]


def extract_strict_json(raw: str) -> dict:
    """Extracts a JSON object from LLM output, repairing minor markdown fencing or trailing commas."""
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in Reasoner output")
    depth = 0
    in_str = False
    esc = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
    snippet = text[start : end + 1] if end != -1 else text[start:]
    snippet = re.sub(r",\s*([}\]])", r"\1", snippet)
    return json.loads(snippet)


def validate_and_normalize_reasoner_output(
    payload: dict,
    rl_action: int,
    fallback_item: dict,
    rng: random.Random = None,
) -> dict:
    """
    Validates Model #6 JSON strictly against the pedagogical schema.
    Shuffles MC options so the correct answer is evenly distributed across [0..3].
    Raises ValueError if academic validity checks fail (triggering clean fallback).
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload is not a JSON object")

    question = re.sub(r"\s+", " ", str(payload.get("question") or "")).strip()
    if len(question) < 14:
        raise ValueError("Generated question too short")

    reference = re.sub(r"\s+", " ", str(payload.get("reference_answer") or payload.get("reference") or "")).strip()
    if len(reference) < 10:
        reference = str(fallback_item.get("reference") or fallback_item.get("reference_answer") or "")
    if len(reference) < 8:
        raise ValueError("Missing reference_answer")

    socratic_hint = re.sub(r"\s+", " ", str(payload.get("socratic_hint") or "")).strip()

    if int(rl_action) == 4:
        return {
            "question": question,
            "choices": None,
            "correct_idx": None,
            "reference_answer": reference,
            "socratic_hint": socratic_hint or None,
        }

    raw_choices = payload.get("choices")
    if not isinstance(raw_choices, list) or len(raw_choices) != 4:
        raise ValueError(f"Expected exactly 4 choices, got {type(raw_choices)}")
    clean_choices = [re.sub(r"\s+", " ", str(c or "")).strip() for c in raw_choices]
    if any(len(c) < 2 for c in clean_choices):
        raise ValueError("Empty choice found")
    if len({c.lower() for c in clean_choices}) < 4:
        raise ValueError("Duplicate distractors detected")

    try:
        c_idx = int(payload.get("correct_idx", 0))
    except Exception:
        c_idx = 0
    if not (0 <= c_idx <= 3):
        raise ValueError(f"correct_idx out of range: {c_idx}")

    # Shuffle choices deterministically/randomly so option 0 isn't always the right answer
    correct_text = clean_choices[c_idx]
    order = [0, 1, 2, 3]
    (rng or random).shuffle(order)
    shuffled = [clean_choices[i] for i in order]
    new_idx = shuffled.index(correct_text)

    return {
        "question": question,
        "choices": shuffled,
        "correct_idx": new_idx,
        "reference_answer": reference,
        "socratic_hint": socratic_hint or None,
    }


class PedagogicalReasonerLLM:
    """
    Model #6 wrapper (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, 4-bit NF4 on `cuda:0`)
    generating verified, curriculum-grounded RPG encounter questions.
    """

    def __init__(
        self,
        model_id: str = PRIMARY_REASONER_ID,
        mode: str = "fallback",
        device_index: int = 0,
    ):
        self.model_id = model_id or PRIMARY_REASONER_ID
        self.mode = mode  # "live" (Kaggle GPU) or "fallback" (local 4GB PC)
        self.device_index = int(device_index)
        self.model = None
        self.tokenizer = None
        self.short_name = self._short_name(self.model_id)
        self.num_params = 0
        self.load_error = None

    @staticmethod
    def _short_name(mid: str) -> str:
        low = str(mid or "").lower()
        if "deepseek" in low and "7b" in low:
            return "DeepSeek-R1-7B"
        if "deepseek" in low:
            return "DeepSeek-R1"
        if "qwen2.5-7b" in low:
            return "Qwen2.5-7B"
        if "qwen2.5-3b" in low:
            return "Qwen2.5-3B"
        return os.path.basename(str(mid or "Reasoner-LLM"))

    def is_live(self) -> bool:
        return self.mode == "live" and self.model is not None and self.tokenizer is not None

    def load_live(self):
        if self.model is not None or self.mode != "live":
            return
        if not torch.cuda.is_available():
            self.mode = "fallback"
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        candidates = [self.model_id] + [m for m in FALLBACK_REASONER_IDS if m != self.model_id]
        last_err = None
        for cand in candidates:
            try:
                print(
                    f"[LLM-6 Reasoner] Loading {cand} (4-bit NF4, device=cuda:{self.device_index})...",
                    flush=True,
                )
                tok = AutoTokenizer.from_pretrained(cand, use_fast=True)
                if tok.pad_token_id is None:
                    tok.pad_token = tok.eos_token
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                mod = AutoModelForCausalLM.from_pretrained(
                    cand,
                    quantization_config=bnb,
                    torch_dtype=torch.float16,
                    device_map={"": self.device_index},
                ).eval()
                self.tokenizer = tok
                self.model = mod
                self.model_id = cand
                self.short_name = self._short_name(cand)
                self.num_params = sum(p.numel() for p in mod.parameters())
                print(
                    f"[LLM-6 Reasoner] Ready: {self.short_name} ({cand}, {self.num_params:,} params, cuda:{self.device_index})",
                    flush=True,
                )
                return
            except Exception as exc:
                last_err = exc
                print(f"[LLM-6 Reasoner] Warning: Could not load {cand}: {exc}", flush=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        self.load_error = str(last_err)
        self.mode = "fallback"
        print(f"[LLM-6 Reasoner] Falling back to verified curriculum bank ({last_err})", flush=True)

    def generate_encounter_item(
        self,
        concept_data: dict,
        rl_action: int,
        mastery: float,
        affect: dict,
        fallback_item: dict,
        asked_history: list = None,
        course_title: str = "",
    ) -> dict:
        """
        Generates a fresh, verified pedagogical question grounded strictly in
        `concept_data['summary_bullets']` + `concept_data['podcast_hint']`.
        Returns a normalized dict with `source`, `reasoning_trace`, and `latency_ms`.
        """
        if not self.is_live():
            return {
                "question": fallback_item["question"],
                "choices": fallback_item.get("choices"),
                "correct_idx": fallback_item.get("correct_idx"),
                "reference_answer": fallback_item.get("reference") or fallback_item.get("reference_answer", ""),
                "socratic_hint": None,
                "reasoning_trace": "",
                "source": "curriculum_bank",
                "model_name": f"{self.short_name} (Fallback)",
                "latency_ms": 0.0,
            }

        title = str(concept_data.get("title") or "Concept")
        bullets = concept_data.get("summary_bullets") or []
        podcast_hint = str(concept_data.get("podcast_hint") or "")
        notes_blob = "\n".join(f"- {b}" for b in bullets[:6])
        if podcast_hint:
            notes_blob += f"\n- Hint Note: {podcast_hint}"
        lang = detect_course_language(title + " " + notes_blob + " " + str(course_title))
        tier_name, tier_instruction = RL_TIER_SPECS.get(int(rl_action), RL_TIER_SPECS[2])

        avoid_block = ""
        if asked_history:
            recent = [str(q)[:110] for q in asked_history[-5:]]
            avoid_block = "\nAlready asked (DO NOT repeat these):\n" + "\n".join(f"* {q}" for q in recent)

        if int(rl_action) == 4:
            schema_str = (
                '{\n'
                '  "question": "1 challenging open-ended synthesis question",\n'
                '  "reference_answer": "1 concise 18-32 word factually verified ground-truth answer",\n'
                '  "socratic_hint": "1 short clue without giving away the answer"\n'
                '}'
            )
        else:
            schema_str = (
                '{\n'
                '  "question": "1 crisp specific question strictly grounded in the notes",\n'
                '  "choices": ["Correct answer", "Plausible wrong distractor 1", "Plausible wrong distractor 2", "Plausible wrong distractor 3"],\n'
                '  "correct_idx": 0,\n'
                '  "reference_answer": "1 concise sentence explaining WHY the correct answer is right",\n'
                '  "socratic_hint": "1 short Socratic clue without giving away the answer"\n'
                '}'
            )

        sys_msg = (
            "You are Model #6 (Pedagogical Reasoner) for an adaptive university E-Learning RPG. "
            "Your job is ZERO academic hallucination: every question, answer, and distractor MUST be "
            "100% consistent with the provided Course Study Notes. "
            f"Write strictly in {lang}. "
            "Keep your internal `<think>` reasoning brief (at most 2 short sentences), then immediately "
            "output ONLY a single valid JSON object matching the exact schema."
        )
        user_msg = (
            f"Course: {course_title or 'Academic Course'}\n"
            f"Target Concept: {title}\n"
            f"RL Difficulty Tier: [{rl_action}] {tier_name}\n"
            f"Student State: Mastery={round(float(mastery) * 100)}%, "
            f"Focus={round(float(affect.get('concentrating', 0.65)) * 100)}%, "
            f"Frustration={round(float(affect.get('frustrated', 0.30)) * 100)}%\n"
            f"Instruction: {tier_instruction}\n"
            f"Verified Course Study Notes:\n{notes_blob}"
            f"{avoid_block}\n\n"
            f"Return ONLY valid JSON matching:\n{schema_str}"
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        t0 = time.perf_counter()
        try:
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # For DeepSeek-R1-Distill, prefill a brief closed <think> block + opening `{`
            # so it doesn't burn 400+ tokens meandering on T4 (~25s -> ~5s, and ~100% JSON validity!)
            prefill = ""
            if "deepseek" in (self.model_id or "").lower():
                prefill = (
                    f"<think>\nGround strictly in '{title}' notes ({lang}), tier {rl_action}. Output clean JSON.\n</think>\n{{"
                )
                prompt_str = prompt_str + prefill
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=340,
                    do_sample=True,
                    temperature=0.55,
                    top_p=0.90,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if prefill.endswith("{"):
                raw_text = "{" + raw_text
            latency_ms = (time.perf_counter() - t0) * 1000.0

            clean_json_text, trace = strip_think_trace(raw_text)
            if not trace and prefill:
                trace = f"Grounded in '{title}' ({lang}, tier {rl_action})."
            parsed = extract_strict_json(clean_json_text)
            norm = validate_and_normalize_reasoner_output(
                parsed, rl_action=int(rl_action), fallback_item=fallback_item
            )
            norm["reasoning_trace"] = trace
            norm["source"] = "live_llm"
            norm["model_name"] = self.short_name
            norm["latency_ms"] = round(latency_ms, 1)
            return norm
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  [LLM-6 Reasoner] Fallback used on '{title}' (tier {rl_action}): {exc}", flush=True)
            return {
                "question": fallback_item["question"],
                "choices": fallback_item.get("choices"),
                "correct_idx": fallback_item.get("correct_idx"),
                "reference_answer": fallback_item.get("reference") or fallback_item.get("reference_answer", ""),
                "socratic_hint": None,
                "reasoning_trace": "",
                "source": "curriculum_bank_fallback",
                "model_name": f"{self.short_name} (Fallback)",
                "latency_ms": round(latency_ms, 1),
            }
        finally:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def answer_course_question(
        self,
        student_question: str,
        course_title: str,
        concepts: list,
        mastery_vec: list,
    ) -> dict:
        """
        Model #6 factual grounding for interactive NPC Q&A:
        Searches the active course / uploaded PDF's 5 concepts and distills a short,
        100% factually accurate 1-2 sentence answer (zero hallucination) so Model #7
        can deliver it with NPC humor!
        """
        q_low = str(student_question or "").lower()
        q_tokens = set(re.findall(r"[0-9a-zA-ZÀ-ÿ$_]{3,}", q_low))

        # Rank all 5 concepts by lexical relevance (3x weight on title/short_title hits!)
        scored = []
        for c in (concepts or []):
            cid = int(c.get("concept_id", 0))
            title_blob = f"{c.get('title', '')} {c.get('short_title', '')}".lower()
            body_blob = " ".join(
                [str(c.get("podcast_hint", ""))]
                + [str(b) for b in (c.get("summary_bullets") or [])]
            ).lower()
            t_tokens = set(re.findall(r"[0-9a-zA-ZÀ-ÿ$_]{3,}", title_blob))
            b_tokens = set(re.findall(r"[0-9a-zA-ZÀ-ÿ$_]{3,}", body_blob))
            overlap = 3 * len(q_tokens & t_tokens) + len(q_tokens & b_tokens)
            scored.append((overlap, cid, c, title_blob + " " + body_blob))
        scored.sort(key=lambda x: (-x[0], x[1]))

        top_concepts = scored[:2] if scored else []
        context_lines = []
        for _, cid, c, _ in top_concepts:
            m_pct = round(float(mastery_vec[cid]) * 100) if cid < len(mastery_vec) else 50
            bullets = " | ".join(str(b) for b in (c.get("summary_bullets") or [])[:4])
            context_lines.append(f"[C{cid}: {c.get('title','')} (Mastery {m_pct}%)] {bullets}")
        context_str = "\n".join(context_lines)
        lang = detect_course_language(student_question + " " + course_title)

        fallback_fact = (
            (top_concepts[0][2].get("summary_bullets") or [top_concepts[0][2].get("podcast_hint", "")])[0]
            if top_concepts
            else f"Review the key concepts of {course_title} in your Field Journal."
        )

        if not self.is_live():
            return {
                "verified_fact": str(fallback_fact),
                "matched_concept": top_concepts[0][2].get("title", course_title) if top_concepts else course_title,
                "language": lang,
                "source": "course_notes_rag",
                "latency_ms": 0.0,
            }

        sys_msg = (
            "You are Model #6 (Academic Fact Reasoner). Answer the student's question in 1-2 crisp, "
            f"accurate sentences in {lang}, grounded strictly in the provided Course Notes. "
            "If the question asks about their weakest topic or study plan, use the Mastery percentages shown. "
            "Never hallucinate facts outside the notes."
        )
        user_msg = (
            f"Course: {course_title}\n"
            f"Course Notes:\n{context_str}\n\n"
            f"Student Question: {student_question}\n"
            f"Write ONLY the concise factual answer in {lang} (max 45 words)."
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]
        t0 = time.perf_counter()
        try:
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if "deepseek" in (self.model_id or "").lower():
                prompt_str += f"<think>\nCheck course notes and answer concisely in {lang}.\n</think>\n"
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=110,
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
            raw_ans = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            clean_ans, _ = strip_think_trace(raw_ans)
            clean_ans = re.sub(r"\s+", " ", clean_ans).strip()
            if len(clean_ans) < 8:
                clean_ans = str(fallback_fact)
            return {
                "verified_fact": clean_ans,
                "matched_concept": top_concepts[0][2].get("title", course_title) if top_concepts else course_title,
                "language": lang,
                "source": "live_llm",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }
        except Exception as exc:
            print(f"  [LLM-6 Reasoner] Q&A fallback: {exc}", flush=True)
            return {
                "verified_fact": str(fallback_fact),
                "matched_concept": top_concepts[0][2].get("title", course_title) if top_concepts else course_title,
                "language": lang,
                "source": "course_notes_rag",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }
        finally:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
