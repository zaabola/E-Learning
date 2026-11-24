"""
Model #7: Retro-GBA Persona & Narrative Stylizer LLM (`Models/LLM/scripts/persona_stylizer.py`)
===============================================================================================
Powered by `NousResearch/Meta-Llama-3.1-8B-Instruct` (100% ungated Meta-Llama-3.1-8B)
with automatic loading of the trained QLoRA adapter from
`Models/LLM/checkpoints/llama31_rpg_stylizer_qlora/` when available!

Smart 2x T4 VRAM Placement:
  - Uses 4-bit NF4 double-quantization with `device_map="auto"` and calibrated
    `max_memory={0: "13.2GiB", 1: "11.2GiB"}` so Model #6 (`DeepSeek-R1-7B`, cuda:0),
    Model #7 (`Llama-3.1-8B` + QLoRA), Model #5 (`Qwen3-VL-8B`, cuda:1), Model #4
    (`LLMDet-Large`, cuda:0), and Model #3 (`DeBERTa-v3-Small`, cuda:0) all coexist
    simultaneously on Kaggle 2x Tesla T4 (30 GB) with zero OOM!
  - Automatic fallback chain: `NousResearch/Meta-Llama-3.1-8B-Instruct` ->
    `unsloth/Llama-3.2-3B-Instruct` -> deterministic persona template bank.

Responsibilities:
  - Takes the Pedagogical Reasoner's VERIFIED topic/fact + RPG character persona
    (`tr_rex`, `tr_cal`, `tr_mira`, `tr_rocco`, `tr_mason`, `wild`, `prof_ada`)
    and injects witty, punchy, retro-GBA character dialogue (each line <= 135 chars)
    WITHOUT ever altering the verified academic question, choices, or answer.
"""

import json
import os
import random
import re
import time
import torch

from pedagogical_reasoner import detect_course_language, extract_strict_json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_ADAPTER_DIR = os.path.join(LLM_ROOT, "checkpoints", "llama31_rpg_stylizer_qlora")

PRIMARY_STYLIZER_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"
FALLBACK_STYLIZER_IDS = [
    "unsloth/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

STYLIZER_SYSTEM_PROMPT = (
    "You are the Retro-GBA RPG Persona & Narrative Stylizer for KnowledgeQuest. "
    "Given a speaker persona, academic concept, learner state (DKT mastery, streak, affect), "
    "and event type, output ONLY a strict JSON object with short, witty, in-character retro RPG "
    "dialogue lines (each <= 130 characters, no markdown, no emojis) in the requested language. "
    "NEVER invent false academic claims or contradict the provided verified fact."
)

PERSONA_PROFILES = {
    "tr_rex": {
        "speaker": "REX",
        "archetype": "cocky, data-obsessed rival who tracks every decimal of your stats and hates losing",
    },
    "tr_cal": {
        "speaker": "CAL",
        "archetype": "energetic trail scout who treats every battle like a thrilling field experiment",
    },
    "tr_mira": {
        "speaker": "MIRA",
        "archetype": "razor-sharp, sarcastic strategist who tests precision and penalizes sloppy guesses",
    },
    "tr_rocco": {
        "speaker": "ROCCO",
        "archetype": "gruff, practical veteran researcher who respects gritty effort over flashy talk",
    },
    "tr_mason": {
        "speaker": "MASON",
        "archetype": "stoic, imposing Gym Leader who speaks in terse stone-and-foundation metaphors",
    },
    "prof_ada": {
        "speaker": "ADA",
        "archetype": "warm, encouraging, gently witty lead professor who translates DKT metrics into clear study strategy",
    },
    "wild": {
        "speaker": "NARRATOR",
        "archetype": "classic punchy GBA battle narrator with dry retro-RPG charm",
    },
}

FALLBACK_BANTER_EN = {
    "tr_rex": {
        "intro_taunt": "REX: My model says you'll slip on {topic}. Prove my variance wrong!",
        "on_correct": "REX: Tsk! Lucky convergence... alright, that {topic} strike was sharp.",
        "on_partial": "REX: Half-baked gradient! Almost nailed {topic}, try sharper next turn.",
        "on_incorrect": "REX: Ha! Total overfitting! Review {topic} before challenging me again.",
        "hint_intro": "REX: Fine, take this {topic} clue so our duel stays interesting!",
    },
    "tr_cal": {
        "intro_taunt": "CAL: Awesome! A live {topic} field test—let's see what you've got!",
        "on_correct": "CAL: Bullseye! That {topic} strike lit up the whole trail!",
        "on_partial": "CAL: So close! One more detail on {topic} and you've got it!",
        "on_incorrect": "CAL: Whoops, tripped on a root! Shake it off and rethink {topic}!",
        "hint_intro": "CAL: Peek at my trail notes—this {topic} clue will steer you right!",
    },
    "tr_mira": {
        "intro_taunt": "MIRA: Sloppy guesses fail fast here. Show me true {topic} clarity.",
        "on_correct": "MIRA: Crisp, disciplined, accurate. You earned that strike.",
        "on_partial": "MIRA: Almost sharp, yet missing the crux of {topic}. Refine it.",
        "on_incorrect": "MIRA: Too hasty. Re-examine {topic} instead of guessing blind.",
        "hint_intro": "MIRA: Pause and analyze—anchor your logic with this clue first.",
    },
    "tr_rocco": {
        "intro_taunt": "ROCCO: Enough theory—let's see if your {topic} holds up under fire!",
        "on_correct": "ROCCO: Hah! Solid hit! You've got real {topic} muscle behind that!",
        "on_partial": "ROCCO: Scratched the armor—put a bit more backbone into {topic}!",
        "on_incorrect": "ROCCO: Clang! Bounced right off. Study {topic} and hit back harder!",
        "hint_intro": "ROCCO: Take a breather, kid—this {topic} pointer will steady your aim.",
    },
    "tr_mason": {
        "intro_taunt": "MASON: Granite yields only to mastery. Prove your command of {topic}.",
        "on_correct": "MASON: Tremendous! Your {topic} mastery cracked my granite guard!",
        "on_partial": "MASON: A hairline fracture in the stone. Deepen your {topic} synthesis.",
        "on_incorrect": "MASON: Stone does not budge for guesswork. Fortify your {topic} roots.",
        "hint_intro": "MASON: Even masons check the blueprint. Study this {topic} principle.",
    },
    "wild": {
        "intro_taunt": "Field alert! Channel your {topic} insight to supercharge your move!",
        "on_correct": "Critical focus! Your {topic} reasoning powered straight through!",
        "on_partial": "Glancing hit! Almost nailed {topic}—tighten up that answer!",
        "on_incorrect": "Focus slipped! Review the {topic} takeaway and strike back!",
        "hint_intro": "Prof. Ada's radio crackles with a timely {topic} hint!",
    },
}

FALLBACK_BANTER_FR = {
    "tr_rex": {
        "intro_taunt": "REX : Mon modèle prédit que tu vas coincer sur {topic}. Prouve le contraire !",
        "on_correct": "REX : Tsk ! Convergence inattendue... joli coup sur {topic}.",
        "on_partial": "REX : Presque ! Il manque un détail clé sur {topic} pour me battre.",
        "on_incorrect": "REX : Ha ! Sur-apprentissage ! Révise {topic} avant de me défier.",
        "hint_intro": "REX : Prends cet indice sur {topic} pour que le duel reste sérieux !",
    },
    "tr_cal": {
        "intro_taunt": "CAL : Super ! Un défi terrain sur {topic}—montre ce que tu sais !",
        "on_correct": "CAL : Dans le mille ! Ton attaque {topic} était imparable !",
        "on_partial": "CAL : Tout près ! Un détail de plus sur {topic} et c'est gagné !",
        "on_incorrect": "CAL : Petit faux pas ! Respire et revois l'essentiel de {topic}.",
        "hint_intro": "CAL : Jette un œil à cet indice sur {topic}, ça va t'aider !",
    },
    "tr_mira": {
        "intro_taunt": "MIRA : L'à-peu-près échoue vite ici. Montre ta précision sur {topic}.",
        "on_correct": "MIRA : Net, rigoureux, exact. Tu as mérité ce coup critique.",
        "on_partial": "MIRA : Bonne intuition, mais il manque le point décisif sur {topic}.",
        "on_incorrect": "MIRA : Trop précipité. Reprends {topic} calmement sans deviner.",
        "hint_intro": "MIRA : Prends une seconde et appuie-toi sur cet indice sur {topic}.",
    },
    "tr_rocco": {
        "intro_taunt": "ROCCO : Assez de blabla—voyons si ton {topic} tient le choc !",
        "on_correct": "ROCCO : Hah ! Solide ! Tu maîtrises {topic} comme un vrai pro !",
        "on_partial": "ROCCO : Pas mal ! Complète encore ton idée sur {topic} !",
        "on_incorrect": "ROCCO : Raté ! Relève-toi, relis {topic} et frappe plus fort !",
        "hint_intro": "ROCCO : Tiens, un tuyau de terrain sur {topic} pour recadrer ton tir.",
    },
    "tr_mason": {
        "intro_taunt": "MASON : Le granit ne cède qu'à la maîtrise. Prouve ta valeur sur {topic}.",
        "on_correct": "MASON : Remarquable ! Ta logique sur {topic} a fendu ma garde !",
        "on_partial": "MASON : Une fissure dans le roc. Approfondis ta synthèse sur {topic}.",
        "on_incorrect": "MASON : La pierre ignore la chance. Consolide tes bases sur {topic}.",
        "hint_intro": "MASON : Même le bâtisseur vérifie ses plans. Médite cet indice.",
    },
    "wild": {
        "intro_taunt": "Alerte terrain ! Mobilise {topic} pour charger ton attaque !",
        "on_correct": "Coup critique ! Ta maîtrise de {topic} a percé la défense !",
        "on_partial": "Coup partiel ! Tu approches du cœur de {topic}—encore un effort !",
        "on_incorrect": "Concentration rompue ! Lis l'explication sur {topic} et riposte !",
        "hint_intro": "Signal radio d'ADA ! Voici un indice rapide sur {topic}.",
    },
}


def _clip_gba_line(text: str, limit: int = 136) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip().strip('"').strip("'")
    if len(s) <= limit:
        return s
    return s[: limit - 3].rstrip(" ,;:-") + "..."


class PersonaStylizerLLM:
    """
    Model #7 wrapper (`NousResearch/Meta-Llama-3.1-8B-Instruct` + optional QLoRA adapter)
    generating crisp retro-GBA character banter & Prof. Ada DKT-aware briefings.
    """

    def __init__(
        self,
        model_id: str = PRIMARY_STYLIZER_ID,
        adapter_dir: str = DEFAULT_ADAPTER_DIR,
        mode: str = "fallback",
    ):
        self.model_id = model_id or PRIMARY_STYLIZER_ID
        self.adapter_dir = adapter_dir or DEFAULT_ADAPTER_DIR
        self.mode = mode  # "live" (Kaggle GPU) or "fallback" (local 4GB PC)
        self.model = None
        self.tokenizer = None
        self.qlora_active = False
        self.short_name = self._short_name(self.model_id, False)
        self.num_params = 0
        self.load_error = None

    @staticmethod
    def _short_name(mid: str, qlora: bool) -> str:
        low = str(mid or "").lower()
        base = "Llama-3.1-8B"
        if "3.2-3b" in low:
            base = "Llama-3.2-3B"
        elif "qwen2.5-3b" in low:
            base = "Qwen2.5-3B"
        elif "3.1-8b" not in low:
            base = os.path.basename(str(mid or "Stylizer-LLM"))
        return f"{base} QLoRA" if qlora else base

    def is_live(self) -> bool:
        return self.mode == "live" and self.model is not None and self.tokenizer is not None

    def load_live(self):
        if self.model is not None or self.mode != "live":
            return
        if not torch.cuda.is_available():
            self.mode = "fallback"
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        num_gpus = torch.cuda.device_count()
        candidates = [self.model_id] + [m for m in FALLBACK_STYLIZER_IDS if m != self.model_id]
        last_err = None

        for cand in candidates:
            try:
                # Calibrated 2x T4 memory cap: keeps cuda:0 <= 13.2 GiB and leaves >= 5.5 GiB
                # free on cuda:1 for Qwen3-VL-8B high-res PDF ingestion!
                if num_gpus >= 2:
                    dev_map = "auto"
                    max_mem = {0: "13.2GiB", 1: "11.2GiB"}
                else:
                    dev_map = {"": 0}
                    max_mem = None

                print(
                    f"[LLM-7 Stylizer] Loading {cand} (4-bit NF4, gpus={num_gpus}, max_mem={max_mem})...",
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
                load_kwargs = {
                    "quantization_config": bnb,
                    "torch_dtype": torch.float16,
                    "device_map": dev_map,
                }
                if max_mem is not None:
                    load_kwargs["max_memory"] = max_mem

                mod = AutoModelForCausalLM.from_pretrained(cand, **load_kwargs).eval()

                # Automatically attach trained QLoRA adapter if present & compatible!
                qlora_loaded = False
                adapter_cfg = os.path.join(self.adapter_dir, "adapter_config.json")
                if os.path.exists(adapter_cfg):
                    try:
                        with open(adapter_cfg, "r", encoding="utf-8") as af:
                            acfg = json.load(af)
                        base_in_cfg = str(acfg.get("base_model_name_or_path", "")).lower()
                        # Only attach if base architecture matches (e.g. Llama-3.1-8B adapter -> Llama-3.1-8B)
                        if (not base_in_cfg) or (cand.lower().split("/")[-1] in base_in_cfg) or (base_in_cfg.split("/")[-1] in cand.lower()):
                            from peft import PeftModel
                            mod = PeftModel.from_pretrained(mod, self.adapter_dir).eval()
                            qlora_loaded = True
                            print(f"[LLM-7 Stylizer] Attached trained RPG QLoRA adapter from {self.adapter_dir}!", flush=True)
                    except Exception as peft_exc:
                        print(f"[LLM-7 Stylizer] Warning: Could not attach QLoRA adapter ({peft_exc}); using base model.", flush=True)

                self.tokenizer = tok
                self.model = mod
                self.model_id = cand
                self.qlora_active = qlora_loaded
                self.short_name = self._short_name(cand, qlora_loaded)
                self.num_params = sum(p.numel() for p in mod.parameters())
                placed = set(getattr(mod, "hf_device_map", {}).values()) or {str(mod.device)}
                print(
                    f"[LLM-7 Stylizer] Ready: {self.short_name} ({cand}, devices={placed}, {self.num_params:,} params)",
                    flush=True,
                )
                return
            except Exception as exc:
                last_err = exc
                print(f"[LLM-7 Stylizer] Warning: Could not load {cand}: {exc}", flush=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        self.load_error = str(last_err)
        self.mode = "fallback"
        print(f"[LLM-7 Stylizer] Falling back to retro-RPG persona templates ({last_err})", flush=True)

    def _fallback_battle_banter(self, persona_key: str, topic: str, lang: str) -> dict:
        pk = persona_key if persona_key in FALLBACK_BANTER_EN else "wild"
        bank = FALLBACK_BANTER_FR if lang == "French" else FALLBACK_BANTER_EN
        short_topic = str(topic or "this concept")[:32]
        raw = bank[pk]
        return {
            "speaker": PERSONA_PROFILES.get(pk, PERSONA_PROFILES["wild"])["speaker"],
            "intro_taunt": _clip_gba_line(raw["intro_taunt"].format(topic=short_topic)),
            "on_correct": _clip_gba_line(raw["on_correct"].format(topic=short_topic)),
            "on_partial": _clip_gba_line(raw["on_partial"].format(topic=short_topic)),
            "on_incorrect": _clip_gba_line(raw["on_incorrect"].format(topic=short_topic)),
            "hint_intro": _clip_gba_line(raw["hint_intro"].format(topic=short_topic)),
            "source": "persona_template",
            "model_name": f"{self.short_name} (Fallback)",
            "latency_ms": 0.0,
        }

    def generate_battle_banter(
        self,
        persona_key: str,
        topic: str,
        verified_fact: str,
        rl_action: int,
        rl_action_name: str,
        mastery: float,
        affect: dict,
        win_streak: int = 0,
        language: str = None,
    ) -> dict:
        """
        Generates in-character GBA battle dialogue lines (`intro_taunt`, `on_correct`,
        `on_partial`, `on_incorrect`, `hint_intro`) strictly without touching academic facts.
        """
        pk = persona_key if persona_key in PERSONA_PROFILES else "wild"
        lang = language or detect_course_language(f"{topic} {verified_fact}")
        fb = self._fallback_battle_banter(pk, topic, lang)
        if not self.is_live():
            return fb

        profile = PERSONA_PROFILES[pk]
        user_payload = {
            "task": "battle_persona_banter",
            "language": lang,
            "persona_key": pk,
            "speaker": profile["speaker"],
            "archetype": profile["archetype"],
            "topic": str(topic)[:48],
            "verified_fact": str(verified_fact)[:180],
            "rl_tier": f"[{rl_action}] {rl_action_name}",
            "student_mastery_pct": round(float(mastery) * 100),
            "student_frustration": round(float(affect.get("frustrated", 0.30)), 2),
            "win_streak": int(win_streak),
            "required_json_keys": ["intro_taunt", "on_correct", "on_partial", "on_incorrect", "hint_intro"],
        }
        messages = [
            {"role": "system", "content": STYLIZER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        t0 = time.perf_counter()
        try:
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            first_dev = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(first_dev)
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.72,
                    top_p=0.90,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            parsed = extract_strict_json(raw_text)

            out = {
                "speaker": profile["speaker"],
                "intro_taunt": _clip_gba_line(parsed.get("intro_taunt") or fb["intro_taunt"]),
                "on_correct": _clip_gba_line(parsed.get("on_correct") or fb["on_correct"]),
                "on_partial": _clip_gba_line(parsed.get("on_partial") or fb["on_partial"]),
                "on_incorrect": _clip_gba_line(parsed.get("on_incorrect") or fb["on_incorrect"]),
                "hint_intro": _clip_gba_line(parsed.get("hint_intro") or fb["hint_intro"]),
                "source": "live_qlora" if self.qlora_active else "live_llm",
                "model_name": self.short_name,
                "latency_ms": round(latency_ms, 1),
            }
            return out
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  [LLM-7 Stylizer] Fallback banter on {pk}/{topic}: {exc}", flush=True)
            fb["latency_ms"] = round(latency_ms, 1)
            return fb
        finally:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def generate_prof_ada_briefing(
        self,
        course_title: str,
        weakest_topic: str,
        weakest_mastery: float,
        overall_mastery: float,
        badges_earned: int,
        win_streak: int,
        frustration: float,
        verified_fact: str,
        starter_name: str = "your partner",
    ) -> dict:
        """
        Generates 3 short, personalized GBA dialog-box lines for Professor ADA in the
        overworld lab based on live DKT mastery, weakest concept, and affective state.
        """
        lang = detect_course_language(f"{course_title} {weakest_topic} {verified_fact}")
        w_pct = round(float(weakest_mastery) * 100)
        avg_pct = round(float(overall_mastery) * 100)
        fact_short = _clip_gba_line(verified_fact, 96)

        if lang == "French":
            fb_lines = [
                _clip_gba_line(f"ADA : Bon retour avec {starter_name} ! Ta maîtrise DKT sur {course_title[:24]} est à {avg_pct}% ({badges_earned}/5 badges)."),
                _clip_gba_line(f"ADA : Priorité suivante : renforce '{weakest_topic[:32]}' ({w_pct}%) au Labo ou sur le sentier."),
                _clip_gba_line(f"ADA : Repère clé : {fact_short}"),
            ]
        else:
            fb_lines = [
                _clip_gba_line(f"ADA: Great to see you and {starter_name}! Your DKT mastery on {course_title[:24]} is {avg_pct}% ({badges_earned}/5 badges)."),
                _clip_gba_line(f"ADA: Next focus: strengthen '{weakest_topic[:32]}' ({w_pct}%) in the Lab Monitor or along Fernway Trail."),
                _clip_gba_line(f"ADA: Key anchor: {fact_short}"),
            ]

        if not self.is_live():
            return {
                "npc": "prof_ada",
                "speaker": "ADA",
                "lines": fb_lines,
                "source": "persona_template",
                "model_name": f"{self.short_name} (Fallback)",
                "latency_ms": 0.0,
            }

        profile = PERSONA_PROFILES["prof_ada"]
        user_payload = {
            "task": "prof_ada_briefing",
            "language": lang,
            "persona_key": "prof_ada",
            "speaker": profile["speaker"],
            "archetype": profile["archetype"],
            "course_title": str(course_title)[:50],
            "weakest_topic": str(weakest_topic)[:48],
            "weakest_mastery_pct": w_pct,
            "overall_mastery_pct": avg_pct,
            "badges_earned": int(badges_earned),
            "win_streak": int(win_streak),
            "frustration": round(float(frustration), 2),
            "verified_fact": str(verified_fact)[:180],
            "required_json_keys": ["lines"],
        }
        messages = [
            {"role": "system", "content": STYLIZER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        t0 = time.perf_counter()
        try:
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            first_dev = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(first_dev)
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=180,
                    do_sample=True,
                    temperature=0.68,
                    top_p=0.90,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            parsed = extract_strict_json(raw_text)

            raw_lines = parsed.get("lines")
            if isinstance(raw_lines, list) and len(raw_lines) >= 2:
                lines = []
                for ln in raw_lines[:3]:
                    s = _clip_gba_line(ln)
                    if not s.upper().startswith("ADA"):
                        s = _clip_gba_line(f"ADA: {s}")
                    lines.append(s)
            else:
                lines = fb_lines

            return {
                "npc": "prof_ada",
                "speaker": "ADA",
                "lines": lines,
                "source": "live_qlora" if self.qlora_active else "live_llm",
                "model_name": self.short_name,
                "latency_ms": round(latency_ms, 1),
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  [LLM-7 Stylizer] Fallback Prof. Ada briefing: {exc}", flush=True)
            return {
                "npc": "prof_ada",
                "speaker": "ADA",
                "lines": fb_lines,
                "source": "persona_template_fallback",
                "model_name": f"{self.short_name} (Fallback)",
                "latency_ms": round(latency_ms, 1),
            }
        finally:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
