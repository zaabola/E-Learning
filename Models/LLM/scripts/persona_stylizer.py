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

import contextlib
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
        "archetype": "cocky, sarcastic, data-obsessed rival who roasts you for sleeping in, brags about benchmarking the whole course before breakfast, and tracks every decimal of your stats",
    },
    "rival_mom": {
        "speaker": "MRS. VANCE",
        "archetype": "warm, gossipy next-door neighbor (Rex's mom) who lovingly embarrasses Rex by spilling stories about him muttering course terms in his sleep and guzzling espresso before dawn",
    },
    "mom": {
        "speaker": "MOM",
        "archetype": "loving, hilariously overprotective RPG mom who compares tough university concepts to soup, sweaters, and doing the dishes",
    },
    "prof_ada": {
        "speaker": "ADA",
        "archetype": "brilliant, coffee-fueled lead professor who cracks dry academic jokes and translates DKT metrics into clear study strategy",
    },
    "hv1": {
        "speaker": "BRAM",
        "archetype": "dramatic, slightly paranoid Willowbrook old-timer terrified of tall grass and wild bugs who delivers over-the-top doom warnings mixed with sneaky study clues",
    },
    "hv2": {
        "speaker": "LUNA",
        "archetype": "hyper-caffeinated systems-theory fangirl obsessed with underlying rules, elegance, and clean mental models who talks excitedly with nerdy enthusiasm",
    },
    "tr_cal": {
        "speaker": "CAL",
        "archetype": "energetic trail scout who treats every quiz like an extreme outdoor field expedition",
    },
    "tr_mira": {
        "speaker": "MIRA",
        "archetype": "razor-sharp, deadpan sarcastic strategist who roasts sloppy guesses while calmly sipping lemonade",
    },
    "sv1": {
        "speaker": "GARR",
        "archetype": "dramatic, battle-scarred Stonegate Gym veteran who barely survived Mason's boss exam and whispers intense survival tips",
    },
    "sv2": {
        "speaker": "PIPA",
        "archetype": "upbeat, gossipy Stonegate bird-lover who raves about the healing machine and jokes about sleep-deprived students cramming before the gym",
    },
    "nurse": {
        "speaker": "NURSE",
        "archetype": "cheerful, slightly sassy healing-center nurse who heals both tired creatures and fried student brains with witty medical puns",
    },
    "clerk": {
        "speaker": "CLERK",
        "archetype": "fast-talking Poke Mart hustler who pitches potions and study facts like limited-time infomercial deals",
    },
    "tr_rocco": {
        "speaker": "ROCCO",
        "archetype": "gruff, practical veteran researcher who treats tough concepts like lifting heavy boulders",
    },
    "tr_mason": {
        "speaker": "MASON",
        "archetype": "stoic, imposing Granite Gym Leader who speaks in terse stone-and-foundation metaphors and demands deep synthesis",
    },
    "villager": {
        "speaker": "TOWNIE",
        "archetype": "quirky, gossipy Willowbrook villager with deadpan RPG humor who overhears rumors about the course and drops sneaky study clues",
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


def _detect_user_language(text: str, default_lang: str = "English") -> str:
    """
    Detects the language of the player's own chat message strictly (independent
    of `course_title`), so asking in English NEVER forces a French reply.
    """
    s = str(text or "").strip()
    low = s.lower()
    if re.search(r"\b(in english|speak english|english please|en anglais)\b", low):
        return "English"
    if re.search(r"\b(in french|speak french|en français|en francais|parle français|parle francais)\b", low):
        return "French"
    if re.search(r"[éèêëàâçùûîïôœ]", s, re.IGNORECASE):
        return "French"
    fr_hits = len(
        re.findall(
            r"\b(bonjour|salut|quoi|qoui|pourquoi|comment|quel|quelle|quels|cours|aujourd|explique|aide|moi|merci|oui|non|est-ce|c'est|les|des|une|dans|pour|avec|mon|ton|sur)\b",
            low,
        )
    )
    en_hits = len(
        re.findall(
            r"\b(hi|hello|hey|bro|what|why|how|can|you|speak|english|explain|help|me|today|course|about|tell|give|hint|weakest|topic|please|thanks|yes|no|the|this|that|is|are)\b",
            low,
        )
    )
    if fr_hits > en_hits:
        return "French"
    if en_hits > fr_hits:
        return "English"
    return default_lang


def _repair_and_parse_json(raw: str) -> dict:
    """
    Extracts JSON from LLM output and repairs minor truncation (unclosed quote,
    trailing comma, or unclosed braces/brackets) if hit near `max_new_tokens`.
    """
    try:
        return extract_strict_json(raw)
    except Exception:
        s = str(raw or "").strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        start = s.find("{")
        if start == -1:
            raise
        frag = s[start:]
        in_str = False
        esc = False
        stack = []
        for ch in frag:
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
                elif ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]" and stack and stack[-1] == ch:
                    stack.pop()
        if in_str:
            frag += '"'
        frag = re.sub(r",\s*$", "", frag)
        frag += "".join(reversed(stack))
        frag = re.sub(r",\s*([}\]])", r"\1", frag)
        return json.loads(frag)


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

    def _base_model_ctx(self):
        """
        Temporarily disables the specialized battle-taunt QLoRA adapter for open-ended
        humorous NPC chat & multi-line course storytelling, preserving full conversational
        Meta-Llama-3.1-8B-Instruct reasoning and language switching with zero extra VRAM.
        """
        if self.qlora_active and self.model is not None and hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        return contextlib.nullcontext()

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
                    max_new_tokens=230,
                    do_sample=True,
                    temperature=0.72,
                    top_p=0.90,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            parsed = _repair_and_parse_json(raw_text)

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

    def generate_course_story(
        self,
        course_title: str,
        concepts: list,
    ) -> dict:
        """
        Generates a complete, humorous, course-specific RPG storyline & overworld NPC
        script whenever a course loads or a new PDF is uploaded!
        Matches the PDF's language (French or English) and weaves all 5 concepts into
        the town's plot and every NPC's personality.
        """
        titles = [str(c.get("short_title") or c.get("title") or f"Concept {i}") for i, c in enumerate(concepts[:5])]
        while len(titles) < 5:
            titles.append(course_title)
        bullets_blob = " ".join(
            str((c.get("summary_bullets") or [c.get("title", "")])[0]) for c in concepts[:5]
        )
        lang = detect_course_language(f"{course_title} {' '.join(titles)} {bullets_blob}")

        if lang == "French":
            fb = {
                "language": "French",
                "quest_title": f"L'Expédition : {course_title[:38]}",
                "story_intro": [
                    f"ALERTE LABO : Les archives de '{course_title[:36]}' ont disjoncté dans tout Willowbrook !",
                    f"ADA : Les chercheurs CAL, MIRA, ROCCO et MASON gardent les secrets de {titles[0][:24]} jusqu'à {titles[4][:24]}.",
                    "ADA : Parle à chaque habitant et champion pour glaner des indices uniques, puis restaure le réseau !",
                ],
                "npc_greetings": {
                    "prof_ada": f"ADA : Mon 4e café a refroidi sur '{course_title[:24]}'... Viens, j'ai besoin d'un vrai cerveau pour déboguer ça !",
                    "mom": f"MAMAN : Tu révises '{course_title[:24]}' à jeun ?! Prends ta veste et ne confonds pas {titles[0][:18]} avec la soupe !",
                    "rival_mom": f"MME VANCE : Chut ! REX a marmonné '{titles[0][:22]}' toute la nuit en avalant 5 espressos avant l'aube...",
                    "tr_rex": f"REX : Enfin réveillé ? J'ai déjà indexé '{course_title[:22]}' avant mon petit-déj. Choisis vite ta créature !",
                    "hv1": f"BRAM : Malheureux ! Ne mets PAS un pied dans les hautes herbes sans partenaire—des bugs de {titles[1][:20]} rôdent !",
                    "hv2": f"LUNA : Oh trop bien ! T'as vu l'élégance des règles derrière {titles[2][:24]} ? Le chaos n'a aucune chance !",
                    "tr_cal": f"CAL : Sac bouclé, boussole prête ! Sur ce sentier, c'est moi qui teste ton flair sur {titles[1][:22]} !",
                    "tr_mira": f"MIRA : *sirote sa limonade* Ici l'à-peu-près sur {titles[2][:22]} ne passe pas. Pense clair ou rentre réviser.",
                    "sv1": f"GARR : J'ai survécu de justesse à MASON sur {titles[4][:22]}... N'y va pas avec des attaques molles, gamin !",
                    "sv2": f"PIPA : La machine du Centre a requinqué mon Pidgey en deux secondes—dommage qu'elle révise pas {titles[3][:20]} à ma place !",
                    "nurse": f"INFIRMIÈRE : Cerveau en surchauffe sur {titles[1][:22]} ? Installe ton équipe ici, on soigne corps et neurones !",
                    "clerk": f"VENDEUR : Affaire du siècle : 1 Potion achetée, 1 tuyau malin sur {titles[2][:20]} offert ! Qui dit mieux ?",
                    "tr_rocco": f"ROCCO : {titles[3][:24]}, c'est du granit pur, gamin ! Voyons si tes bases plient sous le poids !",
                    "tr_mason": f"MASON : La roche ne cède qu'à la maîtrise totale de {titles[4][:22]}. Montre-moi des fondations solides.",
                    "villager": f"HABITANT : On chuchote partout à Willowbrook que '{course_title[:26]}' va décider du prochain champion !",
                },
                "source": "story_template",
                "model_name": f"{self.short_name} (Fallback)",
                "latency_ms": 0.0,
            }
        else:
            fb = {
                "language": "English",
                "quest_title": f"The {course_title[:36]} Chronicle",
                "story_intro": [
                    f"LAB ALERT: Willowbrook's '{course_title[:34]}' knowledge grid just scrambled itself into chaos!",
                    f"ADA: Scouts CAL, MIRA, ROCCO, and Boss MASON guard the pillars from {titles[0][:22]} to {titles[4][:22]}.",
                    "ADA: Chat with every townie, mom, and trainer for their own take—then battle your way to mastery!",
                ],
                "npc_greetings": {
                    "prof_ada": f"ADA: Cup #4 of coffee went cold decoding '{course_title[:24]}'... Come help me untangle this glorious mess!",
                    "mom": f"MOM: Tackling '{course_title[:24]}' without breakfast?! Grab a sweater and don't mix up {titles[0][:18]} with laundry!",
                    "rival_mom": f"MRS. VANCE: Don't tell REX, but he muttered '{titles[0][:22]}' in his sleep and chugged five espressos at 5 AM!",
                    "tr_rex": f"REX: Finally rolled out of bed? I benchmarked all of '{course_title[:22]}' before sunrise. Pick your partner already!",
                    "hv1": f"BRAM: Hold it right there, kid! Wild {titles[1][:22]} bugs in that tall grass will eat unprepared rookies alive!",
                    "hv2": f"LUNA: Ooh! Have you noticed how cleanly every rule in {titles[2][:22]} fits together?! System design is SO cool!",
                    "tr_cal": f"CAL: Boots laced, field log open! Out here on Fernway Trail, let's stress-test your {titles[1][:22]} instincts!",
                    "tr_mira": f"MIRA: *sips iced lemonade* Hand-waving on {titles[2][:22]} gets shredded fast here. Think sharp or step aside.",
                    "sv1": f"GARR: Listen close—I barely walked out of MASON's {titles[4][:22]} trial intact. Chip damage won't scratch him!",
                    "sv2": f"PIPA: That Center machine healed my Pidgey in two seconds—too bad it can't cram {titles[3][:20]} into my brain too!",
                    "nurse": f"NURSE: Brain overheating from {titles[1][:22]}? Drop off your team here—we fix bruised HP and fried neurons!",
                    "clerk": f"CLERK: Today ONLY: Potions on aisle 2, priceless rumors about {titles[2][:20]} free at checkout! Whaddaya need?",
                    "tr_rocco": f"ROCCO: {titles[3][:24]} takes real grit, kid—none of that glossy textbook fluff. Let's see your backbone!",
                    "tr_mason": f"MASON: Bedrock cares nothing for lucky guesses. Prove unshakable command over {titles[4][:22]} to pass.",
                    "villager": f"TOWNIE: Folks all over Willowbrook are buzzing about '{course_title[:26]}'—ask anyone for local intel!",
                },
                "source": "story_template",
                "model_name": f"{self.short_name} (Fallback)",
                "latency_ms": 0.0,
            }

        if not self.is_live():
            return fb

        concept_list = [f"C{i}: {t}" for i, t in enumerate(titles)]
        sys_msg = (
            "You are Model #7, a hilarious, witty Retro-GBA RPG Storyteller and Persona Stylizer. "
            f"Write strictly in {lang}. Turn the uploaded university course into a funny, memorable retro-RPG adventure "
            "where EVERY SINGLE NPC has a completely distinct voice and opening line (zero repetition!). "
            "Every line must be punchy (<= 132 characters), start with the character's prefix (or match their role), "
            "accurately weave in the course topics, and invite the player to ask questions.\n"
            "Output ONLY a strict JSON object with keys:\n"
            "- `quest_title`: short adventurous RPG quest name\n"
            "- `story_intro`: list of 3 short strings introducing the course crisis\n"
            "- `npc_greetings`: object with ALL 14 DISTINCT keys:\n"
            "  * `prof_ada` (ADA: coffee-addicted brilliant professor welcoming you to the lab)\n"
            "  * `mom` (MOM: overprotective player's mom fussing about sweaters/food while you study)\n"
            "  * `rival_mom` (MRS. VANCE: Rex's gossipy mom spilling embarrassing habits about Rex overstudying)\n"
            "  * `tr_rex` (REX: cocky sarcastic rival in the lab bragging he already mastered the course)\n"
            "  * `hv1` (BRAM: dramatic paranoid old-timer outside town warning about wild bugs in tall grass)\n"
            "  * `hv2` (LUNA: hyper-enthusiastic nerd gushing about the rules/patterns of the course)\n"
            "  * `tr_cal` (CAL: energetic outdoor trail scout ready for field experiments)\n"
            "  * `tr_mira` (MIRA: deadpan sarcastic strategist sipping lemonade and roasting sloppy thinkers)\n"
            "  * `sv1` (GARR: battle-scarred veteran outside the Gym whispering dramatic warnings about Mason)\n"
            "  * `sv2` (PIPA: upbeat Stonegate fan praising the Center machine and joking about cramming)\n"
            "  * `nurse` (NURSE: sassy healing-center nurse fixing both HP and fried student brains)\n"
            "  * `clerk` (CLERK: fast-talking Poke Mart salesman pitching potions + course intel deals)\n"
            "  * `tr_rocco` (ROCCO: gruff practical gym trainer who respects gritty effort)\n"
            "  * `tr_mason` (MASON: stoic Granite Gym Boss demanding rock-solid mastery)"
        )
        user_msg = json.dumps(
            {
                "task": "generate_course_rpg_story",
                "language": lang,
                "course_title": course_title[:60],
                "concepts": concept_list,
                "tone": "14 completely unique, funny, memorable retro Pokemon/GBA RPG character introductions",
            },
            ensure_ascii=False,
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
            first_dev = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(first_dev)
            with self._base_model_ctx():
                with torch.no_grad():
                    out_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=920,
                        do_sample=True,
                        temperature=0.78,
                        top_p=0.92,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            parsed = _repair_and_parse_json(raw_text)

            q_title = _clip_gba_line(parsed.get("quest_title") or fb["quest_title"], 64)
            raw_intro = parsed.get("story_intro")
            intro = (
                [_clip_gba_line(x) for x in raw_intro[:3] if str(x).strip()]
                if isinstance(raw_intro, list) and len(raw_intro) >= 2
                else fb["story_intro"]
            )
            greetings = dict(fb["npc_greetings"])
            raw_g = parsed.get("npc_greetings")
            if isinstance(raw_g, dict):
                for k in list(greetings.keys()):
                    val = raw_g.get(k)
                    if isinstance(val, str) and len(val.strip()) >= 10:
                        greetings[k] = _clip_gba_line(val)

            return {
                "language": lang,
                "quest_title": q_title,
                "story_intro": intro,
                "npc_greetings": greetings,
                "source": "live_qlora" if self.qlora_active else "live_llm",
                "model_name": self.short_name,
                "latency_ms": round(latency_ms, 1),
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  [LLM-7 Stylizer] Course story fallback: {exc}", flush=True)
            fb["latency_ms"] = round(latency_ms, 1)
            return fb
        finally:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def generate_npc_chat_reply(
        self,
        npc_key: str,
        student_question: str,
        verified_fact: str = "",
        course_title: str = "",
        matched_concept: str = "",
        overall_mastery: float = 0.5,
        chat_history: list = None,
        language: str = None,
        concepts: list = None,
        mastery_vec: list = None,
    ) -> dict:
        """
        Interactive Overworld NPC Q&A powered by base `Meta-Llama-3.1-8B-Instruct`
        (temporarily disabling the battle-JSON QLoRA adapter via `_base_model_ctx()`
        so dialogue flows naturally in English OR French!) grounded strictly in the
        active course's 5 concepts + DKT mastery.
        """
        pk = npc_key if npc_key in PERSONA_PROFILES else "villager"
        profile = PERSONA_PROFILES[pk]
        speaker = profile["speaker"]
        course_lang = detect_course_language(f"{course_title} {verified_fact}")
        lang = language or _detect_user_language(student_question, default_lang=course_lang)
        m_pct = round(float(overall_mastery) * 100)

        # Build concise 5-concept cheat-sheet so NPC can summarize the whole course OR dive into any module
        cheat_lines = []
        for idx, c in enumerate((concepts or [])[:5]):
            cid = int(c.get("concept_id", idx))
            c_m = round(float(mastery_vec[cid]) * 100) if (mastery_vec and cid < len(mastery_vec)) else m_pct
            c_title = str(c.get("title") or c.get("short_title") or f"Concept {cid}")
            bullets = c.get("summary_bullets") or []
            b0 = str(bullets[0] if bullets else (c.get("podcast_hint") or ""))[:115]
            cheat_lines.append(f"- C{cid} ({c_m}% mastery): {c_title} — {b0}")
        cheat_sheet = "\n".join(cheat_lines) if cheat_lines else (
            f"- {matched_concept or course_title}: {verified_fact[:180]}"
        )

        # Casual vs course-inquiry detection for clean fallback messages
        low_q = re.sub(r"[^a-z0-9à-ÿ\s]", " ", str(student_question or "").lower()).strip()
        is_casual = bool(
            re.fullmatch(r"(hi|hey|hello|yo|salut|bonjour|coucou|bro|bruh|ok|yes|no|oui|non|thanks|merci)( \w+){0,2}", low_q)
            or re.search(r"\b(speak in english|speak english|en anglais|en français)\b", low_q)
        )
        topic_short = str(matched_concept or course_title or "this course")[:34]
        casual_map_fr = {
            "tr_rex": f"REX : Salut le dormeur ! T'as enfin fini ton café ? Pose une question sur '{course_title[:24]}' et voyons ton niveau !",
            "rival_mom": f"MME VANCE : Oh bonjour ! REX fait déjà ses fiches sur '{course_title[:22]}' depuis 5h... Tu veux un potin ou un indice ?",
            "mom": f"MAMAN : Coucou mon cœur ! T'as mangé avant de plonger dans '{course_title[:22]}' ? Pose-moi tes questions !",
            "prof_ada": f"ADA : Bonjour chercheur ! Pose-moi n'importe quoi sur '{course_title[:24]}' pendant que mon labo chauffe !",
            "hv1": f"BRAM : Psst ! Ne reste pas planté là—demande-moi plutôt comment survivre à '{course_title[:22]}' dans les hautes herbes !",
            "hv2": f"LUNA : Coucou !! T'es aussi fan que moi de la logique derrière '{course_title[:22]}' ? Demande-moi n'importe quelle règle !",
            "sv1": f"GARR : Hé, gamin... Avant d'entrer voir MASON, demande-moi quels pièges de '{course_title[:22]}' m'ont coûté cher !",
            "sv2": f"PIPA : Coucou ! Tu prends une pause révision sur '{course_title[:22]}' ? Balance ta question !",
        }
        casual_map_en = {
            "tr_rex": f"REX: Hey sleepyhead! Done yawning yet? Ask me about '{course_title[:24]}' and watch how fast I solve it.",
            "rival_mom": f"MRS. VANCE: Why hello! REX has been muttering about '{course_title[:22]}' all morning—want gossip or study help?",
            "mom": f"MOM: Sweetheart! Did you eat before tackling '{course_title[:24]}'? Ask me anything you need!",
            "prof_ada": f"ADA: Welcome, researcher! Ask me anything about '{course_title[:24]}' while the lab diagnostics hum!",
            "hv1": f"BRAM: Psst! Don't stand out here clueless—ask me how to survive '{course_title[:22]}' before hitting the tall grass!",
            "hv2": f"LUNA: Hii!! Ready to nerd out over the rules in '{course_title[:22]}'? Ask me how anything connects!",
            "sv1": f"GARR: Keep your voice down, kid... Ask me what '{course_title[:22]}' traps almost flattened me at Stonegate Gym!",
            "sv2": f"PIPA: Hey! Taking a breather before cramming '{course_title[:22]}'? Fire away with a question!",
        }
        if is_casual:
            if lang == "French":
                fb_reply = casual_map_fr.get(
                    pk,
                    f"{speaker} : Hé ! Ça roule ! Pose-moi une question sur '{course_title[:28]}' et je t'aide à plier les dresseurs !",
                )
            else:
                fb_reply = casual_map_en.get(
                    pk,
                    f"{speaker}: Hey there! Ask me anything about '{course_title[:28]}'—concepts, traps, or your weakest spot!",
                )
        else:
            if lang == "French":
                fb_reply = f"{speaker} : Sur {topic_short}, retiens surtout ceci : {verified_fact} (et à {m_pct}%, tu gères !)"
            else:
                fb_reply = f"{speaker}: On {topic_short}, here's the key: {verified_fact} (at {m_pct}% mastery, you've got this!)"

        base_chat_name = f"{self._short_name(self.model_id, False)} (Chat)"
        if not self.is_live():
            return {
                "npc": pk,
                "speaker": speaker,
                "reply": fb_reply,
                "verified_fact": verified_fact,
                "matched_concept": matched_concept,
                "source": "persona_template",
                "model_name": f"{self.short_name} (Fallback)",
                "latency_ms": 0.0,
            }

        sys_msg = (
            f"You are {speaker} in the retro-GBA E-Learning RPG KnowledgeQuest.\n"
            f"Personality: {profile['archetype']}.\n"
            f"Active Course: {course_title} (Player Overall DKT Mastery: {m_pct}%).\n"
            f"Course Cheat-Sheet (use ONLY these verified facts if asked about course content):\n{cheat_sheet}\n\n"
            f"Strict Conversation Rules:\n"
            f"1. Reply STRICTLY in {lang} (if the player speaks English or asks for English, answer in English even if the notes are in French!).\n"
            f"2. Reply in 2-3 punchy, genuinely FUNNY, conversational sentences (max 280 chars total). NO markdown, NO bullet lists, NO JSON braces.\n"
            f"3. Match the player's actual message! If they just say hi / casual banter, joke back in character and invite a question about the course (do NOT dump random definitions). "
            f"If they ask about today's course, a concept, or what to study next, explain it clearly and hilariously from the Cheat-Sheet above."
        )
        hist_msgs = [{"role": "system", "content": sys_msg}]
        for turn in (chat_history or [])[-6:]:
            r = turn.get("role")
            c = str(turn.get("content") or "").strip()
            if r in ("user", "assistant") and c:
                hist_msgs.append({"role": r, "content": c[:260]})
        hist_msgs.append({"role": "user", "content": str(student_question or "").strip()})

        t0 = time.perf_counter()
        try:
            prompt_str = self.tokenizer.apply_chat_template(
                hist_msgs, tokenize=False, add_generation_prompt=True
            )
            first_dev = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(first_dev)
            with self._base_model_ctx():
                with torch.no_grad():
                    out_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.75,
                        top_p=0.92,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
            raw_reply = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            clean_reply = re.sub(r"\s+", " ", raw_reply).strip().strip('"')
            if clean_reply.startswith("{") and "}" in clean_reply:
                try:
                    obj = _repair_and_parse_json(clean_reply)
                    if isinstance(obj, dict):
                        vals = [v for v in obj.values() if isinstance(v, str) and len(v) > 8]
                        if vals:
                            clean_reply = " ".join(vals[:2])
                        elif isinstance(obj.get("lines"), list):
                            clean_reply = " ".join(str(x) for x in obj["lines"][:3])
                except Exception:
                    pass
            if len(clean_reply) < 8:
                clean_reply = fb_reply
            if not clean_reply.upper().startswith(speaker):
                clean_reply = f"{speaker}: {clean_reply}"
            return {
                "npc": pk,
                "speaker": speaker,
                "reply": clean_reply[:340],
                "verified_fact": verified_fact,
                "matched_concept": matched_concept,
                "source": "live_llm",
                "model_name": base_chat_name,
                "latency_ms": round(latency_ms, 1),
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  [LLM-7 Stylizer] NPC chat fallback ({pk}): {exc}", flush=True)
            return {
                "npc": pk,
                "speaker": speaker,
                "reply": fb_reply,
                "verified_fact": verified_fact,
                "matched_concept": matched_concept,
                "source": "persona_template_fallback",
                "model_name": f"{base_chat_name} (Fallback)",
                "latency_ms": round(latency_ms, 1),
            }
        finally:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
