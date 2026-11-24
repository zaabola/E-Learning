"""
Stage 4 Multi-Agent LLM Benchmark Suite (`Models/LLM/scripts/evaluate_stage4_llms.py`)
======================================================================================
Evaluates BOTH Stage 4 LLMs on Kaggle GPUs and writes reproducible report tables
to `Models/LLM/results/llm_stage4_benchmark.csv` and `.json`:

  1. Model #6 — Pedagogical Reasoner (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`):
     - Tests all 5 RL tiers `[0..4]` across English + French academic concepts.
     - Measures: Strict JSON Validity (%), Distractor Uniqueness (%), Language Match (%),
       Reasoning Trace Rate (%), and Mean Generation Latency (s).

  2. Model #7 — Persona Stylizer (`NousResearch/Meta-Llama-3.1-8B-Instruct` Base vs QLoRA):
     - Evaluates on held-out `Models/LLM/data/rpg_persona_test.jsonl` (60 test cases).
     - Measures: Held-Out Test Perplexity (PPL), Strict JSON Validity (%), GBA Box Length
       Compliance (`<= 136 chars`, %), Speaker Voice Tag Accuracy (%), Language Match (%),
       and Mean Generation Latency (s).

Usage on Kaggle (after running `train_persona_qlora.py`):
  !python -u Models/LLM/scripts/evaluate_stage4_llms.py
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(LLM_ROOT, "data")
RESULTS_DIR = os.path.join(LLM_ROOT, "results")
ADAPTER_DIR = os.path.join(LLM_ROOT, "checkpoints", "llama31_rpg_stylizer_qlora")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pedagogical_reasoner import (
    PRIMARY_REASONER_ID,
    PedagogicalReasonerLLM,
    detect_course_language,
)
from persona_stylizer import (
    PRIMARY_STYLIZER_ID,
    STYLIZER_SYSTEM_PROMPT,
    extract_strict_json,
)
from train_persona_qlora import ChatSFTDataset, evaluate_loss, make_collate_fn

REASONER_BENCH_CONCEPTS = [
    {
        "title": "Bounding Boxes, IoU & Non-Maximum Suppression (NMS)",
        "language": "English",
        "summary_bullets": [
            "IoU divides the intersection area of two bounding boxes by their combined union area.",
            "Vanilla IoU loss gives zero gradient when boxes do not overlap; CIoU adds center-distance and aspect-ratio penalties.",
            "NMS sorts detections by confidence, keeps the highest-scoring box, and suppresses same-class boxes above the IoU threshold.",
        ],
        "podcast_hint": "IoU measures overlap over union; NMS removes redundant overlapping duplicate boxes.",
    },
    {
        "title": "Architecture du Serveur Oracle — Mémoire SGA et Processus",
        "language": "French",
        "summary_bullets": [
            "La mémoire SGA est partagée entre tous les processus serveur et contient le cache de tampons de la base de données.",
            "Le processus DBWn écrit les blocs modifiés du cache de tampons vers les fichiers de données sur disque.",
            "Le processus SMON effectue automatiquement la récupération de l'instance après un arrêt brutal.",
        ],
        "podcast_hint": "DBWn écrit les blocs modifiés sur disque, tandis que SMON récupère l'instance après un crash.",
    },
]


def benchmark_reasoner(model_id: str) -> dict:
    print("\n" + "=" * 84)
    print(f"  BENCHMARKING MODEL #6: Pedagogical Reasoner ({model_id})")
    print("=" * 84, flush=True)
    reasoner = PedagogicalReasonerLLM(model_id=model_id, mode="live", device_index=0)
    reasoner.load_live()
    if not reasoner.is_live():
        raise RuntimeError("Could not load live Pedagogical Reasoner on GPU.")

    total = 0
    valid_json = 0
    unique_distractors = 0
    mc_count = 0
    lang_match = 0
    think_present = 0
    latencies = []

    for c_data in REASONER_BENCH_CONCEPTS:
        for rl_act in range(5):
            total += 1
            fb = {
                "question": "Fallback question?",
                "choices": ["A", "B", "C", "D"] if rl_act != 4 else None,
                "correct_idx": 0 if rl_act != 4 else None,
                "reference": c_data["summary_bullets"][0],
            }
            out = reasoner.generate_encounter_item(
                concept_data=c_data,
                rl_action=rl_act,
                mastery=0.45 + 0.08 * rl_act,
                affect={"concentrating": 0.72, "frustrated": 0.25},
                fallback_item=fb,
                course_title=c_data["title"],
            )
            latencies.append(out["latency_ms"] / 1000.0)
            is_live_ok = out["source"] == "live_llm"
            if is_live_ok:
                valid_json += 1
            if rl_act != 4:
                mc_count += 1
                ch = out.get("choices") or []
                if is_live_ok and len(ch) == 4 and len({c.lower() for c in ch}) == 4:
                    unique_distractors += 1
            det_lang = detect_course_language(out["question"] + " " + out["reference_answer"])
            if det_lang == c_data["language"]:
                lang_match += 1
            if out.get("reasoning_trace"):
                think_present += 1
            print(
                f"  [Tier {rl_act} | {c_data['language']:<7}] src={out['source']:<10} | "
                f"{out['latency_ms']/1000.0:.2f}s | Q: {out['question'][:68]}",
                flush=True,
            )

    # Free VRAM before benchmarking Model #7
    del reasoner.model
    reasoner.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": f"Model #6: {reasoner.short_name}",
        "model_id": reasoner.model_id,
        "test_cases": total,
        "json_validity_pct": round(100.0 * valid_json / max(1, total), 2),
        "distractor_uniqueness_pct": round(100.0 * unique_distractors / max(1, mc_count), 2),
        "gba_length_compliance_pct": "-",
        "speaker_voice_match_pct": "-",
        "language_match_pct": round(100.0 * lang_match / max(1, total), 2),
        "heldout_test_ppl": "-",
        "mean_latency_sec": round(sum(latencies) / max(1, len(latencies)), 2),
        "think_trace_pct": round(100.0 * think_present / max(1, total), 2),
    }


def benchmark_stylizer_variant(
    base_model_id: str,
    adapter_dir: str,
    use_qlora: bool,
    test_jsonl: str,
    max_gen_samples: int = 24,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tag = "QLoRA" if use_qlora else "Base"
    print("\n" + "=" * 84)
    print(f"  BENCHMARKING MODEL #7 ({tag}): {base_model_id}")
    print("=" * 84, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    mod = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    if use_qlora:
        from peft import PeftModel
        mod = PeftModel.from_pretrained(mod, adapter_dir)
    mod.eval()

    # 1. Held-out Test Loss & Perplexity on all 60 test samples
    test_ds = ChatSFTDataset(test_jsonl, tok, max_length=512)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=make_collate_fn(tok.pad_token_id))
    test_loss, test_ppl = evaluate_loss(mod, test_loader, device)
    print(f"  [{tag}] Held-out Test Loss={test_loss:.4f} | Test PPL={test_ppl:.2f}", flush=True)

    # 2. Generative Quality Audit on held-out test prompts
    with open(test_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()][:max_gen_samples]

    valid_json = 0
    gba_len_ok = 0
    total_lines = 0
    voice_ok = 0
    lang_ok = 0
    latencies = []

    for row in rows:
        msgs = row["messages"][:-1]
        u_obj = json.loads(msgs[-1]["content"])
        speaker = u_obj.get("speaker", "")
        exp_lang = u_obj.get("language", "English")
        prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt_str, return_tensors="pt").to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = mod.generate(
                **inp,
                max_new_tokens=190,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        latencies.append(time.perf_counter() - t0)
        gen = tok.decode(out_ids[0][inp["input_ids"].shape[1] :], skip_special_tokens=True)

        try:
            parsed = extract_strict_json(gen)
            valid_json += 1
            if u_obj.get("task") == "prof_ada_briefing":
                lines = parsed.get("lines") if isinstance(parsed.get("lines"), list) else []
            else:
                lines = [
                    str(parsed.get(k, ""))
                    for k in ("intro_taunt", "on_correct", "on_partial", "on_incorrect", "hint_intro")
                    if parsed.get(k)
                ]
            if lines:
                for ln in lines:
                    total_lines += 1
                    if len(str(ln).strip()) <= 136:
                        gba_len_ok += 1
                joined = " ".join(str(x) for x in lines)
                if speaker == "NARRATOR" or speaker.upper() in joined.upper():
                    voice_ok += 1
                if detect_course_language(joined) == exp_lang:
                    lang_ok += 1
        except Exception:
            pass

    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = max(1, len(rows))
    return {
        "model": f"Model #7: Llama-3.1-8B ({tag})",
        "model_id": f"{base_model_id} + QLoRA" if use_qlora else base_model_id,
        "test_cases": len(rows),
        "json_validity_pct": round(100.0 * valid_json / n, 2),
        "distractor_uniqueness_pct": "-",
        "gba_length_compliance_pct": round(100.0 * gba_len_ok / max(1, total_lines), 2),
        "speaker_voice_match_pct": round(100.0 * voice_ok / n, 2),
        "language_match_pct": round(100.0 * lang_ok / n, 2),
        "heldout_test_ppl": round(test_ppl, 3),
        "mean_latency_sec": round(sum(latencies) / n, 2),
        "think_trace_pct": "-",
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Stage 4 LLMs (DeepSeek-R1-7B + Llama-3.1-8B Base vs QLoRA)")
    parser.add_argument("--reasoner-model", default=PRIMARY_REASONER_ID)
    parser.add_argument("--stylizer-model", default=PRIMARY_STYLIZER_ID)
    parser.add_argument("--adapter-dir", default=ADAPTER_DIR)
    parser.add_argument("--max-stylizer-samples", type=int, default=24)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_jsonl = os.path.join(DATA_DIR, "rpg_persona_test.jsonl")
    if not os.path.exists(test_jsonl):
        sys.path.insert(0, DATA_DIR)
        import build_persona_dataset
        build_persona_dataset.main()

    summary_rows = []
    summary_rows.append(benchmark_reasoner(args.reasoner_model))
    summary_rows.append(
        benchmark_stylizer_variant(
            args.stylizer_model, args.adapter_dir, False, test_jsonl, args.max_stylizer_samples
        )
    )
    if os.path.exists(os.path.join(args.adapter_dir, "adapter_config.json")):
        summary_rows.append(
            benchmark_stylizer_variant(
                args.stylizer_model, args.adapter_dir, True, test_jsonl, args.max_stylizer_samples
            )
        )
    else:
        print(
            f"\n[Note] No QLoRA adapter found at {args.adapter_dir} yet — "
            "run `python Models/LLM/scripts/train_persona_qlora.py` first to compare Base vs QLoRA!",
            flush=True,
        )

    csv_path = os.path.join(RESULTS_DIR, "llm_stage4_benchmark.csv")
    json_path = os.path.join(RESULTS_DIR, "llm_stage4_benchmark.json")
    fields = [
        "model",
        "model_id",
        "test_cases",
        "json_validity_pct",
        "distractor_uniqueness_pct",
        "gba_length_compliance_pct",
        "speaker_voice_match_pct",
        "language_match_pct",
        "heldout_test_ppl",
        "mean_latency_sec",
        "think_trace_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 84)
    print(f"  STAGE 4 LLM BENCHMARK SAVED -> {csv_path}")
    for r in summary_rows:
        print(
            f"  {r['model']:<32} | JSON={r['json_validity_pct']}% | "
            f"GBA<=136c={r['gba_length_compliance_pct']}% | PPL={r['heldout_test_ppl']} | "
            f"Lat={r['mean_latency_sec']}s"
        )
    print("=" * 84, flush=True)


if __name__ == "__main__":
    main()
