"""
Kaggle QLoRA Fine-Tuning Script for Model #7: Retro-GBA Persona Stylizer
(`Models/LLM/scripts/train_persona_qlora.py`)
=======================================================================
Fine-tunes `NousResearch/Meta-Llama-3.1-8B-Instruct` (100% ungated Meta-Llama-3.1-8B
mirror) using 4-bit NF4 QLoRA (r=16, alpha=32) on `Models/LLM/data/rpg_persona_train.jsonl`
with assistant-only loss masking so the adapter learns crisp, witty retro-GBA character
banter (<= 130 chars/line, strict JSON, EN+FR, 7 distinct RPG voices) without mutating
academic facts.

Usage on Kaggle (GPU T4 / T4 x2, Internet ON):
  !pip install -q -U peft bitsandbytes accelerate transformers
  !python -u Models/LLM/scripts/train_persona_qlora.py

Outputs:
  - Adapter checkpoint : Models/LLM/checkpoints/llama31_rpg_stylizer_qlora/
  - Training metrics   : Models/LLM/results/qlora_training_metrics.json
  - Epoch log CSV      : Models/LLM/results/qlora_training_log.csv
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import torch
from torch.utils.data import Dataset, DataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(LLM_ROOT, "data")
CKPT_DIR = os.path.join(LLM_ROOT, "checkpoints", "llama31_rpg_stylizer_qlora")
RESULTS_DIR = os.path.join(LLM_ROOT, "results")

DEFAULT_BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
FALLBACK_BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"


class ChatSFTDataset(Dataset):
    """
    Tokenizes ChatML / Llama-3 chat examples and masks out system + user tokens
    (`label = -100`) so cross-entropy loss is computed STRICTLY on assistant tokens.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                msgs = row["messages"]
                prompt_text = tokenizer.apply_chat_template(
                    msgs[:-1], tokenize=False, add_generation_prompt=True
                )
                full_text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                full_ids = tokenizer(
                    full_text, add_special_tokens=False, truncation=True, max_length=max_length
                )["input_ids"]
                if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
                    if len(full_ids) < max_length:
                        full_ids = full_ids + [tokenizer.eos_token_id]

                prompt_len = min(len(prompt_ids), len(full_ids))
                labels = [-100] * prompt_len + full_ids[prompt_len:]
                if any(l != -100 for l in labels):
                    self.samples.append(
                        {
                            "input_ids": full_ids,
                            "attention_mask": [1] * len(full_ids),
                            "labels": labels,
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate_fn(pad_token_id: int):
    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attn_masks, labels = [], [], []
        for x in batch:
            pad_n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_token_id] * pad_n)
            attn_masks.append(x["attention_mask"] + [0] * pad_n)
            labels.append(x["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


@torch.no_grad()
def evaluate_loss(model, loader, device) -> tuple:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
            out = model(**batch)
        total_loss += float(out.loss.detach().cpu().item())
        total_steps += 1
    avg_loss = total_loss / max(1, total_steps)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for Model #7 RPG Persona Stylizer")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base HF model ID")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate (default: 2e-4)")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-step micro-batch size (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--max-len", type=int, default=512, help="Max sequence length (default: 512)")
    parser.add_argument("--output-dir", default=CKPT_DIR, help="Output LoRA adapter directory")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    train_jsonl = os.path.join(DATA_DIR, "rpg_persona_train.jsonl")
    val_jsonl = os.path.join(DATA_DIR, "rpg_persona_val.jsonl")
    if not os.path.exists(train_jsonl):
        sys.path.insert(0, DATA_DIR)
        import build_persona_dataset
        build_persona_dataset.main()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 84)
    print(f"  STAGE 4 QLoRA TRAINING — Model #7 Retro-GBA Persona Stylizer")
    print(f"  Base Model : {args.base_model}")
    print(f"  Device     : {device} | Epochs={args.epochs} | LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print("=" * 84, flush=True)

    loaded_model_id = args.base_model
    tokenizer = None
    model = None
    for cand_id in (args.base_model, FALLBACK_BASE_MODEL):
        try:
            print(f"[QLoRA] Loading 4-bit NF4 base model: {cand_id}...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(cand_id, use_fast=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                cand_id,
                quantization_config=bnb_cfg if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map={"": 0} if torch.cuda.is_available() else None,
            )
            loaded_model_id = cand_id
            break
        except Exception as exc:
            print(f"[QLoRA] Warning: Could not load {cand_id}: {exc}", flush=True)

    if model is None:
        raise RuntimeError("Failed to load base LLM for QLoRA training.")

    model.config.use_cache = False
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[QLoRA] Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100.0 * trainable_params / max(1, total_params):.3f}%)",
        flush=True,
    )

    train_ds = ChatSFTDataset(train_jsonl, tokenizer, max_length=args.max_len)
    val_ds = ChatSFTDataset(val_jsonl, tokenizer, max_length=args.max_len)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"[QLoRA] Loaded train={len(train_ds)} | val={len(val_ds)} examples.", flush=True)

    base_val_loss, base_val_ppl = evaluate_loss(model, val_loader, device)
    print(f"[QLoRA] Pre-training Baseline | Val Loss={base_val_loss:.4f} | Val PPL={base_val_ppl:.2f}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    total_opt_steps = max(1, (len(train_loader) * args.epochs) // max(1, args.grad_accum))
    warmup_steps = max(2, int(0.08 * total_opt_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    history = []
    best_val_loss = float("inf")
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        steps_in_epoch = 0
        ep_t0 = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                out = model(**batch)
                loss = out.loss / float(args.grad_accum)
            loss.backward()
            running_loss += float(out.loss.detach().cpu().item())
            steps_in_epoch += 1

            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, steps_in_epoch)
        val_loss, val_ppl = evaluate_loss(model, val_loader, device)
        ep_sec = time.perf_counter() - ep_t0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 3),
            "lr": round(float(scheduler.get_last_lr()[0]), 7),
            "epoch_sec": round(ep_sec, 1),
        }
        history.append(row)
        print(
            f"  [Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| val_ppl={val_ppl:.2f} | time={ep_sec:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  [QLoRA] Saved best adapter checkpoint -> {args.output_dir}", flush=True)

    total_sec = time.perf_counter() - t_start
    metrics = {
        "base_model": loaded_model_id,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "baseline_val_loss": round(base_val_loss, 4),
        "baseline_val_ppl": round(base_val_ppl, 3),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_ppl": round(math.exp(min(best_val_loss, 20.0)), 3),
        "ppl_reduction_pct": round(
            100.0 * (base_val_ppl - math.exp(min(best_val_loss, 20.0))) / max(1e-6, base_val_ppl), 2
        ),
        "total_train_sec": round(total_sec, 1),
        "history": history,
    }
    metrics_json = os.path.join(RESULTS_DIR, "qlora_training_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log_csv = os.path.join(RESULTS_DIR, "qlora_training_log.csv")
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_ppl", "lr", "epoch_sec"])
        writer.writeheader()
        writer.writerows(history)

    print("=" * 84)
    print(
        f"  DONE in {total_sec:.1f}s! Val PPL: {base_val_ppl:.2f} -> {metrics['best_val_ppl']:.2f} "
        f"(-{metrics['ppl_reduction_pct']:.1f}%)"
    )
    print(f"  Adapter saved : {args.output_dir}")
    print(f"  Metrics saved : {metrics_json}")
    print("=" * 84, flush=True)


if __name__ == "__main__":
    main()
