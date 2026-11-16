"""
Shared Training, Multi-Benchmark Evaluation & CSV Logging Utilities
-------------------------------------------------------------------
Trains Cross-Encoder Semantic Answer Graders safely on 4 GB VRAM GPUs
using Ampere `bfloat16` (or `float16` fallback) mixed precision, dynamic
batch padding, differential learning rates, and linear-warmup cosine decay.

Logs standardized metrics to:
  - `Models/AnswerEvaluator/checkpoints/<model>.pt` (+ local `<model>_tokenizer/`)
  - `Models/AnswerEvaluator/results/<model>_training_log.csv`
  - `Models/AnswerEvaluator/results/grader_models_summary.csv`
"""

import csv
import math
import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from grader_dataset import (
    ASAG_SOURCE_IDS,
    PAWS_SOURCE_ID,
    build_grader_dataloaders,
)
from grader_models import (
    CrossEncoderAnswerGrader,
    grade_student_answer,
    load_tokenizer_fast_safe,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def resolve_grader_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grader_root = os.path.dirname(script_dir)
    csv_path = os.path.join(grader_root, 'data', 'master_answer_grading.csv')
    checkpoints_dir = os.path.join(grader_root, 'checkpoints')
    results_dir = os.path.join(grader_root, 'results')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return csv_path, checkpoints_dir, results_dir


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _upsert_summary_csv(summary_csv_path: str, row_dict: dict):
    fieldnames = list(row_dict.keys())
    existing = []
    if os.path.exists(summary_csv_path):
        with open(summary_csv_path, 'r', newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('model') != row_dict['model']:
                    existing.append(r)

    existing.append(row_dict)
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing:
            writer.writerow(r)


def ensure_keyword_baseline_in_summary(summary_csv_path: str, test_rows: list):
    """
    Evaluates a traditional non-neural Keyword-Overlap (Jaccard) grader on the
    exact 22,472 held-out test pairs once and logs it at the top of
    `grader_models_summary.csv` as a concrete scientific baseline.
    """
    if os.path.exists(summary_csv_path):
        with open(summary_csv_path, 'r', newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('model') == 'Baseline-Keyword-Overlap':
                    return

    t0 = time.perf_counter()
    n = len(test_rows)
    y_true_3way = np.empty(n, dtype=np.int64)
    y_pred_3way = np.empty(n, dtype=np.int64)
    y_true_bin = np.empty(n, dtype=np.int64)
    y_true_sc = np.empty(n, dtype=np.float32)
    y_pred_sc = np.empty(n, dtype=np.float32)
    src_ids = np.empty(n, dtype=np.int64)

    for i, item in enumerate(test_rows):
        _, stu, raw_ref, g_lbl, b_lbl, s_sc, src_id = item
        ta = set(_TOKEN_RE.findall(raw_ref.lower()))
        tb = set(_TOKEN_RE.findall(stu.lower()))
        jacc = (len(ta & tb) / max(1, len(ta | tb))) if (ta or tb) else 0.0

        if jacc >= 0.60:
            pred_3 = 2
        elif jacc >= 0.35:
            pred_3 = 1
        else:
            pred_3 = 0

        y_true_3way[i] = g_lbl
        y_pred_3way[i] = pred_3
        y_true_bin[i] = b_lbl
        y_true_sc[i] = s_sc
        y_pred_sc[i] = jacc
        src_ids[i] = src_id

    elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, n)
    acc_3 = float(accuracy_score(y_true_3way, y_pred_3way))
    macro_f1 = float(f1_score(y_true_3way, y_pred_3way, average='macro'))
    bin_auc = float(roc_auc_score(y_true_bin, y_pred_sc))
    bin_acc = float(accuracy_score(y_true_bin, (y_pred_3way == 2).astype(int)))
    pearson_r = _pearson_corr(y_true_sc, y_pred_sc)

    asag_mask = np.isin(src_ids, ASAG_SOURCE_IDS)
    paws_mask = src_ids == PAWS_SOURCE_ID
    asag_acc = float(accuracy_score(y_true_3way[asag_mask], y_pred_3way[asag_mask])) if asag_mask.any() else 0.0
    paws_acc = float(accuracy_score(y_true_bin[paws_mask], (y_pred_3way[paws_mask] == 2).astype(int))) if paws_mask.any() else 0.0

    _upsert_summary_csv(summary_csv_path, {
        'model': 'Baseline-Keyword-Overlap',
        'parameters': 0,
        'epochs': 0,
        'best_epoch': 0,
        'test_acc_3way': round(acc_3, 4),
        'test_macro_f1': round(macro_f1, 4),
        'test_binary_auc': round(bin_auc, 4),
        'test_binary_acc': round(bin_acc, 4),
        'test_score_pearson': round(pearson_r, 4),
        'test_asag_acc': round(asag_acc, 4),
        'test_paws_trap_acc': round(paws_acc, 4),
        'inference_ms': round(elapsed_ms, 3),
        'total_time_sec': 0.0,
        'checkpoint_file': 'N/A',
        'epoch_csv_file': 'N/A',
    })


def build_optimizer_and_scheduler(model, lr: float, total_steps: int, warmup_ratio: float = 0.10, weight_decay: float = 0.01):
    no_decay = ('bias', 'LayerNorm.weight', 'layernorm.weight')
    backbone_decay, backbone_nodecay = [], []
    head_decay, head_nodecay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_nd = any(nd in name for nd in no_decay)
        if name.startswith('backbone.'):
            (backbone_nodecay if is_nd else backbone_decay).append(param)
        else:
            (head_nodecay if is_nd else head_decay).append(param)

    param_groups = [
        {'params': backbone_decay, 'lr': lr, 'weight_decay': weight_decay},
        {'params': backbone_nodecay, 'lr': lr, 'weight_decay': 0.0},
        {'params': head_decay, 'lr': lr * 5.0, 'weight_decay': weight_decay},
        {'params': head_nodecay, 'lr': lr * 5.0, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


@torch.no_grad()
def evaluate_grader(model, dataloader, ce_criterion, reg_criterion, device, amp_dtype, use_amp: bool):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_true_3 = []
    all_pred_3 = []
    all_true_bin = []
    all_prob_corr = []
    all_true_sc = []
    all_pred_sc = []
    all_src = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        token_type_ids = batch['token_type_ids'].to(device, non_blocking=True) if batch['token_type_ids'] is not None else None
        g_labels = batch['grade_labels'].to(device, non_blocking=True)
        s_scores = batch['semantic_scores'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            logits_3way, score_logit = model(input_ids, attention_mask, token_type_ids)

        loss_ce = ce_criterion(logits_3way.float(), g_labels)
        loss_reg = reg_criterion(torch.sigmoid(score_logit.float()), s_scores.float())
        loss = loss_ce + 0.6 * loss_reg

        total_loss += float(loss.item())
        num_batches += 1

        probs_3, pred_sc = CrossEncoderAnswerGrader.calibrate_outputs(logits_3way, score_logit)
        preds_3 = probs_3.argmax(dim=-1)

        all_true_3.append(g_labels.cpu().numpy())
        all_pred_3.append(preds_3.cpu().numpy())
        all_true_bin.append(batch['binary_labels'].numpy())
        all_prob_corr.append(probs_3[:, 2].cpu().numpy())
        all_true_sc.append(batch['semantic_scores'].numpy())
        all_pred_sc.append(pred_sc.cpu().numpy())
        all_src.append(batch['source_ids'].numpy())

    y_true_3 = np.concatenate(all_true_3)
    y_pred_3 = np.concatenate(all_pred_3)
    y_true_bin = np.concatenate(all_true_bin)
    y_prob_corr = np.nan_to_num(np.concatenate(all_prob_corr), nan=0.0, posinf=1.0, neginf=0.0)
    y_true_sc = np.concatenate(all_true_sc)
    y_pred_sc = np.nan_to_num(np.concatenate(all_pred_sc), nan=0.0, posinf=1.0, neginf=0.0)
    src_ids = np.concatenate(all_src)

    acc_3 = float(accuracy_score(y_true_3, y_pred_3))
    macro_f1 = float(f1_score(y_true_3, y_pred_3, average='macro'))
    bin_auc = float(roc_auc_score(y_true_bin, y_prob_corr))
    bin_acc = float(accuracy_score(y_true_bin, (y_pred_3 == 2).astype(int)))
    pearson_r = _pearson_corr(y_true_sc, y_pred_sc)

    asag_mask = np.isin(src_ids, ASAG_SOURCE_IDS)
    paws_mask = src_ids == PAWS_SOURCE_ID
    asag_acc = float(accuracy_score(y_true_3[asag_mask], y_pred_3[asag_mask])) if asag_mask.any() else 0.0
    paws_acc = float(accuracy_score(y_true_bin[paws_mask], (y_pred_3[paws_mask] == 2).astype(int))) if paws_mask.any() else 0.0

    return {
        'loss': total_loss / max(1, num_batches),
        'acc_3way': acc_3,
        'macro_f1': macro_f1,
        'binary_auc': bin_auc,
        'binary_acc': bin_acc,
        'score_pearson': pearson_r,
        'asag_acc': asag_acc,
        'paws_trap_acc': paws_acc,
    }


def benchmark_single_answer_latency_ms(model, tokenizer, device, trials: int = 80) -> float:
    sample_q = "Why does gradient clipping stabilize recurrent neural network training?"
    sample_ref = "Gradient clipping rescales exploding gradients when their norm exceeds a threshold, preventing unstable weight updates."
    sample_stu = "It caps overly large gradients during backprop through time so weights do not blow up."

    for _ in range(10):
        grade_student_answer(model, tokenizer, sample_ref, sample_stu, question=sample_q, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(trials):
        grade_student_answer(model, tokenizer, sample_ref, sample_stu, question=sample_q, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return ((time.perf_counter() - t0) * 1000.0) / trials


def train_and_evaluate_cross_encoder(
    model_name: str,
    backbone_name: str,
    checkpoint_filename: str,
    tokenizer_dirname: str,
    csv_filename: str,
    epochs: int = 4,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    grad_accum_steps: int = 1,
    lr: float = 3e-5,
    max_length: int = 160,
    dropout: float = 0.15,
    asag_train_upsample: int = 2,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    prec_label = 'BF16 TensorCore' if use_bf16 else ('FP16 Mixed' if use_amp else 'FP32 CPU')
    gpu_label = f" ({torch.cuda.get_device_name(0)} | {prec_label})" if use_amp else ""

    csv_path, checkpoints_dir, results_dir = resolve_grader_paths()
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_filename)
    tokenizer_save_dir = os.path.join(checkpoints_dir, tokenizer_dirname)
    epoch_csv_path = os.path.join(results_dir, csv_filename)
    summary_csv_path = os.path.join(results_dir, 'grader_models_summary.csv')

    print("=" * 84)
    print(f"  STANDALONE CROSS-ENCODER TRAINING: {model_name}")
    print("=" * 84)
    print(f"Device            : {device}{gpu_label}")
    print(f"Backbone          : {backbone_name}")
    print(f"Dataset           : {csv_path}")
    print(f"Checkpoint Save   : {checkpoint_path}")
    print(f"Tokenizer Folder  : {tokenizer_save_dir}")
    print(f"Epoch CSV Log     : {epoch_csv_path}")
    print(f"Batch Size        : {batch_size} (grad_accum={grad_accum_steps} -> effective={batch_size * grad_accum_steps})\n")

    tokenizer = load_tokenizer_fast_safe(backbone_name)
    train_loader, test_loader, _, test_rows = build_grader_dataloaders(
        csv_path=csv_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        max_length=max_length,
        asag_train_upsample=asag_train_upsample,
        seed=seed,
    )

    # Ensure traditional keyword baseline is logged in summary CSV for comparison
    ensure_keyword_baseline_in_summary(summary_csv_path, test_rows)

    model = CrossEncoderAnswerGrader(
        backbone_name=backbone_name,
        dropout=dropout,
        pretrained=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nInitialized {model_name} | Hidden Size: {model.hidden_size} | Trainable Params: {param_count:,}")
    print("-" * 84)

    steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum_steps))
    total_opt_steps = steps_per_epoch * epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr=lr, total_steps=total_opt_steps)

    # Mild class weight [1.0, 1.35, 1.0] balances the 20.5% `partial` class
    class_weights = torch.tensor([1.0, 1.35, 1.0], dtype=torch.float32, device=device)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    reg_criterion = nn.SmoothL1Loss(beta=0.10)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and not use_bf16))

    best_f1 = -1.0
    best_metrics = None
    best_epoch = 0
    best_weights = None
    epoch_rows = []
    total_start = time.time()

    num_batches = len(train_loader)
    log_interval = max(1, num_batches // 5)

    for epoch in range(1, epochs + 1):
        model.train()
        ep_t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(train_loader, start=1):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(device, non_blocking=True) if batch['token_type_ids'] is not None else None
            g_labels = batch['grade_labels'].to(device, non_blocking=True)
            s_scores = batch['semantic_scores'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                logits_3way, score_logit = model(input_ids, attention_mask, token_type_ids)

            loss_ce = ce_criterion(logits_3way.float(), g_labels)
            loss_reg = reg_criterion(torch.sigmoid(score_logit.float()), s_scores.float())
            raw_loss = loss_ce + 0.6 * loss_reg
            loss = raw_loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(raw_loss.item())

            if (step_idx % grad_accum_steps == 0) or (step_idx == num_batches):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step_idx % log_interval == 0 or step_idx == num_batches:
                avg_so_far = running_loss / step_idx
                cur_lr = optimizer.param_groups[0]['lr']
                vram_str = ""
                if use_amp:
                    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    vram_str = f" | Peak VRAM: {vram_gb:.2f} GB"
                print(
                    f"  Epoch {epoch}/{epochs} [{step_idx:>4d}/{num_batches:<4d}] | "
                    f"Train Loss: {avg_so_far:.4f} | LR: {cur_lr:.2e}{vram_str}"
                )

        train_loss = running_loss / max(1, num_batches)
        eval_m = evaluate_grader(model, test_loader, ce_criterion, reg_criterion, device, amp_dtype, use_amp)
        ep_sec = time.time() - ep_t0

        is_best = eval_m['macro_f1'] > best_f1
        if is_best:
            best_f1 = eval_m['macro_f1']
            best_metrics = eval_m
            best_epoch = epoch
            best_weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        star = " *BEST*" if is_best else ""
        print(
            f"--> Epoch [{epoch:02d}/{epochs:02d}] ({ep_sec:5.1f}s) | "
            f"3-Way Acc: {eval_m['acc_3way']*100:5.2f}% | Macro-F1: {eval_m['macro_f1']:.4f} | "
            f"Bin AUC: {eval_m['binary_auc']:.4f} | ASAG: {eval_m['asag_acc']*100:5.2f}% | "
            f"PAWS Traps: {eval_m['paws_trap_acc']*100:5.2f}%{star}\n"
        )

        epoch_rows.append({
            'model': model_name,
            'epoch': epoch,
            'train_loss': round(train_loss, 5),
            'test_loss': round(eval_m['loss'], 5),
            'test_acc_3way': round(eval_m['acc_3way'], 5),
            'test_macro_f1': round(eval_m['macro_f1'], 5),
            'test_binary_auc': round(eval_m['binary_auc'], 5),
            'test_binary_acc': round(eval_m['binary_acc'], 5),
            'test_score_pearson': round(eval_m['score_pearson'], 5),
            'test_asag_acc': round(eval_m['asag_acc'], 5),
            'test_paws_trap_acc': round(eval_m['paws_trap_acc'], 5),
            'epoch_time_sec': round(ep_sec, 2),
            'is_best_f1': is_best,
        })

    total_time = time.time() - total_start

    # Restore best weights & benchmark single-answer latency
    model.load_state_dict(best_weights)
    latency_ms = benchmark_single_answer_latency_ms(model, tokenizer, device)

    # 1. Save local offline tokenizer copy
    os.makedirs(tokenizer_save_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_dir)

    # 2. Save self-contained PyTorch checkpoint (.pt)
    torch.save({
        'model_name': model_name,
        'backbone_name': backbone_name,
        'backbone_config_dict': model.backbone.config.to_dict(),
        'tokenizer_dir': tokenizer_dirname,
        'dropout': dropout,
        'max_length': max_length,
        'param_count': param_count,
        'best_epoch': best_epoch,
        'best_metrics': best_metrics,
        'inference_ms': round(latency_ms, 3),
        'state_dict': best_weights,
    }, checkpoint_path)

    # 3. Write epoch-by-epoch CSV log
    with open(epoch_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    # 4. Upsert side-by-side summary CSV
    _upsert_summary_csv(summary_csv_path, {
        'model': model_name,
        'parameters': param_count,
        'epochs': epochs,
        'best_epoch': best_epoch,
        'test_acc_3way': round(best_metrics['acc_3way'], 4),
        'test_macro_f1': round(best_metrics['macro_f1'], 4),
        'test_binary_auc': round(best_metrics['binary_auc'], 4),
        'test_binary_acc': round(best_metrics['binary_acc'], 4),
        'test_score_pearson': round(best_metrics['score_pearson'], 4),
        'test_asag_acc': round(best_metrics['asag_acc'], 4),
        'test_paws_trap_acc': round(best_metrics['paws_trap_acc'], 4),
        'inference_ms': round(latency_ms, 3),
        'total_time_sec': round(total_time, 2),
        'checkpoint_file': checkpoint_filename,
        'epoch_csv_file': csv_filename,
    })

    print("=" * 84)
    print(
        f"[DONE] {model_name} finished in {total_time:.1f}s (Best Epoch {best_epoch}) | "
        f"Macro-F1: {best_metrics['macro_f1']:.4f} | Bin AUC: {best_metrics['binary_auc']:.4f} | "
        f"Latency: {latency_ms:.2f} ms/answer"
    )
    print(f"  -> Saved Model Checkpoint : {checkpoint_path}")
    print(f"  -> Saved Local Tokenizer  : {tokenizer_save_dir}")
    print(f"  -> Saved Epoch CSV Log    : {epoch_csv_path}")
    print(f"  -> Updated Summary CSV    : {summary_csv_path}")
    print("=" * 84)
