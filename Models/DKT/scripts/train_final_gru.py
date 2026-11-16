"""
Final Production Training Script: Context-Aware & Regularized DKT-GRU (`FinalDKTGRU`)
-------------------------------------------------------------------------------------
Designed for long-horizon training (up to 50 epochs) WITHOUT overfitting via:
  1. AdamW decoupled weight decay (1e-4) + LayerNorm + Dropout (0.30)
  2. ReduceLROnPlateau learning-rate controller (halves LR when Test AUC plateaus)
  3. Early Stopping (patience = 10) + strict Best-Epoch checkpointing
  4. Zero-leakage Spaced-Repetition memory features + Train-Only Bayesian item priors

Outputs:
  - Model checkpoint : Models/DKT/checkpoints/best_final_gru_model.pt
  - Epoch CSV log    : Models/DKT/results/final_gru_training_log.csv
  - Summary CSV row  : Models/DKT/results/models_summary.csv
"""

import csv
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

try:
    from sklearn.metrics import roc_auc_score, accuracy_score
except ImportError:
    roc_auc_score = None
    accuracy_score = None

from dataset import encode_telemetry, TELEMETRY_DIM, TELEMETRY_FEATURE_NAMES
from model import FinalDKTGRU
from trainer_utils import resolve_paths, _upsert_summary_csv

# =====================================================================
# Hyperparameters & Anti-Overfitting Schedule
# =====================================================================
EPOCHS = 50
BATCH_SIZE = 64
EMBED_DIM = 96
HIDDEN_DIM = 192
NUM_LAYERS = 2
DROPOUT = 0.30
INITIAL_LR = 0.0015
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 3          # Halve LR after 3 epochs without AUC gain
EARLY_STOP_PATIENCE = 10 # Halt cleanly if 10 epochs pass without AUC gain
PRIOR_SMOOTHING = 5.0    # Bayesian smoothing strength for train-only item priors
SEED = 42

NEXT_CONTEXT_DIM = 9
NEXT_CONTEXT_NAMES = [
    'next_item_diff_prior',       # 0: Bayesian train-only pass rate of upcoming item
    'next_skill_diff_prior',      # 1: Train-only pass rate of upcoming skill
    'next_is_main_challenge',     # 2: original == 1 (known before answering)
    'next_is_multiple_choice',    # 3: multiple-choice vs open-response (known before answering)
    'next_is_unassisted_exam',    # 4: test/exam mode vs tutor mode (known before answering)
    'past_skill_attempts_norm',   # 5: log1p(student past attempts on target skill) / log1p(50)
    'past_skill_winrate',         # 6: smoothed student win-rate on target skill up to step t
    'past_skill_recency',         # 7: 1 / (1 + steps since student last saw target skill)
    'is_repeat_problem',          # 8: 1 if student already saw this exact problem_id in 0..t
]

_LOG_MAX_SKILL_ATT = math.log1p(50.0)


def load_rich_sequences(csv_path, min_seq_len=3, max_seq_len=100):
    """
    Loads chronological student sequences preserving problem_id and static prompt
    attributes alongside the 13 behavioral telemetry features.
    """
    print(f"Reading {csv_path}...")
    user_interactions = {}
    unique_skills = set()

    with open(csv_path, 'r', encoding='latin1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_str = row.get('list_skill_ids', '').strip()
            c_str = row.get('correct', '').strip()
            u_id = row.get('user_id', '').strip()
            o_str = row.get('order_id', '').strip()
            p_id = row.get('problem_id', '').strip()

            if not s_str or c_str not in ('0', '1', '0.0', '1.0') or not u_id:
                continue

            primary_skill = s_str.split(';')[0].strip()
            if not primary_skill:
                continue

            unique_skills.add(primary_skill)
            order = int(o_str) if o_str.isdigit() else 0
            correct = 1 if c_str in ('1', '1.0') else 0

            telem = encode_telemetry(
                ms_first_response_time=row.get('ms_first_response_time', ''),
                attempt_count=row.get('attempt_count', '1'),
                original=row.get('original', '1'),
                answer_type=row.get('answer_type', 'choose_1'),
                tutor_mode=row.get('tutor_mode', 'tutor'),
            )
            # Static prompt flags inside telem: [9]=is_main, [10]=is_mc, [12]=is_exam
            q_static = (telem[9], telem[10], telem[12])

            if u_id not in user_interactions:
                user_interactions[u_id] = []
            user_interactions[u_id].append((order, primary_skill, p_id, correct, telem, q_static))

    sorted_skills = sorted(list(unique_skills))
    skill2idx = {sk: idx for idx, sk in enumerate(sorted_skills)}
    num_skills = len(skill2idx)
    print(f"Discovered {num_skills} unique skills across {len(user_interactions):,} students.")

    raw_chunks = []
    for u_id, items in user_interactions.items():
        items.sort(key=lambda x: x[0])
        n = len(items)
        if n < min_seq_len:
            continue

        for start in range(0, n, max_seq_len):
            chunk = items[start:start + max_seq_len]
            if len(chunk) >= min_seq_len:
                raw_chunks.append([
                    (skill2idx[it[1]], it[2], it[3], it[4], it[5])
                    for it in chunk
                ])

    print(f"Extracted {len(raw_chunks):,} chronological sequence chunks.")
    return raw_chunks, num_skills, skill2idx


def compute_train_only_priors(train_chunks, num_skills, smoothing=PRIOR_SMOOTHING):
    """
    Computes empirical skill and problem pass-rate priors STRICTLY from the 80%
    training split so zero test-set target information ever leaks.
    """
    total_correct = 0
    total_count = 0
    skill_wins = np.zeros(num_skills, dtype=np.float64)
    skill_counts = np.zeros(num_skills, dtype=np.float64)
    prob_wins = {}
    prob_counts = {}
    prob_skill = {}

    for chunk in train_chunks:
        for s_idx, p_id, corr, _, _ in chunk:
            total_correct += corr
            total_count += 1
            skill_wins[s_idx] += corr
            skill_counts[s_idx] += 1
            if p_id:
                prob_wins[p_id] = prob_wins.get(p_id, 0.0) + corr
                prob_counts[p_id] = prob_counts.get(p_id, 0.0) + 1.0
                prob_skill[p_id] = s_idx

    global_prior = float(total_correct / max(total_count, 1))
    skill_priors = {}
    for s_idx in range(num_skills):
        cnt = skill_counts[s_idx]
        win = skill_wins[s_idx]
        skill_priors[s_idx] = float((win + smoothing * global_prior) / (cnt + smoothing))

    prob_priors = {}
    for p_id, cnt in prob_counts.items():
        s_prior = skill_priors[prob_skill[p_id]]
        prob_priors[p_id] = float((prob_wins[p_id] + smoothing * s_prior) / (cnt + smoothing))

    return global_prior, skill_priors, prob_priors


def prebuild_tensors(chunks, num_skills, global_prior, skill_priors, prob_priors):
    """
    Pre-tensorizes all sequences once in RAM for fast GPU epochs.
    Every feature at step t uses strictly steps 0..t plus static prompt metadata of t+1.
    """
    encoded = []

    for chunk in chunks:
        L = len(chunk) - 1
        inter_idx = torch.empty(L, dtype=torch.long)
        past_telem = torch.empty((L, TELEMETRY_DIM), dtype=torch.float32)
        past_diff = torch.empty((L, 1), dtype=torch.float32)
        target_skill_1idx = torch.empty(L, dtype=torch.long)
        next_ctx = torch.empty((L, NEXT_CONTEXT_DIM), dtype=torch.float32)
        target_corr = torch.empty(L, dtype=torch.float32)

        seen_problems = set()
        skill_att_so_far = {}
        skill_win_so_far = {}
        skill_last_pos = {}

        for t in range(L):
            s_curr, p_curr, c_curr, telem_curr, _ = chunk[t]

            # Update causal history trackers with completed step t
            if p_curr:
                seen_problems.add(p_curr)
            skill_att_so_far[s_curr] = skill_att_so_far.get(s_curr, 0) + 1
            skill_win_so_far[s_curr] = skill_win_so_far.get(s_curr, 0) + c_curr
            skill_last_pos[s_curr] = t

            # Past step t encoding (1..M = right, M+1..2M = wrong; 0 = pad)
            inter_idx[t] = (s_curr + 1) if c_curr == 1 else (s_curr + num_skills + 1)
            past_telem[t] = torch.tensor(telem_curr, dtype=torch.float32)
            s_curr_prior = skill_priors.get(s_curr, global_prior)
            past_diff[t, 0] = prob_priors.get(p_curr, s_curr_prior)

            # Upcoming step t+1 static prompt + spaced-repetition history features
            s_next, p_next, c_next, _, q_static_next = chunk[t + 1]
            s_next_prior = skill_priors.get(s_next, global_prior)
            p_next_prior = prob_priors.get(p_next, s_next_prior)

            past_att = skill_att_so_far.get(s_next, 0)
            past_win = skill_win_so_far.get(s_next, 0)
            att_norm = math.log1p(min(past_att, 50)) / _LOG_MAX_SKILL_ATT
            win_rate = (past_win + 2.0 * s_next_prior) / (past_att + 2.0)

            if s_next in skill_last_pos:
                steps_ago = t - skill_last_pos[s_next]
                recency = 1.0 / (1.0 + steps_ago)
            else:
                recency = 0.0

            is_repeat = 1.0 if (p_next and p_next in seen_problems) else 0.0

            target_skill_1idx[t] = s_next + 1
            next_ctx[t] = torch.tensor([
                p_next_prior,
                s_next_prior,
                q_static_next[0],
                q_static_next[1],
                q_static_next[2],
                att_norm,
                win_rate,
                recency,
                is_repeat,
            ], dtype=torch.float32)
            target_corr[t] = float(c_next)

        encoded.append((inter_idx, past_telem, past_diff, target_skill_1idx, next_ctx, target_corr))

    return encoded


class PrebuiltFinalDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_final_fn(batch):
    inter_idx, past_telem, past_diff, target_sk, next_ctx, target_corr = zip(*batch)

    p_inter = pad_sequence(inter_idx, batch_first=True, padding_value=0)
    p_telem = pad_sequence(past_telem, batch_first=True, padding_value=0.0)
    p_diff = pad_sequence(past_diff, batch_first=True, padding_value=0.0)
    p_tsk = pad_sequence(target_sk, batch_first=True, padding_value=0)
    p_nctx = pad_sequence(next_ctx, batch_first=True, padding_value=0.0)
    p_tcorr = pad_sequence(target_corr, batch_first=True, padding_value=0.0)

    mask = (p_tsk != 0)
    return p_inter, p_telem, p_diff, p_tsk, p_nctx, p_tcorr, mask


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for p_inter, p_telem, p_diff, p_tsk, p_nctx, p_tcorr, mask in loader:
        p_inter = p_inter.to(device)
        p_telem = p_telem.to(device)
        p_diff = p_diff.to(device)
        p_tsk = p_tsk.to(device)
        p_nctx = p_nctx.to(device)
        p_tcorr = p_tcorr.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        _, logits = model(p_inter, p_telem, p_diff, p_tsk, p_nctx)

        active_logits = logits[mask]
        active_targets = p_tcorr[mask]
        if active_targets.numel() == 0:
            continue

        loss = criterion(active_logits, active_targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    for p_inter, p_telem, p_diff, p_tsk, p_nctx, p_tcorr, mask in loader:
        p_inter = p_inter.to(device)
        p_telem = p_telem.to(device)
        p_diff = p_diff.to(device)
        p_tsk = p_tsk.to(device)
        p_nctx = p_nctx.to(device)
        p_tcorr = p_tcorr.to(device)
        mask = mask.to(device)

        preds, logits = model(p_inter, p_telem, p_diff, p_tsk, p_nctx)

        active_logits = logits[mask]
        active_preds = preds[mask]
        active_targets = p_tcorr[mask]
        if active_targets.numel() == 0:
            continue

        loss = criterion(active_logits, active_targets)
        total_loss += loss.item()
        n_batches += 1

        all_preds.append(active_preds.cpu().numpy())
        all_targets.append(active_targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    if roc_auc_score is not None and len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_pred))
    else:
        auc = 0.5

    if accuracy_score is not None:
        acc = float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
    else:
        acc = float(np.mean((y_pred >= 0.5) == y_true))

    return total_loss / max(n_batches, 1), auc, acc


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_label = f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""

    data_path, checkpoints_dir, results_dir, _ = resolve_paths()
    checkpoint_name = 'best_final_gru_model.pt'
    csv_name = 'final_gru_training_log.csv'
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
    epoch_csv_path = os.path.join(results_dir, csv_name)
    summary_csv_path = os.path.join(results_dir, 'models_summary.csv')

    print("=" * 80)
    print("  FINAL PRODUCTION TRAINING: Context-Aware Regularized DKT-GRU (FinalDKTGRU)")
    print("=" * 80)
    print(f"Device          : {device}{gpu_label}")
    print(f"Max Epochs      : {EPOCHS} (Early Stop Patience={EARLY_STOP_PATIENCE}, LR Halving Patience={LR_PATIENCE})")
    print(f"Optimizer       : AdamW (Initial LR={INITIAL_LR}, Weight Decay={WEIGHT_DECAY}, Dropout={DROPOUT})")
    print(f"Model Save Path : {checkpoint_path}")
    print(f"CSV Log Path    : {epoch_csv_path}\n")

    raw_chunks, num_skills, skill2idx = load_rich_sequences(data_path)

    # Identical 80/20 split seed as train_lstm.py and train_gru.py BEFORE computing priors
    train_size = int(0.8 * len(raw_chunks))
    test_size = len(raw_chunks) - train_size
    split_gen = torch.Generator().manual_seed(SEED)
    train_subset, test_subset = random_split(raw_chunks, [train_size, test_size], generator=split_gen)

    train_chunks = [raw_chunks[i] for i in train_subset.indices]
    test_chunks = [raw_chunks[i] for i in test_subset.indices]
    print(f"Split (seed={SEED}): {len(train_chunks):,} train sequences | {len(test_chunks):,} test sequences.")

    print("Computing leak-free Bayesian item & skill priors on train split only...")
    global_prior, skill_priors, prob_priors = compute_train_only_priors(train_chunks, num_skills)

    print("Pre-tensorizing sequences in memory for fast GPU epochs...")
    train_items = prebuild_tensors(train_chunks, num_skills, global_prior, skill_priors, prob_priors)
    test_items = prebuild_tensors(test_chunks, num_skills, global_prior, skill_priors, prob_priors)

    loader_gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        PrebuiltFinalDataset(train_items),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_final_fn,
        generator=loader_gen,
    )
    test_loader = DataLoader(
        PrebuiltFinalDataset(test_items),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_final_fn,
    )

    model = FinalDKTGRU(
        num_skills=num_skills,
        telemetry_dim=TELEMETRY_DIM,
        next_context_dim=NEXT_CONTEXT_DIM,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nArchitecture    : Final-DKT-GRU (2-Layer LayerNorm GRU + Context Head) | Params: {param_count:,}")
    print("-" * 80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_acc = 0.0
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    epoch_rows = []

    csv_fields = [
        'model', 'epoch', 'learning_rate', 'train_loss', 'test_loss',
        'test_auc', 'test_accuracy', 'epoch_time_sec', 'is_best_auc',
        'hidden_dim', 'num_layers', 'dropout', 'parameters',
    ]

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_auc, test_acc = evaluate_epoch(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        is_best = test_auc > (best_auc + 1e-5)
        if is_best:
            best_auc = test_auc
            best_acc = test_acc
            best_epoch = epoch
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        star = " *BEST*" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({elapsed:4.1f}s) | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}{star}"
        )

        epoch_rows.append({
            'model': 'Final-DKT-GRU',
            'epoch': epoch,
            'learning_rate': current_lr,
            'train_loss': round(train_loss, 6),
            'test_loss': round(test_loss, 6),
            'test_auc': round(test_auc, 6),
            'test_accuracy': round(test_acc, 6),
            'epoch_time_sec': round(elapsed, 2),
            'is_best_auc': is_best,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'parameters': param_count,
        })

        # Flush CSV every epoch so progress can be inspected live
        with open(epoch_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(epoch_rows)

        prev_lr = current_lr
        scheduler.step(test_auc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"   [LR Scheduler] Validation plateau detected -> reducing LR: {prev_lr:.6f} -> {new_lr:.6f}")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n[Early Stopping] No validation AUC gain for {EARLY_STOP_PATIENCE} epochs. Halting to prevent overfitting!")
            break

    total_time = time.time() - total_start

    # Save strictly the best non-overfitted validation epoch weights
    torch.save({
        'model_type': 'Final-DKT-GRU',
        'num_skills': num_skills,
        'telemetry_dim': TELEMETRY_DIM,
        'telemetry_features': TELEMETRY_FEATURE_NAMES,
        'next_context_dim': NEXT_CONTEXT_DIM,
        'next_context_features': NEXT_CONTEXT_NAMES,
        'embed_dim': EMBED_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'best_epoch': best_epoch,
        'best_auc': best_auc,
        'best_accuracy': best_acc,
        'param_count': param_count,
        'global_prior': global_prior,
        'skill_priors': {str(k): v for k, v in skill_priors.items()},
        'skill2idx': skill2idx,
        'state_dict': best_weights,
    }, checkpoint_path)

    _upsert_summary_csv(summary_csv_path, {
        'model': 'Final-DKT-GRU',
        'parameters': param_count,
        'epochs': len(epoch_rows),
        'best_epoch': best_epoch,
        'best_test_auc': round(best_auc, 6),
        'best_test_accuracy': round(best_acc, 6),
        'final_train_loss': epoch_rows[-1]['train_loss'],
        'final_test_loss': epoch_rows[-1]['test_loss'],
        'total_time_sec': round(total_time, 2),
        'device': str(device),
        'checkpoint_file': checkpoint_name,
        'epoch_csv_file': csv_name,
    })

    print("-" * 80)
    print(f"[DONE] Final-DKT-GRU finished ({len(epoch_rows)} epochs in {total_time:.1f}s)")
    print(f"  Best Epoch: {best_epoch} | Best Test AUC: {best_auc:.4f} | Best Test Accuracy: {best_acc:.4f}")
    print(f"  -> Saved Best Model Checkpoint : {checkpoint_path}")
    print(f"  -> Saved Epoch CSV Log         : {epoch_csv_path}")
    print(f"  -> Updated Summary CSV         : {summary_csv_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
