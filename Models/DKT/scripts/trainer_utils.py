import csv
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

try:
    from sklearn.metrics import roc_auc_score, accuracy_score
except ImportError:
    roc_auc_score = None
    accuracy_score = None

from dataset import (
    preprocess_assistments,
    DKTDataset,
    collate_dkt_fn,
    TELEMETRY_DIM,
    TELEMETRY_FEATURE_NAMES,
)


def resolve_paths():
    """
    Resolves dataset, checkpoints, and CSV results directories whether executed
    from project root, Models/DKT/scripts/, or a flat Kaggle workspace.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dkt_root = os.path.dirname(script_dir)

    parent_csv = os.path.join(dkt_root, 'data', 'assistments_2009_2010.csv')
    local_csv = os.path.join(script_dir, 'data', 'assistments_2009_2010.csv')
    data_path = parent_csv if os.path.exists(parent_csv) else local_csv

    base_out = dkt_root if os.path.exists(os.path.join(dkt_root, 'data')) else script_dir
    checkpoints_dir = os.path.join(base_out, 'checkpoints')
    results_dir = os.path.join(base_out, 'results')

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    mapping_path = os.path.join(checkpoints_dir, 'skill_mapping.json')
    return data_path, checkpoints_dir, results_dir, mapping_path


def build_dataloaders(data_path, mapping_path, batch_size=64, seed=42, min_seq_len=3, max_seq_len=100):
    """
    Preprocesses ASSISTments data and creates a deterministic 80/20 train/test split
    using `seed` so separate model scripts evaluate on the exact same test sequences.
    """
    sequences, num_skills, skill2idx = preprocess_assistments(
        data_path,
        mapping_save_path=mapping_path,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
    )

    dataset = DKTDataset(sequences, num_skills, telemetry_dim=TELEMETRY_DIM)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=generator)
    print(f"Split (seed={seed}): {train_size:,} train sequences | {test_size:,} test sequences.")

    train_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_dkt_fn,
        generator=train_gen,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dkt_fn,
    )
    return train_loader, test_loader, num_skills, skill2idx


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, target_skills, target_corrects, mask in dataloader:
        inputs = inputs.to(device)
        target_skills = target_skills.to(device)
        target_corrects = target_corrects.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        _, logits = model(inputs)

        safe_target_skills = target_skills.clone()
        safe_target_skills[~mask] = 0
        gathered_logits = torch.gather(logits, dim=2, index=safe_target_skills.unsqueeze(2)).squeeze(2)

        active_logits = gathered_logits[mask]
        active_targets = target_corrects[mask]

        if active_targets.numel() == 0:
            continue

        loss = criterion(active_logits, active_targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    for inputs, target_skills, target_corrects, mask in dataloader:
        inputs = inputs.to(device)
        target_skills = target_skills.to(device)
        target_corrects = target_corrects.to(device)
        mask = mask.to(device)

        preds, logits = model(inputs)

        safe_target_skills = target_skills.clone()
        safe_target_skills[~mask] = 0
        gathered_logits = torch.gather(logits, dim=2, index=safe_target_skills.unsqueeze(2)).squeeze(2)
        gathered_preds = torch.gather(preds, dim=2, index=safe_target_skills.unsqueeze(2)).squeeze(2)

        active_logits = gathered_logits[mask]
        active_preds = gathered_preds[mask]
        active_targets = target_corrects[mask]

        if active_targets.numel() == 0:
            continue

        loss = criterion(active_logits, active_targets)
        total_loss += loss.item()
        num_batches += 1

        all_preds.extend(active_preds.cpu().numpy().tolist())
        all_targets.extend(active_targets.cpu().numpy().tolist())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    if roc_auc_score is not None and len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_pred))
    else:
        auc = 0.5

    if accuracy_score is not None:
        acc = float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
    else:
        acc = float(np.mean((y_pred >= 0.5) == y_true))

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, auc, acc


def _upsert_summary_csv(summary_csv_path, summary_row):
    """
    Updates or appends the finished model's summary metrics inside models_summary.csv
    so both models can be compared side-by-side in one table after running each script.
    """
    fieldnames = list(summary_row.keys())
    existing_rows = []

    if os.path.exists(summary_csv_path):
        with open(summary_csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get('model') != summary_row['model']:
                    existing_rows.append(r)

    existing_rows.append(summary_row)

    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)


def train_and_save_single_model(
    model_name,
    model_class,
    checkpoint_filename,
    csv_filename,
    epochs=10,
    batch_size=64,
    hidden_dim=128,
    num_layers=1,
    dropout=0.2,
    lr=0.002,
    seed=42,
):
    """
    End-to-end runner for a single DKT architecture:
    1. Preprocesses dataset with identical deterministic 80/20 split (`seed`).
    2. Trains for `epochs` on CUDA/CPU.
    3. Saves best model checkpoint (.pt) in `Models/DKT/checkpoints/`.
    4. Saves epoch-by-epoch training history CSV in `Models/DKT/results/`.
    5. Updates `Models/DKT/results/models_summary.csv` for side-by-side inspection.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_label = f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""

    data_path, checkpoints_dir, results_dir, mapping_path = resolve_paths()
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_filename)
    epoch_csv_path = os.path.join(results_dir, csv_filename)
    summary_csv_path = os.path.join(results_dir, 'models_summary.csv')

    print("=" * 75)
    print(f"  STANDALONE MODEL TRAINING: {model_name}")
    print("=" * 75)
    print(f"Device          : {device}{gpu_label}")
    print(f"Dataset         : {data_path}")
    print(f"Model Save Path : {checkpoint_path}")
    print(f"CSV Log Path    : {epoch_csv_path}")
    print(f"Telemetry Dim   : {TELEMETRY_DIM} features\n")

    train_loader, test_loader, num_skills, skill2idx = build_dataloaders(
        data_path=data_path,
        mapping_path=mapping_path,
        batch_size=batch_size,
        seed=seed,
    )

    model = model_class(
        num_skills=num_skills,
        telemetry_dim=TELEMETRY_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nArchitecture    : {model_name} | Input Dim: {model.input_dim} | Trainable Params: {param_count:,}")
    print("-" * 75)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_acc = 0.0
    best_epoch = 0
    best_weights = None
    epoch_rows = []

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_auc, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        is_best = test_auc > best_auc
        if is_best:
            best_auc = test_auc
            best_acc = test_acc
            best_epoch = epoch
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

        star = " *BEST*" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:5.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}{star}"
        )

        epoch_rows.append({
            'model': model_name,
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'test_loss': round(test_loss, 6),
            'test_auc': round(test_auc, 6),
            'test_accuracy': round(test_acc, 6),
            'epoch_time_sec': round(elapsed, 2),
            'is_best_auc': is_best,
            'learning_rate': lr,
            'hidden_dim': hidden_dim,
            'parameters': param_count,
        })

    total_time = time.time() - total_start

    # 1. Write per-epoch CSV log
    csv_fields = [
        'model', 'epoch', 'train_loss', 'test_loss', 'test_auc',
        'test_accuracy', 'epoch_time_sec', 'is_best_auc',
        'learning_rate', 'hidden_dim', 'parameters',
    ]
    with open(epoch_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(epoch_rows)

    # 2. Save best model checkpoint (.pt)
    torch.save({
        'model_type': model_name,
        'num_skills': num_skills,
        'telemetry_dim': TELEMETRY_DIM,
        'telemetry_features': TELEMETRY_FEATURE_NAMES,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'best_epoch': best_epoch,
        'best_auc': best_auc,
        'best_accuracy': best_acc,
        'param_count': param_count,
        'skill2idx': skill2idx,
        'state_dict': best_weights,
    }, checkpoint_path)

    # 3. Upsert side-by-side summary CSV
    _upsert_summary_csv(summary_csv_path, {
        'model': model_name,
        'parameters': param_count,
        'epochs': epochs,
        'best_epoch': best_epoch,
        'best_test_auc': round(best_auc, 6),
        'best_test_accuracy': round(best_acc, 6),
        'final_train_loss': epoch_rows[-1]['train_loss'],
        'final_test_loss': epoch_rows[-1]['test_loss'],
        'total_time_sec': round(total_time, 2),
        'device': str(device),
        'checkpoint_file': checkpoint_filename,
        'epoch_csv_file': csv_filename,
    })

    print("-" * 75)
    print(f"[DONE] {model_name} finished in {total_time:.1f}s | Best Epoch: {best_epoch} | Best AUC: {best_auc:.4f} | Acc: {best_acc:.4f}")
    print(f"  -> Saved Model Checkpoint : {checkpoint_path}")
    print(f"  -> Saved Epoch CSV Log    : {epoch_csv_path}")
    print(f"  -> Updated Summary CSV    : {summary_csv_path}")
    print("=" * 75)
