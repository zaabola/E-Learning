# Telemetry-Enriched Deep Knowledge Tracing (DKT+) — Standalone Training Suite

This folder contains two **independent training scripts**—one per neural architecture—on the **ASSISTments 2009–2010** dataset enriched with **13 privacy-preserving behavioral telemetry features**, using an identical deterministic `SEED = 42` 80/20 split so you can inspect both CSV logs and saved checkpoints before choosing your winner:

1. **Model 1 (`train_lstm.py`)**: Telemetry-Enriched **DKT-LSTM**
2. **Model 2 (`train_gru.py`)**: Telemetry-Enriched **DKT-GRU**

---

## 1. Scripts in This Folder

* [`train_lstm.py`](file:///c:/Users/zaabola/Desktop/DL-Interactive_L/Models/DKT/scripts/train_lstm.py): Standalone trainer for **DKT-LSTM**.
* [`train_gru.py`](file:///c:/Users/zaabola/Desktop/DL-Interactive_L/Models/DKT/scripts/train_gru.py): Standalone trainer for **DKT-GRU**.
* [`trainer_utils.py`](file:///c:/Users/zaabola/Desktop/DL-Interactive_L/Models/DKT/scripts/trainer_utils.py): Shared training loop, ROC-AUC evaluator, CSV exporter, and checkpoint saver.
* [`model.py`](file:///c:/Users/zaabola/Desktop/DL-Interactive_L/Models/DKT/scripts/model.py): Neural network definitions (`DKTLSTM`, `DKTGRU`, `SAKT`).
* [`dataset.py`](file:///c:/Users/zaabola/Desktop/DL-Interactive_L/Models/DKT/scripts/dataset.py): ASSISTments preprocessing + 13-feature `encode_telemetry()`.

---

## 2. How to Run Each Model

Activate `.venv` from the project root, then run each script in turn:

```powershell
.\.venv\Scripts\Activate.ps1

# 1. Train & save DKT-LSTM
python Models/DKT/scripts/train_lstm.py

# 2. Train & save DKT-GRU
python Models/DKT/scripts/train_gru.py
```

---

## 3. Saved Outputs

### Model Checkpoints (`Models/DKT/checkpoints/`)
* `dkt_lstm_model.pt` — Trained **DKT-LSTM** best-epoch weights + metadata.
* `dkt_gru_model.pt` — Trained **DKT-GRU** best-epoch weights + metadata.
* `skill_mapping.json` — Skill index map and empirical difficulty tiers.

### Training CSV Logs (`Models/DKT/results/`)
* `dkt_lstm_training_log.csv` — Epoch-by-epoch metrics (`train_loss`, `test_loss`, `test_auc`, `test_accuracy`, `epoch_time_sec`, `is_best_auc`) for **DKT-LSTM**.
* `dkt_gru_training_log.csv` — Epoch-by-epoch metrics for **DKT-GRU**.
* `models_summary.csv` — 2-row side-by-side summary table comparing both finished models.
