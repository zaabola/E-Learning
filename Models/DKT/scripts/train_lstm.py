"""
Standalone Training Script 1: Telemetry-Enriched DKT-LSTM
---------------------------------------------------------
Trains ONLY the DKT-LSTM architecture on ASSISTments 2009-2010 and exports:
  - Model checkpoint : Models/DKT/checkpoints/dkt_lstm_model.pt
  - Epoch CSV log    : Models/DKT/results/dkt_lstm_training_log.csv
  - Summary CSV row  : Models/DKT/results/models_summary.csv
"""

from model import DKTLSTM
from trainer_utils import train_and_save_single_model

# Hyperparameters (must match train_gru.py for a fair head-to-head comparison)
EPOCHS = 10
BATCH_SIZE = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.2
LEARNING_RATE = 0.002
SEED = 42


def main():
    train_and_save_single_model(
        model_name='DKT-LSTM',
        model_class=DKTLSTM,
        checkpoint_filename='dkt_lstm_model.pt',
        csv_filename='dkt_lstm_training_log.csv',
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        lr=LEARNING_RATE,
        seed=SEED,
    )


if __name__ == '__main__':
    main()
