"""
Standalone Cross-Encoder Trainer #2: MiniLM-L12-H384 (`microsoft/MiniLM-L12-H384-uncased`)
------------------------------------------------------------------------------------------
Architecture: Deep Self-Attention Distilled Transformer (12 layers, 384 hidden, ~33.7M params).
Strength: ~4x fewer parameters than DeBERTa-v3-Small and ultra-low inference latency
          (~2-4 ms per answer), testing speed/memory efficiency vs. linguistic depth.

Outputs:
  - `Models/AnswerEvaluator/checkpoints/minilm_grader.pt`
  - `Models/AnswerEvaluator/checkpoints/minilm_tokenizer/`
  - `Models/AnswerEvaluator/results/minilm_training_log.csv`
  - `Models/AnswerEvaluator/results/grader_models_summary.csv`
"""

import argparse
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from grader_trainer_utils import train_and_evaluate_cross_encoder


def main():
    parser = argparse.ArgumentParser(description="Train MiniLM-L12 Semantic Answer Evaluator")
    parser.add_argument('--epochs', type=int, default=4, help='Number of fine-tuning epochs (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size safe for 4GB VRAM (default: 64)')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--lr', type=float, default=4e-5, help='Backbone learning rate (default: 4e-5)')
    parser.add_argument('--max_length', type=int, default=160, help='Max token sequence length (default: 160)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    train_and_evaluate_cross_encoder(
        model_name='MiniLM-L12-H384',
        backbone_name='microsoft/MiniLM-L12-H384-uncased',
        checkpoint_filename='minilm_grader.pt',
        tokenizer_dirname='minilm_tokenizer',
        csv_filename='minilm_training_log.csv',
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=128,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
