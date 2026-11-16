"""
Standalone Cross-Encoder Trainer #1: DeBERTa-v3-Small (`microsoft/deberta-v3-small`)
------------------------------------------------------------------------------------
Architecture: Disentangled Relative Positional Attention (6 layers, 768 hidden).
Strength: Specifically excels at catching subtle negations, cause-effect inversions,
          and Google PAWS word-order traps while grading open-text explanations.

Outputs:
  - `Models/AnswerEvaluator/checkpoints/deberta_v3_grader.pt`
  - `Models/AnswerEvaluator/checkpoints/deberta_v3_tokenizer/`
  - `Models/AnswerEvaluator/results/deberta_training_log.csv`
  - `Models/AnswerEvaluator/results/grader_models_summary.csv`
"""

import argparse
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from grader_trainer_utils import train_and_evaluate_cross_encoder


def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa-v3-Small Semantic Answer Evaluator")
    parser.add_argument('--epochs', type=int, default=4, help='Number of fine-tuning epochs (default: 4)')
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size safe for 4GB VRAM with FP32 master weights (default: 16)')
    parser.add_argument('--grad_accum', type=int, default=2, help='Gradient accumulation steps -> effective batch 32 (default: 2)')
    parser.add_argument('--lr', type=float, default=3e-5, help='Backbone learning rate (default: 3e-5)')
    parser.add_argument('--max_length', type=int, default=160, help='Max token sequence length (default: 160)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    train_and_evaluate_cross_encoder(
        model_name='DeBERTa-v3-Small',
        backbone_name='microsoft/deberta-v3-small',
        checkpoint_filename='deberta_v3_grader.pt',
        tokenizer_dirname='deberta_v3_tokenizer',
        csv_filename='deberta_training_log.csv',
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=64,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
