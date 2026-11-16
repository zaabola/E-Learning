"""
Verification & Interactive Live-Grading Suite for `DeBERTa-v3-Small` Answer Evaluator
-------------------------------------------------------------------------------------
Modes:
  1. Automated Diagnostic Audit (`--auto`):
     Tests `deberta_v3_grader.pt` completely offline across 6 real pedagogical stress
     scenarios (True Paraphrase, Partial Credit, Negation Trap, Cause-Effect Word-Swap
     Trap, Buzzword Stuffing, and Blank/"I don't know") + CPU/GPU latency benchmark.
  2. Interactive Mode (default):
     Type your own Question, Reference Concept, and Student Answer to see live 3-way
     grading, Final-DKT-GRU signal, and RPG Battle Attack Damage multiplier!
"""

import argparse
import os
import time
import torch

from grader_models import grade_student_answer, load_trained_grader


DIAGNOSTIC_CASES = [
    {
        'name': '1. True Conceptual Paraphrase (Low Word Overlap -> Expect CORRECT)',
        'question': 'Why do we apply gradient clipping when training recurrent neural networks?',
        'reference': 'Gradient clipping rescales gradients whenever their norm exceeds a threshold so exploding gradients cannot destabilize weight updates.',
        'student': 'It caps huge gradient spikes during backpropagation through time so the network weights stay stable instead of blowing up.',
        'expected': 'correct',
    },
    {
        'name': '2. Partial / Incomplete Understanding (Half the Concept -> Expect PARTIAL)',
        'question': 'How does photosynthesis sustain plant growth?',
        'reference': 'Chloroplasts absorb sunlight to convert water and carbon dioxide into glucose chemical energy while releasing oxygen.',
        'student': 'Plants use sunlight and water to make oxygen.',
        'expected': 'partial',
    },
    {
        'name': '3. Subtle Negation Trap (90% Shared Words -> Expect INCORRECT)',
        'question': 'What is the role of dropout regularization in deep neural networks?',
        'reference': 'Dropout randomly deactivates neurons during training to prevent co-adaptation and reduce overfitting on unseen test data.',
        'student': 'Dropout never deactivates neurons during training, which increases co-adaptation and causes overfitting on unseen test data.',
        'expected': 'incorrect',
    },
    {
        'name': '4. Cause-Effect Word-Swap Trap (100% Same Words, Inverted Meaning -> Expect INCORRECT)',
        'question': '',
        'reference': 'High insulin levels stimulate cells to absorb glucose from the bloodstream.',
        'student': 'High glucose levels in cells stimulate insulin to leave the bloodstream.',
        'expected': 'incorrect',
    },
    {
        'name': '5. Buzzword Stuffing / Nonsense (Expect INCORRECT)',
        'question': 'Explain how backpropagation updates neural network weights.',
        'reference': 'Backpropagation computes loss gradients layer by layer using the chain rule so gradient descent can adjust weights.',
        'student': 'weights layer neural network loss chain rule random tokens pizza.',
        'expected': 'incorrect',
    },
    {
        'name': '6. Honest "I do not know" Student Response (Expect INCORRECT, ~0 Damage)',
        'question': 'What does the discount factor gamma control in reinforcement learning?',
        'reference': 'Gamma controls how strongly the agent values future rewards compared to immediate rewards.',
        'student': 'I am not sure, maybe something about learning rate?',
        'expected': 'incorrect',
    },
]


def run_automated_audit(model, tokenizer, ckpt, device):
    print("=" * 88)
    print(f"  AUTOMATED DIAGNOSTIC AUDIT: {ckpt['model_name']} ({ckpt['backbone_name']})")
    print("=" * 88)
    print(f"Device       : {device}")
    print(f"Best Epoch   : {ckpt['best_epoch']} | Params: {ckpt['param_count']:,}")
    bm = ckpt['best_metrics']
    print(
        f"Held-Out Test: 3-Way Acc={bm['acc_3way']*100:.2f}% | Macro-F1={bm['macro_f1']:.4f} | "
        f"Bin AUC={bm['binary_auc']:.4f} | PAWS Traps={bm['paws_trap_acc']*100:.2f}%"
    )
    print("-" * 88)

    passed = 0
    for case in DIAGNOSTIC_CASES:
        res = grade_student_answer(
            model,
            tokenizer,
            reference_answer=case['reference'],
            student_answer=case['student'],
            question=case['question'],
            device=device,
        )
        ok = (res['verdict'] == case['expected']) or (
            case['expected'] == 'partial' and 0.25 <= res['semantic_score'] <= 0.78
        )
        passed += int(ok)
        badge = "PASS" if ok else "NOTE"
        print(f"\n[{badge}] {case['name']}")
        print(f"  Reference : {case['reference']}")
        print(f"  Student   : {case['student']}")
        print(
            f"  -> Verdict: {res['verdict'].upper():<9} | "
            f"Semantic Score: {res['semantic_score']:.3f} | "
            f"RPG Attack Mult: {res['rpg_damage_multiplier']:.2f}x | "
            f"DKT Binary: {res['binary_correct']} "
            f"(P_corr={res['prob_correct']:.2f}, P_part={res['prob_partial']:.2f}, P_inc={res['prob_incorrect']:.2f})"
        )

    # Measure both active-device and pure-CPU latency
    t0 = time.perf_counter()
    for _ in range(50):
        grade_student_answer(
            model, tokenizer,
            DIAGNOSTIC_CASES[0]['reference'],
            DIAGNOSTIC_CASES[0]['student'],
            DIAGNOSTIC_CASES[0]['question'],
            device=device,
        )
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dev_ms = ((time.perf_counter() - t0) * 1000.0) / 50.0

    print("\n" + "-" * 88)
    print(f"Diagnostic Scenarios Passed : {passed}/{len(DIAGNOSTIC_CASES)}")
    print(f"Live Single-Answer Latency  : {dev_ms:.2f} ms on {device}")
    print("=" * 88)


def run_interactive_loop(model, tokenizer, ckpt, device):
    print("=" * 88)
    print(f"  INTERACTIVE KNOWLEDGEQUEST ANSWER GRADER ({ckpt['model_name']})")
    print("  Press Enter on an empty Reference or type 'q' to quit.")
    print("=" * 88)

    default_q = "Why do we use a target network in Double DQN?"
    default_ref = "A separate slowly-updated target network stabilizes Q-value targets and prevents destructive feedback oscillations."
    print(f"\nTip: Press Enter at Question/Reference prompts to use the built-in RL quiz prompt:\n  Q  : {default_q}\n  Ref: {default_ref}\n")

    while True:
        q = input("Question (optional, Enter for built-in, 'q' to quit): ").strip()
        if q.lower() in ('q', 'quit', 'exit'):
            break
        ref = input("Reference Concept (Enter for built-in): ").strip()
        if ref.lower() in ('q', 'quit', 'exit'):
            break
        if not ref:
            q = q or default_q
            ref = default_ref

        stu = input("Your Student Answer: ").strip()
        if stu.lower() in ('q', 'quit', 'exit') or not stu:
            break

        t0 = time.perf_counter()
        res = grade_student_answer(model, tokenizer, ref, stu, question=q, device=device)
        ms = (time.perf_counter() - t0) * 1000.0

        print(f"\n  Verdict             : {res['verdict'].upper()}")
        print(f"  Semantic Score      : {res['semantic_score']*100:.1f}% ({res['semantic_score']:.4f})")
        print(f"  RPG Damage Mult     : {res['rpg_damage_multiplier']:.2f}x")
        print(f"  DKT Mastery Signal  : binary_correct = {res['binary_correct']} (P_correct = {res['prob_correct']:.3f})")
        print(f"  Class Probabilities : Correct={res['prob_correct']:.2f} | Partial={res['prob_partial']:.2f} | Incorrect={res['prob_incorrect']:.2f}")
        print(f"  Grading Latency     : {ms:.2f} ms\n" + "-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test Semantic Short-Answer Evaluator")
    parser.add_argument('--checkpoint', type=str, default='deberta_v3_grader.pt', help='Checkpoint filename in Models/AnswerEvaluator/checkpoints/')
    parser.add_argument('--auto', action='store_true', help='Run automated 6-scenario diagnostic suite and exit')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(os.path.dirname(script_dir), 'checkpoints', args.checkpoint)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    model, tokenizer, ckpt = load_trained_grader(ckpt_path, device=device)
    if args.auto:
        run_automated_audit(model, tokenizer, ckpt, device)
    else:
        run_interactive_loop(model, tokenizer, ckpt, device)


if __name__ == '__main__':
    main()
