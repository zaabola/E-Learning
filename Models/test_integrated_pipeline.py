"""
End-to-End 3-Model Integration & Playable Terminal RPG Suite
------------------------------------------------------------
Connects and tests ALL THREE custom-trained KnowledgeQuest models together:
  1. Model #1 (`Final-DKT-GRU`)      -> `Models/DKT/checkpoints/best_final_gru_model.pt`
  2. Model #2 (`Double-DQN` GM)      -> `Models/RL/checkpoints/double_dqn_gamemaster.pt`
  3. Model #3 (`DeBERTa-v3-Small`)   -> `Models/AnswerEvaluator/checkpoints/deberta_v3_grader.pt`

Usage:
  # Automated closed-loop & full 5-badge quest completion audit:
  .\.venv\Scripts\python.exe Models/test_integrated_pipeline.py --auto

  # Live interactive Terminal RPG Battle powered by all 3 models:
  .\.venv\Scripts\python.exe Models/test_integrated_pipeline.py
"""

import argparse
import os
import sys
import time
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from engine.knowledgequest_engine import KnowledgeQuestAIEngine


def bar(val: float, width: int = 14) -> str:
    v = float(np.clip(val, 0.0, 1.0))
    filled = int(round(v * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {v*100:5.1f}%"


def print_engine_banner(engine: KnowledgeQuestAIEngine):
    print("=" * 94)
    print("   KNOWLEDGEQUEST UNIFIED 4-MODEL AI ENGINE (100% LOCAL, WEBCAM-FREE)")
    print("=" * 94)
    print(
        f"  [Model #1 - Cognitive Brain]  {engine.dkt_info['name']:<20} | "
        f"Test AUC: {engine.dkt_info['auc']:.4f} | Params: {engine.dkt_info['params']:,}"
    )
    print(
        f"  [Model #2 - Adaptive GM]      {engine.rl_info['name']:<20} | "
        f"Eval Reward: {engine.rl_info['reward']:.2f} | Params: {engine.rl_info['params']:,}"
    )
    print(
        f"  [Model #3 - Answer Grader]    {engine.grader_info['name']:<20} | "
        f"3-Way Acc: {engine.grader_info['acc_3way']*100:.2f}% | PAWS Traps: {engine.grader_info['paws_trap_acc']*100:.2f}% ({engine.grader_info['device']})"
    )
    print(
        f"  [Model #4 - Vision Grounding] {engine.dino_info['name']:<20} | "
        f"Mean Box IoU: {engine.dino_info['mean_box_iou']:.3f} | {engine.dino_info['puzzles']} Lab Diagrams ({engine.dino_info['device']})"
    )
    print(f"  All 4 checkpoints loaded & GPU-warmed offline in {engine.boot_time_sec:.2f}s.")
    print("=" * 94)


def print_session_hud(session):
    c_stats = session.history[-1]['rpg_combat'] if session.history else {
        'player_hp': session.player_hp, 'guardian_hp': session.guardian_hp, 'total_xp': session.xp
    }
    badges_n = sum(session.consolidated)
    print(
        f"\n  [RPG HUD] Player HP: {c_stats['player_hp']:>3d}/100 | "
        f"Concept Guardian HP: {c_stats['guardian_hp']:>3d}/100 | "
        f"Badges: {badges_n}/5 | XP: {c_stats['total_xp']}"
    )
    print(
        f"  [AFFECT]  Concentrating: {session.concentrating*100:4.1f}% | "
        f"Frustrated: {session.frustrated*100:4.1f}% | Bored: {session.bored*100:4.1f}% (Behavioral Telemetry Only)"
    )
    print("  [DKT 5-CONCEPT KNOWLEDGE GRAPH MASTERY & BADGES]")
    for c in session.curriculum:
        cid = c['concept_id']
        badge = "[*]" if session.consolidated[cid] else "[ ]"
        pre = f" (prereq: {c['prereqs']})" if c['prereqs'] else " (foundation)"
        print(f"    {badge} C{cid} {c['title']:<40}: {bar(session.mastery[cid])}{pre}")


SYNTHESIS_ANSWERS = {
    0: "Multiplying linear matrices together just collapses into a single linear transformation, so non-linear activations like ReLU or GELU are needed to learn curved decision boundaries.",
    1: "Too small a learning rate converges very slowly, while an overly large learning rate overshoots the valley minimum and causes the loss to oscillate or diverge.",
    2: "Repeated chain-rule multiplications across many layers cause gradients to shrink exponentially or blow up; gating preserves gradient flow and gradient clipping caps huge norms.",
    3: "Dropout randomly mutes neurons during training so features cannot co-adapt and overfit, and during evaluation dropout is turned off so all learned weights are active.",
    4: "Keyword overlap is fooled by negations and inverted word order sharing the same words, whereas a Transformer Cross-Encoder uses self-attention across both sentences to judge contextual meaning.",
}


def run_automated_closed_loop_test(engine: KnowledgeQuestAIEngine):
    print("\n>>> TEST PART A: FULL NATURAL 5-BADGE COURSE PLAYTHROUGH (NO FORCED OVERRIDES!) <<<")
    session = engine.create_session(profile='average')

    deberta_calls = 0
    consecutive_repeats = 0
    last_question = None

    while not session.quest_completed and session.turn < 25:
        enc = session.plan_next_encounter()
        if enc['question'] == last_question:
            consecutive_repeats += 1
        last_question = enc['question']

        if enc['choices'] is not None:
            ans = str(int(enc['correct_idx']) + 1)
            sec = 9.5
        else:
            ans = SYNTHESIS_ANSWERS[enc['target_concept_id']]
            sec = 14.2

        res = session.submit_turn_response(enc, user_response=ans, response_sec=sec, retries=1)
        if res['used_neural_grader']:
            deberta_calls += 1

        badge_tag = " -> [* BADGE EARNED!]" if res['badge_just_earned'] else ""
        print(
            f"  Turn {res['turn']:02d} | C{res['target_concept_id']} | "
            f"[{res['rl_action']}] {res['rl_action_name']:<31} | "
            f"Grader: {res['grader']['graded_by']:<24} ({res['grader']['verdict'].upper()}, {res['grader']['semantic_score']*100:5.1f}%) | "
            f"Mastery: {res['mastery_after'][res['target_concept_id']]*100:5.1f}% | "
            f"{res['latencies_ms']['total_ai_turn_ms']:5.1f}ms{badge_tag}"
        )

    print_session_hud(session)

    # Part B: Verify Negation Trap Rejection + Prereq Gap Redirection
    print("\n>>> TEST PART B: ADVERSARIAL NEGATION TRAP & PREREQ INTERCEPTION CHECKS <<<")
    s2 = engine.create_session(profile='struggling')
    enc_trap = s2.plan_next_encounter(focus_concept_id=0, force_action=4)
    res_trap = s2.submit_turn_response(
        enc_trap,
        user_response="Non-linear activations are never needed because stacked linear layers never collapse into a single linear function.",
        response_sec=35.0,
        retries=2,
    )
    trap_ok = res_trap['grader']['verdict'] == 'incorrect' and not res_trap['badge_just_earned']
    print(
        f"  [{'PASS' if trap_ok else 'FAIL'}] Negation Trap Rejected -> Verdict={res_trap['grader']['verdict'].upper()} "
        f"(Score={res_trap['grader']['semantic_score']:.3f}, Badge Earned={res_trap['badge_just_earned']})"
    )

    s2.mastery = np.array([0.82, 0.34, 0.78, 0.52, 0.45], dtype=np.float32)
    enc_pre = s2.plan_next_encounter(focus_concept_id=3)
    pre_ok = enc_pre['rl_action'] == 3 and enc_pre['target_concept_id'] == 1
    print(
        f"  [{'PASS' if pre_ok else 'FAIL'}] Weak Prereq Interception -> Challenging C3 with C1=34% triggered "
        f"[{enc_pre['rl_action']}] {enc_pre['rl_action_name']} on C{enc_pre['target_concept_id']}"
    )

    # Part C: Verify Grounding DINO Visual Diagram Calibration -> Final-DKT-GRU Loop
    dino_res = s2.submit_diagram_interaction(
        puzzle_id='c2_p1',
        user_box=[34, 92, 262, 370],
        response_sec=6.4,
    )
    dino_ok = dino_res['verdict'] == 'correct' and dino_res['iou'] >= 0.80
    print(
        f"  [{'PASS' if dino_ok else 'FAIL'}] Grounding DINO Lab Puzzle -> [{dino_res['target_label']}] "
        f"Verdict={dino_res['verdict'].upper()} (IoU={dino_res['iou']:.2f}, Conf={dino_res['dino_confidence']:.2f}) | "
        f"C2 Mastery: {dino_res['mastery_before'][2]*100:.1f}% -> {dino_res['mastery_after'][2]*100:.1f}%"
    )

    print("\n" + "=" * 94)
    print(f"  QUEST COMPLETED NATURALLY      : {session.quest_completed} in {session.turn} turns (Badges: {sum(session.consolidated)}/5)")
    print(f"  AUTOMATIC DeBERTa BOSS TRIALS  : {deberta_calls} open-text synthesis evaluations triggered")
    print(f"  CONSECUTIVE QUESTION REPEATS   : {consecutive_repeats}")
    print("=" * 94)


def run_interactive_terminal_rpg(engine: KnowledgeQuestAIEngine):
    print("\nChoose a starting student profile to see how the RL Game Master adapts:")
    print("  [1] Struggling Learner  (Low initial mastery + high frustration -> triggers Podcast Scaffolds)")
    print("  [2] Average Learner     (Balanced starting mastery -> ~13-15 turns to earn all 5 Badges) [DEFAULT]")
    print("  [3] Advanced Learner    (Strong foundations -> rapidly unlocks Open-Text Badge Finishers)")
    p_in = input("\nSelect profile (1/2/3, Enter=2, or 'q' to quit): ").strip().lower()
    if p_in in ('q', 'quit', 'exit'):
        return
    profile = {'1': 'struggling', '2': 'average', '3': 'advanced'}.get(p_in, 'average')

    session = engine.create_session(profile=profile)
    print(
        f"\nStarted Live Quest (`{profile.upper()}` profile). Earn all 5 [★] Concept Synthesis Badges to clear the course!"
        "\nType 'q' at any prompt to quit, or 'boss' to jump straight to a Concept's Open-Text Badge Finisher."
    )

    while not session.quest_completed:
        print_session_hud(session)
        enc = session.plan_next_encounter()

        print("\n" + "-" * 94)
        q_str = " | ".join(f"A{i}:{v:.2f}" for i, v in enumerate(enc['rl_q_values']))
        print(f"  Turn #{enc['turn']} | RL Game Master -> [{enc['rl_action']}] {enc['rl_action_name']}  ({q_str})")
        print(f"  Policy Trigger : {enc['decision_reason']}")
        print(f"  Target Concept : C{enc['target_concept_id']} - {enc['target_concept_title']}")

        if enc['podcast_hint']:
            print(f"\n  🎧 AI PODCAST AUDIO HINT:\n     {enc['podcast_hint']}")

        print(f"\n  QUESTION: {enc['question']}")
        if enc['choices'] is not None:
            for idx, ch in enumerate(enc['choices'], start=1):
                print(f"    [{idx}] {ch}")
            prompt_msg = "\n  Your Move (Enter 1-4, OR type an explanation in your own words, 'boss' for Open-Text, 'q' to quit): "
        else:
            print("    ⚔️  [CONCEPT BADGE FINISHER — Explain in your own words! Graded live by DeBERTa-v3-Small]")
            prompt_msg = "\n  Type Your Explanation ('q' to quit): "

        t_prompt = time.perf_counter()
        ans = input(prompt_msg).strip()
        resp_sec = time.perf_counter() - t_prompt

        if ans.lower() in ('q', 'quit', 'exit'):
            print("\nExiting KnowledgeQuest Terminal Session. Great battle!")
            return
        if ans.lower() == 'boss':
            enc = session.plan_next_encounter(force_action=4)
            print(f"\n  ⚔️  [CONCEPT BADGE FINISHER] C{enc['target_concept_id']} - {enc['target_concept_title']}")
            print(f"  BOSS QUESTION: {enc['question']}")
            t_prompt = time.perf_counter()
            ans = input("\n  Type Your Explanation ('q' to quit): ").strip()
            resp_sec = time.perf_counter() - t_prompt
            if ans.lower() in ('q', 'quit', 'exit'):
                return

        res = session.submit_turn_response(enc, user_response=ans, response_sec=resp_sec, retries=1)
        g = res['grader']
        c = res['rpg_combat']
        lat = res['latencies_ms']
        cid = res['target_concept_id']

        print(f"\n  >>> TURN #{res['turn']} BATTLE & AI TELEMETRY REPORT <<<")
        print(
            f"  Graded By       : {g['graded_by']} -> {g['verdict'].upper()} "
            f"(Semantic Score: {g['semantic_score']*100:.1f}%)"
        )
        if not res['used_neural_grader'] and g['verdict'] == 'incorrect':
            print(f"  Concept Summary : {enc['reference_answer']}")
        elif res['used_neural_grader']:
            print(
                f"  DeBERTa Probs   : Correct={g['prob_correct']:.2f} | "
                f"Partial={g['prob_partial']:.2f} | Incorrect={g['prob_incorrect']:.2f}"
            )
            if g['verdict'] != 'correct':
                print(f"  Reference Guide : {enc['reference_answer']}")

        print(
            f"  RPG Combat      : You dealt {c['player_damage_dealt']} DMG (+{c['xp_gained']} XP) | "
            f"Guardian counterattacked for {c['enemy_counter_damage']} DMG"
        )
        print(
            f"  Final-DKT-GRU   : C{cid} Mastery {res['mastery_before'][cid]*100:.1f}% -> "
            f"{res['mastery_after'][cid]*100:.1f}% (Response time: {res['response_sec']:.1f}s)"
        )
        print(
            f"  AI Speed        : RL={lat['rl_gamemaster_ms']:.2f}ms | "
            f"DeBERTa={lat['deberta_grader_ms']:.2f}ms | DKT={lat['final_dkt_gru_ms']:.2f}ms "
            f"(Total: {lat['total_ai_turn_ms']:.2f}ms)"
        )
        if res['badge_just_earned']:
            print(f"\n  ★ CONCEPT BADGE EARNED: C{cid} - {res['target_concept_title']}! ({res['badges_earned_count']}/5 Badges)")

    # Victory Celebration!
    print_session_hud(session)
    deberta_n = sum(1 for h in session.history if h['used_neural_grader'])
    print("\n" + "=" * 94)
    print("  🏆 VICTORY! ALL 5 KNOWLEDGE GRAPH CONCEPT BADGES MASTERED! 🏆")
    print(
        f"  Total Turns: {session.turn} | Final Course Mastery: {float(np.mean(session.mastery))*100:.1f}% | "
        f"Open-Text Boss Trials Graded: {deberta_n} | Total XP: {session.xp}"
    )
    print("=" * 94)


def main():
    parser = argparse.ArgumentParser(description="KnowledgeQuest Unified 3-Model Terminal Tester")
    parser.add_argument('--auto', action='store_true', help='Run automated full-course integration audit')
    parser.add_argument('--cpu', action='store_true', help='Run all 3 models on CPU')
    args = parser.parse_args()

    engine = KnowledgeQuestAIEngine(grader_device='cpu' if args.cpu else None)
    print_engine_banner(engine)

    if args.auto:
        run_automated_closed_loop_test(engine)
    else:
        run_interactive_terminal_rpg(engine)


if __name__ == '__main__':
    main()
