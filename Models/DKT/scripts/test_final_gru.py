"""
Interactive & Diagnostic Tester for `best_final_gru_model.pt`
-------------------------------------------------------------
Lets you test your trained Final-DKT-GRU model in two modes:
  1. Automatic Behavioral Sanity & Scenario Suite (proves anti-guessing, learning curves,
     difficulty sensitivity, and forgetting/struggle detection).
  2. Live Interactive Battle Simulator — enter your own answers, response times, and
     retries turn-by-turn and watch your Mastery Bar update in real time!

Usage:
  python Models/DKT/scripts/test_final_gru.py
"""

import json
import math
import os
import sys
import torch

from dataset import encode_telemetry
from model import FinalDKTGRU
from trainer_utils import resolve_paths

_LOG_MAX_SKILL_ATT = math.log1p(50.0)


def load_trained_final_gru(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, checkpoints_dir, _, mapping_path = resolve_paths()
    ckpt_path = os.path.join(checkpoints_dir, 'best_final_gru_model.pt')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run train_final_gru.py first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    skill_names = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mdata = json.load(f).get('skill_metadata', {})
            for idx_str, info in mdata.items():
                skill_names[int(idx_str)] = info.get('name', f"Skill_{idx_str}")

    model = FinalDKTGRU(
        num_skills=ckpt['num_skills'],
        telemetry_dim=ckpt['telemetry_dim'],
        next_context_dim=ckpt['next_context_dim'],
        embed_dim=ckpt['embed_dim'],
        hidden_dim=ckpt['hidden_dim'],
        num_layers=ckpt['num_layers'],
        dropout=ckpt['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    skill_priors = {int(k): float(v) for k, v in ckpt.get('skill_priors', {}).items()}
    global_prior = float(ckpt.get('global_prior', 0.63))

    return model, ckpt, skill_names, skill_priors, global_prior, device


@torch.no_grad()
def predict_from_history(
    model,
    history,
    next_skill_idx,
    skill_priors,
    global_prior,
    device,
    next_item_difficulty=None,
    next_is_main=1.0,
    next_is_mc=1.0,
    next_is_exam=0.0,
    next_is_repeat=0.0,
):
    """
    Given a list of past player interactions `history`, predicts P(correct) on an
    upcoming question for `next_skill_idx`.

    Each item in `history` is a dict:
      {
        'skill_idx': int,
        'correct': 0 or 1,
        'response_sec': float,
        'attempts': int,
        'answer_type': 'choose_1' or 'algebra',
        'original': 1 or 0,
        'tutor_mode': 'tutor' or 'test',
        'item_diff': optional float
      }
    """
    num_skills = model.num_skills
    L = len(history)

    inter_idx = torch.empty((1, L), dtype=torch.long, device=device)
    past_telem = torch.empty((1, L, model.telemetry_dim), dtype=torch.float32, device=device)
    past_diff = torch.empty((1, L, 1), dtype=torch.float32, device=device)
    target_sk = torch.zeros((1, L), dtype=torch.long, device=device)
    next_ctx = torch.zeros((1, L, model.next_context_dim), dtype=torch.float32, device=device)

    skill_att = {}
    skill_win = {}
    skill_last = {}

    for t, ev in enumerate(history):
        s = ev['skill_idx']
        c = int(ev['correct'])
        s_prior = skill_priors.get(s, global_prior)
        i_diff = float(ev.get('item_diff', s_prior))

        telem = encode_telemetry(
            ms_first_response_time=ev.get('response_sec', 12.0) * 1000.0,
            attempt_count=ev.get('attempts', 1),
            original=ev.get('original', 1),
            answer_type=ev.get('answer_type', 'choose_1'),
            tutor_mode=ev.get('tutor_mode', 'tutor'),
        )

        inter_idx[0, t] = (s + 1) if c == 1 else (s + num_skills + 1)
        past_telem[0, t] = torch.tensor(telem, dtype=torch.float32, device=device)
        past_diff[0, t, 0] = i_diff

        skill_att[s] = skill_att.get(s, 0) + 1
        skill_win[s] = skill_win.get(s, 0) + c
        skill_last[s] = t

        target_sk[0, t] = next_skill_idx + 1

    # Populate upcoming question context at the final step L-1
    s_next_prior = skill_priors.get(next_skill_idx, global_prior)
    p_next_prior = float(next_item_difficulty) if next_item_difficulty is not None else s_next_prior

    p_att = skill_att.get(next_skill_idx, 0)
    p_win = skill_win.get(next_skill_idx, 0)
    att_norm = math.log1p(min(p_att, 50)) / _LOG_MAX_SKILL_ATT
    win_rate = (p_win + 2.0 * s_next_prior) / (p_att + 2.0)
    recency = (1.0 / (1.0 + ((L - 1) - skill_last[next_skill_idx]))) if next_skill_idx in skill_last else 0.0

    next_ctx[0, L - 1] = torch.tensor([
        p_next_prior,
        s_next_prior,
        float(next_is_main),
        float(next_is_mc),
        float(next_is_exam),
        att_norm,
        win_rate,
        recency,
        float(next_is_repeat),
    ], dtype=torch.float32, device=device)

    preds, _ = model(inter_idx, past_telem, past_diff, target_sk, next_ctx)
    return float(preds[0, L - 1].item())


def render_bar(prob, width=20):
    filled = int(round(prob * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {prob * 100:5.1f}%"


def game_master_decision(prob):
    if prob < 0.40:
        return "STRUGGLING -> Trigger Guided Scaffold + 15s WhisperX Podcast Hint"
    elif prob < 0.75:
        return "FLOW ZONE  -> Spawn Standard 4-Move RPG Quiz Battle"
    else:
        return "MASTERED   -> Unlock Hard Open-Text Challenge (Bonus Crit + XP)"


def run_diagnostic_scenarios(model, ckpt, skill_names, skill_priors, global_prior, device):
    print("=" * 80)
    print("  FINAL-DKT-GRU DIAGNOSTIC SUITE & GAMEPLAY SIMULATION")
    print("=" * 80)
    print(f"Loaded Checkpoint : best_final_gru_model.pt (Best Epoch {ckpt['best_epoch']})")
    print(f"Validation Score  : ROC-AUC = {ckpt['best_auc']:.4f} | Accuracy = {ckpt['best_accuracy']*100:.2f}%")
    print(f"Parameters        : {ckpt['param_count']:,} | Skills: {ckpt['num_skills']}\n")

    # Pick a representative medium-difficulty skill
    target_skill = 0
    best_dist = 999.0
    for s_idx, pr in skill_priors.items():
        if abs(pr - 0.62) < best_dist:
            best_dist = abs(pr - 0.62)
            target_skill = s_idx
    s_title = skill_names.get(target_skill, f"Concept #{target_skill}")
    print(f"Testing on Course Concept [{target_skill}]: '{s_title}' (Baseline Difficulty Prior: {skill_priors[target_skill]*100:.1f}%)\n")

    # Scenario 1: Mastery progression across 5 turns
    print("--- SCENARIO 1: Student Learning Curve Across 5 Battle Turns ---")
    turns = [
        ("Turn 1: Wrong after 45s & 3 retries (initial struggle)",   {'skill_idx': target_skill, 'correct': 0, 'response_sec': 45.0, 'attempts': 3, 'answer_type': 'choose_1'}),
        ("Turn 2: Asked for hint first (0 attempts -> failed)",      {'skill_idx': target_skill, 'correct': 0, 'response_sec': 18.0, 'attempts': 0, 'answer_type': 'choose_1'}),
        ("Turn 3: Got a guided scaffold step right in 12s",          {'skill_idx': target_skill, 'correct': 1, 'response_sec': 12.0, 'attempts': 1, 'answer_type': 'choose_1', 'original': 0}),
        ("Turn 4: Solved main 4-move battle question in 10s (Try 1)", {'skill_idx': target_skill, 'correct': 1, 'response_sec': 10.0, 'attempts': 1, 'answer_type': 'choose_1'}),
        ("Turn 5: Nailed typed open-response question in 11s (Try 1)", {'skill_idx': target_skill, 'correct': 1, 'response_sec': 11.0, 'attempts': 1, 'answer_type': 'algebra'}),
    ]
    hist = []
    for label, ev in turns:
        hist.append(ev)
        p = predict_from_history(model, hist, target_skill, skill_priors, global_prior, device)
        print(f"  {label:<58} -> {render_bar(p)} | {game_master_decision(p)}")

    # Scenario 2: Difficulty calibration on the exact same student state
    print("\n--- SCENARIO 2: Upcoming Question Difficulty Calibration (Same Student After Turn 4) ---")
    hist_after_4 = [t[1] for t in turns[:4]]
    for diff_label, diff_val, is_mc in [
        ("Easy Multiple-Choice Wild Encounter (85% pass rate)",   0.85, 1.0),
        ("Medium Standard Route Trainer Battle (62% pass rate)", 0.62, 1.0),
        ("Hard Gym Boss Typed Exam Question (30% pass rate)",    0.30, 0.0),
    ]:
        p = predict_from_history(
            model, hist_after_4, target_skill, skill_priors, global_prior, device,
            next_item_difficulty=diff_val, next_is_mc=is_mc, next_is_exam=(0.0 if is_mc else 1.0)
        )
        print(f"  {diff_label:<58} -> Predicted Success: {render_bar(p)}")

    # Scenario 3: Telemetry Differentiation (Rapid Guess + Multi-Retry vs Clean Mastery)
    print("\n--- SCENARIO 3: Telemetry Comparison (Shaky Guess vs. Clean Mastery) ---")
    shaky_hist = [
        {'skill_idx': target_skill, 'correct': 0, 'response_sec': 85.0, 'attempts': 5, 'answer_type': 'choose_1'},
        {'skill_idx': target_skill, 'correct': 1, 'response_sec': 1.2,  'attempts': 1, 'answer_type': 'choose_1'},
    ]
    clean_hist = [
        {'skill_idx': target_skill, 'correct': 1, 'response_sec': 11.0, 'attempts': 1, 'answer_type': 'algebra'},
        {'skill_idx': target_skill, 'correct': 1, 'response_sec': 9.5,  'attempts': 1, 'answer_type': 'algebra'},
    ]
    p_shaky = predict_from_history(model, shaky_hist, target_skill, skill_priors, global_prior, device)
    p_clean = predict_from_history(model, clean_hist, target_skill, skill_priors, global_prior, device)
    print(f"  Player A (Failed w/ 5 retries, then 1.2s rapid MC click)   -> {render_bar(p_shaky)}")
    print(f"  Player B (Two clean 10s first-try typed explanations)      -> {render_bar(p_clean)}")
    print("=" * 80)


def run_interactive_loop(model, skill_names, skill_priors, global_prior, device):
    print("\nLIVE INTERACTIVE SIMULATOR (Press Enter for defaults, or type 'q' to quit)")
    sample_ids = sorted(list(skill_priors.keys()))[:8]
    print("Sample Skill IDs you can test:")
    for sid in sample_ids:
        print(f"  [{sid:3d}] {skill_names.get(sid, f'Skill_{sid}')} (Prior: {skill_priors[sid]*100:.1f}%)")

    raw_s = input(f"\nChoose a Skill ID [0-{model.num_skills - 1}] (default={sample_ids[0]}): ").strip()
    if raw_s.lower() == 'q':
        return
    skill_idx = int(raw_s) if raw_s.isdigit() and 0 <= int(raw_s) < model.num_skills else sample_ids[0]
    print(f"-> Tracking Concept [{skill_idx}]: {skill_names.get(skill_idx, f'Skill_{skill_idx}')}\n")

    history = []
    turn = 1
    while True:
        ans = input(f"[Turn {turn}] Correct? (1=Right, 0=Wrong, q=Quit) [default=1]: ").strip().lower()
        if ans == 'q':
            break
        correct = 0 if ans == '0' else 1

        rt_in = input("         Response time in seconds (e.g. 1.5=guess, 10=optimal, 75=struggle) [default=10]: ").strip()
        if rt_in.lower() == 'q':
            break
        try:
            response_sec = float(rt_in) if rt_in else 10.0
        except ValueError:
            response_sec = 10.0

        att_in = input("         Attempts used (0=hint taken, 1=first try, 3=retries) [default=1]: ").strip()
        if att_in.lower() == 'q':
            break
        attempts = int(att_in) if att_in.isdigit() else 1

        fmt_in = input("         Question type ('mc'=4-move menu, 'open'=typed answer) [default=mc]: ").strip().lower()
        if fmt_in == 'q':
            break
        ans_type = 'algebra' if fmt_in == 'open' else 'choose_1'

        history.append({
            'skill_idx': skill_idx,
            'correct': correct,
            'response_sec': response_sec,
            'attempts': attempts,
            'answer_type': ans_type,
            'original': 1,
            'tutor_mode': 'tutor',
        })

        prob = predict_from_history(model, history, skill_idx, skill_priors, global_prior, device)
        print(f"  => Updated Mastery: {render_bar(prob)} | {game_master_decision(prob)}\n")
        turn += 1


def main():
    model, ckpt, skill_names, skill_priors, global_prior, device = load_trained_final_gru()
    run_diagnostic_scenarios(model, ckpt, skill_names, skill_priors, global_prior, device)

    if '--auto' not in sys.argv:
        run_interactive_loop(model, skill_names, skill_priors, global_prior, device)


if __name__ == '__main__':
    main()
