"""
Diagnostic & Live Interactive Tester for the Trained RL Game Master
-------------------------------------------------------------------
Defaults to your top-scoring champion checkpoint:
  `Models/RL/checkpoints/double_dqn_gamemaster.pt` (Reward = 19.23)

Modes:
  1. Automatic Diagnostic Suite:
     - Benchmarks live CPU/GPU decision latency (< 0.5 ms).
     - Probes exact Q-values & chosen actions across distinct student states.
     - Replays full trajectories for a Struggling Student vs. an Advanced Student.
  2. Live Interactive Game Master Playground:
     - Enter custom student mastery, fail streak, frustration, & boredom values
       and inspect what encounter the RL Game Master spawns in KnowledgeQuest!

Usage:
  python Models/RL/scripts/test_rl_gamemaster.py
"""

import os
import sys
import time
import numpy as np
import torch

from student_env import (
    KnowledgeQuestStudentEnv,
    load_shared_dkt_and_affect,
    STATE_DIM,
    NUM_ACTIONS,
    ACTION_NAMES,
)
from models import DuelingDoubleDQN, PPOActorCritic
from rl_trainer_utils import resolve_rl_paths


def load_rl_gamemaster(ckpt_filename='double_dqn_gamemaster.pt', device=None):
    if device is None:
        device = torch.device('cpu')

    checkpoints_dir, _ = resolve_rl_paths()
    ckpt_path = os.path.join(checkpoints_dir, ckpt_filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"RL checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_type = ckpt['model_type']

    if 'PPO' in m_type:
        model = PPOActorCritic(
            state_dim=ckpt['state_dim'],
            num_actions=ckpt['num_actions'],
            hidden_dim=ckpt['hidden_dim'],
        ).to(device)
    else:
        model = DuelingDoubleDQN(
            state_dim=ckpt['state_dim'],
            num_actions=ckpt['num_actions'],
            hidden_dim=ckpt['hidden_dim'],
        ).to(device)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt, ckpt_path


def render_bar(val, width=16):
    filled = int(round(float(np.clip(val, 0.0, 1.0)) * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {val * 100:5.1f}%"


def run_diagnostics(model, ckpt, ckpt_path):
    print("=" * 88)
    print("  RL ADAPTIVE GAME MASTER DIAGNOSTIC & SIMULATION SUITE")
    print("=" * 88)
    m_info = ckpt.get('best_eval_metrics', {})
    print(f"Loaded Champion   : {os.path.basename(ckpt_path)} ({ckpt['model_type']}, Ep {ckpt['best_episode']})")
    print(
        f"Benchmark Score   : Reward = {m_info.get('mean_reward', 0):.2f} | "
        f"Mastery = {m_info.get('final_mastery', 0)*100:.1f}% | "
        f"Concentrating = {m_info.get('mean_concentrating', 0)*100:.1f}% | "
        f"Frustrated = {m_info.get('mean_frustrated', 0)*100:.1f}%"
    )

    # 1. Inference Latency Benchmark
    dummy_state = np.array([0.55, 0.50, 0.70, 0.0, 0.2, 0.0, 0.45, 0.2, 0.2, 0.75, 0.30, 0.15], dtype=np.float32)
    _ = model.select_action(dummy_state)
    t0 = time.perf_counter()
    for _ in range(500):
        _ = model.select_action(dummy_state)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / 500.0
    print(f"CPU Decision Speed: {avg_ms:.3f} ms per turn (Zero game lag!)\n")

    # 2. Diagnostic State Probes
    print("--- TEST 1: Game Master Action Selection Across Student States ---")
    probes = [
        (
            "State A: Struggling & Tilted (25% mastery, 4 fails in a row, 78% frustrated)",
            np.array([0.25, 0.28, 0.65, 0.00, 0.0, 0.80, 0.85, 0.80, 0.10, 0.52, 0.78, 0.20], dtype=np.float32),
        ),
        (
            "State B: Prerequisite Deficit (52% focus mastery, weak 35% prereq -> gap=0.27)",
            np.array([0.52, 0.46, 0.35, 0.27, 0.2, 0.20, 0.50, 0.40, 0.65, 0.68, 0.38, 0.18], dtype=np.float32),
        ),
        (
            "State C: Steady Flow Learner (58% mastery, 2-win streak, 85% concentrating)",
            np.array([0.58, 0.56, 0.78, 0.00, 0.4, 0.00, 0.45, 0.20, 0.20, 0.85, 0.28, 0.12], dtype=np.float32),
        ),
        (
            "State D: High-Mastery Ready Learner (82% mastery, 4-win streak, prereqs solid)",
            np.array([0.82, 0.79, 0.85, 0.00, 0.8, 0.00, 0.32, 0.20, 0.15, 0.82, 0.24, 0.26], dtype=np.float32),
        ),
    ]

    for title, s_vec in probes:
        act = model.select_action(s_vec)
        if isinstance(model, DuelingDoubleDQN):
            with torch.no_grad():
                q = model(torch.as_tensor(s_vec).unsqueeze(0))[0].numpy()
            q_str = " | ".join(f"A{i}:{q[i]:.2f}" for i in range(NUM_ACTIONS))
        else:
            q_str = ""
        print(f"  {title}")
        print(f"    -> Chosen Action [{act}]: {ACTION_NAMES[act]:<33} ({q_str})")

    # 3. Full Trajectory Simulation: Struggling vs Advanced Student
    shared_res = load_shared_dkt_and_affect()
    env = KnowledgeQuestStudentEnv(shared_resources=shared_res, max_steps=10, seed=7)

    for arch in ('struggling', 'advanced'):
        print(f"\n--- TEST 2: 10-Turn Live Episode Replay ({arch.upper()} Student) ---")
        s = env.reset(archetype=arch)
        start_m = float(np.mean(env.mastery))
        print(f"  Initial Course Mastery: {render_bar(start_m)} | Frust: {env.frustrated*100:4.1f}% | Bored: {env.bored*100:4.1f}%")
        for step_i in range(1, 11):
            a = model.select_action(s)
            s, r, _, info = env.step(a)
            res_tag = "PASS" if info['correct'] == 1 else "FAIL"
            print(
                f"  Turn {step_i:02d} | GM -> [{a}] {ACTION_NAMES[a]:<31} | "
                f"Topic {info['target_concept']} ({res_tag}) | "
                f"Mastery: {info['mean_mastery']*100:5.1f}% | Frust: {info['frustrated']*100:4.1f}%"
            )
    print("=" * 88)


def run_interactive(model):
    print("\nLIVE INTERACTIVE GAME MASTER TESTER (Press Enter for defaults, or 'q' to quit)")
    while True:
        m_in = input("\nEnter Student Concept Mastery % (0-100, e.g. 25=weak, 60=mid, 85=strong) [default=45]: ").strip()
        if m_in.lower() == 'q':
            break
        try:
            mast = float(m_in) / 100.0 if m_in else 0.45
        except ValueError:
            mast = 0.45
        mast = float(np.clip(mast, 0.05, 0.98))

        pre_in = input("Enter Weakest Prerequisite Mastery % (0-100, <60 means prereq gap!) [default=75]: ").strip()
        if pre_in.lower() == 'q':
            break
        try:
            min_pre = float(pre_in) / 100.0 if pre_in else 0.75
        except ValueError:
            min_pre = 0.75
        min_pre = float(np.clip(min_pre, 0.05, 0.98))
        pre_gap = max(0.0, 0.62 - min_pre)

        fs_in = input("Enter Current Fail Streak (0-5 consecutive wrong answers) [default=0]: ").strip()
        if fs_in.lower() == 'q':
            break
        fail_s = min(int(fs_in), 5) / 5.0 if fs_in.isdigit() else 0.0
        win_s = 0.4 if fail_s == 0.0 else 0.0

        fr_in = input("Enter Student Frustration % (0-100, e.g. 30=calm, 75=tilted) [default=32]: ").strip()
        if fr_in.lower() == 'q':
            break
        try:
            frust = float(fr_in) / 100.0 if fr_in else 0.32
        except ValueError:
            frust = 0.32
        frust = float(np.clip(frust, 0.05, 0.98))

        state_vec = np.array([
            mast,
            mast,
            min_pre,
            pre_gap,
            win_s,
            fail_s,
            0.75 if fail_s > 0 else 0.45,
            fail_s,
            0.25,
            max(0.40, 0.85 - 0.4 * frust),
            frust,
            0.15,
        ], dtype=np.float32)

        act = model.select_action(state_vec)
        print(f"  => RL Game Master Spawns Action [{act}]: {ACTION_NAMES[act]}")


def main():
    ckpt_file = 'double_dqn_gamemaster.pt'
    for arg in sys.argv[1:]:
        if arg.endswith('.pt'):
            ckpt_file = arg

    model, ckpt, ckpt_path = load_rl_gamemaster(ckpt_file)
    run_diagnostics(model, ckpt, ckpt_path)

    if '--auto' not in sys.argv:
        run_interactive(model)


if __name__ == '__main__':
    main()
