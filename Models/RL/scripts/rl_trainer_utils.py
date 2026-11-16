"""
Shared RL Evaluation, Baseline Benchmarking & CSV Logging Utilities
-------------------------------------------------------------------
Evaluates RL policies on a deterministic 60-student benchmark cohort
(20 struggling + 20 average + 20 advanced learners) and logs standardized
metrics to `Models/RL/results/` and `Models/RL/checkpoints/`.
"""

import csv
import os
import numpy as np

from student_env import (
    KnowledgeQuestStudentEnv,
    load_shared_dkt_and_affect,
    NUM_ACTIONS,
)


def resolve_rl_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rl_root = os.path.dirname(script_dir)
    checkpoints_dir = os.path.join(rl_root, 'checkpoints')
    results_dir = os.path.join(rl_root, 'results')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return checkpoints_dir, results_dir


def evaluate_policy(policy_fn, shared_resources, eval_seed=999, episodes_per_archetype=20):
    """
    Evaluates `policy_fn(state_np) -> int` on 60 deterministic held-out student
    episodes (20 struggling, 20 average, 20 advanced).
    """
    env = KnowledgeQuestStudentEnv(shared_resources=shared_resources, max_steps=25, seed=eval_seed)

    ep_rewards = []
    final_masteries = []
    pass_flags = []
    conc_vals = []
    frust_vals = []
    bored_vals = []
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)

    for archetype in ('struggling', 'average', 'advanced'):
        for _ in range(episodes_per_archetype):
            state = env.reset(archetype=archetype)
            done = False
            ep_r = 0.0
            ep_conc, ep_frust, ep_bored = [], [], []

            while not done:
                act = int(policy_fn(state))
                action_counts[act] += 1
                state, reward, done, info = env.step(act)
                ep_r += reward
                ep_conc.append(info['concentrating'])
                ep_frust.append(info['frustrated'])
                ep_bored.append(info['bored'])

            ep_rewards.append(ep_r)
            final_m = info['mean_mastery']
            final_masteries.append(final_m)
            pass_flags.append(1.0 if final_m >= 0.75 else 0.0)
            conc_vals.append(float(np.mean(ep_conc)))
            frust_vals.append(float(np.mean(ep_frust)))
            bored_vals.append(float(np.mean(ep_bored)))

    act_dist = (action_counts / max(1, action_counts.sum())).tolist()
    return {
        'mean_reward': float(np.mean(ep_rewards)),
        'final_mastery': float(np.mean(final_masteries)),
        'course_pass_rate': float(np.mean(pass_flags)),
        'mean_concentrating': float(np.mean(conc_vals)),
        'mean_frustrated': float(np.mean(frust_vals)),
        'mean_bored': float(np.mean(bored_vals)),
        'action_distribution': [round(p, 4) for p in act_dist],
    }


def upsert_rl_summary_csv(summary_csv_path, row_dict):
    fieldnames = list(row_dict.keys())
    existing = []

    if os.path.exists(summary_csv_path):
        with open(summary_csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get('model') != row_dict['model']:
                    existing.append(r)

    existing.append(row_dict)

    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing:
            writer.writerow(r)


def ensure_baselines_in_summary(summary_csv_path, shared_resources):
    """
    Computes non-adaptive baselines (`Random-Policy` and `Static-Medium-Only`) once
    and registers them at the top of `rl_models_summary.csv` so you can see exactly
    how much RL outperforms non-adaptive game design!
    """
    existing_names = set()
    if os.path.exists(summary_csv_path):
        with open(summary_csv_path, 'r', newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                existing_names.add(r.get('model', ''))

    rng = np.random.default_rng(123)
    if 'Baseline-Random' not in existing_names:
        m_rand = evaluate_policy(lambda s: int(rng.integers(0, NUM_ACTIONS)), shared_resources)
        upsert_rl_summary_csv(summary_csv_path, {
            'model': 'Baseline-Random',
            'parameters': 0,
            'episodes': 0,
            'best_eval_reward': round(m_rand['mean_reward'], 4),
            'final_course_mastery': round(m_rand['final_mastery'], 4),
            'course_pass_rate': round(m_rand['course_pass_rate'], 4),
            'mean_concentrating': round(m_rand['mean_concentrating'], 4),
            'mean_frustrated': round(m_rand['mean_frustrated'], 4),
            'mean_bored': round(m_rand['mean_bored'], 4),
            'total_time_sec': 0.0,
            'checkpoint_file': 'N/A',
            'training_csv_file': 'N/A',
        })

    if 'Baseline-Static-Medium' not in existing_names:
        m_med = evaluate_policy(lambda s: 2, shared_resources)
        upsert_rl_summary_csv(summary_csv_path, {
            'model': 'Baseline-Static-Medium',
            'parameters': 0,
            'episodes': 0,
            'best_eval_reward': round(m_med['mean_reward'], 4),
            'final_course_mastery': round(m_med['final_mastery'], 4),
            'course_pass_rate': round(m_med['course_pass_rate'], 4),
            'mean_concentrating': round(m_med['mean_concentrating'], 4),
            'mean_frustrated': round(m_med['mean_frustrated'], 4),
            'mean_bored': round(m_med['mean_bored'], 4),
            'total_time_sec': 0.0,
            'checkpoint_file': 'N/A',
            'training_csv_file': 'N/A',
        })
