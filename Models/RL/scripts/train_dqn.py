"""
Standalone RL Training Script 1: Dueling Double-DQN Game Master
---------------------------------------------------------------
Trains a value-based Dueling Double Deep Q-Network (`DuelingDoubleDQN`)
with Experience Replay, Polyak Target Updates, and Cosine LR Scheduling.

Saved Outputs:
  - Model checkpoint : Models/RL/checkpoints/double_dqn_gamemaster.pt
  - Training CSV log : Models/RL/results/dqn_training_log.csv
  - Summary CSV row  : Models/RL/results/rl_models_summary.csv
"""

from collections import deque
import csv
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn

from student_env import (
    KnowledgeQuestStudentEnv,
    load_shared_dkt_and_affect,
    STATE_DIM,
    NUM_ACTIONS,
    STATE_FEATURE_NAMES,
    ACTION_NAMES,
)
from models import DuelingDoubleDQN
from rl_trainer_utils import (
    resolve_rl_paths,
    evaluate_policy,
    upsert_rl_summary_csv,
    ensure_baselines_in_summary,
)

# =====================================================================
# Hyperparameters (Matched 800-Episode Budget with train_ppo.py)
# =====================================================================
TOTAL_EPISODES = 800
EVAL_INTERVAL = 40
MAX_STEPS_PER_EP = 25
HIDDEN_DIM = 128
GAMMA = 0.96
LEARNING_RATE = 1e-3
MIN_LR = 1e-4
BATCH_SIZE = 64
REPLAY_CAPACITY = 15_000
WARMUP_STEPS = 400
TAU = 0.02              # Soft target network Polyak update rate
EPS_START = 1.00
EPS_END = 0.05
EPS_DECAY_EPISODES = 550
SEED = 42


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints_dir, results_dir = resolve_rl_paths()
    ckpt_name = 'double_dqn_gamemaster.pt'
    csv_name = 'dqn_training_log.csv'
    ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
    csv_path = os.path.join(results_dir, csv_name)
    summary_path = os.path.join(results_dir, 'rl_models_summary.csv')

    print("=" * 82)
    print("  STANDALONE RL TRAINING 1: Dueling Double-DQN Adaptive Game Master")
    print("=" * 82)
    print(f"Device          : {device}")
    print(f"Episodes        : {TOTAL_EPISODES} ({TOTAL_EPISODES * MAX_STEPS_PER_EP:,} student turns)")
    print(f"Model Save Path : {ckpt_path}")
    print(f"CSV Log Path    : {csv_path}\n")

    shared_res = load_shared_dkt_and_affect()
    ensure_baselines_in_summary(summary_path, shared_res)

    env = KnowledgeQuestStudentEnv(shared_resources=shared_res, max_steps=MAX_STEPS_PER_EP, seed=SEED)

    online_net = DuelingDoubleDQN(state_dim=STATE_DIM, num_actions=NUM_ACTIONS, hidden_dim=HIDDEN_DIM).to(device)
    target_net = DuelingDoubleDQN(state_dim=STATE_DIM, num_actions=NUM_ACTIONS, hidden_dim=HIDDEN_DIM).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    param_count = sum(p.numel() for p in online_net.parameters() if p.requires_grad)
    print(f"Architecture    : Dueling Double-DQN | State Dim: {STATE_DIM} | Actions: {NUM_ACTIONS} | Params: {param_count:,}")
    print("-" * 82)

    optimizer = torch.optim.AdamW(online_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPISODES, eta_min=MIN_LR)
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(REPLAY_CAPACITY)

    best_eval_reward = -1e9
    best_eval_metrics = None
    best_weights = None
    best_episode = 0

    csv_fields = [
        'model', 'episode', 'epsilon', 'learning_rate', 'recent_train_reward',
        'eval_mean_reward', 'eval_final_mastery', 'eval_course_pass_rate',
        'eval_mean_concentrating', 'eval_mean_frustrated', 'eval_mean_bored',
        'is_best', 'elapsed_sec',
    ]
    log_rows = []
    recent_rewards = deque(maxlen=EVAL_INTERVAL)
    t_start = time.time()

    for ep in range(1, TOTAL_EPISODES + 1):
        frac = min(1.0, (ep - 1) / max(1, EPS_DECAY_EPISODES))
        eps = EPS_START + frac * (EPS_END - EPS_START)

        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            if random.random() < eps:
                action = random.randrange(NUM_ACTIONS)
            else:
                action = online_net.select_action(state, device=device)

            next_state, reward, done, _ = env.step(action)
            replay.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            if len(replay) >= WARMUP_STEPS:
                b_s, b_a, b_r, b_ns, b_d = replay.sample(BATCH_SIZE)
                t_s = torch.as_tensor(b_s, device=device)
                t_a = torch.as_tensor(b_a, device=device).unsqueeze(1)
                t_r = torch.as_tensor(b_r, device=device)
                t_ns = torch.as_tensor(b_ns, device=device)
                t_d = torch.as_tensor(b_d, device=device)

                q_sa = online_net(t_s).gather(1, t_a).squeeze(1)

                with torch.no_grad():
                    # Double-DQN: select greedy action using online_net, evaluate using target_net
                    next_actions = online_net(t_ns).argmax(dim=1, keepdim=True)
                    next_q = target_net(t_ns).gather(1, next_actions).squeeze(1)
                    target_q = t_r + GAMMA * (1.0 - t_d) * next_q

                loss = loss_fn(q_sa, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
                optimizer.step()

                # Polyak soft update of target network
                with torch.no_grad():
                    for p_on, p_tg in zip(online_net.parameters(), target_net.parameters()):
                        p_tg.data.mul_(1.0 - TAU).add_(TAU * p_on.data)

        recent_rewards.append(ep_reward)
        scheduler.step()

        if ep % EVAL_INTERVAL == 0:
            eval_m = evaluate_policy(
                lambda s: online_net.select_action(s, device=device),
                shared_res,
            )
            is_best = eval_m['mean_reward'] > best_eval_reward
            if is_best:
                best_eval_reward = eval_m['mean_reward']
                best_eval_metrics = eval_m
                best_episode = ep
                best_weights = {k: v.cpu() for k, v in online_net.state_dict().items()}

            elapsed = time.time() - t_start
            cur_lr = optimizer.param_groups[0]['lr']
            star = " *BEST*" if is_best else ""
            print(
                f"Ep [{ep:03d}/{TOTAL_EPISODES}] ({elapsed:4.1f}s) | Eps: {eps:.2f} | "
                f"Eval Reward: {eval_m['mean_reward']:6.2f} | Mastery: {eval_m['final_mastery']*100:5.1f}% | "
                f"Pass: {eval_m['course_pass_rate']*100:5.1f}% | Frust: {eval_m['mean_frustrated']:.3f}{star}"
            )

            log_rows.append({
                'model': 'Double-DQN',
                'episode': ep,
                'epsilon': round(eps, 4),
                'learning_rate': round(cur_lr, 6),
                'recent_train_reward': round(float(np.mean(recent_rewards)), 4),
                'eval_mean_reward': round(eval_m['mean_reward'], 4),
                'eval_final_mastery': round(eval_m['final_mastery'], 4),
                'eval_course_pass_rate': round(eval_m['course_pass_rate'], 4),
                'eval_mean_concentrating': round(eval_m['mean_concentrating'], 4),
                'eval_mean_frustrated': round(eval_m['mean_frustrated'], 4),
                'eval_mean_bored': round(eval_m['mean_bored'], 4),
                'is_best': is_best,
                'elapsed_sec': round(elapsed, 2),
            })

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writeheader()
                writer.writerows(log_rows)

    total_time = time.time() - t_start

    torch.save({
        'model_type': 'Double-DQN',
        'state_dim': STATE_DIM,
        'num_actions': NUM_ACTIONS,
        'hidden_dim': HIDDEN_DIM,
        'state_features': STATE_FEATURE_NAMES,
        'action_names': ACTION_NAMES,
        'best_episode': best_episode,
        'best_eval_metrics': best_eval_metrics,
        'param_count': param_count,
        'state_dict': best_weights,
    }, ckpt_path)

    upsert_rl_summary_csv(summary_path, {
        'model': 'Double-DQN',
        'parameters': param_count,
        'episodes': TOTAL_EPISODES,
        'best_eval_reward': round(best_eval_metrics['mean_reward'], 4),
        'final_course_mastery': round(best_eval_metrics['final_mastery'], 4),
        'course_pass_rate': round(best_eval_metrics['course_pass_rate'], 4),
        'mean_concentrating': round(best_eval_metrics['mean_concentrating'], 4),
        'mean_frustrated': round(best_eval_metrics['mean_frustrated'], 4),
        'mean_bored': round(best_eval_metrics['mean_bored'], 4),
        'total_time_sec': round(total_time, 2),
        'checkpoint_file': ckpt_name,
        'training_csv_file': csv_name,
    })

    print("-" * 82)
    print(f"[DONE] Double-DQN finished in {total_time:.1f}s (Best Ep {best_episode})")
    print(f"  Best Eval Reward: {best_eval_metrics['mean_reward']:.2f} | Final Mastery: {best_eval_metrics['final_mastery']*100:.1f}% | Pass Rate: {best_eval_metrics['course_pass_rate']*100:.1f}%")
    print(f"  -> Saved Model Checkpoint : {ckpt_path}")
    print(f"  -> Saved Training CSV Log : {csv_path}")
    print(f"  -> Updated Summary CSV    : {summary_path}")
    print("=" * 82)


if __name__ == '__main__':
    main()
