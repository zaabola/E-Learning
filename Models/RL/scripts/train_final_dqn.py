"""
Final Production RL Training Script: 3-Step Dueling Double-DQN (`Final-Double-DQN`)
-----------------------------------------------------------------------------------
Trains a higher-capacity Dueling Double-DQN Game Master over a long horizon
(up to 2,500 student episodes = 62,500 turns) with:
  1. 3-Step Bootstrapped Returns (propagates multi-turn pedagogical credit fast:
     e.g., Prerequisite Review / Podcast Hint -> Flow Battle -> Hard Synthesis)
  2. Deeper Dueling Network (`hidden_dim = 192`) + 40,000-transition Replay Buffer
  3. ReduceLROnPlateau LR controller (halves LR when validation reward plateaus)
  4. Early Stopping (`patience = 12` eval checkpoints) + strict Best-Checkpoint saving

Saved Outputs:
  - Model checkpoint : Models/RL/checkpoints/best_final_dqn_gamemaster.pt
  - Training CSV log : Models/RL/results/final_dqn_training_log.csv
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
# Long-Horizon Hyperparameters & Anti-Overfitting Controls
# =====================================================================
MAX_EPISODES = 2500
EVAL_INTERVAL = 40
MAX_STEPS_PER_EP = 25
HIDDEN_DIM = 192
N_STEP = 3                  # 3-step multi-turn pedagogical return
GAMMA = 0.96
INITIAL_LR = 8e-4
MIN_LR = 5e-5
LR_PATIENCE = 4             # Halve LR after 4 evals without reward gain
EARLY_STOP_PATIENCE = 12    # Stop if 12 evals (480 eps) pass with no new best
BATCH_SIZE = 128
TRAIN_EVERY_STEPS = 2       # Lower-variance batched updates every 2 steps
REPLAY_CAPACITY = 40_000
WARMUP_STEPS = 600
TAU = 0.015
EPS_START = 1.00
EPS_END = 0.03
EPS_DECAY_EPISODES = 900
SEED = 42


class NStepReplayBuffer:
    """
    Stores n-step transitions (s_t, a_t, R_t^(n), s_{t+n}, done_{t+n}, gamma_n)
    for faster multi-turn credit assignment in sequential tutoring.
    """
    def __init__(self, capacity, n_step=3, gamma=0.96):
        self.buf = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_queue = deque()

    def _pop_n_step(self):
        s0, a0, _, _, _ = self.n_queue[0]
        cum_r = 0.0
        discount = 1.0
        last_ns = self.n_queue[-1][3]
        last_done = 0.0

        for _, _, r, ns, d in self.n_queue:
            cum_r += discount * r
            discount *= self.gamma
            last_ns = ns
            if d > 0.5:
                last_done = 1.0
                break

        self.buf.append((s0, a0, cum_r, last_ns, last_done, discount))
        self.n_queue.popleft()

    def push(self, s, a, r, ns, done):
        self.n_queue.append((s, a, r, ns, done))
        if len(self.n_queue) >= self.n_step:
            self._pop_n_step()
        if done > 0.5:
            while self.n_queue:
                self._pop_n_step()

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d, gn = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
            np.array(gn, dtype=np.float32),
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
    gpu_label = f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""

    checkpoints_dir, results_dir = resolve_rl_paths()
    ckpt_name = 'best_final_dqn_gamemaster.pt'
    csv_name = 'final_dqn_training_log.csv'
    ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
    csv_path = os.path.join(results_dir, csv_name)
    summary_path = os.path.join(results_dir, 'rl_models_summary.csv')

    print("=" * 86)
    print("  FINAL PRODUCTION RL TRAINING: 3-Step Dueling Double-DQN (`Final-Double-DQN`)")
    print("=" * 86)
    print(f"Device          : {device}{gpu_label}")
    print(f"Max Episodes    : {MAX_EPISODES} (Early Stop Patience={EARLY_STOP_PATIENCE} evals, LR Patience={LR_PATIENCE} evals)")
    print(f"Architecture    : {N_STEP}-Step Dueling Double-DQN (Hidden={HIDDEN_DIM}, Replay={REPLAY_CAPACITY:,})")
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
    print(f"Trainable Params: {param_count:,}")
    print("-" * 86)

    optimizer = torch.optim.AdamW(online_net.parameters(), lr=INITIAL_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )
    loss_fn = nn.SmoothL1Loss()
    replay = NStepReplayBuffer(capacity=REPLAY_CAPACITY, n_step=N_STEP, gamma=GAMMA)

    best_eval_reward = -1e9
    best_eval_metrics = None
    best_weights = None
    best_episode = 0
    evals_without_gain = 0
    global_steps = 0

    csv_fields = [
        'model', 'episode', 'epsilon', 'learning_rate', 'recent_train_reward',
        'eval_mean_reward', 'eval_final_mastery', 'eval_course_pass_rate',
        'eval_mean_concentrating', 'eval_mean_frustrated', 'eval_mean_bored',
        'is_best', 'elapsed_sec',
    ]
    log_rows = []
    recent_rewards = deque(maxlen=EVAL_INTERVAL)
    t_start = time.time()

    for ep in range(1, MAX_EPISODES + 1):
        frac = min(1.0, (ep - 1) / max(1, EPS_DECAY_EPISODES))
        eps = EPS_START + frac * (EPS_END - EPS_START)

        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            global_steps += 1
            if random.random() < eps:
                action = random.randrange(NUM_ACTIONS)
            else:
                action = online_net.select_action(state, device=device)

            next_state, reward, done, _ = env.step(action)
            replay.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            if len(replay) >= WARMUP_STEPS and (global_steps % TRAIN_EVERY_STEPS == 0):
                b_s, b_a, b_r, b_ns, b_d, b_gn = replay.sample(BATCH_SIZE)
                t_s = torch.as_tensor(b_s, device=device)
                t_a = torch.as_tensor(b_a, device=device).unsqueeze(1)
                t_r = torch.as_tensor(b_r, device=device)
                t_ns = torch.as_tensor(b_ns, device=device)
                t_d = torch.as_tensor(b_d, device=device)
                t_gn = torch.as_tensor(b_gn, device=device)

                q_sa = online_net(t_s).gather(1, t_a).squeeze(1)

                with torch.no_grad():
                    next_actions = online_net(t_ns).argmax(dim=1, keepdim=True)
                    next_q = target_net(t_ns).gather(1, next_actions).squeeze(1)
                    target_q = t_r + t_gn * (1.0 - t_d) * next_q

                loss = loss_fn(q_sa, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    for p_on, p_tg in zip(online_net.parameters(), target_net.parameters()):
                        p_tg.data.mul_(1.0 - TAU).add_(TAU * p_on.data)

        recent_rewards.append(ep_reward)

        if ep % EVAL_INTERVAL == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            eval_m = evaluate_policy(
                lambda s: online_net.select_action(s, device=device),
                shared_res,
            )

            is_best = eval_m['mean_reward'] > (best_eval_reward + 1e-4)
            if is_best:
                best_eval_reward = eval_m['mean_reward']
                best_eval_metrics = eval_m
                best_episode = ep
                best_weights = {k: v.cpu() for k, v in online_net.state_dict().items()}
                evals_without_gain = 0
            elif ep >= EPS_DECAY_EPISODES:
                # Only start counting early-stop patience after exploration phase finishes!
                evals_without_gain += 1

            elapsed = time.time() - t_start
            star = " *BEST*" if is_best else ""
            print(
                f"Ep [{ep:04d}/{MAX_EPISODES}] ({elapsed:5.1f}s) | Eps: {eps:.2f} | LR: {cur_lr:.5f} | "
                f"Reward: {eval_m['mean_reward']:6.2f} | Mastery: {eval_m['final_mastery']*100:5.1f}% | "
                f"Pass: {eval_m['course_pass_rate']*100:5.1f}% | Frust: {eval_m['mean_frustrated']:.3f}{star}"
            )

            log_rows.append({
                'model': 'Final-Double-DQN',
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

            # Step LR controller once exploration phase settles below 0.25
            if eps <= 0.25:
                prev_lr = cur_lr
                scheduler.step(eval_m['mean_reward'])
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < prev_lr:
                    print(f"   [LR Scheduler] Reward plateau -> reducing LR: {prev_lr:.5f} -> {new_lr:.5f}")

            if evals_without_gain >= EARLY_STOP_PATIENCE:
                print(f"\n[Early Stopping] Policy converged (no gain over {EARLY_STOP_PATIENCE} evals post-exploration). Halting cleanly!")
                break

    total_time = time.time() - t_start

    torch.save({
        'model_type': 'Final-Double-DQN',
        'state_dim': STATE_DIM,
        'num_actions': NUM_ACTIONS,
        'hidden_dim': HIDDEN_DIM,
        'n_step': N_STEP,
        'state_features': STATE_FEATURE_NAMES,
        'action_names': ACTION_NAMES,
        'best_episode': best_episode,
        'best_eval_metrics': best_eval_metrics,
        'param_count': param_count,
        'state_dict': best_weights,
    }, ckpt_path)

    upsert_rl_summary_csv(summary_path, {
        'model': 'Final-Double-DQN',
        'parameters': param_count,
        'episodes': ep,
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

    print("-" * 86)
    print(f"[DONE] Final-Double-DQN finished ({ep} episodes in {total_time:.1f}s | Best Ep {best_episode})")
    print(
        f"  Best Eval Reward: {best_eval_metrics['mean_reward']:.2f} | "
        f"Final Mastery: {best_eval_metrics['final_mastery']*100:.1f}% | "
        f"Pass Rate: {best_eval_metrics['course_pass_rate']*100:.1f}% | "
        f"Action Mix: {best_eval_metrics['action_distribution']}"
    )
    print(f"  -> Saved Best Model Checkpoint : {ckpt_path}")
    print(f"  -> Saved Training CSV Log      : {csv_path}")
    print(f"  -> Updated Summary CSV         : {summary_path}")
    print("=" * 86)


if __name__ == '__main__':
    main()
