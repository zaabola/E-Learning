"""
Standalone RL Training Script 2: PPO Actor-Critic Game Master
-------------------------------------------------------------
Trains a policy-gradient Proximal Policy Optimization (`PPOActorCritic`)
agent with Generalized Advantage Estimation (GAE), Clipped Surrogate Loss,
Entropy Regularization, and Linear LR Annealing.

Saved Outputs:
  - Model checkpoint : Models/RL/checkpoints/ppo_gamemaster.pt
  - Training CSV log : Models/RL/results/ppo_training_log.csv
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
from models import PPOActorCritic
from rl_trainer_utils import (
    resolve_rl_paths,
    evaluate_policy,
    upsert_rl_summary_csv,
    ensure_baselines_in_summary,
)

# =====================================================================
# Hyperparameters (Matched 800-Episode Budget with train_dqn.py)
# =====================================================================
TOTAL_EPISODES = 800
ROLLOUT_EPISODES = 20     # 20 episodes * 25 steps = 500 steps per PPO update
EVAL_INTERVAL = 40
MAX_STEPS_PER_EP = 25
HIDDEN_DIM = 128
GAMMA = 0.96
GAE_LAMBDA = 0.95
CLIP_EPS = 0.20
PPO_EPOCHS = 4
MINIBATCH_SIZE = 125
LEARNING_RATE = 8e-4
MIN_LR = 8e-5
ENTROPY_COEF = 0.015
VALUE_COEF = 0.50
SEED = 42


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        next_val = values[t + 1] if (t + 1 < T and next_non_terminal > 0.5) else 0.0
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints_dir, results_dir = resolve_rl_paths()
    ckpt_name = 'ppo_gamemaster.pt'
    csv_name = 'ppo_training_log.csv'
    ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
    csv_path = os.path.join(results_dir, csv_name)
    summary_path = os.path.join(results_dir, 'rl_models_summary.csv')

    print("=" * 82)
    print("  STANDALONE RL TRAINING 2: PPO Actor-Critic Adaptive Game Master")
    print("=" * 82)
    print(f"Device          : {device}")
    print(f"Episodes        : {TOTAL_EPISODES} ({TOTAL_EPISODES * MAX_STEPS_PER_EP:,} student turns)")
    print(f"Model Save Path : {ckpt_path}")
    print(f"CSV Log Path    : {csv_path}\n")

    shared_res = load_shared_dkt_and_affect()
    ensure_baselines_in_summary(summary_path, shared_res)

    env = KnowledgeQuestStudentEnv(shared_resources=shared_res, max_steps=MAX_STEPS_PER_EP, seed=SEED)

    model = PPOActorCritic(state_dim=STATE_DIM, num_actions=NUM_ACTIONS, hidden_dim=HIDDEN_DIM).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture    : PPO Actor-Critic | State Dim: {STATE_DIM} | Actions: {NUM_ACTIONS} | Params: {param_count:,}")
    print("-" * 82)

    num_updates = TOTAL_EPISODES // ROLLOUT_EPISODES
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_updates, eta_min=MIN_LR)

    best_eval_reward = -1e9
    best_eval_metrics = None
    best_weights = None
    best_episode = 0

    csv_fields = [
        'model', 'episode', 'learning_rate', 'recent_train_reward',
        'eval_mean_reward', 'eval_final_mastery', 'eval_course_pass_rate',
        'eval_mean_concentrating', 'eval_mean_frustrated', 'eval_mean_bored',
        'is_best', 'elapsed_sec',
    ]
    log_rows = []
    recent_rewards = deque(maxlen=EVAL_INTERVAL)
    t_start = time.time()

    ep_counter = 0

    for _ in range(num_updates):
        buf_s, buf_a, buf_logp, buf_r, buf_v, buf_d = [], [], [], [], [], []

        model.eval()
        for _ in range(ROLLOUT_EPISODES):
            ep_counter += 1
            state = env.reset()
            done = False
            ep_r = 0.0

            while not done:
                s_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    act_t, logp_t, _, val_t = model.get_action_and_value(s_t)

                action = int(act_t.item())
                next_state, reward, done, _ = env.step(action)

                buf_s.append(state)
                buf_a.append(action)
                buf_logp.append(float(logp_t.item()))
                buf_r.append(float(reward))
                buf_v.append(float(val_t.item()))
                buf_d.append(1.0 if done else 0.0)

                state = next_state
                ep_r += reward

            recent_rewards.append(ep_r)

        # Compute GAE advantages & returns
        adv_np, ret_np = compute_gae(
            np.array(buf_r, dtype=np.float32),
            np.array(buf_v, dtype=np.float32),
            np.array(buf_d, dtype=np.float32),
        )

        t_s = torch.as_tensor(np.array(buf_s, dtype=np.float32), device=device)
        t_a = torch.as_tensor(np.array(buf_a, dtype=np.int64), device=device)
        t_old_logp = torch.as_tensor(np.array(buf_logp, dtype=np.float32), device=device)
        t_adv = torch.as_tensor(adv_np, device=device)
        t_ret = torch.as_tensor(ret_np, device=device)

        # PPO minibatch optimization epochs
        model.train()
        n_steps = len(buf_s)
        indices = np.arange(n_steps)

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, n_steps, MINIBATCH_SIZE):
                mb = indices[start:start + MINIBATCH_SIZE]
                _, new_logp, entropy, new_val = model.get_action_and_value(t_s[mb], t_a[mb])

                mb_adv = t_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                ratio = torch.exp(new_logp - t_old_logp[mb])
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * ((new_val - t_ret[mb]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        scheduler.step()

        if ep_counter % EVAL_INTERVAL == 0:
            eval_m = evaluate_policy(
                lambda s: model.select_action(s, device=device, deterministic=True),
                shared_res,
            )
            is_best = eval_m['mean_reward'] > best_eval_reward
            if is_best:
                best_eval_reward = eval_m['mean_reward']
                best_eval_metrics = eval_m
                best_episode = ep_counter
                best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

            elapsed = time.time() - t_start
            cur_lr = optimizer.param_groups[0]['lr']
            star = " *BEST*" if is_best else ""
            print(
                f"Ep [{ep_counter:03d}/{TOTAL_EPISODES}] ({elapsed:4.1f}s) | LR: {cur_lr:.5f} | "
                f"Eval Reward: {eval_m['mean_reward']:6.2f} | Mastery: {eval_m['final_mastery']*100:5.1f}% | "
                f"Pass: {eval_m['course_pass_rate']*100:5.1f}% | Frust: {eval_m['mean_frustrated']:.3f}{star}"
            )

            log_rows.append({
                'model': 'PPO-ActorCritic',
                'episode': ep_counter,
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
        'model_type': 'PPO-ActorCritic',
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
        'model': 'PPO-ActorCritic',
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
    print(f"[DONE] PPO-ActorCritic finished in {total_time:.1f}s (Best Ep {best_episode})")
    print(f"  Best Eval Reward: {best_eval_metrics['mean_reward']:.2f} | Final Mastery: {best_eval_metrics['final_mastery']*100:.1f}% | Pass Rate: {best_eval_metrics['course_pass_rate']*100:.1f}%")
    print(f"  -> Saved Model Checkpoint : {ckpt_path}")
    print(f"  -> Saved Training CSV Log : {csv_path}")
    print(f"  -> Updated Summary CSV    : {summary_path}")
    print("=" * 82)


if __name__ == '__main__':
    main()
