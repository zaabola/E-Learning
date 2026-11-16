# Reinforcement Learning Adaptive Game Master (`Models/RL/`)

This module implements and benchmarks **two Deep Reinforcement Learning Game Master architectures** for **KnowledgeQuest**, grounded in:
1. **Your Trained `Final-DKT-GRU`** (`../DKT/checkpoints/best_final_gru_model.pt`) as the simulated student's cognitive mastery engine.
2. **6,117,947 Real Affective Classroom Interactions** (`../RL/eda_results/affective_calibration.json`) extracted from `2012-2013-data-with-predictions-4-final.csv` (`CONCENTRATING`, `FRUSTRATED`, `CONFUSED`, `BORED`).

---

## 1. Files

* [`../eda_affective.py`](file:///c:/Users/zaabola/Desktop/DL-Interactive_L/Models/RL/eda_affective.py): Streams the 2.87 GB 2012–2013 dataset, audits all 35 columns, and exports `eda_results/affective_calibration.json`.
* [`student_env.py`](file:///c:/Users/zaabola/Desktop/DL-Interactive_L/Models/RL/scripts/student_env.py): 5-concept Knowledge Graph curriculum simulator (`STATE_DIM = 12`, `NUM_ACTIONS = 5`).
* [`models.py`](file:///c:/Users/zaabola/Desktop/DL-Interactive_L/Models/RL/scripts/models.py): Defines `DuelingDoubleDQN` (value-based RL) and `PPOActorCritic` (policy-gradient RL).
* [`train_dqn.py`](file:///c:/Users/zaabola/Desktop/DL-Interactive_L/Models/RL/scripts/train_dqn.py): **Run #1** — Standalone trainer for **Dueling Double-DQN**.
* [`train_ppo.py`](file:///c:/Users/zaabola/Desktop/DL-Interactive_L/Models/RL/scripts/train_ppo.py): **Run #2** — Standalone trainer for **PPO Actor-Critic**.

---

## 2. How to Run Both RL Models

With `(.venv)` active in the project root:

```powershell
# 1. Train & save Dueling Double-DQN
python Models/RL/scripts/train_dqn.py

# 2. Train & save PPO Actor-Critic
python Models/RL/scripts/train_ppo.py
```

---

## 3. Saved Outputs

* **Checkpoints (`Models/RL/checkpoints/`)**:
  * `double_dqn_gamemaster.pt`
  * `ppo_gamemaster.pt`
* **Training CSV Logs (`Models/RL/results/`)**:
  * `dqn_training_log.csv`
  * `ppo_training_log.csv`
  * `rl_models_summary.csv` *(Compares `Baseline-Random`, `Baseline-Static-Medium`, `Double-DQN`, and `PPO-ActorCritic` side-by-side!)*
