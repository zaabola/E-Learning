"""
Data-Driven & DKT-Backed Student Environment (`KnowledgeQuestStudentEnv`)
-------------------------------------------------------------------------
Simulates a student playing through a 5-concept KnowledgeQuest curriculum zone:
  - Cognitive Brain: Uses your trained `best_final_gru_model.pt` (`FinalDKTGRU`)
    via fast incremental GRU hidden-state updates to compute live mastery vectors
    and realistic question-answering probabilities.
  - Affective Response: Uses `Models/RL/eda_results/affective_calibration.json`
    extracted from ~6M ASSISTments 2012-2013 interactions to model realistic
    CONCENTRATING, FRUSTRATED, CONFUSED, and BORED dynamics (100% webcam-free).
"""

import json
import math
import os
import sys
import numpy as np
import torch

# Allow importing FinalDKTGRU & encode_telemetry from Models/DKT/scripts
_RL_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_ROOT = os.path.dirname(_RL_SCRIPTS_DIR)
_MODELS_ROOT = os.path.dirname(_RL_ROOT)
_DKT_SCRIPTS_DIR = os.path.join(_MODELS_ROOT, 'DKT', 'scripts')
if _DKT_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _DKT_SCRIPTS_DIR)

from dataset import encode_telemetry
from model import FinalDKTGRU

STATE_DIM = 12
NUM_ACTIONS = 5

STATE_FEATURE_NAMES = [
    'focus_concept_mastery',   # 0: DKT mastery [0, 1] of current frontier concept
    'mean_course_mastery',     # 1: Mean mastery [0, 1] across all 5 curriculum concepts
    'min_prereq_mastery',      # 2: Mastery [0, 1] of weakest prerequisite for focus concept
    'prereq_deficit',          # 3: max(0, 0.60 - min_prereq_mastery) prerequisite gap
    'win_streak_norm',         # 4: min(win_streak, 5) / 5
    'fail_streak_norm',        # 5: min(fail_streak, 5) / 5
    'last_latency_norm',       # 6: log1p(last_sec) / log1p(120)
    'last_retries_norm',       # 7: min(last_retries, 5) / 5
    'max_decay_recency',       # 8: max steps neglected among learned concepts / 12
    'est_concentrating',       # 9: Behavioral concentration index [0, 1]
    'est_frustrated',          # 10: Behavioral frustration index [0, 1]
    'est_bored',               # 11: Behavioral boredom index [0, 1]
]

ACTION_NAMES = {
    0: 'Scaffold + 15s Podcast Hint',
    1: 'Easy 4-Move Wild Encounter',
    2: 'Medium Flow Trainer Battle',
    3: 'Spaced-Repetition Prereq Review',
    4: 'Hard Open-Text Boss Challenge',
}

# 5-concept Knowledge Graph DAG:
#   0 (Foundation A), 1 (Foundation B) -> 2 (req 0), 3 (req 1) -> 4 (Boss Synthesis, req 2 & 3)
PREREQ_GRAPH = {
    0: [],
    1: [],
    2: [0],
    3: [1],
    4: [2, 3],
}

_LOG_MAX_SKILL_ATT = math.log1p(50.0)
_LOG_120 = math.log1p(120.0)


def load_shared_dkt_and_affect():
    """
    Loads `best_final_gru_model.pt` on CPU (fast 0.2ms single-step GRU inference)
    and `affective_calibration.json` once so multiple RL episodes share weights cleanly.
    """
    ckpt_path = os.path.join(_MODELS_ROOT, 'DKT', 'checkpoints', 'best_final_gru_model.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Trained DKT model not found at {ckpt_path}")

    cpu = torch.device('cpu')
    ckpt = torch.load(ckpt_path, map_location=cpu, weights_only=False)

    dkt_model = FinalDKTGRU(
        num_skills=ckpt['num_skills'],
        telemetry_dim=ckpt['telemetry_dim'],
        next_context_dim=ckpt['next_context_dim'],
        embed_dim=ckpt['embed_dim'],
        hidden_dim=ckpt['hidden_dim'],
        num_layers=ckpt['num_layers'],
        dropout=ckpt['dropout'],
    ).to(cpu)
    dkt_model.load_state_dict(ckpt['state_dict'])
    dkt_model.eval()

    skill_priors = {int(k): float(v) for k, v in ckpt.get('skill_priors', {}).items()}
    global_prior = float(ckpt.get('global_prior', 0.63))

    # Pick 5 well-calibrated concept IDs spanning foundational -> advanced priors
    sorted_by_ease = sorted(skill_priors.items(), key=lambda kv: kv[1], reverse=True)
    n_sk = len(sorted_by_ease)
    course_skill_ids = [
        sorted_by_ease[int(n_sk * 0.25)][0],  # Concept 0: Foundation A (~72% prior)
        sorted_by_ease[int(n_sk * 0.35)][0],  # Concept 1: Foundation B (~68% prior)
        sorted_by_ease[int(n_sk * 0.50)][0],  # Concept 2: Route 1 C    (~63% prior)
        sorted_by_ease[int(n_sk * 0.65)][0],  # Concept 3: Route 1 D    (~57% prior)
        sorted_by_ease[int(n_sk * 0.80)][0],  # Concept 4: Boss E       (~48% prior)
    ]

    calib_path = os.path.join(_RL_ROOT, 'eda_results', 'affective_calibration.json')
    affect_calib = None
    if os.path.exists(calib_path):
        with open(calib_path, 'r', encoding='utf-8') as f:
            affect_calib = json.load(f)

    return {
        'dkt_model': dkt_model,
        'skill_priors': skill_priors,
        'global_prior': global_prior,
        'course_skill_ids': course_skill_ids,
        'affect_calib': affect_calib,
    }


class KnowledgeQuestStudentEnv:
    """
    OpenAI-Gym-style pedagogical environment for training and evaluating RL Game Masters.
    """
    def __init__(self, shared_resources=None, max_steps=25, seed=42):
        self.res = shared_resources if shared_resources is not None else load_shared_dkt_and_affect()
        self.dkt = self.res['dkt_model']
        self.skill_priors = self.res['skill_priors']
        self.global_prior = self.res['global_prior']
        self.course_skills = self.res['course_skill_ids']
        self.calib_grid = (self.res['affect_calib'] or {}).get('calibration_grid', {})

        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Precompute 1-indexed tensor of the 5 course skills for batched mastery readout
        self._course_target_tensor = torch.tensor(
            [[s + 1 for s in self.course_skills]], dtype=torch.long
        )
        self.reset()

    def _lookup_empirical_affect(self, correct, sec, hint_used, retries):
        if sec < 5.0:
            z = 'rapid_<5s'
        elif sec <= 15.0:
            z = 'flow_5_15s'
        elif sec <= 60.0:
            z = 'deliberate_15_60s'
        else:
            z = 'struggle_>60s'

        h_bit = 1 if hint_used else 0
        r_bit = 1 if retries >= 2 else 0
        key = f"c{int(correct)}_{z}_h{h_bit}_r{r_bit}"

        cell = self.calib_grid.get(key)
        if cell:
            return cell['concentrating'], cell['frustrated'], cell['bored']

        # Fallback empirical values if calibration JSON isn't present yet
        if correct == 1:
            return 0.68, 0.37, (0.38 if sec < 5.0 else 0.24)
        return 0.56, (0.52 if retries >= 2 else 0.44), 0.31

    @torch.no_grad()
    def _step_dkt_hidden(self, skill_id, correct, response_sec, attempts, is_main, is_mc, is_exam, item_diff):
        num_skills = self.dkt.num_skills
        inter_val = (skill_id + 1) if correct == 1 else (skill_id + num_skills + 1)
        inter_t = torch.tensor([[inter_val]], dtype=torch.long)

        telem = encode_telemetry(
            ms_first_response_time=response_sec * 1000.0,
            attempt_count=attempts,
            original=1 if is_main >= 0.5 else 0,
            answer_type='choose_1' if is_mc >= 0.5 else 'algebra',
            tutor_mode='test' if is_exam >= 0.5 else 'tutor',
        )
        telem_t = torch.tensor([[telem]], dtype=torch.float32)
        diff_t = torch.tensor([[[float(item_diff)]]], dtype=torch.float32)

        e_inter = self.dkt.inter_emb(inter_t)
        step_x = torch.cat([e_inter, telem_t, diff_t], dim=-1)
        step_h = self.dkt.input_proj(step_x)
        gru_out, self.h_state = self.dkt.gru(step_h, self.h_state)
        self.last_h_norm = self.dkt.state_norm(gru_out)  # (1, 1, hidden_dim)

    @torch.no_grad()
    def _query_course_mastery(self):
        """
        Queries `FinalDKTGRU` readout head across all 5 course concepts simultaneously
        on a standardized benchmark prompt (diff=0.60, main=1, mc=1, exam=0), adjusted
        by forgetting decay if a concept has been neglected for many turns.
        """
        h_rep = self.last_h_norm.expand(1, 5, self.dkt.hidden_dim)
        e_targets = self.dkt.skill_emb(self._course_target_tensor)

        ctx_rows = []
        for c_idx, s_id in enumerate(self.course_skills):
            s_prior = self.skill_priors.get(s_id, self.global_prior)
            p_att = self.concept_attempts[c_idx]
            p_win = self.concept_wins[c_idx]
            att_norm = math.log1p(min(p_att, 50)) / _LOG_MAX_SKILL_ATT
            win_rate = (p_win + 2.0 * s_prior) / (p_att + 2.0)

            if self.concept_last_step[c_idx] >= 0:
                steps_ago = max(0, self.step_count - self.concept_last_step[c_idx])
                recency = 1.0 / (1.0 + steps_ago)
            else:
                steps_ago = 0
                recency = 0.0

            ctx_rows.append([
                0.60,     # Standardized reference difficulty
                s_prior,
                1.0,      # Main question
                1.0,      # Standard 4-move prompt
                0.0,      # Practice mode
                att_norm,
                win_rate,
                recency,
                0.0,
            ])

        ctx_t = torch.tensor([ctx_rows], dtype=torch.float32)
        joint = torch.cat([h_rep, e_targets, ctx_t], dim=-1)
        probs = torch.sigmoid(self.dkt.readout(joint).squeeze(-1))[0].numpy()

        # Combine neural DKT mastery with structural latent retention & forgetting decay
        effective = np.zeros(5, dtype=np.float32)
        for c_idx in range(5):
            neglect = (self.step_count - self.concept_last_step[c_idx]) if self.concept_last_step[c_idx] >= 0 else 0
            decay = 0.015 * max(0, neglect - 5)
            blended = 0.55 * float(probs[c_idx]) + 0.45 * self.latent_skill[c_idx] - decay
            effective[c_idx] = float(np.clip(blended, 0.08, 0.98))

        return effective

    def _select_frontier_concept(self):
        for c_idx in range(5):
            if self.mastery[c_idx] < 0.74:
                return c_idx
        return 4

    def _build_state(self):
        self.focus_idx = self._select_frontier_concept()
        focus_m = self.mastery[self.focus_idx]
        mean_m = float(np.mean(self.mastery))

        prereqs = PREREQ_GRAPH[self.focus_idx]
        if prereqs:
            min_pre = float(min(self.mastery[p] for p in prereqs))
            pre_gap = max(0.0, 0.62 - min_pre)
        else:
            min_pre = 1.0
            pre_gap = 0.0

        neglects = [
            (self.step_count - self.concept_last_step[i])
            for i in range(5) if self.concept_last_step[i] >= 0
        ]
        max_neglect = min(max(neglects) if neglects else 0, 12) / 12.0

        return np.array([
            focus_m,
            mean_m,
            min_pre,
            pre_gap,
            min(self.win_streak, 5) / 5.0,
            min(self.fail_streak, 5) / 5.0,
            math.log1p(min(self.last_sec, 120.0)) / _LOG_120,
            min(self.last_retries, 5) / 5.0,
            max_neglect,
            self.concentrating,
            self.frustrated,
            self.bored,
        ], dtype=np.float32)

    def reset(self, archetype=None):
        """
        Resets the environment for a new 25-turn student episode.
        `archetype`: None (random), 'struggling', 'average', or 'advanced'.
        """
        if archetype is None:
            archetype = self.rng.choice(['struggling', 'average', 'advanced'], p=[0.35, 0.40, 0.25])
        self.archetype = archetype

        self.step_count = 0
        self.win_streak = 0
        self.fail_streak = 0
        self.last_sec = 14.0
        self.last_retries = 1
        self.concentrating = 0.65
        self.frustrated = 0.36
        self.bored = 0.22

        self.concept_attempts = [0] * 5
        self.concept_wins = [0] * 5
        self.concept_last_step = [-1] * 5

        self.h_state = torch.zeros((self.dkt.num_layers, 1, self.dkt.hidden_dim), dtype=torch.float32)
        self.last_h_norm = torch.zeros((1, 1, self.dkt.hidden_dim), dtype=torch.float32)

        if archetype == 'struggling':
            self.latent_skill = self.rng.uniform(0.18, 0.32, size=5).astype(np.float32)
            self.learn_rate_mult = 0.85
            # Seed 1 initial failed attempt so DKT hidden state reflects early struggle
            self._step_dkt_hidden(self.course_skills[0], 0, 48.0, 3, 1.0, 1.0, 0.0, 0.62)
            self.fail_streak = 1
            self.frustrated = 0.48
        elif archetype == 'advanced':
            self.latent_skill = self.rng.uniform(0.52, 0.68, size=5).astype(np.float32)
            self.learn_rate_mult = 1.18
            self._step_dkt_hidden(self.course_skills[0], 1, 9.5, 1, 1.0, 1.0, 0.0, 0.62)
            self.win_streak = 1
        else:
            self.latent_skill = self.rng.uniform(0.32, 0.48, size=5).astype(np.float32)
            self.learn_rate_mult = 1.00
            self._step_dkt_hidden(self.course_skills[0], 1, 15.0, 1, 1.0, 1.0, 0.0, 0.75)

        self.mastery = self._query_course_mastery()
        return self._build_state()

    def step(self, action):
        prev_mean_mastery = float(np.mean(self.mastery))
        self.step_count += 1
        focus_c = self._select_frontier_concept()

        # Map RL Action -> KnowledgeQuest encounter configuration
        hint_used = False
        is_repeat = 0.0
        if action == 0:
            # Guided Scaffold + 15s WhisperX Podcast Audio Hint
            target_c = focus_c
            item_diff, is_main, is_mc, is_exam = 0.86, 0.0, 1.0, 0.0
            hint_used = True
        elif action == 1:
            # Easy 4-Move Wild Encounter
            target_c = focus_c
            item_diff, is_main, is_mc, is_exam = 0.78, 1.0, 1.0, 0.0
        elif action == 2:
            # Medium Flow-Zone Trainer Battle
            target_c = focus_c
            item_diff, is_main, is_mc, is_exam = 0.62, 1.0, 1.0, 0.0
        elif action == 3:
            # Spaced-Repetition Prerequisite Review
            prereqs = PREREQ_GRAPH[focus_c]
            if prereqs:
                target_c = min(prereqs, key=lambda p: self.mastery[p])
            else:
                target_c = int(np.argmin(self.mastery[:max(1, focus_c + 1)]))
            item_diff, is_main, is_mc, is_exam = 0.68, 1.0, 1.0, 0.0
            is_repeat = 1.0
        elif action == 4:
            # Hard Open-Text / Diagram Boss Challenge (DeBERTa-v3 graded)
            target_c = focus_c
            item_diff, is_main, is_mc, is_exam = 0.35, 1.0, 0.0, 1.0
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check prerequisite readiness penalty
        prereqs = PREREQ_GRAPH[target_c]
        min_pre = min((self.mastery[p] for p in prereqs), default=1.0)
        prereq_penalty = max(0.0, 0.60 - min_pre)

        # Compute true success probability combining current mastery, item difficulty, affect, & prereqs
        c_mast = self.mastery[target_c]
        logit = 3.2 * (c_mast - (1.0 - item_diff)) - 1.8 * prereq_penalty - 0.7 * max(0.0, self.frustrated - 0.45)
        if hint_used:
            logit += 0.75
        if is_mc < 0.5:
            logit -= 0.25  # No 25% multiple-choice guessing floor on typed open responses
        p_success = float(1.0 / (1.0 + math.exp(-logit)))
        p_success = float(np.clip(p_success, 0.05, 0.96))

        correct = 1 if self.rng.random() < p_success else 0

        # Simulate realistic behavioral telemetry (response latency + retries)
        mismatch = (1.0 - item_diff) - c_mast
        if hint_used:
            sec = float(self.rng.uniform(8.0, 16.0))
            retries = 1 if correct == 1 else 2
        elif correct == 1 and mismatch < -0.25:
            # Way too easy -> rapid click / boredom risk
            sec = float(self.rng.uniform(2.2, 5.5))
            retries = 1
        elif correct == 1:
            # Optimal Flow cognitive window (5-15s)
            sec = float(self.rng.uniform(6.5, 15.0))
            retries = 1
        else:
            # Failed -> struggle latency & multiple attempts
            sec = float(self.rng.uniform(28.0, 82.0))
            retries = int(self.rng.integers(2, 5))

        # Update concept learning progression (Zone of Proximal Development peaks when p_success in [0.45, 0.80])
        if correct == 1:
            if action == 4:
                base_gain = 0.125  # High-effort open-response synthesis yields largest mastery jump
            elif action == 0:
                base_gain = 0.055  # Podcast hint + scaffold builds initial foothold
            elif action == 1:
                base_gain = 0.065 if c_mast < 0.65 else 0.020
            else:
                base_gain = 0.090
            # Diminishing gain if question was trivially easy for an already-mastered student
            if c_mast > 0.78 and action in (0, 1):
                base_gain *= 0.25
            self.latent_skill[target_c] = min(0.97, self.latent_skill[target_c] + base_gain * self.learn_rate_mult * (1.0 - 0.45 * prereq_penalty))
            self.win_streak += 1
            self.fail_streak = 0
        else:
            # Podcast scaffold cushions failure; failing a premature Boss Challenge hurts confidence
            slip = 0.005 if hint_used else (0.030 if action == 4 else 0.012)
            self.latent_skill[target_c] = max(0.10, self.latent_skill[target_c] - slip)
            self.fail_streak += 1
            self.win_streak = 0

        # Mild natural decay on unpracticed concepts
        for i in range(5):
            if i != target_c and self.concept_last_step[i] >= 0 and (self.step_count - self.concept_last_step[i]) > 6:
                self.latent_skill[i] = max(0.15, self.latent_skill[i] - 0.008)

        self.concept_attempts[target_c] += 1
        self.concept_wins[target_c] += correct
        self.concept_last_step[target_c] = self.step_count
        self.last_sec = sec
        self.last_retries = retries

        # Advance neural Final-DKT-GRU hidden state and recompute course mastery vector
        self._step_dkt_hidden(
            self.course_skills[target_c], correct, sec, retries, is_main, is_mc, is_exam, item_diff
        )
        self.mastery = self._query_course_mastery()

        # Update affective states anchored in 2012-2013 empirical calibration + momentum
        emp_conc, emp_frust, emp_bored = self._lookup_empirical_affect(correct, sec, hint_used, retries)

        frust_delta = 0.0
        bored_delta = 0.0
        conc_delta = 0.0

        if hint_used:
            # Podcast hint rescues frustrated learners, but bores high-mastery learners!
            if self.fail_streak == 0 and c_mast > 0.62:
                bored_delta += 0.24
                conc_delta -= 0.10
            else:
                frust_delta -= 0.16
                conc_delta += 0.08
        elif action == 1 and c_mast > 0.70:
            bored_delta += 0.20
        elif action == 4 and (c_mast < 0.55 or prereq_penalty > 0.08) and correct == 0:
            frust_delta += 0.24
            conc_delta -= 0.12
        elif 0.45 <= p_success <= 0.80 and correct == 1:
            conc_delta += 0.10
            frust_delta -= 0.08
            bored_delta -= 0.08

        if self.fail_streak >= 2:
            frust_delta += 0.07 * self.fail_streak

        self.concentrating = float(np.clip(0.55 * self.concentrating + 0.45 * emp_conc + conc_delta, 0.05, 0.98))
        self.frustrated = float(np.clip(0.55 * self.frustrated + 0.45 * emp_frust + frust_delta, 0.05, 0.98))
        self.bored = float(np.clip(0.55 * self.bored + 0.45 * emp_bored + bored_delta, 0.05, 0.98))

        # Pedagogical Flow Reward R_t
        new_mean_mastery = float(np.mean(self.mastery))
        delta_mastery = new_mean_mastery - prev_mean_mastery
        in_flow_zone = 1.0 if (0.45 <= p_success <= 0.82 and not (hint_used and c_mast > 0.62)) else 0.0
        prereq_violation = prereq_penalty if action != 3 else 0.0

        reward = (
            6.0 * delta_mastery
            + 0.45 * self.concentrating
            + 0.35 * in_flow_zone
            - 0.65 * max(0.0, self.frustrated - 0.38)
            - 0.55 * max(0.0, self.bored - 0.28)
            - 0.60 * prereq_violation
        )

        done = self.step_count >= self.max_steps
        if done and new_mean_mastery >= 0.75:
            reward += 1.5  # Course mastery completion bonus

        next_state = self._build_state()
        info = {
            'target_concept': target_c,
            'correct': correct,
            'p_success': round(p_success, 4),
            'mean_mastery': round(new_mean_mastery, 4),
            'concentrating': round(self.concentrating, 4),
            'frustrated': round(self.frustrated, 4),
            'bored': round(self.bored, 4),
        }
        return next_state, float(reward), done, info
