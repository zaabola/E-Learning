"""
Unified KnowledgeQuest Closed-Loop AI Engine (`KnowledgeQuestAIEngine`)
-----------------------------------------------------------------------
Connects all 3 custom-trained AI models into a single real-time (<30ms/turn),
100% webcam-free, local-offline Intelligent Tutoring RPG brain:

  [Model #2: RL Game Master]  `Models/RL/checkpoints/double_dqn_gamemaster.pt`
       |  Reads 12-dim (DKT Mastery + Prereq Gap + Behavioral Affect) state
       v  Selects optimal next encounter [0..4] in ~0.25 ms (+ Badge Synthesis Gate)
  [Active RPG Encounter]  (Rotating Multi-Variant Pool: Podcast / Wild / Flow / Prereq / Boss)
       |  Player answers (measures response_sec & retries — zero webcam!)
       v
  [Model #3: Answer Grader]   `Models/AnswerEvaluator/checkpoints/deberta_v3_grader.pt`
       |  Grades open-text answers in ~12-25 ms -> 3-way verdict + [0,1] RPG Damage
       v
  [Model #1: Final-DKT-GRU]   `Models/DKT/checkpoints/best_final_gru_model.pt`
       |  Encodes 13-dim telemetry + correctness -> updates 2-layer GRU memory h_t
       v  Recomputes 5-concept DAG mastery + Consolidates Earned Concept Badges [★]!
"""

import json
import math
import os
import sys
import time
import numpy as np
import torch

_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_ROOT = os.path.dirname(_ENGINE_DIR)

for _sub in (
    _ENGINE_DIR,
    os.path.join(_MODELS_ROOT, 'DKT', 'scripts'),
    os.path.join(_MODELS_ROOT, 'RL', 'scripts'),
    os.path.join(_MODELS_ROOT, 'AnswerEvaluator', 'scripts'),
    os.path.join(_MODELS_ROOT, 'GroundingDINO', 'scripts'),
    os.path.join(_MODELS_ROOT, 'VLM', 'scripts'),
    os.path.join(_MODELS_ROOT, 'LLM', 'scripts'),
):
    if _sub not in sys.path:
        sys.path.insert(0, _sub)

from dataset import encode_telemetry
from model import FinalDKTGRU
from models import DuelingDoubleDQN, PPOActorCritic
from grader_models import grade_student_answer, load_trained_grader
from dino_evaluator import GroundingDINODiagramEvaluator
from vlm_diagram_engine import QwenVLDiagramEngine
from stage4_llm_orchestrator import Stage4LLMOrchestrator
from custom_course_ingestor import ingest_custom_course_package
from student_env import ACTION_NAMES, STATE_FEATURE_NAMES
from curriculum_bank import (
    DEFAULT_CURRICULUM,
    PREREQ_GRAPH,
    list_available_courses,
    load_course_package,
)

_LOG_MAX_SKILL_ATT = math.log1p(50.0)
_LOG_120 = math.log1p(120.0)


class KnowledgeQuestAIEngine:
    """
    Loads all 3 champion checkpoints once and serves stateful `PlayerSession` instances
    for both terminal simulation/playtesting and the FastAPI web backend.
    """

    def __init__(
        self,
        dkt_ckpt_path: str = None,
        rl_ckpt_path: str = None,
        grader_ckpt_path: str = None,
        affect_calib_path: str = None,
        grader_device: str = None,
        vlm_mode: str = 'hybrid',
        vlm_model_id: str = 'Qwen/Qwen3-VL-8B-Instruct',
        llm_mode: str = 'fallback',
        reasoner_model_id: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        stylizer_model_id: str = 'NousResearch/Meta-Llama-3.1-8B-Instruct',
    ):
        t0 = time.perf_counter()
        cpu = torch.device('cpu')
        if grader_device is None:
            self.grader_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.grader_device = torch.device(grader_device)

        dkt_ckpt_path = dkt_ckpt_path or os.path.join(_MODELS_ROOT, 'DKT', 'checkpoints', 'best_final_gru_model.pt')
        rl_ckpt_path = rl_ckpt_path or os.path.join(_MODELS_ROOT, 'RL', 'checkpoints', 'double_dqn_gamemaster.pt')
        grader_ckpt_path = grader_ckpt_path or os.path.join(_MODELS_ROOT, 'AnswerEvaluator', 'checkpoints', 'deberta_v3_grader.pt')
        affect_calib_path = affect_calib_path or os.path.join(_MODELS_ROOT, 'RL', 'eda_results', 'affective_calibration.json')

        # 1. Load Model #1: Final-DKT-GRU (Cognitive Knowledge Tracing)
        dkt_ckpt = torch.load(dkt_ckpt_path, map_location=cpu, weights_only=False)
        self.dkt_model = FinalDKTGRU(
            num_skills=dkt_ckpt['num_skills'],
            telemetry_dim=dkt_ckpt['telemetry_dim'],
            next_context_dim=dkt_ckpt['next_context_dim'],
            embed_dim=dkt_ckpt['embed_dim'],
            hidden_dim=dkt_ckpt['hidden_dim'],
            num_layers=dkt_ckpt['num_layers'],
            dropout=dkt_ckpt['dropout'],
        ).to(cpu)
        self.dkt_model.load_state_dict(dkt_ckpt['state_dict'])
        self.dkt_model.eval()
        self.dkt_info = {
            'name': dkt_ckpt.get('model_type', 'Final-DKT-GRU'),
            'auc': dkt_ckpt.get('best_auc', 0.7996),
            'params': dkt_ckpt.get('param_count', 0),
        }

        self.skill_priors = {int(k): float(v) for k, v in dkt_ckpt.get('skill_priors', {}).items()}
        self.global_prior = float(dkt_ckpt.get('global_prior', 0.63))
        sorted_by_ease = sorted(self.skill_priors.items(), key=lambda kv: kv[1], reverse=True)
        n_sk = len(sorted_by_ease)
        self.course_skill_ids = [
            sorted_by_ease[int(n_sk * 0.25)][0],
            sorted_by_ease[int(n_sk * 0.35)][0],
            sorted_by_ease[int(n_sk * 0.50)][0],
            sorted_by_ease[int(n_sk * 0.65)][0],
            sorted_by_ease[int(n_sk * 0.80)][0],
        ]

        # 2. Load Model #2: Double-DQN RL Adaptive Game Master + Affective Calibration
        rl_ckpt = torch.load(rl_ckpt_path, map_location=cpu, weights_only=False)
        if 'PPO' in rl_ckpt.get('model_type', ''):
            self.rl_model = PPOActorCritic(
                state_dim=rl_ckpt['state_dim'],
                num_actions=rl_ckpt['num_actions'],
                hidden_dim=rl_ckpt['hidden_dim'],
            ).to(cpu)
        else:
            self.rl_model = DuelingDoubleDQN(
                state_dim=rl_ckpt['state_dim'],
                num_actions=rl_ckpt['num_actions'],
                hidden_dim=rl_ckpt['hidden_dim'],
            ).to(cpu)
        self.rl_model.load_state_dict(rl_ckpt['state_dict'])
        self.rl_model.eval()
        self.rl_info = {
            'name': rl_ckpt.get('model_type', 'Double-DQN'),
            'reward': rl_ckpt.get('best_eval_metrics', {}).get('mean_reward', 19.23),
            'params': rl_ckpt.get('param_count', 35590),
        }

        self.calib_grid = {}
        if os.path.exists(affect_calib_path):
            with open(affect_calib_path, 'r', encoding='utf-8') as f:
                self.calib_grid = json.load(f).get('calibration_grid', {})

        # 3. Load Model #3: DeBERTa-v3-Small Cross-Encoder Short-Answer Evaluator
        self.grader_model, self.grader_tokenizer, grader_ckpt = load_trained_grader(
            grader_ckpt_path, device=self.grader_device
        )
        bm = grader_ckpt.get('best_metrics', {})
        self.grader_info = {
            'name': grader_ckpt.get('model_name', 'DeBERTa-v3-Small'),
            'acc_3way': bm.get('acc_3way', 0.7984),
            'macro_f1': bm.get('macro_f1', 0.7703),
            'paws_trap_acc': bm.get('paws_trap_acc', 0.9361),
            'params': grader_ckpt.get('param_count', 142490884),
            'device': str(self.grader_device),
        }
        # Pre-warm CUDA/CPU kernels once so live gameplay turns have zero cold-start lag
        grade_student_answer(
            self.grader_model,
            self.grader_tokenizer,
            reference_answer='Warmup reference.',
            student_answer='Warmup answer.',
            device=self.grader_device,
        )

        # 4. Load Model #4 (Stage 2 Vision): Grounding DINO Zero-Shot Diagram Evaluator
        self.dino_evaluator = GroundingDINODiagramEvaluator(device=self.grader_device, prewarm=True)
        self.dino_info = {
            'name': getattr(self.dino_evaluator, 'model_display_name', 'Grounding-DINO-Tiny'),
            'params': self.dino_evaluator.num_params,
            'device': str(self.grader_device),
            'puzzles': len(self.dino_evaluator.puzzles_list),
            'mean_box_iou': 0.894,
        }

        # 5. Load Model #5 (Stage 1/2 Multimodal VLM): Qwen3-VL-8B (fallback Qwen2.5-VL-7B)
        self.vlm_engine = QwenVLDiagramEngine(
            model_id=vlm_model_id,
            mode=vlm_mode,
            device=str(self.grader_device),
        )
        self.dino_evaluator.active_vlm_name = getattr(self.vlm_engine, 'short_model_name', 'Qwen3-VL-8B')
        self.vlm_info = {
            'name': self.vlm_engine.model_id,
            'short_name': getattr(self.vlm_engine, 'short_model_name', 'Qwen3-VL-8B'),
            'mode': self.vlm_engine.mode,
            'diagrams': len(self.vlm_engine.diagrams_list),
            'vqa_challenges': len(self.vlm_engine.diagrams_list),
        }
        # 6 & 7. Load Stage 4 Distributed 2-LLM Pipeline:
        #   - Model #6: Pedagogical Reasoner (`DeepSeek-R1-Distill-Qwen-7B`, cuda:0)
        #   - Model #7: Persona Stylizer (`Meta-Llama-3.1-8B-Instruct` + QLoRA, auto 2x T4)
        self.llm_orchestrator = Stage4LLMOrchestrator(
            llm_mode=llm_mode,
            reasoner_model_id=reasoner_model_id,
            stylizer_model_id=stylizer_model_id,
        )
        self.llm_info = self.llm_orchestrator.get_info()
        self.boot_time_sec = round(time.perf_counter() - t0, 2)

    def create_session(self, profile: str = 'average', curriculum: list = None, course_id: str = None):
        course_meta = None
        if curriculum is None and course_id is not None:
            pkg = load_course_package(course_id)
            curriculum = pkg['concepts']
            course_meta = pkg
            if pkg.get('puzzles'):
                self.dino_evaluator.puzzles_list = pkg['puzzles']
                self.dino_evaluator.puzzles = {p['puzzle_id']: p for p in pkg['puzzles']}
                self.dino_evaluator._grounding_cache.clear()
            else:
                self.dino_evaluator._load_puzzles()
                self.dino_evaluator._grounding_cache.clear()

            if pkg.get('vlm_diagrams'):
                self.vlm_engine.diagrams_list = pkg['vlm_diagrams']
                self.vlm_engine.diagrams_by_id = {d['diagram_id']: d for d in pkg['vlm_diagrams']}
                self.vlm_engine.diagrams_by_concept = {int(d['concept_id']): d for d in pkg['vlm_diagrams']}
            else:
                self.vlm_engine._load_cached_extractions()

        sess = PlayerSession(
            self,
            profile=profile,
            curriculum=curriculum or DEFAULT_CURRICULUM,
            course_meta=course_meta,
        )
        sess.prefetch_next_turns(persona_key='wild')
        sess.prefetch_next_turns(persona_key='tr_rex')
        return sess

    def ingest_new_course(
        self,
        course_title: str,
        raw_notes: str = '',
        image_base64: str = None,
        file_name: str = None,
        file_base64: str = None,
        remote_vlm_url: str = None,
        profile: str = 'average',
        progress_cb=None,
    ):
        self.llm_orchestrator.set_vlm_busy(True)
        try:
            pkg = ingest_custom_course_package(
                course_title=course_title,
                raw_notes=raw_notes,
                image_base64=image_base64,
                file_name=file_name,
                file_base64=file_base64,
                remote_vlm_url=remote_vlm_url,
                vlm_engine=self.vlm_engine,
                dino_evaluator=self.dino_evaluator,
                progress_cb=progress_cb,
            )
        finally:
            self.llm_orchestrator.clear_caches()
            self.llm_orchestrator.set_vlm_busy(False)

        sess = self.create_session(profile=profile, course_id=pkg['course_id'])
        # Generate the custom humorous RPG Storyline + NPC script strictly from the uploaded PDF!
        if progress_cb:
            try:
                progress_cb(94, f"Stage 4 LLMs writing a custom RPG story & NPC dialogue for '{pkg['title']}'...")
            except Exception:
                pass
        try:
            sess.course_story = self.llm_orchestrator.get_or_generate_course_story(
                course_id=sess.course_id,
                course_title=sess.course_title,
                concepts=sess.curriculum,
                allow_blocking=True,
            )
        except Exception as story_exc:
            print(f"  [STAGE-4 STORY WARN] {story_exc}", flush=True)
        return sess, pkg


class PlayerSession:
    """
    Live per-student cognitive, affective, and RPG combat session orchestrated
    jointly by Final-DKT-GRU, Double-DQN, and DeBERTa-v3-Small.
    """

    def __init__(
        self,
        engine: KnowledgeQuestAIEngine,
        profile: str = 'average',
        curriculum: list = None,
        course_meta: dict = None,
    ):
        self.engine = engine
        self.profile = profile
        self.curriculum = curriculum or DEFAULT_CURRICULUM
        self.course_id = (course_meta or {}).get('course_id', 'dl_foundations')
        self.course_title = (course_meta or {}).get('title', 'Deep Learning & Neural Foundations')
        self.course_subtitle = (course_meta or {}).get(
            'subtitle', 'Activations, Gradient Descent, Backprop, Regularization & Attention'
        )
        self.course_mentor = (course_meta or {}).get('mentor', 'Prof. Ada (Willowbrook Research Lab)')
        self.active_encounter = None
        self.prereq_graph = {
            c['concept_id']: list(c.get('prereqs', PREREQ_GRAPH.get(c['concept_id'], [])))
            for c in self.curriculum
        }
        self._course_target_tensor = torch.tensor(
            [[s + 1 for s in engine.course_skill_ids]], dtype=torch.long
        )

        self.turn = 0
        self.win_streak = 0
        self.fail_streak = 0
        self.last_sec = 12.0
        self.last_retries = 1
        self.concentrating = 0.68
        self.frustrated = 0.32
        self.bored = 0.20

        self.player_hp = 100
        self.guardian_hp = 100
        self.xp = 0

        self.concept_attempts = [0] * 5
        self.concept_wins = [0] * 5
        self.concept_last_step = [-1] * 5
        self.consolidated = [False] * 5       # Earned [★] Concept Synthesis Badges!
        self.quest_completed = False
        self._variant_cursors = {}            # Per-(concept_id, tier) anti-repeat rotation
        self.asked_questions = []             # Recent questions passed to Model #6 Reasoner to prevent repeats
        self.npc_chat_histories = {}          # Per-NPC multi-turn Q&A history for humorous overworld chat
        self.course_story = None              # Live LLM-generated RPG story & NPC greetings for this course

        dkt = engine.dkt_model
        self.h_state = torch.zeros((dkt.num_layers, 1, dkt.hidden_dim), dtype=torch.float32)
        self.last_h_norm = torch.zeros((1, 1, dkt.hidden_dim), dtype=torch.float32)

        if profile == 'struggling':
            self.empirical_evidence = np.array([0.24, 0.24, 0.22, 0.22, 0.20], dtype=np.float32)
            self._step_dkt_hidden(engine.course_skill_ids[0], 0, 48.0, 3, 1.0, 1.0, 0.0, 0.62)
            self.fail_streak = 1
            self.frustrated = 0.48
        elif profile == 'advanced':
            self.empirical_evidence = np.array([0.62, 0.60, 0.56, 0.55, 0.52], dtype=np.float32)
            self._step_dkt_hidden(engine.course_skill_ids[0], 1, 9.5, 1, 1.0, 1.0, 0.0, 0.62)
            self.win_streak = 1
        else:
            self.empirical_evidence = np.array([0.40, 0.38, 0.36, 0.36, 0.34], dtype=np.float32)
            self._step_dkt_hidden(engine.course_skill_ids[0], 1, 14.0, 1, 1.0, 1.0, 0.0, 0.75)

        self.mastery = self._query_dkt_mastery()
        self.history = []

    def _pick_question_variant(self, c_data: dict, tier_key: str) -> dict:
        """
        Selects the next unseen question variant from `c_data[tier_key]` (supports
        both list-of-dicts pools and single-dict entries).
        """
        pool = c_data[tier_key]
        if isinstance(pool, dict):
            return pool
        cid = int(c_data['concept_id'])
        key = (cid, tier_key)
        idx = self._variant_cursors.get(key, 0)
        chosen = pool[idx % len(pool)]
        self._variant_cursors[key] = idx + 1
        return chosen

    @torch.no_grad()
    def _step_dkt_hidden(self, skill_id, correct, response_sec, attempts, is_main, is_mc, is_exam, item_diff):
        dkt = self.engine.dkt_model
        num_skills = dkt.num_skills
        inter_val = (skill_id + 1) if correct == 1 else (skill_id + num_skills + 1)
        inter_t = torch.tensor([[inter_val]], dtype=torch.long)

        telem = encode_telemetry(
            ms_first_response_time=max(200.0, float(response_sec) * 1000.0),
            attempt_count=max(1, int(attempts)),
            original=1 if is_main >= 0.5 else 0,
            answer_type='choose_1' if is_mc >= 0.5 else 'algebra',
            tutor_mode='test' if is_exam >= 0.5 else 'tutor',
        )
        telem_t = torch.tensor([[telem]], dtype=torch.float32)
        diff_t = torch.tensor([[[float(item_diff)]]], dtype=torch.float32)

        e_inter = dkt.inter_emb(inter_t)
        step_x = torch.cat([e_inter, telem_t, diff_t], dim=-1)
        step_h = dkt.input_proj(step_x)
        gru_out, self.h_state = dkt.gru(step_h, self.h_state)
        self.last_h_norm = dkt.state_norm(gru_out)

    @torch.no_grad()
    def _query_dkt_mastery(self) -> np.ndarray:
        dkt = self.engine.dkt_model
        h_rep = self.last_h_norm.expand(1, 5, dkt.hidden_dim)
        e_targets = dkt.skill_emb(self._course_target_tensor)

        ctx_rows = []
        for c_idx, s_id in enumerate(self.engine.course_skill_ids):
            s_prior = self.engine.skill_priors.get(s_id, self.engine.global_prior)
            p_att = self.concept_attempts[c_idx]
            p_win = self.concept_wins[c_idx]
            att_norm = math.log1p(min(p_att, 50)) / _LOG_MAX_SKILL_ATT
            win_rate = (p_win + 2.0 * s_prior) / (p_att + 2.0)

            if self.concept_last_step[c_idx] >= 0:
                steps_ago = max(0, self.turn - self.concept_last_step[c_idx])
                recency = 1.0 / (1.0 + steps_ago)
            else:
                recency = 0.0

            ctx_rows.append([0.60, s_prior, 1.0, 1.0, 0.0, att_norm, win_rate, recency, 0.0])

        ctx_t = torch.tensor([ctx_rows], dtype=torch.float32)
        joint = torch.cat([h_rep, e_targets, ctx_t], dim=-1)
        neural_probs = torch.sigmoid(dkt.readout(joint).squeeze(-1))[0].numpy()

        effective = np.zeros(5, dtype=np.float32)
        for c_idx in range(5):
            neglect = (self.turn - self.concept_last_step[c_idx]) if self.concept_last_step[c_idx] >= 0 else 0
            if self.consolidated[c_idx]:
                # Consolidated concepts (proven via Open-Text Synthesis) retain long-term memory!
                decay = 0.003 * max(0, neglect - 8)
                blended = 0.55 * float(neural_probs[c_idx]) + 0.45 * float(self.empirical_evidence[c_idx]) - decay
                effective[c_idx] = float(np.clip(blended, 0.76, 0.98))
            else:
                decay = 0.012 * max(0, neglect - 6)
                blended = 0.55 * float(neural_probs[c_idx]) + 0.45 * float(self.empirical_evidence[c_idx]) - decay
                effective[c_idx] = float(np.clip(blended, 0.08, 0.98))
        return effective

    def _select_frontier_concept(self) -> int:
        """
        Advances in topological order across un-consolidated concepts so earning a
        [★] Concept Badge permanently locks progress and prevents whack-a-mole loops.
        """
        for c_idx in range(5):
            if not self.consolidated[c_idx]:
                return c_idx
        return 4

    def get_rl_state_vector(self, focus_concept_id: int = None) -> np.ndarray:
        focus_idx = int(focus_concept_id) if focus_concept_id is not None else self._select_frontier_concept()
        focus_m = float(self.mastery[focus_idx])
        mean_m = float(np.mean(self.mastery))

        prereqs = self.prereq_graph.get(focus_idx, [])
        if prereqs:
            min_pre = float(min(self.mastery[p] for p in prereqs))
            pre_gap = max(0.0, 0.62 - min_pre)
        else:
            min_pre = 1.0
            pre_gap = 0.0

        neglects = [
            (self.turn - self.concept_last_step[i])
            for i in range(5) if self.concept_last_step[i] >= 0 and not self.consolidated[i]
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

    def _lookup_empirical_affect(self, correct: int, sec: float, hint_used: bool, retries: int):
        if sec < 5.0:
            z = 'rapid_<5s'
        elif sec <= 15.0:
            z = 'flow_5_15s'
        elif sec <= 60.0:
            z = 'deliberate_15_60s'
        else:
            z = 'struggle_>60s'

        key = f"c{int(correct)}_{z}_h{1 if hint_used else 0}_r{1 if retries >= 2 else 0}"
        cell = self.engine.calib_grid.get(key)
        if cell:
            return cell['concentrating'], cell['frustrated'], cell['bored']
        if correct == 1:
            return 0.68, 0.35, (0.36 if sec < 5.0 else 0.22)
        return 0.56, (0.52 if retries >= 2 else 0.44), 0.30

    def _build_prof_ada_spec(self, starter_name: str = 'your partner') -> dict:
        weakest_c = int(np.argmin(self.mastery))
        c_weak = self.curriculum[weakest_c]
        bullets = c_weak.get('summary_bullets') or []
        fact = bullets[0] if bullets else (c_weak.get('podcast_hint') or c_weak.get('title', ''))
        return {
            'course_id': self.course_id,
            'course_title': self.course_title,
            'weakest_concept_id': weakest_c,
            'weakest_topic': c_weak.get('short_title') or c_weak.get('title') or f'Concept {weakest_c}',
            'weakest_mastery': float(self.mastery[weakest_c]),
            'overall_mastery': float(np.mean(self.mastery)),
            'badges_earned': int(sum(self.consolidated)),
            'win_streak': int(self.win_streak),
            'frustration': float(self.frustrated),
            'verified_fact': str(fact),
            'starter_name': str(starter_name or 'your partner'),
        }

    def _select_rl_decision(self, focus_concept_id: int = None, force_action: int = None) -> tuple:
        focus_c = int(focus_concept_id) if focus_concept_id is not None else self._select_frontier_concept()
        state_vec = self.get_rl_state_vector(focus_concept_id=focus_c)

        rl_model = self.engine.rl_model
        s_t = torch.as_tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        if isinstance(rl_model, DuelingDoubleDQN):
            q_vals = rl_model(s_t)[0].numpy().tolist()
        else:
            logits, _ = rl_model(s_t)
            q_vals = torch.softmax(logits, dim=-1)[0].numpy().tolist()

        prereqs = self.prereq_graph.get(focus_c, [])
        has_weak_prereq = bool(prereqs) and min(self.mastery[p] for p in prereqs) < 0.68

        effective_q = list(q_vals)
        if not has_weak_prereq:
            effective_q[3] = -1e9
        chosen_action = int(np.argmax(effective_q))

        decision_reason = 'Double-DQN Optimal Q-Value'
        if force_action is None and has_weak_prereq and min(self.mastery[p] for p in prereqs) < 0.52:
            chosen_action = 3
            decision_reason = 'Prerequisite Gap (<52%) -> Redirected to Prereq Review!'
        elif (
            force_action is None
            and not self.consolidated[focus_c]
            and not has_weak_prereq
            and self.fail_streak == 0
        ):
            if self.mastery[focus_c] >= 0.68:
                chosen_action = 4
                decision_reason = 'Mastery >= 68% -> Unlocked Open-Text Badge Finisher!'
            elif self.bored >= 0.28 and self.mastery[focus_c] >= 0.62:
                chosen_action = 4
                decision_reason = 'High Boredom Detected -> Escalated to Open-Text Challenge!'

        if force_action is not None:
            chosen_action = int(force_action)
            decision_reason = 'Manual Override'

        if chosen_action == 3 and prereqs:
            target_c = min(prereqs, key=lambda p: self.mastery[p])
        else:
            target_c = focus_c
        return focus_c, target_c, chosen_action, decision_reason, q_vals, state_vec

    def prefetch_next_turns(self, persona_key: str = 'wild'):
        """Schedules asynchronous Stage 4 LLM prefetching so upcoming battles & Prof. Ada feel instant."""
        orch = getattr(self.engine, 'llm_orchestrator', None)
        if orch is None or not orch.is_live():
            return
        try:
            _, target_c, chosen_action, _, _, _ = self._select_rl_decision()
            c_data = self.curriculum[target_c]
            tier_key = (
                'boss_open' if chosen_action == 4 else (
                    'medium_mc' if (chosen_action == 2 or (chosen_action == 3 and self.mastery[target_c] >= 0.45)) else 'easy_mc'
                )
            )
            pool = c_data[tier_key]
            fb_item = pool if isinstance(pool, dict) else pool[0]
            spec = {
                'course_id': self.course_id,
                'course_title': self.course_title,
                'concept_id': target_c,
                'concept_data': c_data,
                'rl_action': chosen_action,
                'rl_action_name': ACTION_NAMES[chosen_action],
                'mastery': float(self.mastery[target_c]),
                'affect': {
                    'concentrating': float(self.concentrating),
                    'frustrated': float(self.frustrated),
                    'bored': float(self.bored),
                },
                'win_streak': int(self.win_streak),
                'fallback_item': fb_item,
                'persona_key': persona_key or 'wild',
                'asked_history': list(self.asked_questions[-6:]),
            }
            orch.schedule_course_story_prefetch(self.course_id, self.course_title, self.curriculum)
            orch.schedule_encounter_prefetch(spec)
            orch.schedule_prof_ada_prefetch(self._build_prof_ada_spec())
        except Exception:
            pass

    def get_course_story(self, allow_blocking: bool = False) -> dict:
        if self.course_story is not None and self.course_story.get('source', '').startswith('live'):
            return self.course_story
        orch = getattr(self.engine, 'llm_orchestrator', None)
        if orch is not None:
            story = orch.get_or_generate_course_story(
                course_id=self.course_id,
                course_title=self.course_title,
                concepts=self.curriculum,
                allow_blocking=allow_blocking,
            )
            if story.get('source', '').startswith('live') or self.course_story is None:
                self.course_story = story
            return self.course_story
        return self.course_story or {}

    def chat_with_npc(self, npc_key: str, student_question: str) -> dict:
        pk = str(npc_key or 'prof_ada')
        q_clean = str(student_question or '').strip()
        hist = self.npc_chat_histories.setdefault(pk, [])
        orch = getattr(self.engine, 'llm_orchestrator', None)
        if orch is not None:
            res = orch.chat_with_npc(
                npc_key=pk,
                student_question=q_clean,
                course_title=self.course_title,
                concepts=self.curriculum,
                mastery_vec=[float(v) for v in self.mastery],
                chat_history=hist,
            )
        else:
            res = {
                'npc': pk,
                'speaker': pk.upper(),
                'reply': f"{pk.upper()}: Check your Field Journal for notes on {self.course_title}!",
                'verified_fact': '',
                'matched_concept': self.course_title,
                'source': 'fallback',
                'model_name': 'Template',
                'latency_ms': 0.0,
            }
        hist.append({'role': 'user', 'content': q_clean})
        hist.append({'role': 'assistant', 'content': res['reply']})
        if len(hist) > 8:
            self.npc_chat_histories[pk] = hist[-8:]
        return res

    def get_npc_dialogue(self, npc_key: str = 'prof_ada', starter_name: str = 'your partner') -> dict:
        """Generates live DKT-driven retro-GBA dialogue for Prof. ADA or an overworld trainer."""
        orch = getattr(self.engine, 'llm_orchestrator', None)
        if npc_key == 'prof_ada' or not npc_key:
            ada_spec = self._build_prof_ada_spec(starter_name=starter_name)
            if orch is not None:
                return orch.get_or_generate_prof_ada(ada_spec, allow_blocking=True)
            return {
                'npc': 'prof_ada',
                'speaker': 'ADA',
                'lines': [
                    f"ADA: Your mastery on {self.course_title[:26]} is {round(float(np.mean(self.mastery))*100)}%.",
                    f"ADA: Focus next on {ada_spec['weakest_topic']} ({round(ada_spec['weakest_mastery']*100)}%).",
                ],
                'source': 'persona_template',
                'model_name': 'Template',
                'cache_hit': False,
            }
        # Trainer stylized pre-battle taunt
        trainer_c_map = {'tr_rex': 0, 'tr_cal': 1, 'tr_mira': 2, 'tr_rocco': 3, 'tr_mason': 4}
        cid = trainer_c_map.get(npc_key, self._select_frontier_concept())
        c_data = self.curriculum[cid]
        bullets = c_data.get('summary_bullets') or []
        fact = bullets[0] if bullets else c_data.get('title', '')
        if orch is not None:
            banter = orch.stylizer.generate_battle_banter(
                persona_key=npc_key,
                topic=c_data.get('short_title') or c_data.get('title') or f'C{cid}',
                verified_fact=fact,
                rl_action=4 if npc_key == 'tr_mason' else 2,
                rl_action_name=ACTION_NAMES[4 if npc_key == 'tr_mason' else 2],
                mastery=float(self.mastery[cid]),
                affect={'concentrating': self.concentrating, 'frustrated': self.frustrated, 'bored': self.bored},
                win_streak=self.win_streak,
            )
            return {
                'npc': npc_key,
                'speaker': banter['speaker'],
                'lines': [banter['intro_taunt']],
                'source': banter['source'],
                'model_name': banter['model_name'],
                'cache_hit': False,
            }
        return {'npc': npc_key, 'speaker': npc_key.upper(), 'lines': [], 'source': 'none', 'model_name': 'Template', 'cache_hit': False}

    @torch.no_grad()
    def plan_next_encounter(
        self,
        focus_concept_id: int = None,
        force_action: int = None,
        persona_key: str = 'wild',
    ) -> dict:
        """
        Runs Model #2 (`Double-DQN`) on the current 12-dim cognitive+affective state,
        applies valid-action & Concept-Badge Synthesis gating, and dispatches through
        Stage 4 (`Model #6 DeepSeek-R1-7B` + `Model #7 Llama-3.1-8B QLoRA`) with
        instant background prefetching and verified curriculum fallback.
        """
        t0 = time.perf_counter()
        focus_c, target_c, chosen_action, decision_reason, q_vals, state_vec = self._select_rl_decision(
            focus_concept_id=focus_concept_id, force_action=force_action
        )
        rl_ms = (time.perf_counter() - t0) * 1000.0

        hint_text = None
        hint_audio = None
        c_data = self.curriculum[target_c]
        if chosen_action == 0:
            hint_text = c_data['podcast_hint']
            hint_audio = c_data.get('podcast_audio')
            q_item = self._pick_question_variant(c_data, 'easy_mc')
            mode = 'scaffold_mc'
            item_diff, is_main, is_mc, is_exam = 0.86, 0.0, 1.0, 0.0
        elif chosen_action == 1:
            q_item = self._pick_question_variant(c_data, 'easy_mc')
            mode = 'easy_mc'
            item_diff, is_main, is_mc, is_exam = 0.78, 1.0, 1.0, 0.0
        elif chosen_action == 2:
            q_item = self._pick_question_variant(c_data, 'medium_mc')
            mode = 'medium_mc'
            item_diff, is_main, is_mc, is_exam = 0.62, 1.0, 1.0, 0.0
        elif chosen_action == 3:
            tier = 'medium_mc' if self.mastery[target_c] >= 0.45 else 'easy_mc'
            q_item = self._pick_question_variant(c_data, tier)
            mode = 'prereq_review_mc'
            item_diff, is_main, is_mc, is_exam = 0.68, 1.0, 1.0, 0.0
        elif chosen_action == 4:
            q_item = self._pick_question_variant(c_data, 'boss_open')
            mode = 'boss_open_text'
            item_diff, is_main, is_mc, is_exam = 0.35, 1.0, 0.0, 1.0
        else:
            raise ValueError(f"Unsupported RL action: {chosen_action}")

        # Stage 4 Multi-Agent LLM Pipeline (Model #6 Reasoner -> Model #7 Persona Stylizer)
        p_key = str(persona_key or 'wild')
        orch = getattr(self.engine, 'llm_orchestrator', None)
        llm_bundle = None
        if orch is not None:
            spec = {
                'course_id': self.course_id,
                'course_title': self.course_title,
                'concept_id': target_c,
                'concept_data': c_data,
                'rl_action': chosen_action,
                'rl_action_name': ACTION_NAMES[chosen_action],
                'mastery': float(self.mastery[target_c]),
                'affect': {
                    'concentrating': float(self.concentrating),
                    'frustrated': float(self.frustrated),
                    'bored': float(self.bored),
                },
                'win_streak': int(self.win_streak),
                'fallback_item': q_item,
                'persona_key': p_key,
                'asked_history': list(self.asked_questions[-6:]),
            }
            llm_bundle = orch.get_or_generate_encounter(spec, allow_blocking=True)

        r_out = llm_bundle['reasoner'] if llm_bundle else None
        s_out = llm_bundle['stylizer'] if llm_bundle else None

        final_q = r_out['question'] if r_out else q_item['question']
        final_choices = r_out['choices'] if r_out else q_item.get('choices')
        final_cidx = r_out['correct_idx'] if r_out else q_item.get('correct_idx')
        final_ref = r_out['reference_answer'] if r_out else q_item['reference']
        socratic_hint = r_out.get('socratic_hint') if r_out else None

        # Enrich available hint with Model #6's targeted Socratic hint (or keep course podcast hint)
        avail_hint = socratic_hint or c_data.get('podcast_hint')
        if chosen_action == 0 and socratic_hint:
            hint_text = socratic_hint

        self.asked_questions.append(final_q)
        if len(self.asked_questions) > 12:
            self.asked_questions = self.asked_questions[-12:]

        enc = {
            'turn': self.turn + 1,
            'rl_action': chosen_action,
            'rl_action_name': ACTION_NAMES[chosen_action],
            'decision_reason': decision_reason,
            'rl_q_values': [round(v, 3) for v in q_vals],
            'rl_latency_ms': round(rl_ms, 3),
            'state_vector': [round(float(x), 4) for x in state_vec],
            'focus_concept_id': focus_c,
            'target_concept_id': target_c,
            'target_concept_title': c_data['title'],
            'target_concept_short': c_data.get('short_title', c_data['title']),
            'target_mastery': round(float(self.mastery[target_c]), 4),
            'mode': mode,
            'podcast_hint': hint_text,
            'podcast_audio': hint_audio,
            'available_podcast_hint': avail_hint,
            'available_podcast_audio': c_data.get('podcast_audio'),
            'socratic_hint': socratic_hint,
            'question': final_q,
            'choices': final_choices,
            'correct_idx': final_cidx,
            'reference_answer': final_ref,
            'persona_key': p_key,
            'persona_banter': s_out,
            'llm_meta': {
                'reasoner_model': r_out['model_name'] if r_out else 'Curriculum Bank',
                'reasoner_source': r_out['source'] if r_out else 'curriculum_bank',
                'reasoner_ms': r_out['latency_ms'] if r_out else 0.0,
                'stylizer_model': s_out['model_name'] if s_out else 'Template',
                'stylizer_source': s_out['source'] if s_out else 'persona_template',
                'stylizer_ms': s_out['latency_ms'] if s_out else 0.0,
                'cache_hit': bool(llm_bundle.get('cache_hit')) if llm_bundle else False,
                'reasoning_trace': r_out.get('reasoning_trace', '') if r_out else '',
            },
            'telemetry_meta': {
                'item_diff': item_diff,
                'is_main': is_main,
                'is_mc': is_mc,
                'is_exam': is_exam,
            },
        }
        self.active_encounter = enc
        return enc

    def submit_turn_response(
        self,
        encounter: dict,
        user_response: str,
        response_sec: float,
        retries: int = 1,
        hint_used_override: bool = False,
    ) -> dict:
        """
        Grades the player's response (via MC selection OR Model #3 `DeBERTa-v3-Small`
        if open-text), updates Model #1 `Final-DKT-GRU` hidden state + 5-concept mastery,
        awards [★] Concept Synthesis Badges, and computes RPG combat damage.
        """
        self.turn += 1
        target_c = int(encounter['target_concept_id'])
        action = int(encounter['rl_action'])
        meta = encounter['telemetry_meta']
        hint_used = (encounter['podcast_hint'] is not None) or bool(hint_used_override)
        choices = encounter.get('choices')

        cleaned = str(user_response).strip()
        used_neural_grader = False
        grader_ms = 0.0

        if choices is not None and cleaned in ('1', '2', '3', '4'):
            picked_idx = int(cleaned) - 1
            is_right = picked_idx == int(encounter['correct_idx'])
            verdict = 'correct' if is_right else 'incorrect'
            binary_correct = 1 if is_right else 0
            sem_score = 1.0 if is_right else 0.0
            dmg_mult = 1.0 if is_right else 0.0
            grader_detail = {
                'graded_by': '4-Move Tactical Selector',
                'verdict': verdict,
                'semantic_score': sem_score,
                'prob_correct': 1.0 if is_right else 0.0,
                'prob_partial': 0.0,
                'prob_incorrect': 0.0 if is_right else 1.0,
            }
            is_mc_actual = 1.0
        else:
            t_g = time.perf_counter()
            g_res = grade_student_answer(
                self.engine.grader_model,
                self.engine.grader_tokenizer,
                reference_answer=encounter['reference_answer'],
                student_answer=cleaned if cleaned else 'no answer',
                question=encounter['question'],
                device=self.engine.grader_device,
            )
            grader_ms = (time.perf_counter() - t_g) * 1000.0
            used_neural_grader = True
            verdict = g_res['verdict']
            binary_correct = g_res['binary_correct']
            sem_score = g_res['semantic_score']
            dmg_mult = g_res['rpg_damage_multiplier']
            grader_detail = {
                'graded_by': self.engine.grader_info['name'],
                **g_res,
            }
            is_mc_actual = 0.0

        prev_mastery_vec = self.mastery.copy()
        c_mast_before = float(prev_mastery_vec[target_c])
        prereqs = self.prereq_graph.get(target_c, [])
        min_pre = min((float(prev_mastery_vec[p]) for p in prereqs), default=1.0)
        prereq_penalty = max(0.0, 0.60 - min_pre)

        if verdict == 'correct':
            base_gain = 0.18 if is_mc_actual < 0.5 else (0.08 if hint_used else 0.125)
            if is_mc_actual >= 0.5 and response_sec < 1.8:
                base_gain *= 0.35
            gain = base_gain * max(0.65, sem_score) * (1.0 - 0.40 * prereq_penalty)
            self.empirical_evidence[target_c] = min(0.97, self.empirical_evidence[target_c] + gain)
            self.win_streak += 1
            self.fail_streak = 0
        elif verdict == 'partial':
            gain = 0.075 * sem_score
            self.empirical_evidence[target_c] = min(0.88, self.empirical_evidence[target_c] + gain)
            self.win_streak = 0
            self.fail_streak = 0
        else:
            slip = 0.008 if hint_used else (0.030 if action == 4 else 0.018)
            self.empirical_evidence[target_c] = max(0.10, self.empirical_evidence[target_c] - slip)
            self.fail_streak += 1
            self.win_streak = 0

        self.concept_attempts[target_c] += 1
        self.concept_wins[target_c] += binary_correct
        self.concept_last_step[target_c] = self.turn
        self.last_sec = float(response_sec)
        self.last_retries = max(1, int(retries))

        t_dkt = time.perf_counter()
        dkt_signal = 1 if (binary_correct == 1 or (verdict == 'partial' and sem_score >= 0.62)) else 0
        self._step_dkt_hidden(
            skill_id=self.engine.course_skill_ids[target_c],
            correct=dkt_signal,
            response_sec=response_sec,
            attempts=retries,
            is_main=meta['is_main'],
            is_mc=is_mc_actual,
            is_exam=meta['is_exam'],
            item_diff=meta['item_diff'],
        )

        # Check if the student just earned the [★] Concept Synthesis Badge via Open-Text Grading!
        badge_just_earned = False
        if (
            not self.consolidated[target_c]
            and used_neural_grader
            and (verdict == 'correct' or (verdict == 'partial' and sem_score >= 0.68 and c_mast_before >= 0.72))
        ):
            self.consolidated[target_c] = True
            self.empirical_evidence[target_c] = max(self.empirical_evidence[target_c], 0.86)
            badge_just_earned = True

        self.mastery = self._query_dkt_mastery()
        self.quest_completed = all(self.consolidated)
        dkt_ms = (time.perf_counter() - t_dkt) * 1000.0

        # Update behavioral affect (100% webcam-free)
        emp_conc, emp_frust, emp_bored = self._lookup_empirical_affect(
            dkt_signal, response_sec, hint_used, retries
        )
        frust_delta, bored_delta, conc_delta = 0.0, 0.0, 0.0
        if hint_used:
            if self.fail_streak == 0 and c_mast_before > 0.65:
                bored_delta += 0.18
            else:
                frust_delta -= 0.16
                conc_delta += 0.08
        elif verdict == 'incorrect' and self.fail_streak >= 2:
            frust_delta += 0.09 * self.fail_streak
        elif verdict == 'correct':
            conc_delta += 0.08
            frust_delta -= 0.08
            if is_mc_actual < 0.5:
                # Open-text synthesis actively engages the learner and clears boredom!
                bored_delta -= 0.15
            elif response_sec < 3.0 and c_mast_before > 0.72:
                bored_delta += 0.14
            else:
                bored_delta -= 0.06

        self.concentrating = float(np.clip(0.55 * self.concentrating + 0.45 * emp_conc + conc_delta, 0.05, 0.98))
        self.frustrated = float(np.clip(0.55 * self.frustrated + 0.45 * emp_frust + frust_delta, 0.05, 0.98))
        self.bored = float(np.clip(0.55 * self.bored + 0.45 * emp_bored + bored_delta, 0.05, 0.98))

        # RPG Combat Damage & Guardian Stage Progression
        base_power = 50 if action == 4 else (28 if action in (2, 3) else 22)
        player_dmg = int(round(base_power * dmg_mult))
        enemy_dmg = 0 if verdict == 'correct' else (6 if verdict == 'partial' else (8 if hint_used else 16))
        xp_gained = int(round(player_dmg * 1.5)) + (75 if badge_just_earned else 0)

        if badge_just_earned:
            # Earning a Concept Badge defeats the current Concept Guardian outright!
            self.guardian_hp = 100
        else:
            # Without Open-Text Synthesis, Guardian endures at 1 HP ("Shielded — Requires Open-Text Finisher!")
            self.guardian_hp = max(1, self.guardian_hp - player_dmg)
        self.player_hp = max(1, self.player_hp - enemy_dmg)
        self.xp += xp_gained

        p_banter = encounter.get('persona_banter') or {}
        if verdict == 'correct':
            persona_reaction = p_banter.get('on_correct')
        elif verdict == 'partial':
            persona_reaction = p_banter.get('on_partial')
        else:
            persona_reaction = p_banter.get('on_incorrect')

        record = {
            'turn': self.turn,
            'rl_action': action,
            'rl_action_name': encounter['rl_action_name'],
            'decision_reason': encounter['decision_reason'],
            'target_concept_id': target_c,
            'target_concept_title': encounter['target_concept_title'],
            'used_neural_grader': used_neural_grader,
            'grader': grader_detail,
            'reference_answer': encounter['reference_answer'],
            'correct_idx': encounter.get('correct_idx'),
            'persona_reaction': persona_reaction,
            'persona_speaker': p_banter.get('speaker'),
            'llm_meta': encounter.get('llm_meta'),
            'badge_just_earned': badge_just_earned,
            'badges_earned_count': int(sum(self.consolidated)),
            'consolidated': list(self.consolidated),
            'quest_completed': self.quest_completed,
            'response_sec': round(float(response_sec), 2),
            'mastery_before': [round(float(v), 4) for v in prev_mastery_vec],
            'mastery_after': [round(float(v), 4) for v in self.mastery],
            'mean_mastery_after': round(float(np.mean(self.mastery)), 4),
            'affect': {
                'concentrating': round(self.concentrating, 4),
                'frustrated': round(self.frustrated, 4),
                'bored': round(self.bored, 4),
            },
            'rpg_combat': {
                'player_damage_dealt': player_dmg,
                'enemy_counter_damage': enemy_dmg,
                'xp_gained': xp_gained,
                'player_hp': self.player_hp,
                'guardian_hp': self.guardian_hp,
                'total_xp': self.xp,
            },
            'latencies_ms': {
                'rl_gamemaster_ms': encounter['rl_latency_ms'],
                'deberta_grader_ms': round(grader_ms, 2),
                'final_dkt_gru_ms': round(dkt_ms, 2),
                'total_ai_turn_ms': round(encounter['rl_latency_ms'] + grader_ms + dkt_ms, 2),
            },
        }
        self.history.append(record)
        self.active_encounter = None
        # Kick off non-blocking background Stage 4 prefetch for the NEXT turn!
        self.prefetch_next_turns(persona_key=encounter.get('persona_key', 'wild'))
        return record

    def get_diagram_puzzles(self, concept_id: int = None) -> list:
        raw_puzzles = self.engine.dino_evaluator.list_puzzles(concept_id=concept_id)
        enriched = []
        for p in raw_puzzles:
            item = dict(p)
            cid = int(item.get('concept_id', 0))
            vlm_d = self.engine.vlm_engine.diagrams_by_concept.get(cid, {})
            vqa = vlm_d.get('vqa_challenge', {})
            item['vlm_model'] = self.engine.vlm_engine.model_id
            item['vlm_visual_summary'] = vlm_d.get('visual_summary', item.get('explanation', ''))
            item['vlm_vqa_question'] = vqa.get('question', '')
            item['vlm_vqa_reference'] = vqa.get('reference_answer', '')
            enriched.append(item)
        return enriched

    def submit_diagram_interaction(
        self,
        puzzle_id: str,
        click_x: float = None,
        click_y: float = None,
        user_box: list = None,
        response_sec: float = 8.0,
    ) -> dict:
        """
        Evaluates a student's click or dragged bounding box on an in-game laboratory
        technical diagram via Grounding DINO and feeds the result into Final-DKT-GRU.
        """
        self.turn += 1
        d_res = self.engine.dino_evaluator.evaluate_student_interaction(
            puzzle_id=puzzle_id,
            click_x=click_x,
            click_y=click_y,
            user_box=user_box,
        )
        target_c = int(d_res['concept_id'])
        verdict = d_res['verdict']
        sem_score = float(d_res['semantic_score'])
        prev_mastery_vec = self.mastery.copy()

        if verdict == 'correct':
            gain = 0.14 * max(0.75, sem_score)
            base_ev = max(float(self.empirical_evidence[target_c]), float(prev_mastery_vec[target_c]))
            self.empirical_evidence[target_c] = min(0.96, base_ev + gain)
            self.win_streak += 1
            self.fail_streak = 0
            binary_correct = 1
        elif verdict == 'partial':
            gain = 0.06 * sem_score
            base_ev = max(float(self.empirical_evidence[target_c]), float(prev_mastery_vec[target_c]))
            self.empirical_evidence[target_c] = min(0.88, base_ev + gain)
            self.win_streak = 0
            self.fail_streak = 0
            binary_correct = 0
        else:
            self.empirical_evidence[target_c] = max(0.10, self.empirical_evidence[target_c] - 0.015)
            self.fail_streak += 1
            self.win_streak = 0
            binary_correct = 0

        self.concept_attempts[target_c] += 1
        self.concept_wins[target_c] += binary_correct
        self.concept_last_step[target_c] = self.turn

        t_dkt = time.perf_counter()
        self._step_dkt_hidden(
            skill_id=self.engine.course_skill_ids[target_c],
            correct=binary_correct,
            response_sec=float(response_sec),
            attempts=1,
            is_main=1.0,
            is_mc=0.0,
            is_exam=0.0,
            item_diff=0.52,
        )
        self.mastery = self._query_dkt_mastery()
        if verdict == 'correct':
            self.mastery[target_c] = min(
                0.96, max(float(self.mastery[target_c]), float(prev_mastery_vec[target_c]) + 0.08)
            )
        elif verdict == 'partial':
            self.mastery[target_c] = min(
                0.90, max(float(self.mastery[target_c]), float(prev_mastery_vec[target_c]) + 0.03)
            )
        dkt_ms = (time.perf_counter() - t_dkt) * 1000.0

        if verdict == 'correct':
            self.concentrating = float(np.clip(self.concentrating + 0.06, 0.05, 0.98))
            self.frustrated = float(np.clip(self.frustrated - 0.07, 0.05, 0.98))
            self.bored = float(np.clip(self.bored - 0.08, 0.05, 0.98))
        elif verdict == 'incorrect':
            self.frustrated = float(np.clip(self.frustrated + 0.06, 0.05, 0.98))

        xp_gained = 45 if verdict == 'correct' else (20 if verdict == 'partial' else 0)
        self.xp += xp_gained

        out = {
            **d_res,
            'turn': self.turn,
            'response_sec': round(float(response_sec), 2),
            'mastery_before': [round(float(v), 4) for v in prev_mastery_vec],
            'mastery_after': [round(float(v), 4) for v in self.mastery],
            'xp_gained': xp_gained,
            'affect': {
                'concentrating': round(self.concentrating, 4),
                'frustrated': round(self.frustrated, 4),
                'bored': round(self.bored, 4),
            },
            'latencies_ms': {
                'grounding_dino_ms': d_res['dino_latency_ms'],
                'final_dkt_gru_ms': round(dkt_ms, 2),
                'total_ai_turn_ms': round(d_res['dino_latency_ms'] + dkt_ms, 2),
            },
        }
        return out

    def get_vlm_diagram_insight(self, concept_id: int = None) -> dict:
        """Returns Qwen-VL-7B visual comprehension + VQA challenge for the target concept diagram."""
        target_c = int(concept_id) if concept_id is not None else self._select_frontier_concept()
        return self.engine.vlm_engine.comprehend_diagram(target_c)

    def submit_vlm_vqa_answer(
        self,
        concept_id: int,
        student_answer: str,
        response_sec: float = 11.5,
    ) -> dict:
        """
        Grades an open-text Visual Question Answering (VQA) explanation on a VLM-comprehended
        diagram using DeBERTa-v3-Small and updates Final-DKT-GRU mastery.
        """
        target_c = int(concept_id)
        prev_mastery_vec = list(self.mastery)
        vqa_res = self.engine.vlm_engine.evaluate_vqa_with_deberta(
            concept_id=target_c,
            student_answer=student_answer,
            grader_model=self.engine.grader_model,
            grader_tokenizer=self.engine.grader_tokenizer,
            grade_fn=grade_student_answer,
            device=self.engine.grader_device,
        )
        t_dkt = time.perf_counter()
        self._step_dkt_hidden(
            skill_id=self.engine.course_skill_ids[target_c],
            correct=int(vqa_res['is_correct']),
            response_sec=float(response_sec),
            attempts=1,
            is_main=1.0,
            is_mc=0.0,
            is_exam=1.0,
            item_diff=0.62,
        )
        self.mastery = self._query_dkt_mastery()
        if vqa_res['verdict'] == 'correct':
            self.mastery[target_c] = min(
                0.96, max(float(self.mastery[target_c]), float(prev_mastery_vec[target_c]) + 0.08)
            )
        dkt_ms = (time.perf_counter() - t_dkt) * 1000.0
        return {
            **vqa_res,
            'mastery_before': [round(float(v), 4) for v in prev_mastery_vec],
            'mastery_after': [round(float(v), 4) for v in self.mastery],
            'dkt_latency_ms': round(dkt_ms, 2),
        }

    def sync_progress(self, consolidated: list = None, mastery: list = None):
        if consolidated and len(consolidated) == 5:
            self.consolidated = [bool(x) for x in consolidated]
            for i, b in enumerate(self.consolidated):
                if b:
                    self.empirical_evidence[i] = max(self.empirical_evidence[i], 0.86)
        if mastery and len(mastery) == 5:
            for i, m in enumerate(mastery):
                self.empirical_evidence[i] = float(np.clip(m, 0.12, 0.96))
        self.mastery = self._query_dkt_mastery()
        self.quest_completed = all(self.consolidated)

    def get_status_snapshot(self) -> dict:
        concepts_out = []
        for c in self.curriculum:
            cid = int(c['concept_id'])
            concepts_out.append({
                'concept_id': cid,
                'title': c['title'],
                'short_title': c.get('short_title', c['title']),
                'prereqs': list(self.prereq_graph.get(cid, [])),
                'mastery': round(float(self.mastery[cid]), 4),
                'badge': bool(self.consolidated[cid]),
                'attempts': int(self.concept_attempts[cid]),
                'wins': int(self.concept_wins[cid]),
                'podcast_hint': c.get('podcast_hint', ''),
                'podcast_audio': c.get('podcast_audio'),
                'summary_bullets': c.get('summary_bullets', []),
            })
        return {
            'online': True,
            'course_id': self.course_id,
            'course_title': self.course_title,
            'course_subtitle': self.course_subtitle,
            'course_mentor': self.course_mentor,
            'course_story': self.get_course_story(allow_blocking=False),
            'available_courses': list_available_courses(),
            'profile': self.profile,
            'turn': self.turn,
            'frontier_concept_id': self._select_frontier_concept(),
            'mastery': [round(float(v), 4) for v in self.mastery],
            'mean_mastery': round(float(np.mean(self.mastery)), 4),
            'consolidated': list(self.consolidated),
            'badges_earned_count': int(sum(self.consolidated)),
            'quest_completed': self.quest_completed,
            'affect': {
                'concentrating': round(self.concentrating, 4),
                'frustrated': round(self.frustrated, 4),
                'bored': round(self.bored, 4),
            },
            'win_streak': self.win_streak,
            'fail_streak': self.fail_streak,
            'xp': self.xp,
            'concepts': concepts_out,
            'models': {
                'dkt': self.engine.dkt_info,
                'rl': self.engine.rl_info,
                'grader': self.engine.grader_info,
                'dino': self.engine.dino_info,
                'vlm': self.engine.vlm_info,
                'llm': self.engine.llm_info,
                'boot_time_sec': self.engine.boot_time_sec,
            },
            'last_turn': self.history[-1] if self.history else None,
        }
