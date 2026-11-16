import csv
import json
import math
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

TELEMETRY_DIM = 13
TELEMETRY_FEATURE_NAMES = [
    'log_response_time_norm',   # 0: log1p(sec) / log1p(300), clipped to [0, 300s]
    'rt_rapid_guess',           # 1: < 5s (rapid guessing zone)
    'rt_optimal_flow',          # 2: 5s - 15s (optimal cognitive window)
    'rt_deliberate',            # 3: 15s - 60s (deliberate problem solving)
    'rt_prolonged_struggle',    # 4: > 60s (extended struggle / hesitation)
    'log_attempt_count_norm',   # 5: log1p(min(attempts, 20)) / log1p(20)
    'att_zero_hint_first',      # 6: attempt_count == 0 (requested hint before trying)
    'att_first_try',            # 7: attempt_count == 1 (clean single submission)
    'att_multi_retry',          # 8: attempt_count >= 2 (multiple retries)
    'is_main_challenge',        # 9: original == 1 (main problem vs 0 = scaffold sub-step)
    'is_multiple_choice',       # 10: choose_1 / choose_n / rank (4-move RPG menu)
    'is_open_response',         # 11: algebra / fill_in_1 / open_response / external (typed)
    'is_unassisted_exam',       # 12: tutor_mode in {'test', 'pre', 'post'} (boss/exam mode)
]

_LOG_MAX_RT = math.log1p(300.0)
_LOG_MAX_ATT = math.log1p(20.0)
_MC_ANSWER_TYPES = {'choose_1', 'choose_n', 'rank'}
_EXAM_TUTOR_MODES = {'test', 'pre', 'post'}


def encode_telemetry(
    ms_first_response_time=20900.0,
    attempt_count=1,
    original=1,
    answer_type='choose_1',
    tutor_mode='tutor',
):
    """
    Converts behavioral interaction metadata from ASSISTments logs (or live KnowledgeQuest
    gameplay events) into a bounded 13-dimensional [0, 1] telemetry feature vector.
    """
    # 1. Response time latency (default 20.9s median if missing; clip negative glitches)
    try:
        rt_ms = float(ms_first_response_time) if str(ms_first_response_time).strip() != '' else 20900.0
    except (ValueError, TypeError):
        rt_ms = 20900.0

    sec = max(0.0, rt_ms / 1000.0)
    sec_clipped = min(sec, 300.0)
    log_rt_norm = math.log1p(sec_clipped) / _LOG_MAX_RT

    rt_rapid = 1.0 if sec < 5.0 else 0.0
    rt_optimal = 1.0 if (5.0 <= sec <= 15.0) else 0.0
    rt_deliberate = 1.0 if (15.0 < sec <= 60.0) else 0.0
    rt_struggle = 1.0 if sec > 60.0 else 0.0

    # 2. Attempt persistence (clip spam-bot outliers > 20)
    try:
        att = int(float(attempt_count)) if str(attempt_count).strip() != '' else 1
    except (ValueError, TypeError):
        att = 1
    att = max(0, att)
    log_att_norm = math.log1p(min(att, 20)) / _LOG_MAX_ATT

    att_zero = 1.0 if att == 0 else 0.0
    att_one = 1.0 if att == 1 else 0.0
    att_multi = 1.0 if att >= 2 else 0.0

    # 3. Main challenge vs. guided scaffold sub-question
    orig_str = str(original).strip().lower()
    is_main = 1.0 if orig_str in ('1', '1.0', 'true') else 0.0

    # 4. Question widget format (Multiple-choice menu vs. Free-text constructed response)
    ans_str = str(answer_type).strip().lower()
    is_mc = 1.0 if ans_str in _MC_ANSWER_TYPES else 0.0
    is_open = 0.0 if is_mc == 1.0 else 1.0

    # 5. Assessment mode (Unassisted test/boss exam vs. assisted practice)
    mode_str = str(tutor_mode).strip().lower()
    is_exam = 1.0 if mode_str in _EXAM_TUTOR_MODES else 0.0

    return [
        log_rt_norm,
        rt_rapid,
        rt_optimal,
        rt_deliberate,
        rt_struggle,
        log_att_norm,
        att_zero,
        att_one,
        att_multi,
        is_main,
        is_mc,
        is_open,
        is_exam,
    ]


def preprocess_assistments(csv_path, mapping_save_path=None, min_seq_len=3, max_seq_len=100):
    """
    Parses ASSISTments 2009-2010 interaction logs with full behavioral telemetry.
    Groups by student (user_id), sorts by order_id, chunks sequences to max_seq_len,
    and returns list of (skill_seq, correct_seq, telemetry_seq) tuples along with
    the skill mapping dictionary.
    """
    print(f"Reading {csv_path}...")
    user_interactions = {}
    unique_skills = set()
    skill_names = {}
    skill_totals = {}
    skill_corrects = {}

    with open(csv_path, 'r', encoding='latin1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_str = row.get('list_skill_ids', '').strip()
            c_str = row.get('correct', '').strip()
            u_id = row.get('user_id', '').strip()
            o_str = row.get('order_id', '').strip()

            # Filter out missing skills and fractional partial-credit anomalies
            if not s_str or c_str not in ('0', '1', '0.0', '1.0') or not u_id:
                continue

            # If multiple skills are semicolon-concatenated (e.g. "10;12"), extract primary skill
            primary_skill = s_str.split(';')[0].strip()
            if not primary_skill:
                continue

            s_name_raw = row.get('list_skills', '').strip()
            primary_name = s_name_raw.split(';')[0].strip() if s_name_raw else f"Skill_{primary_skill}"

            unique_skills.add(primary_skill)
            skill_names[primary_skill] = primary_name

            order = int(o_str) if o_str.isdigit() else 0
            correct = 1 if c_str in ('1', '1.0') else 0

            skill_totals[primary_skill] = skill_totals.get(primary_skill, 0) + 1
            skill_corrects[primary_skill] = skill_corrects.get(primary_skill, 0) + correct

            telemetry = encode_telemetry(
                ms_first_response_time=row.get('ms_first_response_time', ''),
                attempt_count=row.get('attempt_count', '1'),
                original=row.get('original', '1'),
                answer_type=row.get('answer_type', 'choose_1'),
                tutor_mode=row.get('tutor_mode', 'tutor'),
            )

            if u_id not in user_interactions:
                user_interactions[u_id] = []
            user_interactions[u_id].append((order, primary_skill, correct, telemetry))

    # Map raw skill IDs to clean contiguous 0-indexed integers
    sorted_skills = sorted(list(unique_skills))
    skill2idx = {sk: idx for idx, sk in enumerate(sorted_skills)}
    num_skills = len(skill2idx)
    print(f"Discovered {num_skills} unique skills across {len(user_interactions)} students.")

    # Compute empirical per-skill accuracy so runtime courses can align concept difficulty
    skill_metadata = {}
    for sk, idx in skill2idx.items():
        tot = skill_totals.get(sk, 1)
        acc = skill_corrects.get(sk, 0) / max(tot, 1)
        skill_metadata[str(idx)] = {
            'raw_skill_id': sk,
            'name': skill_names.get(sk, f"Skill_{sk}"),
            'attempts': tot,
            'empirical_accuracy': round(acc, 4),
        }

    if mapping_save_path:
        parent = os.path.dirname(mapping_save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(mapping_save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'num_skills': num_skills,
                'telemetry_dim': TELEMETRY_DIM,
                'telemetry_features': TELEMETRY_FEATURE_NAMES,
                'skill2idx': skill2idx,
                'skill_metadata': skill_metadata,
            }, f, indent=2)
        print(f"Saved skill & difficulty mapping to {mapping_save_path}")

    # Build chronological sequence chunks
    sequences = []
    for u_id, items in user_interactions.items():
        items.sort(key=lambda x: x[0])
        skills = [skill2idx[it[1]] for it in items]
        corrects = [it[2] for it in items]
        telemetries = [it[3] for it in items]

        n = len(skills)
        if n < min_seq_len:
            continue

        for start in range(0, n, max_seq_len):
            chunk_skills = skills[start:start + max_seq_len]
            chunk_corrects = corrects[start:start + max_seq_len]
            chunk_telemetries = telemetries[start:start + max_seq_len]
            if len(chunk_skills) >= min_seq_len:
                sequences.append((chunk_skills, chunk_corrects, chunk_telemetries))

    print(f"Extracted {len(sequences)} valid telemetry-enriched learning sequences.")
    return sequences, num_skills, skill2idx


class DKTDataset(Dataset):
    """
    PyTorch Dataset providing telemetry-enriched input vectors of size (2 * num_skills + TELEMETRY_DIM)
    at step t and target pairs (target_skill, target_correct) at step t+1.
    """
    def __init__(self, sequences, num_skills, telemetry_dim=TELEMETRY_DIM):
        self.sequences = sequences
        self.num_skills = num_skills
        self.telemetry_dim = telemetry_dim
        self.input_dim = 2 * num_skills + telemetry_dim

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        skills, corrects, telemetries = self.sequences[idx]

        # Step t (0 .. T-2) predicts future step t+1 (1 .. T-1) -> zero target leakage
        in_skills = skills[:-1]
        in_corrects = corrects[:-1]
        in_telemetries = telemetries[:-1]

        target_skills = skills[1:]
        target_corrects = corrects[1:]

        x = torch.zeros(len(in_skills), self.input_dim, dtype=torch.float32)
        base_telem_offset = 2 * self.num_skills

        for t, (s, c, telem) in enumerate(zip(in_skills, in_corrects, in_telemetries)):
            feat_idx = s if c == 1 else s + self.num_skills
            x[t, feat_idx] = 1.0
            if self.telemetry_dim > 0:
                x[t, base_telem_offset:] = torch.tensor(telem, dtype=torch.float32)

        target_skills = torch.tensor(target_skills, dtype=torch.long)
        target_corrects = torch.tensor(target_corrects, dtype=torch.float32)

        return x, target_skills, target_corrects


def collate_dkt_fn(batch):
    """
    Pads variable-length sequences to the max length in batch and generates active step masks.
    """
    inputs, target_skills, target_corrects = zip(*batch)

    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    padded_target_skills = pad_sequence(target_skills, batch_first=True, padding_value=-1)
    padded_target_corrects = pad_sequence(target_corrects, batch_first=True, padding_value=0.0)

    mask = (padded_target_skills != -1)

    return padded_inputs, padded_target_skills, padded_target_corrects, mask
