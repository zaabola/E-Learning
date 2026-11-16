"""
Multi-Domain Master Dataset Builder for Semantic Short-Answer Evaluation
------------------------------------------------------------------------
Downloads and unifies 5 peer-reviewed NLP & Educational Grading benchmarks
into `Models/AnswerEvaluator/data/master_answer_grading.csv` (~99,000 pairs):

  1. `SciEntsBank` (SemEval-2013 Student Response Analysis - Science ASAG)
  2. `Beetle`      (SemEval-2013 Interactive Tutor Student Responses)
  3. `MultiNLI`    (Multi-Genre Entailment, Partial/Neutral, & Contradiction)
  4. `Google PAWS` (Adversarial Word-Swap & Buzzword-Stuffing Traps)
  5. `STS-B`       (Continuous 0.0-1.0 Semantic Similarity Calibration)
"""

import csv
import os
import random

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from datasets import load_dataset


def clean_text(s):
    if s is None:
        return ''
    return ' '.join(str(s).replace('\r', ' ').replace('\n', ' ').split()).strip()


def map_semeval_5way(lbl_int):
    # ['correct', 'contradictory', 'partially_correct_incomplete', 'irrelevant', 'non_domain']
    if lbl_int == 0:
        return 'correct', 1, 1.0
    elif lbl_int == 2:
        return 'partial', 0, 0.5
    else:
        return 'incorrect', 0, 0.0


def build_master_dataset(output_csv_path, seed=42):
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    rows = []

    # 1. SciEntsBank (Real Classroom Student Short Answers)
    print("[1/5] Downloading `nkazi/SciEntsBank` (Classroom Science Short Answers)...")
    ds_sci = load_dataset('nkazi/SciEntsBank')
    for split_name in ds_sci.keys():
        part = 'train' if split_name == 'train' else 'test'
        for r in ds_sci[split_name]:
            q = clean_text(r.get('question', ''))
            ref = clean_text(r.get('reference_answer', ''))
            stu = clean_text(r.get('student_answer', ''))
            if not ref or not stu:
                continue
            g_lbl, b_corr, s_score = map_semeval_5way(int(r['label']))
            rows.append(('SciEntsBank_ASAG', part, q, ref, stu, g_lbl, b_corr, s_score))

    # 2. Beetle (Interactive Tutoring Dialogue Student Answers)
    print("[2/5] Downloading `nkazi/Beetle` (Interactive Tutor Student Answers)...")
    ds_bee = load_dataset('nkazi/Beetle')
    for split_name in ds_bee.keys():
        part = 'train' if split_name == 'train' else 'test'
        for r in ds_bee[split_name]:
            q = clean_text(r.get('question', ''))
            ref = clean_text(r.get('reference_answer', ''))
            stu = clean_text(r.get('student_answer', ''))
            if not ref or not stu:
                continue
            g_lbl, b_corr, s_score = map_semeval_5way(int(r['label']))
            rows.append(('Beetle_Tutor_ASAG', part, q, ref, stu, g_lbl, b_corr, s_score))

    # 3. Google PAWS (Adversarial Word-Order & Buzzword Traps)
    print("[3/5] Downloading `google-research-datasets/paws` (Adversarial Word-Order Traps)...")
    ds_paws = load_dataset('google-research-datasets/paws', 'labeled_final')
    paws_train = list(ds_paws['train']) + list(ds_paws['validation'])
    rng.shuffle(paws_train)
    for r in paws_train[:26_000]:
        ref = clean_text(r['sentence1'])
        stu = clean_text(r['sentence2'])
        if not ref or not stu:
            continue
        is_para = int(r['label']) == 1
        rows.append((
            'Google_PAWS_Adversarial', 'train', '',
            ref, stu,
            'correct' if is_para else 'incorrect',
            1 if is_para else 0,
            1.0 if is_para else 0.0,
        ))
    for r in ds_paws['test']:
        ref = clean_text(r['sentence1'])
        stu = clean_text(r['sentence2'])
        if not ref or not stu:
            continue
        is_para = int(r['label']) == 1
        rows.append((
            'Google_PAWS_Adversarial', 'test', '',
            ref, stu,
            'correct' if is_para else 'incorrect',
            1 if is_para else 0,
            1.0 if is_para else 0.0,
        ))

    # 4. MultiNLI (Multi-Genre Entailment, Neutral/Partial, & Contradiction)
    print("[4/5] Downloading `nyu-mll/glue` (`mnli` Academic & Multi-Domain NLI)...")
    ds_mnli = load_dataset('nyu-mll/glue', 'mnli')
    mnli_train_indices = list(range(len(ds_mnli['train'])))
    rng.shuffle(mnli_train_indices)
    for idx in mnli_train_indices[:35_000]:
        r = ds_mnli['train'][idx]
        lbl = int(r['label'])
        if lbl not in (0, 1, 2):
            continue
        ref = clean_text(r['premise'])
        stu = clean_text(r['hypothesis'])
        if not ref or not stu:
            continue
        # 0 = entailment, 1 = neutral (incomplete/partial), 2 = contradiction
        if lbl == 0:
            g_lbl, b_corr, s_score = 'correct', 1, 1.0
        elif lbl == 1:
            g_lbl, b_corr, s_score = 'partial', 0, 0.45
        else:
            g_lbl, b_corr, s_score = 'incorrect', 0, 0.0
        rows.append(('MultiNLI_MultiDomain', 'train', '', ref, stu, g_lbl, b_corr, s_score))

    mnli_val = list(ds_mnli['validation_matched']) + list(ds_mnli['validation_mismatched'])
    rng.shuffle(mnli_val)
    for r in mnli_val[:6_000]:
        lbl = int(r['label'])
        if lbl not in (0, 1, 2):
            continue
        ref = clean_text(r['premise'])
        stu = clean_text(r['hypothesis'])
        if not ref or not stu:
            continue
        if lbl == 0:
            g_lbl, b_corr, s_score = 'correct', 1, 1.0
        elif lbl == 1:
            g_lbl, b_corr, s_score = 'partial', 0, 0.45
        else:
            g_lbl, b_corr, s_score = 'incorrect', 0, 0.0
        rows.append(('MultiNLI_MultiDomain', 'test', '', ref, stu, g_lbl, b_corr, s_score))

    # 5. STS-Benchmark (Continuous 0.0-1.0 Semantic Similarity Calibration)
    print("[5/5] Downloading `sentence-transformers/stsb` (Graduated 0.0-1.0 Similarity)...")
    ds_sts = load_dataset('sentence-transformers/stsb')
    sample_max = max(float(r['score']) for r in ds_sts['train'])
    scale = 5.0 if sample_max > 1.05 else 1.0

    for split_name in ('train', 'validation', 'test'):
        part = 'test' if split_name == 'test' else 'train'
        for r in ds_sts[split_name]:
            ref = clean_text(r['sentence1'])
            stu = clean_text(r['sentence2'])
            if not ref or not stu:
                continue
            sc = max(0.0, min(1.0, float(r['score']) / scale))
            if sc >= 0.72:
                g_lbl, b_corr = 'correct', 1
            elif sc >= 0.38:
                g_lbl, b_corr = 'partial', 0
            else:
                g_lbl, b_corr = 'incorrect', 0
            rows.append(('STS_Benchmark', part, '', ref, stu, g_lbl, b_corr, round(sc, 4)))

    # Write unified CSV
    print(f"\nWriting {len(rows):,} unified pairs to {output_csv_path}...")
    fieldnames = [
        'pair_id', 'source', 'split', 'question',
        'reference_answer', 'student_answer',
        'grade_label', 'binary_correct', 'semantic_score',
    ]
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for idx, r in enumerate(rows, start=1):
            writer.writerow([idx, *r])

    size_mb = os.path.getsize(output_csv_path) / (1024 * 1024)
    print(f"[DONE] Saved {len(rows):,} pairs ({size_mb:.2f} MB) -> {output_csv_path}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_csv = os.path.join(base_dir, 'data', 'master_answer_grading.csv')
    build_master_dataset(out_csv)
