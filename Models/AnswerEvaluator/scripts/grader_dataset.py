"""
Dataset & Dynamic-Padding DataLoader Pipeline for Semantic Answer Evaluation
----------------------------------------------------------------------------
Loads `Models/AnswerEvaluator/data/master_answer_grading.csv` (99,631 pairs),
formats Question + Reference + Student Answer cross-encoder inputs, bulk-tokenizes
with Rust fast tokenizers, and pads each mini-batch dynamically to multiples of 8
for maximum Tensor-Core speed and minimal VRAM usage on 4 GB GPUs.
"""

import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

LABEL2ID = {'incorrect': 0, 'partial': 1, 'correct': 2}
ID2LABEL = {0: 'incorrect', 1: 'partial', 2: 'correct'}

SOURCE2ID = {
    'SciEntsBank_ASAG': 0,
    'Beetle_Tutor_ASAG': 1,
    'Google_PAWS_Adversarial': 2,
    'MultiNLI_MultiDomain': 3,
    'STS_Benchmark': 4,
}
ASAG_SOURCE_IDS = (0, 1)
PAWS_SOURCE_ID = 2


def format_reference_context(question: str, reference_answer: str) -> str:
    """
    Formats Segment A of the Cross-Encoder input `[CLS] Segment_A [SEP] Student_Answer [SEP]`.
    Includes the prompt question whenever available (classroom ASAG items).
    """
    q = (question or '').strip()
    ref = (reference_answer or '').strip()
    if q:
        return f"Question: {q} | Reference: {ref}"
    return f"Reference: {ref}"


def load_master_csv_rows(csv_path: str, asag_train_upsample: int = 2):
    """
    Reads `master_answer_grading.csv` safely (preserving literal strings like 'None'
    or 'NA' typed by real students without converting them to NaN).

    Args:
        csv_path: Path to `master_answer_grading.csv`.
        asag_train_upsample: Multiplier for real classroom student Q&A rows
            (`SciEntsBank` + `Beetle`) inside the TRAIN split ONLY. Test split is
            never modified or duplicated.
    """
    train_rows = []
    test_rows = []

    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            ref = r['reference_answer'].strip()
            stu = r['student_answer'].strip()
            if not ref or not stu:
                continue

            src = r['source'].strip()
            src_id = SOURCE2ID.get(src, -1)
            lbl_str = r['grade_label'].strip().lower()
            if lbl_str not in LABEL2ID:
                continue

            item = (
                format_reference_context(r['question'], ref),
                stu,
                ref,  # raw reference kept for keyword baseline
                LABEL2ID[lbl_str],
                int(r['binary_correct']),
                float(r['semantic_score']),
                src_id,
            )

            if r['split'].strip() == 'train':
                repeats = asag_train_upsample if src_id in ASAG_SOURCE_IDS else 1
                for _ in range(max(1, repeats)):
                    train_rows.append(item)
            else:
                test_rows.append(item)

    return train_rows, test_rows


class PretokenizedGraderDataset(Dataset):
    """
    Bulk-tokenizes `(context, student_answer)` pairs once in fast C++/Rust chunks
    so GPU epochs spend zero time waiting on Python string tokenization.
    """

    def __init__(self, rows, tokenizer, max_length: int = 160, chunk_size: int = 10_000, sort_by_length: bool = False):
        if sort_by_length:
            # Sorting held-out evaluation rows by character length minimizes batch padding waste
            rows = sorted(rows, key=lambda x: len(x[0]) + len(x[1]))

        n = len(rows)
        self.input_ids = []
        self.token_type_ids = []
        self.has_token_type_ids = False

        self.grade_labels = np.empty(n, dtype=np.int64)
        self.binary_labels = np.empty(n, dtype=np.int64)
        self.semantic_scores = np.empty(n, dtype=np.float32)
        self.source_ids = np.empty(n, dtype=np.int64)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            batch_slice = rows[start:end]
            texts_a = [x[0] for x in batch_slice]
            texts_b = [x[1] for x in batch_slice]

            enc = tokenizer(
                texts_a,
                texts_b,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
            )
            self.input_ids.extend(enc['input_ids'])
            if 'token_type_ids' in enc:
                self.has_token_type_ids = True
                self.token_type_ids.extend(enc['token_type_ids'])

            for offset, x in enumerate(batch_slice):
                idx = start + offset
                self.grade_labels[idx] = x[3]
                self.binary_labels[idx] = x[4]
                self.semantic_scores[idx] = x[5]
                self.source_ids[idx] = x[6]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        tt = self.token_type_ids[idx] if self.has_token_type_ids else None
        return (
            self.input_ids[idx],
            tt,
            self.grade_labels[idx],
            self.binary_labels[idx],
            self.semantic_scores[idx],
            self.source_ids[idx],
        )


class DynamicPadCollator:
    """
    Pads each batch dynamically to the longest sequence in that mini-batch
    (rounded up to a multiple of 8 for Ampere Tensor Core alignment).
    """

    def __init__(self, pad_token_id: int = 0, pad_to_multiple_of: int = 8):
        self.pad_token_id = int(pad_token_id)
        self.mult = int(pad_to_multiple_of)

    def __call__(self, batch):
        bsz = len(batch)
        max_len = max(len(x[0]) for x in batch)
        if self.mult > 1 and (max_len % self.mult) != 0:
            max_len = ((max_len // self.mult) + 1) * self.mult

        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        has_tt = batch[0][1] is not None
        token_type_ids = torch.zeros((bsz, max_len), dtype=torch.long) if has_tt else None

        grade_labels = torch.empty(bsz, dtype=torch.long)
        binary_labels = torch.empty(bsz, dtype=torch.long)
        semantic_scores = torch.empty(bsz, dtype=torch.float32)
        source_ids = torch.empty(bsz, dtype=torch.long)

        for i, (ids, tt, g_lbl, b_lbl, s_sc, src_id) in enumerate(batch):
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            if has_tt and tt is not None:
                token_type_ids[i, :seq_len] = torch.tensor(tt, dtype=torch.long)

            grade_labels[i] = int(g_lbl)
            binary_labels[i] = int(b_lbl)
            semantic_scores[i] = float(s_sc)
            source_ids[i] = int(src_id)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'grade_labels': grade_labels,
            'binary_labels': binary_labels,
            'semantic_scores': semantic_scores,
            'source_ids': source_ids,
        }


def build_grader_dataloaders(
    csv_path: str,
    tokenizer,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    max_length: int = 160,
    asag_train_upsample: int = 2,
    seed: int = 42,
):
    train_rows, test_rows = load_master_csv_rows(csv_path, asag_train_upsample=asag_train_upsample)

    print(f"Tokenizing {len(train_rows):,} train pairs & {len(test_rows):,} held-out test pairs (max_length={max_length})...")
    train_ds = PretokenizedGraderDataset(train_rows, tokenizer, max_length=max_length, sort_by_length=False)
    test_ds = PretokenizedGraderDataset(test_rows, tokenizer, max_length=max_length, sort_by_length=True)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collator = DynamicPadCollator(pad_token_id=pad_id, pad_to_multiple_of=8)

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=gen,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, train_rows, test_rows
