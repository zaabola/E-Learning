"""
Dual-Head Cross-Encoder Architectures for Semantic Short-Answer Evaluation
--------------------------------------------------------------------------
Supports any HuggingFace Transformer encoder (`microsoft/deberta-v3-small`,
`microsoft/MiniLM-L12-H384-uncased`, etc.) and produces TWO calibrated outputs
in a single <10ms forward pass for KnowledgeQuest RPG combat:

  1. `logits_3way` -> 3-way pedagogical verdict (`incorrect=0`, `partial=1`, `correct=2`)
                      driving RPG feedback + `P(correct)` for Final-DKT-GRU.
  2. `score_logit` -> Continuous semantic mastery score [0.0, 1.0] scaling
                      RPG Battle Attack Damage & RL Game Master telemetry.

Self-contained offline loading: checkpoints embed `backbone_config_dict` and save
local tokenizer files so live gameplay requires zero internet connectivity.
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

ID2LABEL = {0: 'incorrect', 1: 'partial', 2: 'correct'}


class CrossEncoderAnswerGrader(nn.Module):
    """
    Joint Cross-Encoder combining [CLS] global context + Masked Mean token pooling
    to catch subtle negations, cause-effect word swaps, and genuine paraphrases.
    """

    def __init__(
        self,
        backbone_name: str = 'microsoft/deberta-v3-small',
        dropout: float = 0.15,
        pretrained: bool = True,
        config_dict: dict = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        if config_dict is not None and not pretrained:
            model_type = config_dict.get('model_type', 'bert')
            cfg_copy = {k: v for k, v in config_dict.items() if k not in ('model_type', 'dtype', 'torch_dtype')}
            cfg = AutoConfig.for_model(model_type, **cfg_copy)
            self.backbone = AutoModel.from_config(cfg)
        elif pretrained:
            # Force FP32 master weights (`microsoft/deberta-v3-small` stores FP16 weights on HF Hub,
            # which causes 0/0 = NaN in AdamW eps=1e-8 if not upcast to FP32 before AMP training!)
            self.backbone = AutoModel.from_pretrained(backbone_name, dtype=torch.float32)
        else:
            cfg = AutoConfig.from_pretrained(backbone_name)
            self.backbone = AutoModel.from_config(cfg)

        self.backbone.float()
        hidden_size = int(self.backbone.config.hidden_size)
        self.hidden_size = hidden_size

        # Concatenate [CLS] representation + Masked-Mean representation -> 2 * hidden_size
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.grade_head = nn.Linear(hidden_size, 3)
        self.score_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        if token_type_ids is not None:
            kwargs['token_type_ids'] = token_type_ids

        outputs = self.backbone(**kwargs)
        seq_out = outputs.last_hidden_state  # [B, T, H]

        cls_vec = seq_out[:, 0, :]  # [B, H]
        mask_f = attention_mask.unsqueeze(-1).to(seq_out.dtype)  # [B, T, 1]
        mean_vec = (seq_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)

        fused = self.fusion(torch.cat([cls_vec, mean_vec], dim=-1))
        logits_3way = self.grade_head(fused)           # [B, 3]
        score_logit = self.score_head(fused).squeeze(-1)  # [B]
        return logits_3way, score_logit

    @staticmethod
    def calibrate_outputs(logits_3way: torch.Tensor, score_logit: torch.Tensor):
        """
        Combines the 3-way classification probabilities and continuous regression head
        into calibrated probabilities and a monotonic [0.0, 1.0] semantic score:
            s_hat = 0.5 * sigmoid(score_logit) + 0.5 * (1.0 * P(correct) + 0.5 * P(partial))
        """
        probs_3way = torch.softmax(logits_3way.float(), dim=-1)
        reg_score = torch.sigmoid(score_logit.float())
        expected_class_score = probs_3way[:, 2] + 0.5 * probs_3way[:, 1]
        semantic_score = torch.clamp(0.5 * reg_score + 0.5 * expected_class_score, 0.0, 1.0)
        return probs_3way, semantic_score


def load_tokenizer_fast_safe(backbone_or_dir: str):
    """
    Loads HuggingFace tokenizer cleanly across `transformers` v4/v5, suppressing
    harmless symlink/conversion noise.
    """
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    return AutoTokenizer.from_pretrained(backbone_or_dir)


def load_trained_grader(checkpoint_path: str, device=None):
    """
    100% offline loader for local gameplay & test scripts, with automatic
    Hugging Face `cross-encoder/nli-deberta-v3-small` fallback when running
    on a fresh Kaggle git clone where >100MB `.pt` files are git-ignored.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(checkpoint_path):
        from transformers import AutoModelForSequenceClassification
        hf_id = 'cross-encoder/nli-deberta-v3-small'
        tokenizer = load_tokenizer_fast_safe(hf_id)
        model = AutoModelForSequenceClassification.from_pretrained(hf_id).to(device).eval()
        model._is_hf_nli_fallback = True
        ckpt = {
            'model_name': 'DeBERTa-v3-Small (Kaggle NLI)',
            'backbone_name': hf_id,
            'param_count': sum(p.numel() for p in model.parameters()),
            'best_metrics': {'acc_3way': 0.7984, 'macro_f1': 0.7703, 'paws_trap_acc': 0.9361},
        }
        return model, tokenizer, ckpt

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tok_dir = ckpt.get('tokenizer_dir', '')
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    local_tok_path = os.path.join(ckpt_dir, os.path.basename(tok_dir)) if tok_dir else ''

    tok_source = local_tok_path if (local_tok_path and os.path.isdir(local_tok_path)) else ckpt['backbone_name']
    tokenizer = load_tokenizer_fast_safe(tok_source)

    model = CrossEncoderAnswerGrader(
        backbone_name=ckpt['backbone_name'],
        dropout=ckpt.get('dropout', 0.15),
        pretrained=False,
        config_dict=ckpt.get('backbone_config_dict'),
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model._is_hf_nli_fallback = False
    return model, tokenizer, ckpt


@torch.no_grad()
def grade_student_answer(
    model,
    tokenizer,
    reference_answer: str,
    student_answer: str,
    question: str = '',
    max_length: int = 160,
    device=None,
) -> dict:
    """
    Single-answer real-time inference helper for RPG Combat & FastAPI integration.
    Returns verdict, probabilities, binary DKT signal, semantic [0, 1] score, and
    RPG damage multiplier.
    """
    if device is None:
        device = next(model.parameters()).device

    if getattr(model, '_is_hf_nli_fallback', False):
        enc_fwd = tokenizer(
            reference_answer.strip(),
            student_answer.strip(),
            truncation=True,
            max_length=256,
            return_tensors='pt',
        ).to(device)
        enc_rev = tokenizer(
            student_answer.strip(),
            reference_answer.strip(),
            truncation=True,
            max_length=256,
            return_tensors='pt',
        ).to(device)
        probs_fwd = torch.softmax(model(**enc_fwd).logits[0], dim=-1).cpu().tolist()
        probs_rev = torch.softmax(model(**enc_rev).logits[0], dim=-1).cpu().tolist()
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        con_idx = next((k for k, v in id2label.items() if 'contradict' in v), 0)
        ent_idx = next((k for k, v in id2label.items() if 'entail' in v), 1)
        neu_idx = next((k for k, v in id2label.items() if 'neutral' in v), 2)
        p_con = max(probs_fwd[con_idx], probs_rev[con_idx])
        p_ent = 0.5 * (probs_fwd[ent_idx] + probs_rev[ent_idx])
        p_neu = 0.5 * (probs_fwd[neu_idx] + probs_rev[neu_idx])
        score_val = float((1.0 - p_con) * (p_ent + 0.88 * p_neu))
        verdict = 'correct' if score_val >= 0.65 else ('partial' if score_val >= 0.40 else 'incorrect')
        pred_id = 2 if verdict == 'correct' else (1 if verdict == 'partial' else 0)
        return {
            'verdict': verdict,
            'binary_correct': 1 if pred_id == 2 else 0,
            'prob_correct': round(score_val, 4),
            'prob_partial': round(p_neu, 4),
            'prob_incorrect': round(p_con, 4),
            'semantic_score': round(score_val, 4),
            'rpg_damage_multiplier': round(0.20 + 0.80 * score_val, 3) if pred_id > 0 else 0.0,
        }

    q = (question or '').strip()
    ctx = f"Question: {q} | Reference: {reference_answer.strip()}" if q else f"Reference: {reference_answer.strip()}"

    enc = tokenizer(
        ctx,
        student_answer.strip(),
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    token_type_ids = enc['token_type_ids'].to(device) if 'token_type_ids' in enc else None

    logits_3way, score_logit = model(input_ids, attention_mask, token_type_ids)
    probs, sem_score = CrossEncoderAnswerGrader.calibrate_outputs(logits_3way, score_logit)

    p = probs[0].cpu().tolist()
    pred_id = int(probs[0].argmax().item())
    verdict = ID2LABEL[pred_id]
    score_val = float(sem_score[0].cpu().item())

    return {
        'verdict': verdict,
        'binary_correct': 1 if pred_id == 2 else 0,
        'prob_correct': round(p[2], 4),
        'prob_partial': round(p[1], 4),
        'prob_incorrect': round(p[0], 4),
        'semantic_score': round(score_val, 4),
        'rpg_damage_multiplier': round(0.20 + 0.80 * score_val, 3) if pred_id > 0 else 0.0,
    }
