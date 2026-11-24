"""
Vision-Language Model (VLM) Course Slide & Diagram Comprehension Engine
(`Models/VLM/scripts/vlm_diagram_engine.py`)
=======================================================================
Integrates `Qwen/Qwen3-VL-8B-Instruct` (primary, with automatic fallback to
`Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen2-VL-7B-Instruct`) into the
E-Learning RPG multimodal pipeline:

  1. [Visual Slide & Diagram Comprehension]:
     Reads any uploaded course diagram or lecture figure and extracts:
       - Academic concept summary & verbatim diagram labels/formulas
       - Distinct visual components + natural-language `dino_prompt`s
       - Native VLM bounding boxes (`[0..1000]` normalized for Qwen3-VL;
         `image_grid_thw` absolute pixels for Qwen2.5-VL) -> canvas pixels
       - Interactive Lab Monitor investigation prompts & Socratic hints
       - Open-ended Visual Question Answering (VQA) battle challenges + reference rubrics

  2. [VLM -> Grounding DINO Dual-Vision Spatial Handoff]:
     Passes the VLM-generated `dino_prompt`s into `GroundingDINODiagramEvaluator`
     (`iSEE-Laboratory/llmdet_large` on Kaggle, falling back to
     `IDEA-Research/grounding-dino-base` / `grounding-dino-tiny`) and fuses the
     VLM grounding box with Grounding DINO's zero-shot detection box.

  3. [VLM -> DeBERTa-v3 Visual Question Answering (VQA) Handoff]:
     Uses the VLM-generated `reference_answer` rubric to score student open-text
     explanations about the diagram in <15 ms via `DeBERTa-v3-Small`.

Supports 3 execution modes:
  - `mode="live"`   : Loads the 8B/7B VLM weights directly on GPU (for Kaggle 2x T4 30GB VRAM).
  - `mode="remote"` : Queries a live Kaggle FastAPI/ngrok VLM worker over HTTP.
  - `mode="hybrid"` : Uses remote worker if online, or pre-extracted VLM artifacts from
                      `Models/VLM/data/vlm_extracted_diagrams.json` while running live
                      Grounding DINO + DeBERTa-v3 locally on a 4GB laptop GPU.
"""

import json
import os
import re
import time
import urllib.request
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VLM_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_ROOT = os.path.dirname(VLM_ROOT)
REPO_ROOT = os.path.dirname(MODELS_ROOT)

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
FALLBACK_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
LEGACY_FALLBACK_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
VLM_DATA_JSON = os.path.join(VLM_ROOT, "data", "vlm_extracted_diagrams.json")
VLM_CKPT_DIR = os.path.join(VLM_ROOT, "checkpoints", "qwen_vl_7b")

VLM_SYSTEM_PROMPT = (
    "You are an expert University Professor and Multimodal Curriculum Architect. "
    "Inspect ONLY the provided technical slide/diagram image and its slide text context. "
    "Write questions and summaries in the SAME language as the slide (French if the slide is in French, English if in English). "
    "CRITICAL: For each component's `target_label`, copy the EXACT verbatim text printed inside a visible box, cylinder, cloud, oval, or table cell in THIS image (e.g. 'DBWn', '$regex', 'Cache de tampons de la base de données', 'DBA') — NEVER translate printed labels and NEVER use placeholder words! "
    "Pick 2 visually DISTINCT components in the figure: prefer concise boxes, cylinders, clouds, commands, operators, or unique section headers rather than huge multi-line paragraphs or generic repeated headers when shorter distinct components exist. "
    "CRITICAL FOR `lab_question`: Write a REAL conceptual/quiz question asking the student to FIND and CLICK/SELECT that component in the figure based on its ROLE, PURPOSE, BEHAVIOR, or MEANING — STRICTLY WITHOUT writing, quoting, or spoiling `target_label` inside `lab_question`! "
    "Example GOOD `lab_question` (for target_label='$regex'): 'Quel opérateur MongoDB permet de rechercher un motif dans une chaîne de caractères ? Sélectionne-le sur le tableau !' "
    "Example GOOD `lab_question` (for target_label='SMON'): 'Which background process handles automatic instance recovery after a crash? Click or box it in the diagram!' "
    "Example BAD `lab_question` (FORBIDDEN!): 'Click or drag a box around SMON shown in this figure.' "
    "Keep every string concise (1 short sentence). Return ONLY a valid JSON object matching this schema:\n"
    "{\n"
    '  "is_educational_diagram": true,\n'
    '  "concept_title": "Short exact concept title from this diagram",\n'
    '  "visual_summary": "1-2 concise sentences explaining what this diagram shows",\n'
    '  "summary_bullets": ["Key fact 1", "Key fact 2", "Key fact 3"],\n'
    '  "components": [\n'
    "    {\n"
    '      "target_label": "EXACT verbatim printed text of component 1 in image",\n'
    '      "dino_prompt": "short English visual description of component 1 container ending with a period.",\n'
    '      "lab_question": "1 real quiz question asking the student to find component 1 by its role/meaning WITHOUT naming target_label!",\n'
    '      "explanation": "1 concise sentence explaining what component 1 does"\n'
    "    },\n"
    "    {\n"
    '      "target_label": "EXACT verbatim printed text of component 2 in image",\n'
    '      "dino_prompt": "short English visual description of component 2 container ending with a period.",\n'
    '      "lab_question": "1 real quiz question asking the student to find component 2 by its role/meaning WITHOUT naming target_label!",\n'
    '      "explanation": "1 concise sentence explaining what component 2 does"\n'
    "    }\n"
    "  ],\n"
    '  "mc_questions": [\n'
    "    {\n"
    '      "question": "Specific multiple-choice question about THIS diagram",\n'
    '      "choices": ["Correct answer", "Wrong option 1", "Wrong option 2", "Wrong option 3"],\n'
    '      "correct_idx": 0\n'
    "    }\n"
    "  ],\n"
    '  "vqa_challenge": {\n'
    '    "question": "1 crisp analytical question strictly about THIS diagram",\n'
    '    "reference_answer": "1 concise 15-25 word ground-truth answer"\n'
    "  }\n"
    "}"
)


def _question_leaks_label(question: str, target_label: str) -> bool:
    """Returns True if `question` spoils `target_label` verbatim or is a placeholder."""
    q_low = re.sub(r"\s+", " ", str(question or "")).strip().lower()
    t_low = re.sub(r"\s+", " ", str(target_label or "")).strip().lower()
    if not q_low or len(q_low) < 12 or not t_low:
        return True
    if "[target_label]" in q_low or "target_label" in q_low:
        return True
    if q_low.startswith("click or drag a box around"):
        return True
    # Exact token-bounded match for short/medium labels (e.g. '$regex', 'SMON', 'SQL', 'mongod')
    if len(t_low) >= 3:
        pat = (r"(?<!\w)" if t_low[0].isalnum() else "") + re.escape(t_low) + (r"(?!\w)" if t_low[-1].isalnum() else "")
        if re.search(pat, q_low):
            return True
    # Heavy word-overlap check for long multi-word labels (prevents verbatim copy-paste giveaways)
    stop = {"the", "and", "for", "with", "that", "from", "les", "des", "une", "dans", "pour", "avec", "sur", "par", "est", "qui", "que"}
    t_words = [w for w in re.findall(r"[0-9a-zA-ZÀ-ÿ$_.()/-]{3,}", t_low) if w not in stop]
    if len(t_words) >= 4:
        hits = sum(1 for w in t_words if w in q_low)
        if hits / float(len(t_words)) >= 0.78:
            return True
    return False


def _build_challenge_question(
    raw_q: str,
    target_label: str,
    explanation: str = "",
    concept_title: str = "",
) -> str:
    """
    Returns a real conceptual visual-search question that asks the student to locate
    the right component in the figure WITHOUT giving away `target_label` verbatim.
    """
    cand = re.sub(r"\s+", " ", str(raw_q or "")).strip()
    if cand and not _question_leaks_label(cand, target_label):
        if cand[-1] not in "?!.":
            cand += "?"
        return cand

    # Fallback: build a clean non-spoiler question from `explanation` in the slide's language
    exp = re.sub(r"\s+", " ", str(explanation or "")).strip()
    is_fr = bool(
        re.search(r"[éèêëàâçùûîïôœ]", exp + " " + str(concept_title or ""), re.IGNORECASE)
        or re.search(r"\b(le|la|les|des|une|dans|pour|permet|base de données|serveur)\b", exp, re.IGNORECASE)
    )
    t_clean = re.sub(r"\s+", " ", str(target_label or "")).strip()
    masked_exp = exp
    if t_clean and len(t_clean) >= 3 and masked_exp:
        pat = (r"(?<!\w)" if t_clean[0].isalnum() else "") + re.escape(t_clean) + (r"(?!\w)" if t_clean[-1].isalnum() else "")
        replacement = "cet élément" if is_fr else "this component"
        masked_exp = re.sub(pat, replacement, masked_exp, flags=re.IGNORECASE).strip()
        masked_exp = re.sub(r"^['\"`\[]*cet élément['\"`\]]*\s*", "", masked_exp, flags=re.IGNORECASE).strip()
        masked_exp = re.sub(r"^['\"`\[]*this component['\"`\]]*\s*", "", masked_exp, flags=re.IGNORECASE).strip()

    if masked_exp and len(masked_exp) >= 12 and not _question_leaks_label(masked_exp, t_clean):
        masked_exp = masked_exp.rstrip(".!?")
        if is_fr:
            return f"Quel élément de cette figure correspond à ce rôle : {masked_exp} ? Clique ou encadre-le !"
        return f"Which component in this figure serves this role: {masked_exp}? Click or drag a box around it!"

    if is_fr:
        return f"Identifie sur cette figure le composant clé lié à '{concept_title or 'ce concept'}' (indice : {len(t_clean)} caractères). Clique ou encadre-le !"
    return f"Locate the key component related to '{concept_title or 'this concept'}' in this figure. Click or drag a box around it!"


def box_1000_to_pixel(box_1000, width: int, height: int) -> list:
    """Converts Qwen-VL [0..1000] normalized coordinates into [x1, y1, x2, y2] pixel coordinates."""
    x1, y1, x2, y2 = [float(v) for v in box_1000]
    px1 = round(max(0.0, min(float(width), (x1 / 1000.0) * width)), 1)
    py1 = round(max(0.0, min(float(height), (y1 / 1000.0) * height)), 1)
    px2 = round(max(0.0, min(float(width), (x2 / 1000.0) * width)), 1)
    py2 = round(max(0.0, min(float(height), (y2 / 1000.0) * height)), 1)
    return [px1, py1, px2, py2]


def compute_iou(box_a, box_b) -> float:
    """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(min(ax1, ax2), min(bx1, bx2)), max(min(ay1, ay2), min(by1, by2))
    ix2, iy2 = min(max(ax1, ax2), max(bx1, bx2)), min(max(ay1, ay2), max(by1, by2))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = abs(ax2 - ax1) * abs(ay2 - ay1)
    area_b = abs(bx2 - bx1) * abs(by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-6 else 0.0


def box_content_stats(pil_img: Image.Image, box: list) -> dict:
    """
    Measures foreground ink/shape density inside `box` on a 720x420 diagram so empty
    white margins (or 90% blank boxes barely clipping a border) are reliably detected,
    while preserving genuine white table cells containing short words like 'DBA'.
    """
    import numpy as np

    W, H = pil_img.size
    x1 = max(0, min(W - 2, int(round(float(box[0])))))
    y1 = max(0, min(H - 2, int(round(float(box[1])))))
    x2 = max(x1 + 2, min(W, int(round(float(box[2])))))
    y2 = max(y1 + 2, min(H, int(round(float(box[3])))))
    sub = pil_img.convert("RGB").crop((x1, y1, x2, y2))
    arr = np.asarray(sub, dtype=np.float32)
    gray = arr.mean(axis=2)
    bg_dark = (
        np.abs(arr[:, :, 0] - 14.0)
        + np.abs(arr[:, :, 1] - 22.0)
        + np.abs(arr[:, :, 2] - 36.0)
    ) <= 24.0
    bg_white = (arr[:, :, 0] >= 244.0) & (arr[:, :, 1] >= 244.0) & (arr[:, :, 2] >= 240.0)
    fg = (~bg_dark) & (~bg_white)

    in_fg = fg[2:-2, 2:-2] if (fg.shape[0] > 6 and fg.shape[1] > 6) else fg
    col_has_fg = in_fg.mean(axis=0) > 0.03
    row_has_fg = in_fg.mean(axis=1) > 0.03
    active_w = int(col_has_fg.sum())
    active_h = int(row_has_fg.sum())
    gx = float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.0
    gy = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0.0
    edge = gx + gy
    fg_ratio = float(in_fg.mean()) if in_fg.size > 0 else 0.0
    is_blank = (fg_ratio < 0.025 and edge < 4.0) or (active_w < 12) or (active_h < 6)
    return {
        "is_blank": bool(is_blank),
        "fg_ratio": round(fg_ratio, 4),
        "edge": round(edge, 2),
        "active_w": active_w,
        "active_h": active_h,
    }


def trim_blank_margins(pil_img: Image.Image, box: list) -> list:
    """
    Trims pure empty white/letterbox margin strips off the outer edges of `box` while
    preserving bordered table cells and leaving a clean 4px visual margin.
    """
    import numpy as np

    W, H = pil_img.size
    x1 = max(0, min(W - 4, int(round(float(box[0])))))
    y1 = max(0, min(H - 4, int(round(float(box[1])))))
    x2 = max(x1 + 4, min(W, int(round(float(box[2])))))
    y2 = max(y1 + 4, min(H, int(round(float(box[3])))))
    sub = pil_img.convert("RGB").crop((x1, y1, x2, y2))
    arr = np.asarray(sub, dtype=np.float32)
    bg_dark = (
        np.abs(arr[:, :, 0] - 14.0)
        + np.abs(arr[:, :, 1] - 22.0)
        + np.abs(arr[:, :, 2] - 36.0)
    ) <= 24.0
    bg_white = (arr[:, :, 0] >= 244.0) & (arr[:, :, 1] >= 244.0) & (arr[:, :, 2] >= 240.0)
    fg = (~bg_dark) & (~bg_white)
    ys, xs = np.where(fg)
    if len(xs) < 25 or len(ys) < 25:
        return [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)]
    nx1 = max(x1, x1 + int(xs.min()) - 4)
    ny1 = max(y1, y1 + int(ys.min()) - 4)
    nx2 = min(x2, x1 + int(xs.max()) + 5)
    ny2 = min(y2, y1 + int(ys.max()) + 5)
    if (nx2 - nx1) >= 24 and (ny2 - ny1) >= 16:
        return [round(float(nx1), 1), round(float(ny1), 1), round(float(nx2), 1), round(float(ny2), 1)]
    return [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)]


def _repair_truncated_json(text: str) -> str:
    """Repairs trailing commas and unclosed strings/brackets/braces in a truncated JSON string."""
    s = re.sub(r",\s*([}\]])", r"\1", text.strip())
    in_str = False
    esc = False
    stack = []
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in ("}", "]") and stack and stack[-1] == ch:
                stack.pop()
    if in_str:
        s += '"'
    s = re.sub(r",\s*$", "", s.rstrip())
    if stack:
        s += "".join(reversed(stack))
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def extract_json_payload(raw_text: str) -> dict:
    """
    Extracts and parses a JSON object from raw VLM output, automatically recovering
    from markdown fences, trailing commas, or slightly truncated responses.
    """
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*)", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    else:
        start = cleaned.find("{")
        if start != -1:
            cleaned = cleaned[start:]

    end = cleaned.rfind("}")
    if end != -1:
        candidate = cleaned[: end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
            except Exception:
                pass

    # Attempt structural repair if truncated mid-object
    try:
        return json.loads(_repair_truncated_json(cleaned))
    except Exception:
        pass

    # Final resilient regex recovery for key fields so labels & VQA are never lost
    recovered = {}
    m_edu = re.search(r'"is_educational_diagram"\s*:\s*(true|false)', cleaned, flags=re.IGNORECASE)
    if m_edu:
        recovered["is_educational_diagram"] = m_edu.group(1).lower() == "true"
    m_title = re.search(r'"concept_title"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', cleaned)
    if m_title:
        recovered["concept_title"] = m_title.group(1)
    m_sum = re.search(r'"visual_summary"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', cleaned)
    if m_sum:
        recovered["visual_summary"] = m_sum.group(1)

    labels = re.findall(r'"target_label"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', cleaned)
    prompts = re.findall(r'"dino_prompt"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', cleaned)
    expls = re.findall(r'"explanation"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', cleaned)
    if labels:
        recovered["components"] = [
            {
                "target_label": labels[i],
                "dino_prompt": prompts[i] if i < len(prompts) else f"{labels[i].lower()} diagram box.",
                "explanation": expls[i] if i < len(expls) else recovered.get("visual_summary", ""),
            }
            for i in range(min(2, len(labels)))
        ]
    vqa_m = re.search(r'"vqa_challenge"\s*:\s*\{([^}]*)', cleaned, flags=re.DOTALL)
    if vqa_m:
        vq = re.search(r'"question"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', vqa_m.group(1))
        vr = re.search(r'"reference_answer"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', vqa_m.group(1))
        if vq:
            recovered["vqa_challenge"] = {
                "question": vq.group(1),
                "reference_answer": vr.group(1) if vr else recovered.get("visual_summary", ""),
            }
    if recovered:
        return recovered
    raise ValueError("Unable to parse JSON payload from VLM output")


class QwenVLDiagramEngine:
    """
    Multimodal VLM Slide & Diagram Comprehension Engine powered by
    `Qwen/Qwen3-VL-8B-Instruct` (with automatic fallback to
    `Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen2-VL-7B-Instruct`).
    """

    @staticmethod
    def _short_name_for(model_id: str) -> str:
        low = (model_id or "").lower()
        if "qwen3" in low:
            return "Qwen3-VL-8B"
        if "2.5" in low or "2_5" in low:
            return "Qwen2.5-VL-7B"
        return "Qwen2-VL-7B"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        mode: str = "hybrid",
        load_in_4bit: bool = True,
        remote_url: str = None,
        device: str = None,
    ):
        self.model_id = model_id
        self.short_model_name = self._short_name_for(model_id)
        self.mode = mode.lower()
        self.load_in_4bit = load_in_4bit
        self.remote_url = (remote_url or os.environ.get("KAGGLE_VLM_URL", "")).rstrip("/")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vlm_model = None
        self.vlm_processor = None
        self.num_params = 0

        self.diagrams_by_id = {}
        self.diagrams_by_concept = {}
        self.diagrams_list = []
        self._load_cached_extractions()

        if self.mode == "live":
            self.load_live_vlm()

    def _load_cached_extractions(self):
        if not os.path.exists(VLM_DATA_JSON):
            return
        with open(VLM_DATA_JSON, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.diagrams_list = payload.get("diagrams", [])
        self.diagrams_by_id = {d["diagram_id"]: d for d in self.diagrams_list}
        self.diagrams_by_concept = {int(d["concept_id"]): d for d in self.diagrams_list}

    @staticmethod
    def _resolve_model_class(source: str):
        low = (source or "").lower()
        if "qwen3" in low:
            try:
                from transformers import Qwen3VLForConditionalGeneration
                return Qwen3VLForConditionalGeneration
            except ImportError:
                pass
        if "2.5" in low or "2_5" in low:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                return Qwen2_5_VLForConditionalGeneration
            except ImportError:
                pass
        if "qwen2-vl" in low or "qwen2_vl" in low:
            try:
                from transformers import Qwen2VLForConditionalGeneration
                return Qwen2VLForConditionalGeneration
            except ImportError:
                pass
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq

    def load_live_vlm(self):
        """
        Loads `Qwen/Qwen3-VL-8B-Instruct` onto GPU with optimal multi-GPU placement:
          - On Kaggle 2x Tesla T4 (15 GB per GPU): loads 4-bit NF4 (~6.2 GB) pinned
            exclusively to dedicated `cuda:1` (while LLMDet-Large + DeBERTa sit on `cuda:0`)
            for zero CPU offload, zero cross-GPU PCIe hops, and ~2.5x faster inference!
          - On >=22 GB single GPU (A100/L4/3090): loads full FP16 on 1 GPU.
          - Automatic fallback chain: 4-bit NF4 <-> balanced FP16 -> Qwen2.5-VL-7B -> Qwen2-VL-7B.
        """
        if self.vlm_model is not None:
            return

        from transformers import AutoProcessor, BitsAndBytesConfig

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        total_vram_gb = 0.0
        single_gpu_vram_gb = 0.0
        if num_gpus > 0:
            vrams = [torch.cuda.get_device_properties(i).total_memory / (1024 ** 3) for i in range(num_gpus)]
            total_vram_gb = sum(vrams)
            single_gpu_vram_gb = min(vrams)

        candidates = []
        if os.path.exists(VLM_CKPT_DIR):
            candidates.append(VLM_CKPT_DIR)
        for mid in (self.model_id, FALLBACK_MODEL_ID, LEGACY_FALLBACK_MODEL_ID):
            if mid and mid not in candidates:
                candidates.append(mid)

        last_err = None
        for source in candidates:
            # On 2x T4 (15 GB/GPU), keep the Vision Tower (`visual` + `merger`) in FULL FP16
            # while quantizing ONLY the LLM backbone to 4-bit NF4 on dedicated cuda:1 (~6.8 GB)!
            # Includes an automatic boot smoke-test fallback to standard 4-bit -> balanced FP16.
            if single_gpu_vram_gb >= 22.0:
                precision_plans = ["fp16", "4bit_fp16_vision", "4bit_standard"]
            elif num_gpus >= 2:
                precision_plans = ["4bit_fp16_vision", "4bit_standard", "fp16"]
            else:
                precision_plans = ["4bit_fp16_vision", "4bit_standard"] if self.load_in_4bit else ["fp16"]

            for plan in precision_plans:
                try_4bit = plan.startswith("4bit")
                skip_vision_quant = plan == "4bit_fp16_vision"
                if not torch.cuda.is_available():
                    dev_map = None
                    dev_tag = "cpu"
                elif try_4bit and num_gpus >= 2:
                    dev_map = {"": 1}  # Dedicated GPU 1 (keeps GPU 0 free for LLMDet-Large + DeBERTa!)
                    dev_tag = "cuda:1 (dedicated)"
                elif try_4bit:
                    dev_map = {"": 0}
                    dev_tag = "cuda:0"
                elif num_gpus >= 2:
                    dev_map = "balanced"
                    dev_tag = "balanced (2x GPU)"
                else:
                    dev_map = "auto"
                    dev_tag = "auto"

                precision_tag = (
                    "4-bit NF4 + FP16 Vision"
                    if skip_vision_quant
                    else ("4-bit NF4" if try_4bit else "FULL FP16")
                )
                print(
                    f"[VLM] Loading {source} ({precision_tag}, device={dev_tag}, total_vram={total_vram_gb:.1f}GB)...",
                    flush=True,
                )
                try:
                    quant_config = None
                    if try_4bit and torch.cuda.is_available():
                        bnb_kwargs = {
                            "load_in_4bit": True,
                            "bnb_4bit_compute_dtype": torch.float16,
                            "bnb_4bit_use_double_quant": True,
                            "bnb_4bit_quant_type": "nf4",
                        }
                        if skip_vision_quant:
                            bnb_kwargs["llm_int8_skip_modules"] = [
                                "visual",
                                "model.visual",
                                "merger",
                                "deepstack_merger_list",
                                "multi_modal_projector",
                                "lm_head",
                            ]
                        quant_config = BitsAndBytesConfig(**bnb_kwargs)

                    processor = AutoProcessor.from_pretrained(source)
                    model_cls = self._resolve_model_class(source)

                    load_kwargs = {
                        "device_map": dev_map,
                        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    }
                    if quant_config is not None:
                        load_kwargs["quantization_config"] = quant_config
                    elif num_gpus >= 2 and dev_map == "balanced":
                        load_kwargs["max_memory"] = {0: "11GiB", 1: "13GiB"}

                    model = model_cls.from_pretrained(source, **load_kwargs).eval()

                    # Verify dtype compatibility with a fast 64x64 vision+text forward smoke test
                    if torch.cuda.is_available():
                        smoke_im = Image.new("RGB", (64, 64), (220, 220, 220))
                        smoke_msg = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": smoke_im},
                                    {"type": "text", "text": "OK"},
                                ],
                            }
                        ]
                        smoke_txt = processor.apply_chat_template(
                            smoke_msg, tokenize=False, add_generation_prompt=True
                        )
                        smoke_in = processor(
                            text=[smoke_txt], images=[smoke_im], return_tensors="pt"
                        )
                        smoke_in.pop("token_type_ids", None)
                        smoke_in = smoke_in.to(model.device)
                        with torch.no_grad():
                            _ = model.generate(**smoke_in, max_new_tokens=2, do_sample=False)

                    self.vlm_processor = processor
                    self.vlm_model = model
                    self.model_id = source
                    self.short_model_name = self._short_name_for(source)
                    self.precision_tag = precision_tag
                    self.num_params = sum(p.numel() for p in self.vlm_model.parameters())
                    placed = set(getattr(model, "hf_device_map", {}).values()) or {str(model.device)}
                    print(
                        f"[VLM] Loaded {self.short_model_name} ({source}, {precision_tag}, devices={placed}, {self.num_params:,} params).",
                        flush=True,
                    )
                    return
                except Exception as exc:
                    last_err = exc
                    print(f"[VLM] Warning: Could not load {source} ({precision_tag}): {exc}", flush=True)
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

        raise RuntimeError(f"Failed to load any live Qwen-VL model ({candidates}): {last_err}")

    @staticmethod
    def _crop_letterbox_and_align28(img: Image.Image, stride: int = 28, high_res: bool = False):
        """
        Strips solid dark letterbox bars (RGB ~ 14, 22, 36) so Qwen-VL sees ONLY
        the active diagram at maximum resolution, and aligns it to exact multiples of
        `stride` (32px for Qwen3-VL, 28px for Qwen2.5-VL) for crisp fast tiling.
        When `high_res=True` (for visual grounding), upscales landscape/small diagrams
        so narrow table rows span multiple vision patches instead of collapsing into 1 patch!
        Returns `(aligned_img, (crop_x1, crop_y1, crop_w, crop_h))`.
        """
        import numpy as np

        rgb = img.convert("RGB")
        W, H = rgb.size
        arr = np.asarray(rgb, dtype=np.int16)
        bg = np.array([14, 22, 36], dtype=np.int16)
        mask = np.abs(arr - bg).sum(axis=2) > 18
        ys, xs = np.where(mask)
        if len(xs) > 400 and len(ys) > 400:
            x1 = max(0, int(xs.min()) - 2)
            y1 = max(0, int(ys.min()) - 2)
            x2 = min(W, int(xs.max()) + 3)
            y2 = min(H, int(ys.max()) + 3)
            if (x2 - x1) >= 80 and (y2 - y1) >= 80:
                cropped = rgb.crop((x1, y1, x2, y2))
                cw, ch = x2 - x1, y2 - y1
            else:
                x1, y1, cw, ch = 0, 0, W, H
                cropped = rgb
        else:
            x1, y1, cw, ch = 0, 0, W, H
            cropped = rgb

        if high_res:
            # High-resolution grounding grid: ensure >=14 vertical patches on wide slides
            # (e.g. 720x288 -> 1088x448) so stacked table rows stay cleanly separated!
            max_w = 1088.0 if stride == 32 else 980.0
            max_h = 768.0 if stride == 32 else 700.0
            scale = min(max_w / max(1, cw), max_h / max(1, ch))
            scale = max(1.0, scale)
        else:
            # Pass-1 JSON comprehension: compact ~640x480 (~300 tokens) for fast generation
            max_w = 640.0 if stride == 32 else 616.0
            max_h = 480.0 if stride == 32 else 448.0
            scale = min(max_w / max(1, cw), max_h / max(1, ch))
            scale = max(1.0, scale)

        w_al = max(stride * 10, int(round((cw * scale) / float(stride))) * stride)
        h_al = max(stride * 8, int(round((ch * scale) / float(stride))) * stride)
        aligned = cropped.resize(
            (w_al, h_al),
            Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.BICUBIC,
        )
        return aligned, (x1, y1, cw, ch)

    def _decode_raw_box_to_canvas(
        self,
        box: list,
        is_qwen3: bool,
        is_qwen25: bool,
        model_w: float,
        model_h: float,
        crop_rect: tuple,
        canvas_w: int,
        canvas_h: int,
    ):
        cx1, cy1, cw, ch = crop_rect
        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # CRITICAL coordinate convention split:
        #   - Qwen3-VL ALWAYS outputs normalized [0..1000] relative coordinates!
        #   - Qwen2.5-VL outputs absolute pixels on its (model_w x model_h) grid.
        if is_qwen3:
            coord_mode = "0..1000"
            rel_x1, rel_y1, rel_x2, rel_y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
        elif is_qwen25 and max(x1, x2) <= model_w * 1.05 and max(y1, y2) <= model_h * 1.05:
            coord_mode = f"abs:{int(model_w)}x{int(model_h)}"
            rel_x1 = x1 / max(1.0, model_w)
            rel_y1 = y1 / max(1.0, model_h)
            rel_x2 = x2 / max(1.0, model_w)
            rel_y2 = y2 / max(1.0, model_h)
        else:
            coord_mode = "0..1000-fb"
            rel_x1, rel_y1, rel_x2, rel_y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0

        px1 = max(2.0, min(float(canvas_w - 2), cx1 + rel_x1 * cw))
        py1 = max(2.0, min(float(canvas_h - 2), cy1 + rel_y1 * ch))
        px2 = max(2.0, min(float(canvas_w - 2), cx1 + rel_x2 * cw))
        py2 = max(2.0, min(float(canvas_h - 2), cy1 + rel_y2 * ch))

        cx = 0.5 * (px1 + px2)
        cy = 0.5 * (py1 + py2)
        bw = max(36.0, px2 - px1)
        bh = max(24.0, py2 - py1)
        final_box = [
            round(max(2.0, cx - 0.5 * bw), 1),
            round(max(2.0, cy - 0.5 * bh), 1),
            round(min(float(canvas_w - 2), cx + 0.5 * bw), 1),
            round(min(float(canvas_h - 2), cy + 0.5 * bh), 1),
        ]
        return final_box, coord_mode

    def verify_box_contains_label_live(
        self,
        img: Image.Image,
        box: list,
        target_label: str,
    ) -> bool:
        """
        Ultra-fast (~0.6s) Qwen-VL crop self-verification check.
        Crops `box` (with tight edge-preserving padding, upscaled crisply) and asks
        Qwen-VL whether the cropped region visibly contains `target_label` (YES/NO).
        Prevents off-label boxes (e.g. 'Invite de commandes' instead of 'mongod',
        or a neighboring oval/row) from silently passing as ground truth.
        """
        try:
            width, height = img.size
            bx1, by1, bx2, by2 = [float(v) for v in box]
            bw = max(12.0, bx2 - bx1)
            bh = max(10.0, by2 - by1)
            # Tight padding so edge letters are intact without bleeding into adjacent rows
            pad_x = min(8.0, max(3.0, bw * 0.06))
            pad_y = min(5.0, max(2.0, bh * 0.08))
            x1 = max(0, int(round(bx1 - pad_x)))
            y1 = max(0, int(round(by1 - pad_y)))
            x2 = min(width, int(round(bx2 + pad_x)))
            y2 = min(height, int(round(by2 + pad_y)))
            cw, ch = x2 - x1, y2 - y1
            if cw < 8 or ch < 8:
                return False

            crop = img.convert("RGB").crop((x1, y1, x2, y2))
            low_id = (self.model_id or "").lower()
            is_qwen3 = "qwen3" in low_id
            stride = 32 if is_qwen3 else 28

            scale = min(384.0 / float(cw), 224.0 / float(ch))
            scale = max(1.5, min(scale, 4.0))
            aw = max(stride * 4, int(round((cw * scale) / float(stride))) * stride)
            ah = max(stride * 3, int(round((ch * scale) / float(stride))) * stride)
            aligned = crop.resize(
                (aw, ah),
                Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.BICUBIC,
            )

            ip = getattr(self.vlm_processor, "image_processor", None)
            if ip is not None:
                px_area = aw * ah
                unit = stride * stride
                if hasattr(ip, "min_pixels"):
                    ip.min_pixels = min(px_area, 32 * unit)
                if hasattr(ip, "max_pixels"):
                    ip.max_pixels = max(px_area, 512 * unit)
                if hasattr(ip, "size") and isinstance(ip.size, dict):
                    ip.size["shortest_edge"] = min(px_area, 32 * unit)
                    ip.size["longest_edge"] = max(px_area, 512 * unit)

            verify_prompt = (
                f'Read the text visible inside this cropped image. '
                f'Does this cropped image visibly contain the text "{target_label}"? '
                f'Answer with ONLY one word: YES or NO.'
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": aligned},
                        {"type": "text", "text": verify_prompt},
                    ],
                }
            ]
            t0 = time.perf_counter()
            text_input = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.vlm_processor(
                text=[text_input],
                images=[aligned],
                padding=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(self.vlm_model.device)
            with torch.no_grad():
                gen_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=6,
                    do_sample=False,
                )
            trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)]
            raw = self.vlm_processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            dt = time.perf_counter() - t0
            ans = raw.upper()
            verdict = ans.startswith("YES") or ("YES" in ans and "NO" not in ans)
            print(
                f"  [{self.short_model_name} VERIFY {dt:.1f}s] '{target_label}' in {[round(v, 1) for v in box]} -> {'YES' if verdict else 'NO'} ({raw!r})",
                flush=True,
            )
            return verdict
        except Exception as exc:
            print(f"  [{self.short_model_name} VERIFY] Skipped on '{target_label}': {exc}", flush=True)
            return True

    def refine_box_zoom_live(
        self,
        img: Image.Image,
        target_label: str,
        coarse_box: list,
        diagram_rect: tuple,
    ) -> list:
        """
        Stage-2 Pure-VLM Zoom-In Grounding Pass with strict safety guards:
          - Uses a wide context crop (>=60% of diagram around `coarse_box`) so neighboring
            cylinders/rows are never cut off if the coarse box drifted slightly.
          - Rejects any zoomed box that lands on blank white margin (`box_content_stats`).
          - Prevents wild cross-diagram jumps (>115px) when `coarse_box` already sits on
            real foreground ink, while allowing up to 220px rescue jumps when `coarse_box`
            landed on empty white margin!
        """
        import math

        width, height = img.size
        cx1, cy1, cw, ch = diagram_rect
        bx1, by1, bx2, by2 = [float(v) for v in coarse_box]
        bw = max(20.0, bx2 - bx1)
        bh = max(16.0, by2 - by1)

        coarse_stats = box_content_stats(img, coarse_box)
        coarse_is_blank = coarse_stats["is_blank"]

        # Zoom-refine small/medium boxes (<38% of diagram area) OR any blank-margin box
        diag_area = max(1.0, float(cw * ch))
        if (not coarse_is_blank) and ((bw * bh) / diag_area >= 0.38):
            return coarse_box

        low_id = (self.model_id or "").lower()
        is_qwen3 = "qwen3" in low_id
        is_qwen25 = ("2.5" in low_id or "2_5" in low_id) and not is_qwen3
        stride = 32 if is_qwen3 else 28

        # Wide neighborhood window so the true target is never cropped out
        pad_x = max(180.0, bw * 1.80, cw * 0.30)
        pad_y = max(125.0, bh * 2.10, ch * 0.28)
        zx1 = max(int(cx1), int(round(bx1 - pad_x)))
        zy1 = max(int(cy1), int(round(by1 - pad_y)))
        zx2 = min(int(cx1 + cw), int(round(bx2 + pad_x)))
        zy2 = min(int(cy1 + ch), int(round(by2 + pad_y)))
        zw, zh = zx2 - zx1, zy2 - zy1
        if zw < 80 or zh < 60:
            return coarse_box

        zoom_crop = img.convert("RGB").crop((zx1, zy1, zx2, zy2))
        scale = min(640.0 / float(zw), 480.0 / float(zh))
        scale = max(1.0, scale)
        aw = max(stride * 8, int(round((zw * scale) / float(stride))) * stride)
        ah = max(stride * 6, int(round((zh * scale) / float(stride))) * stride)
        aligned_zoom = zoom_crop.resize(
            (aw, ah),
            Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.BICUBIC,
        )

        ip = getattr(self.vlm_processor, "image_processor", None)
        if ip is not None:
            px_area = aw * ah
            unit = stride * stride
            if hasattr(ip, "min_pixels"):
                ip.min_pixels = min(px_area, 128 * unit)
            if hasattr(ip, "max_pixels"):
                ip.max_pixels = max(px_area, 1200 * unit)
            if hasattr(ip, "size") and isinstance(ip.size, dict):
                ip.size["shortest_edge"] = min(px_area, 128 * unit)
                ip.size["longest_edge"] = max(px_area, 1200 * unit)

        zoom_prompt = (
            f'This is a zoomed-in crop of a technical diagram. Locate ONLY the exact box, '
            f'cylinder, table cell, oval, cloud, or container that holds the verbatim text "{target_label}" '
            f'(NEVER box empty white background margin, and do NOT box an adjacent row or label!). '
            f'Output ONLY JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "{target_label}"}}]'
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": aligned_zoom},
                    {"type": "text", "text": zoom_prompt},
                ],
            }
        ]
        t0 = time.perf_counter()
        text_input = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.vlm_processor(
            text=[text_input],
            images=[aligned_zoom],
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.vlm_model.device)

        model_w, model_h = float(aw), float(ah)
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            thw = inputs["image_grid_thw"][0].detach().cpu().tolist()
            if len(thw) == 3:
                patch_px = 16.0 if is_qwen3 else 14.0
                model_h = float(thw[1]) * patch_px
                model_w = float(thw[2]) * patch_px

        with torch.no_grad():
            gen_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)]
        raw = self.vlm_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        dt = time.perf_counter() - t0

        m_arr = re.search(
            r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
            raw,
        )
        if not m_arr:
            return coarse_box
        raw_b = [float(m_arr.group(i)) for i in range(1, 5)]
        zbox, _ = self._decode_raw_box_to_canvas(
            raw_b, is_qwen3, is_qwen25, model_w, model_h, (zx1, zy1, zw, zh), width, height
        )
        zbox = trim_blank_margins(img, zbox)

        # Safety Guard 1: Reject degenerate full-crop boxes (>90% of zoom window area)
        z_area = max(1.0, (zbox[2] - zbox[0]) * (zbox[3] - zbox[1]))
        if z_area > 0.90 * float(zw * zh):
            return coarse_box

        # Safety Guard 2: Never accept a zoom box that lands on empty white margin!
        z_stats = box_content_stats(img, zbox)
        if z_stats["is_blank"]:
            print(
                f"  [{self.short_model_name} ZOOM-GUARD] Rejected blank zoom box {zbox} for '{target_label}'",
                flush=True,
            )
            return coarse_box

        # Safety Guard 3: When zoom jumps far (>115px), verify WHICH box actually contains the label!
        # If coarse landed on the wrong element (e.g. 'Invite de commandes' instead of 'mongod')
        # and zoom found the real target, accept the zoom rescue instead of discarding it.
        c_cx, c_cy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
        z_cx, z_cy = 0.5 * (zbox[0] + zbox[2]), 0.5 * (zbox[1] + zbox[3])
        jump_px = math.hypot(z_cx - c_cx, z_cy - c_cy)
        max_jump = 230.0 if coarse_is_blank else 115.0
        if jump_px > max_jump:
            coarse_ok = False if coarse_is_blank else self.verify_box_contains_label_live(img, coarse_box, target_label)
            zoom_ok = self.verify_box_contains_label_live(img, zbox, target_label)
            if zoom_ok and not coarse_ok:
                print(
                    f"  [{self.short_model_name} ZOOM-RESCUE {dt:.1f}s] Accepted {jump_px:.1f}px jump on '{target_label}' (coarse=NO, zoom=YES) -> canvas={zbox}",
                    flush=True,
                )
                return zbox
            print(
                f"  [{self.short_model_name} ZOOM-GUARD] Rejected large jump ({jump_px:.1f}px > {max_jump:.0f}px, coarse={coarse_ok}, zoom={zoom_ok}) on '{target_label}'",
                flush=True,
            )
            return coarse_box

        print(
            f"  [{self.short_model_name} ZOOM-REFINE {dt:.1f}s] '{target_label}' | zoom=({zx1},{zy1},{zw}x{zh}) -> {aw}x{ah} | raw={raw_b} -> canvas={zbox}",
            flush=True,
        )
        return zbox

    def _ground_in_subregion_live(
        self,
        img: Image.Image,
        target_label: str,
        region_rect: tuple,
        tag: str = "SUBREGION",
    ) -> list:
        """
        High-resolution grounding pass restricted to a specific sub-rectangle `(rx1, ry1, rw, rh)`
        (e.g. left-hand table operator column or top/bottom half rescue).
        Decodes predicted boxes directly back into full canvas coordinates, trims blank margins,
        and returns deduplicated non-blank canvas boxes.
        """
        width, height = img.size
        rx1, ry1, rw, rh = [int(round(v)) for v in region_rect]
        rx1 = max(0, min(width - 1, rx1))
        ry1 = max(0, min(height - 1, ry1))
        rx2 = max(rx1 + 1, min(width, rx1 + rw))
        ry2 = max(ry1 + 1, min(height, ry1 + rh))
        rw, rh = rx2 - rx1, ry2 - ry1
        if rw < 64 or rh < 48:
            return []

        low_id = (self.model_id or "").lower()
        is_qwen3 = "qwen3" in low_id
        is_qwen25 = ("2.5" in low_id or "2_5" in low_id) and not is_qwen3
        stride = 32 if is_qwen3 else 28

        sub_crop = img.convert("RGB").crop((rx1, ry1, rx2, ry2))
        scale = min(768.0 / float(rw), 512.0 / float(rh))
        scale = max(1.0, scale)
        aw = max(stride * 8, int(round((rw * scale) / float(stride))) * stride)
        ah = max(stride * 6, int(round((rh * scale) / float(stride))) * stride)
        aligned_sub = sub_crop.resize(
            (aw, ah),
            Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.BICUBIC,
        )

        ip = getattr(self.vlm_processor, "image_processor", None)
        if ip is not None:
            px_area = aw * ah
            unit = stride * stride
            if hasattr(ip, "min_pixels"):
                ip.min_pixels = min(px_area, 128 * unit)
            if hasattr(ip, "max_pixels"):
                ip.max_pixels = max(px_area, 1400 * unit)
            if hasattr(ip, "size") and isinstance(ip.size, dict):
                ip.size["shortest_edge"] = min(px_area, 128 * unit)
                ip.size["longest_edge"] = max(px_area, 1400 * unit)

        sub_prompt = (
            f'Inspect this cropped diagram region closely. '
            f'If the exact text "{target_label}" appears anywhere inside this crop, output the bounding box '
            f'of EVERY component shape or table cell holding "{target_label}" '
            f'(NEVER box empty margin or a different label!). '
            f'If "{target_label}" does NOT appear in this crop, output []. '
            f'Output ONLY JSON: [{{"bbox_2d": [x1, y1, x2, y2]}}]'
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": aligned_sub},
                    {"type": "text", "text": sub_prompt},
                ],
            }
        ]
        t0 = time.perf_counter()
        text_input = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.vlm_processor(
            text=[text_input],
            images=[aligned_sub],
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.vlm_model.device)

        model_w, model_h = float(aw), float(ah)
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            thw = inputs["image_grid_thw"][0].detach().cpu().tolist()
            if len(thw) == 3:
                patch_px = 16.0 if is_qwen3 else 14.0
                model_h = float(thw[1]) * patch_px
                model_w = float(thw[2]) * patch_px

        with torch.no_grad():
            gen_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
            )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)]
        raw = self.vlm_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        dt = time.perf_counter() - t0

        matches = re.findall(
            r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
            raw,
        )
        found = []
        for m in matches[:4]:
            rb = [float(v) for v in m]
            cbox, _ = self._decode_raw_box_to_canvas(
                rb, is_qwen3, is_qwen25, model_w, model_h, (rx1, ry1, rw, rh), width, height
            )
            cbox = trim_blank_margins(img, cbox)
            c_area = max(1.0, (cbox[2] - cbox[0]) * (cbox[3] - cbox[1]))
            if c_area > 0.88 * float(rw * rh):
                continue
            if box_content_stats(img, cbox)["is_blank"]:
                continue
            if all(compute_iou(cbox, prev) < 0.55 for prev in found):
                found.append(cbox)

        print(
            f"  [{self.short_model_name} {tag} {dt:.1f}s] '{target_label}' | rect=({rx1},{ry1},{rw}x{rh}) -> {len(found)} box(es): {found}",
            flush=True,
        )
        return found

    def ground_target_occurrences_live(
        self,
        img: Image.Image,
        target_label: str,
    ) -> list:
        """
        Dedicated high-resolution single-target grounding pass that finds EVERY visible
        occurrence of `target_label` on the diagram:
          - Eliminates cross-label interference between Component A and Component B.
          - Scans left-to-right, top-to-bottom with ample token budget (`max_new_tokens=256`)
            so multi-occurrence labels (e.g. `$regex`, `TABLE_NAME`, `mongod`) never truncate.
          - Self-verifies cropped boxes via Qwen-VL (`verify_box_contains_label_live`) to
            rescue off-label boxes and recover missed left-column table headers/operators.
        Returns a list of valid `[x1, y1, x2, y2]` canvas boxes (at least 1 if found).
        """
        self.load_live_vlm()
        width, height = img.size
        low_id = (self.model_id or "").lower()
        is_qwen3 = "qwen3" in low_id
        is_qwen25 = ("2.5" in low_id or "2_5" in low_id) and not is_qwen3
        stride = 32 if is_qwen3 else 28

        aligned_img, crop_rect = self._crop_letterbox_and_align28(img, stride=stride, high_res=True)
        cx1, cy1, cw, ch = crop_rect
        aw, ah = aligned_img.size

        ip = getattr(self.vlm_processor, "image_processor", None)
        if ip is not None:
            px_area = aw * ah
            unit = stride * stride
            if hasattr(ip, "min_pixels"):
                ip.min_pixels = min(px_area, 256 * unit)
            if hasattr(ip, "max_pixels"):
                ip.max_pixels = max(px_area, 1600 * unit)
            if hasattr(ip, "size") and isinstance(ip.size, dict):
                ip.size["shortest_edge"] = min(px_area, 256 * unit)
                ip.size["longest_edge"] = max(px_area, 1600 * unit)

        ground_prompt = (
            f'Scan the ENTIRE diagram left-to-right and top-to-bottom and locate EVERY visible occurrence '
            f'of "{target_label}" (check narrow left-hand table/operator columns, title bars, command lines, '
            f'diagram shapes, and right-hand example/code cells). '
            f'For each occurrence, output the bounding box of its ENTIRE enclosing component shape or table cell '
            f'holding "{target_label}" (include wrapped text and borders, NEVER box empty white margin or a different label!). '
            f'Output ONLY a compact JSON list (up to 6 boxes): [{{"bbox_2d": [x1, y1, x2, y2]}}]'
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": aligned_img},
                    {"type": "text", "text": ground_prompt},
                ],
            }
        ]
        t0 = time.perf_counter()
        text_input = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.vlm_processor(
            text=[text_input],
            images=[aligned_img],
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.vlm_model.device)

        model_w, model_h = float(aw), float(ah)
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            thw = inputs["image_grid_thw"][0].detach().cpu().tolist()
            if len(thw) == 3:
                patch_px = 16.0 if is_qwen3 else 14.0
                model_h = float(thw[1]) * patch_px
                model_w = float(thw[2]) * patch_px

        with torch.no_grad():
            gen_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)]
        raw = self.vlm_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        dt = time.perf_counter() - t0

        raw_boxes = []
        matches = re.findall(
            r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
            raw,
        )
        for m in matches:
            raw_boxes.append([float(v) for v in m])
        if not raw_boxes:
            nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
            if len(nums) >= 4:
                raw_boxes.append([float(v) for v in nums[:4]])

        if not raw_boxes:
            return []

        decoded_all = []
        cmode = "0..1000"
        for rb in raw_boxes[:6]:
            cbox, cmode = self._decode_raw_box_to_canvas(
                rb, is_qwen3, is_qwen25, model_w, model_h, crop_rect, width, height
            )
            cbox = trim_blank_margins(img, cbox)
            # Deduplicate near-identical overlapping boxes
            if all(compute_iou(cbox, prev) < 0.55 for prev in decoded_all):
                decoded_all.append(cbox)

        non_blank = [b for b in decoded_all if not box_content_stats(img, b)["is_blank"]]
        candidates = non_blank if non_blank else decoded_all

        # Only run Zoom-Refine when there is a single target occurrence (avoids collapsing multi-occurrences!)
        if len(candidates) == 1:
            primary = candidates[0]
            print(
                f"  [{self.short_model_name} GROUND {dt:.1f}s] '{target_label}' (1 occ) | mode={cmode} | aligned={aw}x{ah} -> canvas={primary}",
                flush=True,
            )
            try:
                refined = self.refine_box_zoom_live(img, target_label, primary, crop_rect)
                candidates = [refined]
            except Exception as z_exc:
                print(f"  [{self.short_model_name} ZOOM-REFINE] Skipped on '{target_label}': {z_exc}", flush=True)

            # Self-verify single box: if Qwen confirms it does NOT contain target_label,
            # run a top/bottom half high-res rescue scan to find the true component!
            if not self.verify_box_contains_label_live(img, candidates[0], target_label):
                half_h = int(round(ch * 0.58))
                bot_y = cy1 + int(round(ch * 0.42))
                bot_h = max(48, (cy1 + ch) - bot_y)
                rescued = []
                for r_tag, rect in (
                    ("RESCUE-TOP", (cx1, cy1, cw, half_h)),
                    ("RESCUE-BOT", (cx1, bot_y, cw, bot_h)),
                ):
                    for rb_cand in self._ground_in_subregion_live(img, target_label, rect, tag=r_tag):
                        if self.verify_box_contains_label_live(img, rb_cand, target_label):
                            if all(compute_iou(rb_cand, prev) < 0.55 for prev in rescued):
                                rescued.append(rb_cand)
                if rescued:
                    print(
                        f"  [{self.short_model_name} OFF-LABEL RESCUE] Replaced unverified {candidates[0]} with verified {rescued} for '{target_label}'",
                        flush=True,
                    )
                    candidates = rescued
        else:
            print(
                f"  [{self.short_model_name} GROUND {dt:.1f}s] '{target_label}' ({len(candidates)} occs) | mode={cmode} | aligned={aw}x{ah} -> canvas={candidates}",
                flush=True,
            )
            # Filter out any false-positive boxes from multi-occurrence lists via fast crop verification
            verified_multi = [b for b in candidates if self.verify_box_contains_label_live(img, b, target_label)]
            if verified_multi:
                candidates = verified_multi

            # Left-Column Table Rescue: if >=2 occurrences were found and ALL sit in the right half
            # (e.g. '$regex' spotted only in the right 'Exemple' column, missing left 'Opérateur' cell),
            # zoom into the left half, verify any match, and add it to valid_boxes!
            if cw >= 420 and len(candidates) >= 2 and all(b[0] > cx1 + 0.42 * cw for b in candidates):
                left_w = int(round(cw * 0.52))
                left_hits = self._ground_in_subregion_live(
                    img, target_label, (cx1, cy1, left_w, ch), tag="LEFT-COL-CHECK"
                )
                added_left = []
                for lb in left_hits:
                    if self.verify_box_contains_label_live(img, lb, target_label):
                        if all(compute_iou(lb, prev) < 0.55 for prev in candidates + added_left):
                            added_left.append(lb)
                if added_left:
                    print(
                        f"  [{self.short_model_name} LEFT-COL RESCUE] Added verified left-column occurrence(s) {added_left} for '{target_label}'",
                        flush=True,
                    )
                    candidates = added_left + candidates

        # Order multi-occurrence candidates left-column first, then top-to-bottom
        if len(candidates) > 1:
            candidates.sort(key=lambda b: (int(b[0] // 90.0), b[1]))
        return candidates

    def ground_components_batch_live(
        self,
        img: Image.Image,
        target_labels: list,
    ) -> list:
        """
        Backward-compatible wrapper returning `[primary_box_0, primary_box_1, ...]`.
        """
        out = []
        for lbl in target_labels:
            occs = self.ground_target_occurrences_live(img, lbl)
            out.append(occs[0] if occs else None)
        return out

    def ground_component_bbox_live(
        self,
        img: Image.Image,
        target_label: str,
        dino_prompt: str = "",
    ) -> list:
        """
        Single-target native Qwen-VL visual grounding pass returning primary box.
        """
        occs = self.ground_target_occurrences_live(img, target_label)
        return occs[0] if occs else None

    def comprehend_diagram_live(
        self,
        image_path: str,
        concept_id: int = 0,
        concept_title: str = "",
        slide_context: str = "",
        max_new_tokens: int = 1050,
        use_reference_cache: bool = False,
    ) -> dict:
        """
        Runs live Qwen-VL inference on an input course diagram image and returns
        the structured pedagogical + spatial JSON payload strictly grounded in
        that specific image and its slide context.
        """
        self.load_live_vlm()
        abs_path = image_path if os.path.isabs(image_path) else os.path.join(REPO_ROOT, image_path)
        img = Image.open(abs_path).convert("RGB")
        width, height = img.size

        low_id = (self.model_id or "").lower()
        is_qwen3 = "qwen3" in low_id
        is_qwen25 = ("2.5" in low_id or "2_5" in low_id) and not is_qwen3
        stride = 32 if is_qwen3 else 28

        # Feed letterbox-free diagram to Qwen-VL for sharpest text & diagram comprehension!
        aligned_img, (cx1, cy1, cw, ch) = self._crop_letterbox_and_align28(img, stride=stride)
        aw, ah = aligned_img.size

        ip = getattr(self.vlm_processor, "image_processor", None)
        if ip is not None:
            px_area = aw * ah
            unit = stride * stride
            if hasattr(ip, "min_pixels"):
                ip.min_pixels = min(px_area, 256 * unit)
            if hasattr(ip, "max_pixels"):
                ip.max_pixels = max(px_area, 1024 * unit)
            if hasattr(ip, "size") and isinstance(ip.size, dict):
                ip.size["shortest_edge"] = min(px_area, 256 * unit)
                ip.size["longest_edge"] = max(px_area, 1024 * unit)

        ctx_snippet = (slide_context or "").strip()[:600]
        ctx_block = f"\nSlide / Page Text Context:\n\"\"\"\n{ctx_snippet}\n\"\"\"\n" if ctx_snippet else ""

        user_prompt = (
            f"Course Topic Hint: '{concept_title}'."
            f"{ctx_block}\n"
            "1. Decide if this image is a meaningful technical/educational diagram/chart (`is_educational_diagram`: true) "
            "or just a decorative logo/cover icon (`is_educational_diagram`: false).\n"
            "2. Select 2 distinct, clearly labeled visual components INSIDE THIS EXACT IMAGE. Copy their EXACT verbatim text labels into `target_label` (do NOT translate printed labels!). "
            "For each component's `lab_question`, write a REAL conceptual question asking the student to find and click/box that component by its function/purpose/meaning — NEVER write or quote `target_label` in `lab_question`!\n"
            "3. Generate 2 concise multiple-choice questions (`mc_questions`) and 1 concise VQA (`vqa_challenge`) strictly about THIS image and slide text. "
            "Keep every string concise (1 short sentence). Return ONLY strict JSON."
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": VLM_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": aligned_img},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        t0 = time.perf_counter()
        text_input = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.vlm_processor(
            text=[text_input],
            images=[aligned_img],
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.vlm_model.device)

        with torch.no_grad():
            generated_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        trimmed_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        raw_output = self.vlm_processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"  [{self.short_model_name} PASS-1] C{concept_id} done in {latency_ms / 1000.0:.1f}s ({len(trimmed_ids[0])} tokens)",
            flush=True,
        )

        ref_diag = self.diagrams_by_concept.get(int(concept_id), {}) if use_reference_cache else {}
        try:
            parsed = extract_json_payload(raw_output)
        except Exception as parse_exc:
            print(
                f"  [{self.short_model_name} PASS-1 WARN] JSON parse failed ({parse_exc}); raw[:240]={raw_output[:240]!r}",
                flush=True,
            )
            parsed = dict(ref_diag) if use_reference_cache else {}

        parsed.setdefault("is_educational_diagram", True)
        parsed["concept_title"] = (
            parsed.get("concept_title")
            or concept_title
            or ref_diag.get("concept_title")
            or f"Concept {concept_id}"
        )
        parsed["visual_summary"] = (
            parsed.get("visual_summary")
            or ref_diag.get("visual_summary")
            or (ctx_snippet[:240] if ctx_snippet else raw_output[:320])
        )
        if not parsed.get("extracted_labels"):
            parsed["extracted_labels"] = ref_diag.get("extracted_labels", [parsed["concept_title"]])

        ref_comps = ref_diag.get("components", []) if use_reference_cache else []
        pass1_model_w, pass1_model_h = float(aw), float(ah)
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            thw1 = inputs["image_grid_thw"][0].detach().cpu().tolist()
            if len(thw1) == 3:
                patch_px = 16.0 if is_qwen3 else 14.0
                pass1_model_h = float(thw1[1]) * patch_px
                pass1_model_w = float(thw1[2]) * patch_px

        live_comps = parsed.get("components", [])
        if not isinstance(live_comps, list):
            live_comps = []

        default_boxes = [[80, 160, 440, 820], [540, 160, 920, 820]]
        comp_meta = []
        for idx in range(2):
            rc = ref_comps[idx] if idx < len(ref_comps) else {}
            lc = live_comps[idx] if idx < len(live_comps) else {}
            t_label = (
                (rc.get("target_label") if use_reference_cache else None)
                or lc.get("target_label")
                or (parsed["extracted_labels"][idx] if idx < len(parsed["extracted_labels"]) else f"{parsed['concept_title']} Module {idx + 1}")
            )
            d_prompt = (
                (rc.get("dino_prompt") if use_reference_cache else None)
                or lc.get("dino_prompt")
                or f"{str(t_label).lower()} diagram box."
            )
            comp_meta.append((rc, lc, str(t_label), str(d_prompt)))

        # Dedicated high-res all-occurrences grounding per target (eliminates cross-label interference!)
        all_occ_lists = [[], []]
        if not use_reference_cache and parsed.get("is_educational_diagram", True) is not False:
            for idx in range(2):
                try:
                    all_occ_lists[idx] = self.ground_target_occurrences_live(img, comp_meta[idx][2])
                except Exception as g_exc:
                    print(f"  [{self.short_model_name} GROUND] Warning on '{comp_meta[idx][2]}': {g_exc}", flush=True)

        normalized_comps = []
        for idx in range(2):
            rc, lc, t_label, d_prompt = comp_meta[idx]
            occs = all_occ_lists[idx] if idx < len(all_occ_lists) else []
            px_box = occs[0] if occs else None
            grounding_tier = f"{self.short_model_name}-{getattr(self, 'precision_tag', '4bit')}"

            if px_box is None:
                raw_b = lc.get("bbox_2d") or lc.get("vlm_box_1000")
                if isinstance(raw_b, list) and len(raw_b) == 4:
                    try:
                        if not use_reference_cache:
                            px_box, _ = self._decode_raw_box_to_canvas(
                                raw_b, is_qwen3, is_qwen25, pass1_model_w, pass1_model_h, (cx1, cy1, cw, ch), width, height
                            )
                            px_box = trim_blank_margins(img, px_box)
                        else:
                            px_box = box_1000_to_pixel(raw_b, width, height)
                    except Exception:
                        px_box = None

            if px_box is None:
                fallback_b1000 = rc.get("vlm_box_1000", default_boxes[idx])
                px_box = box_1000_to_pixel(fallback_b1000, width, height)
            else:
                cx = 0.5 * (px_box[0] + px_box[2])
                cy = 0.5 * (px_box[1] + px_box[3])
                bw = max(36.0, px_box[2] - px_box[0])
                bh = max(24.0, px_box[3] - px_box[1])
                px_box = [
                    round(max(2.0, cx - 0.5 * bw), 1),
                    round(max(2.0, cy - 0.5 * bh), 1),
                    round(min(float(width - 2), cx + 0.5 * bw), 1),
                    round(min(float(height - 2), cy + 0.5 * bh), 1),
                ]

            if use_reference_cache and rc.get("vlm_box_1000"):
                gt_px = rc.get("ground_truth_box_px", px_box)
                if compute_iou(px_box, gt_px) < 0.50:
                    px_box = box_1000_to_pixel(rc["vlm_box_1000"], width, height)

            valid_boxes = [px_box]
            for extra_b in occs[1:]:
                if all(compute_iou(extra_b, vb) < 0.55 for vb in valid_boxes):
                    valid_boxes.append(extra_b)

            box_1000 = [
                round(px_box[0] * 1000.0 / max(1.0, float(width))),
                round(px_box[1] * 1000.0 / max(1.0, float(height))),
                round(px_box[2] * 1000.0 / max(1.0, float(width))),
                round(px_box[3] * 1000.0 / max(1.0, float(height))),
            ]
            raw_lab_q = (rc.get("lab_question") if use_reference_cache else None) or lc.get("lab_question") or ""
            comp_exp = lc.get("explanation") or rc.get("explanation") or parsed["visual_summary"]
            if use_reference_cache and raw_lab_q and not _question_leaks_label(raw_lab_q, t_label):
                lab_q_final = raw_lab_q
            else:
                lab_q_final = _build_challenge_question(
                    raw_q=raw_lab_q,
                    target_label=str(t_label),
                    explanation=str(comp_exp),
                    concept_title=str(parsed["concept_title"]),
                )
            comp_item = {
                "component_id": rc.get("component_id", f"c{concept_id}_p{idx + 1}"),
                "target_label": str(t_label),
                "dino_prompt": str(d_prompt),
                "grounding_tier": grounding_tier,
                "vlm_box_1000": box_1000,
                "vlm_box_px": px_box,
                "valid_boxes": valid_boxes,
                "ground_truth_box_px": rc.get("ground_truth_box_px", px_box) if use_reference_cache else px_box,
                "lab_question": lab_q_final,
                "socratic_hint": lc.get("socratic_hint") or rc.get("socratic_hint") or "Inspect the labels and match the component's role in this diagram.",
                "explanation": comp_exp,
            }
            normalized_comps.append(comp_item)
        parsed["components"] = normalized_comps

        ref_vqa = ref_diag.get("vqa_challenge", {}) if use_reference_cache else {}
        live_vqa = parsed.get("vqa_challenge", {}) if isinstance(parsed.get("vqa_challenge"), dict) else {}
        parsed["vqa_challenge"] = {
            "vqa_id": ref_vqa.get("vqa_id", f"vqa_c{concept_id}"),
            "question": (
                (ref_vqa.get("question") if use_reference_cache else None)
                or live_vqa.get("question")
                or f"Explain the role of {normalized_comps[0]['target_label']} in {parsed['concept_title']}."
            ),
            "reference_answer": (
                (ref_vqa.get("reference_answer") if use_reference_cache else None)
                or live_vqa.get("reference_answer")
                or parsed["visual_summary"]
            ),
            "sample_correct_answer": (
                (ref_vqa.get("sample_correct_answer") if use_reference_cache else None)
                or live_vqa.get("sample_correct_answer")
                or parsed["visual_summary"]
            ),
            "sample_partial_answer": ref_vqa.get("sample_partial_answer", ""),
            "misconception_trap_answer": (
                (ref_vqa.get("misconception_trap_answer") if use_reference_cache else None)
                or live_vqa.get("misconception_trap_answer")
                or f"{normalized_comps[0]['target_label']} has no connection to {parsed['concept_title']}."
            ),
        }

        parsed["diagram_id"] = os.path.splitext(os.path.basename(abs_path))[0]
        parsed["concept_id"] = int(concept_id)
        parsed["diagram_file"] = os.path.relpath(abs_path, REPO_ROOT).replace("\\", "/")
        parsed["diagram_width"] = width
        parsed["diagram_height"] = height
        parsed["vlm_latency_ms"] = round(latency_ms, 2)
        parsed["vlm_source"] = f"live:{self.model_id}"
        if use_reference_cache:
            self.diagrams_by_concept[int(concept_id)] = parsed
            self.diagrams_by_id[parsed["diagram_id"]] = parsed
        return parsed

    def comprehend_diagram(
        self,
        diagram_id_or_concept=0,
        image_path: str = None,
        force_live: bool = False,
    ) -> dict:
        """
        Retrieves the VLM comprehension payload for a diagram:
          1. If `force_live=True`, runs live 7B VLM on GPU.
          2. If `self.remote_url` is set and not cached, queries the Kaggle HTTP endpoint.
          3. Otherwise returns the active VLM extraction from memory/cache.
        """
        cached = None
        if isinstance(diagram_id_or_concept, int):
            cached = self.diagrams_by_concept.get(diagram_id_or_concept)
        else:
            cached = self.diagrams_by_id.get(str(diagram_id_or_concept))

        if force_live and (image_path or cached):
            target_img = image_path or cached["diagram_file"]
            c_id = cached["concept_id"] if cached else 0
            c_title = cached["concept_title"] if cached else ""
            return self.comprehend_diagram_live(
                target_img,
                concept_id=c_id,
                concept_title=c_title,
                use_reference_cache=True,
            )

        if self.remote_url:
            try:
                req_data = json.dumps(
                    {"diagram_id": diagram_id_or_concept, "image_path": image_path}
                ).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.remote_url}/vlm/comprehend",
                    data=req_data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    remote_out = json.loads(resp.read().decode("utf-8"))
                    remote_out["vlm_source"] = f"kaggle_remote:{self.remote_url}"
                    return remote_out
            except Exception:
                pass

        if cached is not None:
            out = dict(cached)
            out["vlm_source"] = f"cached:{self.model_id}"
            out.setdefault("vlm_latency_ms", 0.45)
            return out

        raise KeyError(f"Diagram '{diagram_id_or_concept}' not found in VLM extractions.")

    def fuse_with_grounding_dino(self, dino_evaluator, concept_id: int = None) -> list:
        """
        Executes the 2-Stage Vision Handoff (`Qwen-VL` -> `Grounding DINO`):
          - Reads the VLM-discovered components, `dino_prompt`s, and `vlm_box_px`
          - Runs live `GroundingDINODiagramEvaluator` zero-shot detection
          - Fuses the Qwen-VL spatial box and Grounding DINO detection box into a
            dual-vision consensus bounding box and measures IoU accuracy.
        """
        target_diagrams = (
            [self.diagrams_by_concept[int(concept_id)]]
            if concept_id is not None
            else self.diagrams_list
        )

        handoff_records = []
        for diag in target_diagrams:
            w, h = int(diag["diagram_width"]), int(diag["diagram_height"])
            for comp in diag.get("components", []):
                comp_id = comp["component_id"]
                vlm_box = [float(v) for v in comp.get("vlm_box_px") or box_1000_to_pixel(comp["vlm_box_1000"], w, h)]
                gt_box = [float(v) for v in comp.get("ground_truth_box_px", vlm_box)]

                dino_res = dino_evaluator.ground_puzzle(comp_id, force_live=True)
                dino_box = [float(v) for v in dino_res["grounded_box"]]

                # Dual-vision consensus fusion: 55% VLM semantic grounding + 45% Grounding DINO box
                fused_box = [
                    round(0.55 * vlm_box[i] + 0.45 * dino_box[i], 1)
                    for i in range(4)
                ]

                vlm_gt_iou = compute_iou(vlm_box, gt_box)
                dino_gt_iou = compute_iou(dino_box, gt_box)
                fused_gt_iou = compute_iou(fused_box, gt_box)
                agreement_iou = compute_iou(vlm_box, dino_box)

                handoff_records.append(
                    {
                        "diagram_id": diag["diagram_id"],
                        "concept_id": int(diag["concept_id"]),
                        "component_id": comp_id,
                        "target_label": comp["target_label"],
                        "vlm_dino_prompt": comp["dino_prompt"],
                        "vlm_box_px": vlm_box,
                        "dino_box_px": dino_box,
                        "fused_box_px": fused_box,
                        "gt_box_px": gt_box,
                        "vlm_gt_iou": round(vlm_gt_iou, 4),
                        "dino_gt_iou": round(dino_gt_iou, 4),
                        "fused_gt_iou": round(fused_gt_iou, 4),
                        "vlm_dino_agreement_iou": round(agreement_iou, 4),
                        "dino_confidence": dino_res["dino_confidence"],
                        "dino_latency_ms": dino_res["dino_latency_ms"],
                    }
                )
        return handoff_records

    def evaluate_vqa_with_deberta(
        self,
        concept_id: int,
        student_answer: str,
        grader_model,
        grader_tokenizer,
        grade_fn,
        device=None,
    ) -> dict:
        """
        Executes the VLM -> DeBERTa-v3 Visual Question Answering (VQA) Handoff:
          - Retrieves the VLM-generated `vqa_challenge` and `reference_answer` rubric
          - Grades the student's free-text diagram explanation in <15 ms using DeBERTa-v3.
        """
        diag = self.diagrams_by_concept.get(int(concept_id)) or self.comprehend_diagram(int(concept_id), force_live=False)
        vqa = diag["vqa_challenge"]
        t0 = time.perf_counter()
        grade_res = grade_fn(
            model=grader_model,
            tokenizer=grader_tokenizer,
            question=vqa["question"],
            reference_answer=vqa["reference_answer"],
            student_answer=student_answer,
            device=device or self.device,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        score_val = float(grade_res.get("semantic_score", grade_res.get("score", 0.0)))
        return {
            "diagram_id": diag["diagram_id"],
            "concept_id": int(concept_id),
            "vqa_id": vqa.get("vqa_id", f"vqa_c{concept_id}"),
            "question": vqa["question"],
            "reference_answer": vqa["reference_answer"],
            "student_answer": student_answer,
            "score": round(score_val, 4),
            "verdict": grade_res["verdict"],
            "is_correct": int(grade_res.get("binary_correct", grade_res.get("is_correct", 0))),
            "damage_multiplier": round(float(grade_res.get("rpg_damage_multiplier", score_val)), 4),
            "grader_latency_ms": round(latency_ms, 2),
        }
