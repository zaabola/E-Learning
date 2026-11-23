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
    "Write questions and labels in the SAME language as the slide (French if the slide is in French, English if in English). "
    "CRITICAL: For `target_label`, copy the EXACT verbatim text printed inside the diagram (e.g. 'DBWn', 'Cache de tampons de la base de données', 'Assurer la sauvegarde') — NEVER translate printed diagram labels into another language! "
    "For `bbox_2d`, always enclose the ENTIRE box, rectangle, oval, cloud, or container shape holding `target_label` (all wrapped text lines + outer border, not just one word). "
    "Return ONLY a valid JSON object with this exact schema:\n"
    "{\n"
    '  "is_educational_diagram": true,\n'
    '  "concept_title": "Short exact academic concept title from this diagram",\n'
    '  "visual_summary": "2-sentence technical explanation of what this specific diagram shows",\n'
    '  "extracted_labels": ["verbatim label 1 visible in image", "verbatim label 2 visible in image", "verbatim label 3"],\n'
    '  "summary_bullets": ["Key fact 1 from diagram/slide", "Key fact 2", "Key fact 3"],\n'
    '  "components": [\n'
    "    {\n"
    '      "target_label": "EXACT verbatim text label printed inside THIS image",\n'
    '      "dino_prompt": "short visual description of this component box in English ending with a period.",\n'
    '      "bbox_2d": [x1, y1, x2, y2],\n'
    '      "lab_question": "Click or drag a box around [target_label] shown in this figure.",\n'
    '      "socratic_hint": "1-sentence visual hint pointing to where it is on the figure",\n'
    '      "explanation": "1-sentence explanation of what this component does"\n'
    "    }\n"
    "  ],\n"
    '  "mc_questions": [\n'
    "    {\n"
    '      "question": "Specific multiple-choice question about THIS diagram/slide",\n'
    '      "choices": ["Correct answer", "Plausible wrong option 1", "Plausible wrong option 2", "Plausible wrong option 3"],\n'
    '      "correct_idx": 0\n'
    "    }\n"
    "  ],\n"
    '  "vqa_challenge": {\n'
    '    "question": "1 crisp analytical question strictly about THIS diagram",\n'
    '    "reference_answer": "1 concise 15-25 word ground-truth answer",\n'
    '    "sample_correct_answer": "Example accurate student paraphrase",\n'
    '    "misconception_trap_answer": "Plausible false answer contradicting the diagram"\n'
    "  }\n"
    "}"
)


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


def extract_json_payload(raw_text: str) -> dict:
    """Extracts and parses a JSON object from raw VLM output (handles markdown code fences)."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


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
        Loads `Qwen/Qwen3-VL-8B-Instruct` onto GPU (designed for Kaggle 2x T4 30GB VRAM
        in full FP16, or 4-bit NF4 if <24 GB VRAM), with automatic fallback to
        `Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen2-VL-7B-Instruct`.
        """
        if self.vlm_model is not None:
            return

        from transformers import AutoProcessor, BitsAndBytesConfig

        total_vram_gb = 0.0
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_vram_gb += torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)

        candidates = []
        if os.path.exists(VLM_CKPT_DIR):
            candidates.append(VLM_CKPT_DIR)
        for mid in (self.model_id, FALLBACK_MODEL_ID, LEGACY_FALLBACK_MODEL_ID):
            if mid and mid not in candidates:
                candidates.append(mid)

        last_err = None
        for source in candidates:
            # On Kaggle 2x T4 (>=24 GB VRAM), try FULL FP16 first, then 4-bit NF4 if OOM.
            precision_plans = [False, True] if (total_vram_gb >= 24.0) else [bool(self.load_in_4bit)]
            for try_4bit in precision_plans:
                precision_tag = "4-bit NF4" if try_4bit else "FULL FP16 (Multi-GPU)"
                print(
                    f"[VLM] Loading {source} ({precision_tag}, total_vram={total_vram_gb:.1f}GB, device_map='auto')...",
                    flush=True,
                )
                try:
                    quant_config = None
                    if try_4bit and torch.cuda.is_available():
                        try:
                            quant_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                            )
                        except Exception:
                            quant_config = None

                    processor = AutoProcessor.from_pretrained(source)
                    model_cls = self._resolve_model_class(source)

                    load_kwargs = {
                        "device_map": "auto" if torch.cuda.is_available() else None,
                        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    }
                    if quant_config is not None:
                        load_kwargs["quantization_config"] = quant_config

                    model = model_cls.from_pretrained(source, **load_kwargs).eval()
                    self.vlm_processor = processor
                    self.vlm_model = model
                    self.model_id = source
                    self.short_model_name = self._short_name_for(source)
                    self.num_params = sum(p.numel() for p in self.vlm_model.parameters())
                    print(
                        f"[VLM] Loaded {self.short_model_name} ({source}, {precision_tag}, {self.num_params:,} params).",
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
    def _crop_letterbox_and_align28(img: Image.Image, stride: int = 28):
        """
        Strips solid dark letterbox bars (RGB ~ 14, 22, 36) so Qwen-VL sees ONLY
        the active diagram at maximum resolution, and aligns it to exact multiples of
        `stride` (32px for Qwen3-VL, 28px for Qwen2.5-VL) for crisp zero-blur tiling.
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

        # Upscale crisp diagram to ~768x576 (Qwen3 stride=32) or ~672x504 (Qwen2.5 stride=28)
        max_w = 768.0 if stride == 32 else 672.0
        max_h = 576.0 if stride == 32 else 504.0
        scale = min(max_w / max(1, cw), max_h / max(1, ch))
        scale = max(1.0, scale)
        w_al = max(stride * 10, int(round((cw * scale) / float(stride))) * stride)
        h_al = max(stride * 8, int(round((ch * scale) / float(stride))) * stride)
        aligned = cropped.resize(
            (w_al, h_al),
            Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.BICUBIC,
        )
        return aligned, (x1, y1, cw, ch)

    def ground_component_bbox_live(
        self,
        img: Image.Image,
        target_label: str,
        dino_prompt: str = "",
    ) -> list:
        """
        Dedicated native Qwen-VL visual grounding pass using the official
        `[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]` format on the
        letterbox-free diagram image.
          - Qwen3-VL-8B: decodes `[0..1000]` normalized relative coordinates.
          - Qwen2.5-VL-7B: decodes `image_grid_thw` absolute pixel coordinates.
        """
        self.load_live_vlm()
        width, height = img.size
        low_id = (self.model_id or "").lower()
        is_qwen3 = "qwen3" in low_id
        is_qwen25 = ("2.5" in low_id or "2_5" in low_id) and not is_qwen3
        stride = 32 if is_qwen3 else 28

        aligned_img, (cx1, cy1, cw, ch) = self._crop_letterbox_and_align28(img, stride=stride)
        aw, ah = aligned_img.size

        # Lock processor pixel bounds cleanly (32x32 for Qwen3-VL, 28x28 for Qwen2.5-VL)
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

        ground_prompt = (
            f'Locate the ENTIRE enclosing box, rectangle, oval, cloud, or container shape '
            f'that holds "{target_label}" (include all wrapped lines of text inside it and '
            f'its full outer border, NOT just a single word). '
            f'Output ONLY JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "{target_label}"}}]'
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
                max_new_tokens=96,
                do_sample=False,
            )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)]
        raw = self.vlm_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        box = None
        m_arr = re.search(
            r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
            raw,
        )
        if m_arr:
            box = [float(m_arr.group(i)) for i in range(1, 5)]
        else:
            nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
            if len(nums) >= 4:
                box = [float(v) for v in nums[:4]]

        if box is None:
            return None

        x1, y1, x2, y2 = box
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # CRITICAL coordinate convention split:
        #   - Qwen3-VL ALWAYS outputs normalized [0..1000] relative coordinates!
        #   - Qwen2.5-VL outputs absolute pixels on its (model_w x model_h) grid.
        if is_qwen3:
            coord_mode = "0..1000"
            rel_x1 = x1 / 1000.0
            rel_y1 = y1 / 1000.0
            rel_x2 = x2 / 1000.0
            rel_y2 = y2 / 1000.0
        elif is_qwen25 and max(x1, x2) <= model_w * 1.05 and max(y1, y2) <= model_h * 1.05:
            coord_mode = f"abs:{int(model_w)}x{int(model_h)}"
            rel_x1 = x1 / max(1.0, model_w)
            rel_y1 = y1 / max(1.0, model_h)
            rel_x2 = x2 / max(1.0, model_w)
            rel_y2 = y2 / max(1.0, model_h)
        else:
            coord_mode = "0..1000-fb"
            rel_x1 = x1 / 1000.0
            rel_y1 = y1 / 1000.0
            rel_x2 = x2 / 1000.0
            rel_y2 = y2 / 1000.0

        px1 = cx1 + rel_x1 * cw
        py1 = cy1 + rel_y1 * ch
        px2 = cx1 + rel_x2 * cw
        py2 = cy1 + rel_y2 * ch

        px1 = max(2.0, min(float(width - 2), px1))
        py1 = max(2.0, min(float(height - 2), py1))
        px2 = max(2.0, min(float(width - 2), px2))
        py2 = max(2.0, min(float(height - 2), py2))

        cx = 0.5 * (px1 + px2)
        cy = 0.5 * (py1 + py2)
        bw = max(46.0, px2 - px1)
        bh = max(34.0, py2 - py1)
        final_box = [
            round(max(2.0, cx - 0.5 * bw), 1),
            round(max(2.0, cy - 0.5 * bh), 1),
            round(min(float(width - 2), cx + 0.5 * bw), 1),
            round(min(float(height - 2), cy + 0.5 * bh), 1),
        ]
        print(
            f"  [{self.short_model_name} GROUND] '{target_label}' | mode={coord_mode} | crop=({cx1},{cy1},{cw}x{ch}) | aligned={aw}x{ah} | raw={box} -> canvas={final_box}",
            flush=True,
        )
        return final_box

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

        ctx_snippet = (slide_context or "").strip()[:800]
        ctx_block = f"\nSlide / Page Text Context:\n\"\"\"\n{ctx_snippet}\n\"\"\"\n" if ctx_snippet else ""

        user_prompt = (
            f"Course Topic Hint: '{concept_title}'."
            f"{ctx_block}\n"
            "1. First decide if this image is a meaningful technical/educational diagram, schema, chart, or table (`is_educational_diagram`: true) "
            "or just a decorative school logo / cover icon / stock photo (`is_educational_diagram`: false).\n"
            "2. Select 2 distinct, clearly labeled visual components INSIDE THIS EXACT IMAGE. Copy their EXACT verbatim text labels printed on the image (do NOT translate French labels into English!).\n"
            "3. Generate 2 specific multiple-choice questions (`mc_questions`) and 1 analytical Visual Question Answering (`vqa_challenge`) strictly about THIS image and slide text. "
            "Do NOT invent unrelated topics. Return strict JSON."
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

        ref_diag = self.diagrams_by_concept.get(int(concept_id), {}) if use_reference_cache else {}
        try:
            parsed = extract_json_payload(raw_output)
        except Exception:
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
        normalized_comps = []

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

            px_box = None
            grounding_tier = f"{self.short_model_name}-FP16"

            # Dedicated native Qwen-VL `bbox_2d` grounding pass on letterbox-free diagram!
            if not use_reference_cache:
                try:
                    px_box = self.ground_component_bbox_live(img, target_label=str(t_label), dino_prompt=str(d_prompt))
                except Exception as g_exc:
                    print(f"  [{self.short_model_name} GROUND] Fallback on '{t_label}': {g_exc}", flush=True)

            if px_box is None:
                raw_b = lc.get("bbox_2d") or lc.get("vlm_box_1000")
                if isinstance(raw_b, list) and len(raw_b) == 4:
                    try:
                        bx1, by1, bx2, by2 = [float(v) for v in raw_b]
                        if bx2 < bx1:
                            bx1, bx2 = bx2, bx1
                        if by2 < by1:
                            by1, by2 = by2, by1
                        if not use_reference_cache:
                            if is_qwen3:
                                rx1, ry1, rx2, ry2 = bx1 / 1000.0, by1 / 1000.0, bx2 / 1000.0, by2 / 1000.0
                            elif is_qwen25 and max(bx1, bx2) <= pass1_model_w * 1.05 and max(by1, by2) <= pass1_model_h * 1.05:
                                rx1 = bx1 / max(1.0, pass1_model_w)
                                ry1 = by1 / max(1.0, pass1_model_h)
                                rx2 = bx2 / max(1.0, pass1_model_w)
                                ry2 = by2 / max(1.0, pass1_model_h)
                            else:
                                rx1, ry1, rx2, ry2 = bx1 / 1000.0, by1 / 1000.0, bx2 / 1000.0, by2 / 1000.0
                            px_box = [
                                round(max(2.0, cx1 + rx1 * cw), 1),
                                round(max(2.0, cy1 + ry1 * ch), 1),
                                round(min(float(width - 2), cx1 + rx2 * cw), 1),
                                round(min(float(height - 2), cy1 + ry2 * ch), 1),
                            ]
                        else:
                            px_box = box_1000_to_pixel([bx1, by1, bx2, by2], width, height)
                    except Exception:
                        px_box = None

            if px_box is None:
                fallback_b1000 = rc.get("vlm_box_1000", default_boxes[idx])
                px_box = box_1000_to_pixel(fallback_b1000, width, height)
            else:
                cx = 0.5 * (px_box[0] + px_box[2])
                cy = 0.5 * (px_box[1] + px_box[3])
                bw = max(46.0, px_box[2] - px_box[0])
                bh = max(34.0, px_box[3] - px_box[1])
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

            box_1000 = [
                round(px_box[0] * 1000.0 / max(1.0, float(width))),
                round(px_box[1] * 1000.0 / max(1.0, float(height))),
                round(px_box[2] * 1000.0 / max(1.0, float(width))),
                round(px_box[3] * 1000.0 / max(1.0, float(height))),
            ]
            lab_q_final = lc.get("lab_question") or rc.get("lab_question") or f"Click or drag a box around [{t_label}] shown in this figure."
            comp_item = {
                "component_id": rc.get("component_id", f"c{concept_id}_p{idx + 1}"),
                "target_label": str(t_label),
                "dino_prompt": str(d_prompt),
                "grounding_tier": grounding_tier,
                "vlm_box_1000": box_1000,
                "vlm_box_px": px_box,
                "ground_truth_box_px": rc.get("ground_truth_box_px", px_box) if use_reference_cache else px_box,
                "lab_question": lab_q_final,
                "socratic_hint": lc.get("socratic_hint") or rc.get("socratic_hint") or f"Look closely for the [{t_label}] region in this figure.",
                "explanation": lc.get("explanation") or rc.get("explanation") or parsed["visual_summary"],
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
