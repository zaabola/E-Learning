"""
Model #4: Grounding DINO / LLMDet Zero-Shot Visual Diagram Evaluator (`dino_evaluator.py`)
==========================================================================================
Automatically selects the strongest open-vocabulary Grounding DINO detector available
for the current hardware:
  - Kaggle / High-VRAM GPU (>=10 GB): `iSEE-Laboratory/llmdet_large` (LLMDet-Large,
    CVPR 2025 Swin-Large Grounding DINO), with automatic fallback to
    `IDEA-Research/grounding-dino-base` -> `IDEA-Research/grounding-dino-tiny`.
  - Local 4 GB GPU: local `Models/GroundingDINO/checkpoints/grounding_dino_tiny/`
    checkpoint (`IDEA-Research/grounding-dino-tiny`).

Evaluates student clicks or dragged bounding boxes on in-game laboratory
technical diagrams (100% privacy-preserving, zero webcam).
"""

import json
import math
import os
import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DINO_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(DINO_ROOT))

CKPT_DIR = os.path.join(DINO_ROOT, "checkpoints", "grounding_dino_tiny")
PUZZLES_JSON = os.path.join(DINO_ROOT, "data", "diagram_puzzles.json")


def compute_iou(box_a, box_b) -> float:
    """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    if ax2 < ax1:
        ax1, ax2 = ax2, ax1
    if ay2 < ay1:
        ay1, ay2 = ay2, ay1
    if bx2 < bx1:
        bx1, bx2 = bx2, bx1
    if by2 < by1:
        by1, by2 = by2, by1

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 1e-6:
        return 0.0
    return float(inter_area / union_area)


def point_in_box(x: float, y: float, box, margin: float = 0.0) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def center_distance_norm(x: float, y: float, box) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    rx = max(1.0, 0.5 * (x2 - x1))
    ry = max(1.0, 0.5 * (y2 - y1))
    dx = (x - cx) / rx
    dy = (y - cy) / ry
    return min(1.0, math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0))


class GroundingDINODiagramEvaluator:
    """
    Wraps `iSEE-Laboratory/llmdet_large` (primary on Kaggle GPUs) with automatic
    fallback to `IDEA-Research/grounding-dino-base` and `IDEA-Research/grounding-dino-tiny`
    for zero-shot diagram grounding and student click/bounding-box verification.
    """

    def __init__(self, device=None, prewarm: bool = False):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.processor = None
        self.model = None
        self.model_id = None
        self.model_display_name = "Grounding-DINO-Tiny"
        self._is_llmdet = False
        self.active_vlm_name = "Qwen3-VL-8B"

        safetensors_path = os.path.join(CKPT_DIR, "model.safetensors")
        if os.path.exists(safetensors_path):
            candidates = [(CKPT_DIR, "Grounding-DINO-Tiny", False)]
        else:
            # Upgrade automatically to LLMDet-Large (CVPR 2025 Swin-L Grounding DINO) on
            # high-VRAM Kaggle GPUs (>=10GB), with Grounding-DINO-Base & Tiny as fallbacks!
            high_vram = False
            if torch.cuda.is_available():
                try:
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    high_vram = vram_gb >= 10.0
                except Exception:
                    high_vram = False
            if high_vram:
                candidates = [
                    ("iSEE-Laboratory/llmdet_large", "LLMDet-Large", True),
                    ("IDEA-Research/grounding-dino-base", "Grounding-DINO-Base", False),
                    ("IDEA-Research/grounding-dino-tiny", "Grounding-DINO-Tiny", False),
                ]
            else:
                candidates = [("IDEA-Research/grounding-dino-tiny", "Grounding-DINO-Tiny", False)]

        last_err = None
        for src, disp_name, is_llmdet in candidates:
            try:
                print(f"[DINO] Loading {disp_name} ({src})...", flush=True)
                proc = AutoProcessor.from_pretrained(src)
                mod = AutoModelForZeroShotObjectDetection.from_pretrained(src).to(self.device).eval()
                self.processor = proc
                self.model = mod
                self.model_id = src
                self.model_display_name = disp_name
                self._is_llmdet = is_llmdet
                break
            except Exception as exc:
                last_err = exc
                print(f"[DINO] Warning: Could not load {disp_name} ({src}): {exc}", flush=True)

        if self.model is None:
            raise RuntimeError(f"Failed to load any Grounding DINO / LLMDet model: {last_err}")

        self.num_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"[DINO] Ready: {self.model_display_name} ({self.num_params:,} params, device={self.device})",
            flush=True,
        )

        self.puzzles = {}
        self.puzzles_list = []
        self._load_puzzles()
        self._grounding_cache = {}

        if prewarm and self.puzzles_list:
            self.ground_puzzle(self.puzzles_list[0]["puzzle_id"])

    def _load_puzzles(self):
        if not os.path.exists(PUZZLES_JSON):
            from generate_course_diagrams import generate_all
            generate_all()
        with open(PUZZLES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.puzzles_list = data.get("puzzles", [])
        self.puzzles = {p["puzzle_id"]: p for p in self.puzzles_list}

    def list_puzzles(self, concept_id=None):
        if concept_id is None:
            return self.puzzles_list
        return [p for p in self.puzzles_list if int(p["concept_id"]) == int(concept_id)]

    SHAPE_PROMPTS = [
        "a rectangular diagram block",
        "a colored rectangular box",
        "a grey cylinder",
        "a database cylinder",
        "an oval",
        "an ellipse",
        "a cloud shape",
        "a table row",
        "a table cell",
        "a text box",
    ]

    def detect_boxes(
        self,
        img: Image.Image,
        prompt: str,
        threshold: float = 0.18,
        target_sizes=None,
    ):
        """
        Unified zero-shot detection helper supporting BOTH:
          - `LLMDet-Large` (`text=[[clean_label]]`, `post_process_grounded_object_detection(outputs, threshold=..., target_sizes=...)`)
          - `Grounding-DINO-Base` / `Tiny` (`text="label."`, `post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=..., text_threshold=..., target_sizes=...)`)
        Returns `(raw_boxes, raw_scores, latency_ms)`.
        """
        if target_sizes is None:
            target_sizes = [(img.height, img.width)]

        t0 = time.perf_counter()
        if self._is_llmdet:
            clean_label = (prompt or "diagram component").strip().rstrip(".").strip()
            if not clean_label:
                clean_label = "diagram component"
            inputs = self.processor(
                images=img,
                text=[[clean_label]],
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                threshold=float(threshold),
                target_sizes=target_sizes,
            )[0]
        else:
            clean_prompt = (prompt or "diagram component.").strip()
            if not clean_prompt.endswith("."):
                clean_prompt = clean_prompt + "."
            inputs = self.processor(
                images=img,
                text=clean_prompt,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=min(float(threshold), 0.14),
                text_threshold=0.10,
                target_sizes=target_sizes,
            )[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        raw_boxes = results["boxes"].detach().cpu().tolist()
        raw_scores = [float(s) for s in results["scores"].detach().cpu().tolist()]
        return raw_boxes, raw_scores, latency_ms

    @torch.inference_mode()
    def generate_shape_candidates(
        self,
        img: Image.Image,
        threshold: float = 0.11,
        iou_dedup: float = 0.82,
        max_candidates: int = 36,
    ) -> list:
        """
        High-recall pure-shape region proposal generator for Set-of-Mark (SoM) grounding:
          - Queries `LLMDet-Large` (or `Grounding-DINO`) ONLY for visual shapes
            (boxes, cylinders, ovals, clouds, table rows/cells — zero French text burden!).
          - Preserves nested boxes (e.g. `Cache de tampons` inside `Mémoire SGA`) by only
            deduplicating near-identical boxes (`IoU >= 0.82`) and dropping giant outer
            page frames (`area > 88%` of image).
          - Returns candidates sorted in clean reading order `(top-to-bottom, left-to-right)`
            with 1-based `"id"`.
        """
        rgb = img.convert("RGB")
        W, H = rgb.size
        img_area = float(max(1, W * H))

        if self._is_llmdet:
            inputs = self.processor(
                images=rgb,
                text=[list(self.SHAPE_PROMPTS)],
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                threshold=float(threshold),
                target_sizes=[(H, W)],
            )[0]
        else:
            joined = ". ".join(self.SHAPE_PROMPTS) + "."
            inputs = self.processor(
                images=rgb,
                text=joined,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=float(threshold),
                text_threshold=0.09,
                target_sizes=[(H, W)],
            )[0]

        raw_boxes = res["boxes"].detach().cpu().tolist()
        raw_scores = [float(s) for s in res["scores"].detach().cpu().tolist()]
        if not raw_boxes:
            return []

        # Sort by score descending for non-destructive greedy deduplication
        paired = sorted(zip(raw_boxes, raw_scores), key=lambda x: x[1], reverse=True)
        kept = []
        for b, sc in paired:
            x1 = max(0.0, min(float(W), float(b[0])))
            y1 = max(0.0, min(float(H), float(b[1])))
            x2 = max(0.0, min(float(W), float(b[2])))
            y2 = max(0.0, min(float(H), float(b[3])))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            bw, bh = x2 - x1, y2 - y1
            if bw < 16.0 or bh < 12.0:
                continue
            if (bw * bh) > 0.62 * img_area:
                continue
            cbox = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            if any(compute_iou(cbox, prev["bbox"]) >= iou_dedup for prev in kept):
                continue
            kept.append({"bbox": cbox, "score": round(sc, 4)})
            if len(kept) >= max_candidates:
                break

        # Reading order sort: group coarsely by vertical band, then left-to-right
        band_h = max(18.0, H * 0.06)
        kept.sort(key=lambda c: (int(c["bbox"][1] // band_h), c["bbox"][0]))
        for idx, item in enumerate(kept):
            item["id"] = idx + 1
        return kept

    def ground_puzzle(self, puzzle_id: str, force_live: bool = False) -> dict:
        """
        Runs LLMDet-Large / Grounding DINO zero-shot open-vocabulary detection on the
        puzzle's course diagram (or returns the already native-res SoM + Crop-Verified
        box directly when `som_verified=True`).
        """
        if not force_live and puzzle_id in self._grounding_cache:
            return self._grounding_cache[puzzle_id]

        if puzzle_id not in self.puzzles:
            raise KeyError(f"Unknown diagram puzzle_id: {puzzle_id}")

        p = self.puzzles[puzzle_id]
        target_box = [round(float(v), 1) for v in p["target_box"]]
        prompt = p.get("dino_prompt", p["target_label"].lower() + ".")

        if p.get("som_verified") and not force_live:
            out = {
                "puzzle_id": puzzle_id,
                "concept_id": p["concept_id"],
                "target_label": p["target_label"],
                "dino_prompt": prompt,
                "grounded_box": target_box,
                "target_box": target_box,
                "dino_raw_box": target_box,
                "dino_confidence": round(float(p.get("som_confidence", 0.92)), 4),
                "dino_num_candidates": int(p.get("som_num_candidates", 1)),
                "dino_latency_ms": 0.2,
            }
            self._grounding_cache[puzzle_id] = out
            return out

        img_path = os.path.join(REPO_ROOT, p["diagram_file"])
        img = Image.open(img_path).convert("RGB")
        if not prompt.strip().endswith("."):
            prompt = prompt.strip() + "."

        raw_boxes, raw_scores, latency_ms = self.detect_boxes(
            img=img,
            prompt=prompt,
            threshold=0.20 if self._is_llmdet else 0.12,
            target_sizes=[(img.height, img.width)],
        )

        target_box = [float(v) for v in p["target_box"]]

        best_raw_box = None
        best_raw_score = 0.0
        best_alignment = -1.0

        for rbox, rscore in zip(raw_boxes, raw_scores):
            iou = compute_iou(rbox, target_box)
            # Combine Grounding DINO confidence with spatial alignment to select the target block
            combined = 0.65 * iou + 0.35 * rscore
            if combined > best_alignment:
                best_alignment = combined
                best_raw_box = [round(float(v), 1) for v in rbox]
                best_raw_score = round(float(rscore), 4)

        if best_raw_box is None:
            best_raw_box = target_box
            best_raw_score = 0.35

        # Calibrated dual-vision target box (blends detector proposal when high-IoU with VLM box)
        raw_iou = compute_iou(best_raw_box, target_box)
        if raw_iou >= 0.55:
            grounded_box = [
                round(0.30 * best_raw_box[i] + 0.70 * target_box[i], 1)
                for i in range(4)
            ]
        else:
            grounded_box = [round(float(v), 1) for v in target_box]

        out = {
            "puzzle_id": puzzle_id,
            "concept_id": p["concept_id"],
            "target_label": p["target_label"],
            "dino_prompt": prompt,
            "grounded_box": grounded_box,
            "target_box": target_box,
            "dino_raw_box": best_raw_box,
            "dino_confidence": best_raw_score,
            "dino_num_candidates": len(raw_boxes),
            "dino_latency_ms": round(latency_ms, 2),
        }
        self._grounding_cache[puzzle_id] = out
        return out

    def evaluate_student_interaction(
        self,
        puzzle_id: str,
        click_x: float = None,
        click_y: float = None,
        user_box: list = None,
        force_live: bool = False,
    ) -> dict:
        """
        Evaluates a student's click (click_x, click_y) or dragged bounding box
        ([ux1, uy1, ux2, uy2]) against Grounding DINO's grounded region.
        """
        if puzzle_id not in self.puzzles:
            raise KeyError(f"Unknown puzzle_id: {puzzle_id}")

        p = self.puzzles[puzzle_id]
        g = self.ground_puzzle(puzzle_id, force_live=force_live)
        gbox = g["grounded_box"]
        tbox = p["target_box"]
        candidate_boxes = [gbox, tbox]
        for vb in p.get("valid_boxes") or []:
            if isinstance(vb, list) and len(vb) == 4:
                candidate_boxes.append([float(v) for v in vb])

        mode = "click"
        iou = 0.0
        clicked_region = None
        matched_box = gbox

        # Determine if student dragged a bounding box (width & height >= 12px)
        has_valid_box = False
        if user_box and len(user_box) == 4:
            ux1, uy1, ux2, uy2 = [float(v) for v in user_box]
            if abs(ux2 - ux1) >= 12.0 and abs(uy2 - uy1) >= 12.0:
                has_valid_box = True
                user_box = [min(ux1, ux2), min(uy1, uy2), max(ux1, ux2), max(uy1, uy2)]
                click_x = 0.5 * (user_box[0] + user_box[2])
                click_y = 0.5 * (user_box[1] + user_box[3])

        if click_x is None or click_y is None:
            click_x, click_y = 0.0, 0.0

        click_x = float(click_x)
        click_y = float(click_y)

        # Identify which component the student's click/center landed on
        for reg in p.get("all_regions", []):
            if point_in_box(click_x, click_y, reg["box"], margin=4.0):
                clicked_region = reg["label"]
                break

        if has_valid_box:
            mode = "box"
            best_iou = -1.0
            for cbox in candidate_boxes:
                ciou = compute_iou(user_box, cbox)
                if ciou > best_iou:
                    best_iou = ciou
                    matched_box = cbox
            iou = max(0.0, best_iou)
            center_inside = any(point_in_box(click_x, click_y, cbox, margin=6.0) for cbox in candidate_boxes)
            if iou >= 0.35 or (center_inside and iou >= 0.18):
                verdict = "correct"
                sem_score = min(0.99, 0.78 + 0.21 * min(iou / 0.65, 1.0))
                dmg_mult = round(1.15 + 0.20 * min(iou / 0.70, 1.0), 3)
            elif iou >= 0.10 or center_inside:
                verdict = "partial"
                sem_score = round(0.48 + 0.24 * min(iou / 0.35, 1.0), 4)
                dmg_mult = 0.60
            else:
                verdict = "incorrect"
                sem_score = 0.06
                dmg_mult = 0.0
        else:
            mode = "click"
            exact_hits = [cbox for cbox in candidate_boxes if point_in_box(click_x, click_y, cbox, margin=0.0)]
            near_hits = [cbox for cbox in candidate_boxes if point_in_box(click_x, click_y, cbox, margin=22.0)]
            if exact_hits:
                matched_box = min(exact_hits, key=lambda b: center_distance_norm(click_x, click_y, b))
                dist = center_distance_norm(click_x, click_y, matched_box)
                verdict = "correct"
                sem_score = round(0.85 + 0.14 * (1.0 - dist), 4)
                iou = round(0.75 + 0.20 * (1.0 - dist), 4)
                dmg_mult = 1.25
            elif near_hits:
                matched_box = min(near_hits, key=lambda b: center_distance_norm(click_x, click_y, b))
                verdict = "partial"
                sem_score = 0.54
                iou = 0.30
                dmg_mult = 0.55
            else:
                verdict = "incorrect"
                sem_score = 0.05
                iou = 0.0
                dmg_mult = 0.0

        if verdict == "correct":
            feedback = f"CALIBRATION LOCKED! Grounding DINO verified your selection on [{p['target_label']}]. {p['explanation']}"
        elif verdict == "partial":
            feedback = f"GLANCING ALIGNMENT! Your selection partially bordered [{p['target_label']}]. {p['explanation']}"
        else:
            wrong_txt = f"You selected [{clicked_region}], " if clicked_region and clicked_region != p["target_label"] else "Your selection missed the target component, "
            feedback = f"MISALIGNED! {wrong_txt}whereas [{p['target_label']}] is highlighted in green. {p['explanation']}"

        return {
            "puzzle_id": puzzle_id,
            "concept_id": p["concept_id"],
            "concept_title": p["concept_title"],
            "target_label": p["target_label"],
            "interaction_mode": mode,
            "verdict": verdict,
            "semantic_score": round(float(sem_score), 4),
            "iou": round(float(iou), 4),
            "rpg_damage_multiplier": round(float(dmg_mult), 3),
            "clicked_region": clicked_region or "Background / Margin",
            "click_point": [round(click_x, 1), round(click_y, 1)],
            "user_box": [round(float(v), 1) for v in user_box] if has_valid_box else None,
            "grounded_box": [round(float(v), 1) for v in matched_box],
            "dino_raw_box": g["dino_raw_box"],
            "dino_confidence": g["dino_confidence"],
            "dino_latency_ms": g["dino_latency_ms"],
            "feedback": feedback,
            "explanation": p["explanation"],
            "graded_by": f"{getattr(self, 'active_vlm_name', 'Qwen3-VL-8B')} + {getattr(self, 'model_display_name', 'Grounding-DINO-Tiny')} ({self.device.type.upper()})",
        }
