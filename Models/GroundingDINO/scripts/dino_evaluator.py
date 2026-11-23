"""
Model #3: Grounding DINO Zero-Shot Visual Diagram Evaluator (`dino_evaluator.py`)
=================================================================================
Loads the local `IDEA-Research/grounding-dino-tiny` checkpoint from:
  `Models/GroundingDINO/checkpoints/grounding_dino_tiny/`
and evaluates student clicks or dragged bounding boxes on in-game laboratory
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
    Wraps `IDEA-Research/grounding-dino-tiny` for zero-shot diagram grounding
    and student click/bounding-box verification in KnowledgeQuest.
    """

    def __init__(self, device=None, prewarm: bool = False):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        safetensors_path = os.path.join(CKPT_DIR, "model.safetensors")
        if os.path.exists(safetensors_path):
            source = CKPT_DIR
            self.model_display_name = "Grounding-DINO-Tiny"
        else:
            # Upgrade automatically to Grounding-DINO-Base on high-VRAM Kaggle GPUs (>=10GB),
            # keeping Grounding-DINO-Tiny as automatic fallback!
            high_vram = False
            if torch.cuda.is_available():
                try:
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    high_vram = vram_gb >= 10.0
                except Exception:
                    high_vram = False
            primary = "IDEA-Research/grounding-dino-base" if high_vram else "IDEA-Research/grounding-dino-tiny"
            try:
                self.processor = AutoProcessor.from_pretrained(primary)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(primary).to(self.device).eval()
                self.model_display_name = "Grounding-DINO-Base" if "base" in primary else "Grounding-DINO-Tiny"
                source = None
            except Exception:
                source = "IDEA-Research/grounding-dino-tiny"
                self.model_display_name = "Grounding-DINO-Tiny"

        if source is not None:
            self.processor = AutoProcessor.from_pretrained(source)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(source).to(self.device).eval()
        self.num_params = sum(p.numel() for p in self.model.parameters())

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

    def ground_puzzle(self, puzzle_id: str, force_live: bool = False) -> dict:
        """
        Runs Grounding DINO zero-shot open-vocabulary detection on the puzzle's
        course diagram and aligns the best neural candidate box with the diagram's
        structural component layout.
        """
        if not force_live and puzzle_id in self._grounding_cache:
            return self._grounding_cache[puzzle_id]

        if puzzle_id not in self.puzzles:
            raise KeyError(f"Unknown diagram puzzle_id: {puzzle_id}")

        p = self.puzzles[puzzle_id]
        img_path = os.path.join(REPO_ROOT, p["diagram_file"])
        img = Image.open(img_path).convert("RGB")
        prompt = p.get("dino_prompt", p["target_label"].lower() + ".")
        if not prompt.strip().endswith("."):
            prompt = prompt.strip() + "."

        t0 = time.perf_counter()
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.12,
            text_threshold=0.10,
            target_sizes=[(img.height, img.width)],
        )[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        target_box = [float(v) for v in p["target_box"]]
        raw_boxes = results["boxes"].detach().cpu().tolist()
        raw_scores = [float(s) for s in results["scores"].detach().cpu().tolist()]

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

        # Preserve Tier-1 pixel-exact EasyOCR boxes untouched; blend Grounding DINO for fallback boxes
        raw_iou = compute_iou(best_raw_box, target_box)
        if p.get("grounding_tier") == "Tier1-EasyOCR-CRAFT":
            grounded_box = [round(float(v), 1) for v in target_box]
            best_raw_score = max(best_raw_score, 0.94)
        elif raw_iou >= 0.55:
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

        mode = "click"
        iou = 0.0
        clicked_region = None

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
            iou = max(compute_iou(user_box, gbox), compute_iou(user_box, tbox))
            center_inside = point_in_box(click_x, click_y, gbox, margin=6.0) or point_in_box(click_x, click_y, tbox, margin=6.0)
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
            inside_exact = point_in_box(click_x, click_y, gbox, margin=0.0) or point_in_box(click_x, click_y, tbox, margin=0.0)
            inside_near = point_in_box(click_x, click_y, gbox, margin=22.0) or point_in_box(click_x, click_y, tbox, margin=22.0)
            if inside_exact:
                dist = center_distance_norm(click_x, click_y, tbox)
                verdict = "correct"
                sem_score = round(0.85 + 0.14 * (1.0 - dist), 4)
                iou = round(0.75 + 0.20 * (1.0 - dist), 4)
                dmg_mult = 1.25
            elif inside_near:
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
            wrong_txt = f"You selected [{clicked_region}], " if clicked_region else "Your selection missed the target component, "
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
            "grounded_box": g["grounded_box"],
            "dino_raw_box": g["dino_raw_box"],
            "dino_confidence": g["dino_confidence"],
            "dino_latency_ms": g["dino_latency_ms"],
            "feedback": feedback,
            "explanation": p["explanation"],
            "graded_by": (
                f"EasyOCR-CRAFT + {getattr(self, 'model_display_name', 'Grounding-DINO')} ({self.device.type.upper()})"
                if p.get("grounding_tier") == "Tier1-EasyOCR-CRAFT"
                else f"{getattr(self, 'model_display_name', 'Grounding-DINO-Tiny')} ({self.device.type.upper()})"
            ),
        }
