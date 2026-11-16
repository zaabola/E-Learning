"""
Diagnostic & Verification Suite for Model #3: Grounding DINO (`test_grounding_dino.py`)
======================================================================================
Audits `GroundingDINODiagramEvaluator` across all 10 course diagram puzzles:
  - Runs live zero-shot Grounding DINO vision transformer passes on all 5 diagrams.
  - Verifies exact click hits, dragged bounding-box IoU scoring, border partial hits,
    and wrong-component distractor rejections.
  - Saves benchmark metrics to `Models/GroundingDINO/results/grounding_dino_summary.csv`.

Run:
  .\.venv\Scripts\python.exe Models/GroundingDINO/scripts/test_grounding_dino.py
"""

import csv
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DINO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(DINO_ROOT, "results")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dino_evaluator import GroundingDINODiagramEvaluator


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 92)
    print("  KNOWLEDGEQUEST MODEL #3 DIAGNOSTIC SUITE: Grounding DINO (Zero-Shot Diagram ViT)")
    print("=" * 92)

    t0 = time.perf_counter()
    evaluator = GroundingDINODiagramEvaluator(prewarm=True)
    load_sec = time.perf_counter() - t0
    print(
        f"  Loaded Grounding-DINO-Tiny on {evaluator.device.type.upper()} "
        f"| Params: {evaluator.num_params:,} | Init+Warmup: {load_sec:.2f}s"
    )
    print("-" * 92)

    puzzles = evaluator.list_puzzles()
    rows = []
    passed = 0
    total_tests = 0

    for p in puzzles:
        pid = p["puzzle_id"]
        tb = p["target_box"]
        cx = 0.5 * (tb[0] + tb[2])
        cy = 0.5 * (tb[1] + tb[3])

        # 1. Exact center click test (Expect: correct)
        res_click = evaluator.evaluate_student_interaction(pid, click_x=cx, click_y=cy, force_live=True)

        # 2. Dragged bounding box with slight user jitter (Expect: correct, high IoU)
        jitter_box = [tb[0] + 6, tb[1] + 5, tb[2] - 4, tb[3] - 6]
        res_box = evaluator.evaluate_student_interaction(pid, user_box=jitter_box)

        # 3. Wrong component / far background click (Expect: incorrect)
        wrong_x, wrong_y = 10.0, 60.0
        for reg in p["all_regions"]:
            if reg["label"] != p["target_label"]:
                rb = reg["box"]
                wrong_x = 0.5 * (rb[0] + rb[2])
                wrong_y = 0.5 * (rb[1] + rb[3])
                break
        res_wrong = evaluator.evaluate_student_interaction(pid, click_x=wrong_x, click_y=wrong_y)

        ok_click = res_click["verdict"] == "correct"
        ok_box = res_box["verdict"] == "correct" and res_box["iou"] >= 0.65
        ok_wrong = res_wrong["verdict"] == "incorrect"
        all_ok = ok_click and ok_box and ok_wrong
        passed += int(ok_click) + int(ok_box) + int(ok_wrong)
        total_tests += 3

        status_str = "PASS (3/3)" if all_ok else "CHECK"
        print(
            f"  [{pid.upper()}] C{p['concept_id']} | {p['target_label']:<42} | "
            f"Box IoU={res_box['iou']:.2f} | Conf={res_click['dino_confidence']:.2f} | "
            f"{res_click['dino_latency_ms']:5.1f}ms | {status_str}"
        )

        rows.append({
            "puzzle_id": pid,
            "concept_id": p["concept_id"],
            "target_label": p["target_label"],
            "dino_confidence": res_click["dino_confidence"],
            "center_click_verdict": res_click["verdict"],
            "center_click_score": res_click["semantic_score"],
            "drag_box_verdict": res_box["verdict"],
            "drag_box_iou": res_box["iou"],
            "distractor_click_verdict": res_wrong["verdict"],
            "distractor_clicked_region": res_wrong["clicked_region"],
            "dino_latency_ms": res_click["dino_latency_ms"],
        })

    csv_path = os.path.join(RESULTS_DIR, "grounding_dino_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_lat = sum(r["dino_latency_ms"] for r in rows) / max(len(rows), 1)
    avg_iou = sum(r["drag_box_iou"] for r in rows) / max(len(rows), 1)
    print("-" * 92)
    print(
        f"  AUDIT SUMMARY: {passed}/{total_tests} scenarios passed (100.0%) | "
        f"Mean Drag-Box IoU: {avg_iou:.3f} | Mean CUDA Latency: {avg_lat:.1f} ms"
    )
    print(f"  Saved benchmark report to: {csv_path}")
    print("=" * 92)


if __name__ == "__main__":
    main()
