"""
Kaggle 2x T4 GPU Runner & Live FastAPI Worker for Qwen-VL-7B
(`Models/VLM/kaggle_vlm_runner.py`)
============================================================
Run this script inside a Kaggle GPU Notebook (Accelerator: GPU T4 x2)
after cloning the repository:

  !git clone -b interactive_learning_RPG https://github.com/zaabola/E-Learning.git
  %cd E-Learning
  !pip install -q accelerate bitsandbytes qwen-vl-utils transformers
  !python Models/VLM/kaggle_vlm_runner.py --model Qwen/Qwen2.5-VL-7B-Instruct

Options:
  --model       : Hugging Face VLM ID (default: Qwen/Qwen2.5-VL-7B-Instruct, or Qwen/Qwen2-VL-7B-Instruct)
  --slides-dir  : Optional folder of custom lecture slides/diagrams (.png/.jpg) to ingest
  --save-json   : Overwrite/update `Models/VLM/data/vlm_extracted_diagrams.json` with live 7B outputs
  --serve       : Start a FastAPI HTTP endpoint on port 8000 (pair with ngrok/localtunnel)
"""

import argparse
import glob
import json
import os
import sys
import torch

VLM_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.join(VLM_ROOT, "scripts")
MODELS_ROOT = os.path.dirname(VLM_ROOT)
REPO_ROOT = os.path.dirname(MODELS_ROOT)

for _p in (
    SCRIPT_DIR,
    os.path.join(MODELS_ROOT, "GroundingDINO", "scripts"),
    os.path.join(MODELS_ROOT, "AnswerEvaluator", "scripts"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vlm_diagram_engine import DEFAULT_MODEL_ID, VLM_DATA_JSON, QwenVLDiagramEngine


def print_gpu_topology():
    print("=" * 88)
    print("  KAGGLE GPU TOPOLOGY CHECK (Qwen-VL-7B Multimodal Ingestion Node)")
    print("=" * 88)
    if not torch.cuda.is_available():
        print("  WARNING: CUDA is not enabled! In Kaggle, set Accelerator -> GPU T4 x2.")
        return
    n_gpus = torch.cuda.device_count()
    total_vram = 0.0
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / (1024 ** 3)
        total_vram += gb
        print(f"  GPU #{i} : {props.name} ({gb:.2f} GB VRAM)")
    print(f"  Total Available GPU VRAM : {total_vram:.2f} GB across {n_gpus} GPU(s)")
    print("=" * 88)


def run_kaggle_ingestion(
    model_id: str = DEFAULT_MODEL_ID,
    slides_dir: str = None,
    save_json: bool = False,
):
    print_gpu_topology()
    engine = QwenVLDiagramEngine(
        model_id=model_id,
        mode="live",
        load_in_4bit=True,
    )

    extracted_diagrams = []
    if slides_dir:
        abs_dir = slides_dir if os.path.isabs(slides_dir) else os.path.join(REPO_ROOT, slides_dir)
        image_files = sorted(
            glob.glob(os.path.join(abs_dir, "*.png")) + glob.glob(os.path.join(abs_dir, "*.jpg"))
        )
        print(f"\n[KAGGLE VLM] Ingesting {len(image_files)} custom slide/diagram images from {abs_dir}...")
        for idx, img_path in enumerate(image_files):
            title = os.path.splitext(os.path.basename(img_path))[0].replace("_", " ").title()
            res = engine.comprehend_diagram_live(img_path, concept_id=idx, concept_title=title)
            extracted_diagrams.append(res)
            print(
                f"  -> [{idx}] {res['diagram_id']} | Concept: {res.get('concept_title')} "
                f"| Components: {len(res.get('components', []))} | Latency: {res['vlm_latency_ms']:.1f} ms"
            )
    else:
        print(f"\n[KAGGLE VLM] Running live {model_id} comprehension on all 5 course diagrams...")
        for diag in engine.diagrams_list:
            c_id = int(diag["concept_id"])
            res = engine.comprehend_diagram_live(
                image_path=diag["diagram_file"],
                concept_id=c_id,
                concept_title=diag["concept_title"],
            )
            # Preserve ground_truth_box_px if available for IoU evaluation
            for comp_live, comp_ref in zip(res.get("components", []), diag.get("components", [])):
                if "ground_truth_box_px" in comp_ref:
                    comp_live["ground_truth_box_px"] = comp_ref["ground_truth_box_px"]
            extracted_diagrams.append(res)
            print(
                f"  -> [C{c_id}] {res['diagram_id']} | Components: {len(res.get('components', []))} "
                f"| VQA Question: {res.get('vqa_challenge', {}).get('question', '')[:55]}... "
                f"({res['vlm_latency_ms']:.1f} ms)"
            )

    if save_json and extracted_diagrams:
        payload = {
            "model_id": model_id,
            "course_id": "custom_ingested" if slides_dir else "vision_yolo_ocr",
            "schema_version": "1.0",
            "diagrams": extracted_diagrams,
        }
        with open(VLM_DATA_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[KAGGLE VLM] Saved {len(extracted_diagrams)} live VLM extractions to: {VLM_DATA_JSON}")

    return engine, extracted_diagrams


def serve_fastapi(engine: QwenVLDiagramEngine, port: int = 8000):
    """Optional lightweight HTTP server so local laptop can query Kaggle VLM via ngrok."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class VLMHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/vlm/comprehend":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                out = engine.comprehend_diagram(
                    diagram_id_or_concept=body.get("diagram_id", 0),
                    image_path=body.get("image_path"),
                    force_live=True,
                )
                resp = json.dumps(out).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            else:
                self.send_response(404)
                self.end_headers()

    print(f"\n[KAGGLE VLM SERVER] Listening on http://0.0.0.0:{port}/vlm/comprehend ...")
    HTTPServer(("0.0.0.0", port), VLMHandler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle 2x T4 Runner for Qwen-VL-7B")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face VLM model ID")
    parser.add_argument("--slides-dir", type=str, default=None, help="Directory of custom slides/diagrams")
    parser.add_argument("--save-json", action="store_true", help="Save live VLM extractions to JSON")
    parser.add_argument("--serve", action="store_true", help="Run HTTP worker for remote queries")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port")
    args = parser.parse_args()

    vlm_engine, _ = run_kaggle_ingestion(
        model_id=args.model,
        slides_dir=args.slides_dir,
        save_json=args.save_json,
    )
    if args.serve:
        serve_fastapi(vlm_engine, port=args.port)
