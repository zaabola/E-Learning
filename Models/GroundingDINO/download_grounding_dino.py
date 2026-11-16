"""
Downloader & Offline Checkpoint Saver for Model #3: Grounding DINO
==================================================================
Downloads `IDEA-Research/grounding-dino-tiny` (Open-Vocabulary Vision Transformer)
from Hugging Face with live progress bars and saves it locally into:
  `Models/GroundingDINO/checkpoints/grounding_dino_tiny/`

Run in your terminal:
  .\.venv\Scripts\python.exe Models/GroundingDINO/download_grounding_dino.py
"""

import os
import time
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints", "grounding_dino_tiny")


def main():
    print("=" * 84)
    print("  KNOWLEDGEQUEST MODEL #3 DOWNLOADER: Grounding DINO (Open-Vocabulary ViT)")
    print("=" * 84)
    print(f"  Source Model : {MODEL_ID}")
    print(f"  Target Path  : {SAVE_DIR}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Compute Dev  : {device.upper()}")
    print("-" * 84)

    t0 = time.time()
    print("[1/3] Downloading AutoProcessor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("[2/3] Downloading Grounding DINO Zero-Shot Object Detection Transformer...")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)

    os.makedirs(SAVE_DIR, exist_ok=True)
    processor.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"      Saved local checkpoint to: {SAVE_DIR}")
    print(f"      Total Parameters: {num_params:,} | Elapsed: {time.time() - t0:.1f}s")

    print("[3/3] Running quick zero-shot GPU/CPU verification pass...")
    model = model.to(device).eval()
    img = Image.new("RGB", (320, 240), (24, 32, 48))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 50, 240, 180], fill=(70, 160, 230), outline=(255, 220, 80), width=4)

    t1 = time.perf_counter()
    inputs = processor(images=img, text="diagram box.", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.10,
        text_threshold=0.10,
        target_sizes=[(img.height, img.width)],
    )[0]
    latency_ms = (time.perf_counter() - t1) * 1000.0

    print("-" * 84)
    print(f"  SUCCESS! Grounding DINO verified on {device.upper()} in {latency_ms:.1f} ms.")
    print(f"  Detected boxes in synthetic test: {len(results['boxes'])}")
    print("=" * 84)


if __name__ == "__main__":
    main()
