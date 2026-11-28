"""
Turnkey Kaggle 2x T4 GPU Launcher for the Full RPG + 7-Model AI + Oral Coach Engine
(`kaggle_run_game.py`)
===================================================================================
Runs the complete browser RPG + all 7 AI models (`Qwen3-VL-8B-Instruct`
with `Qwen2.5-VL-7B-Instruct` fallback, `LLMDet-Large` with
`Grounding-DINO-Base/Tiny` fallback, `DeBERTa-v3-Small`, `Final-DKT-GRU`,
`Double-DQN`, `DeepSeek-R1-Distill-Qwen-7B`, `Meta-Llama-3.1-8B-Instruct + QLoRA`)
AND the live GPU **Oral Presentation Coach** (MediaPipe + HSEmotion + Silero VAD +
audEERING FP16 + faster-whisper FP16 + Stage-4 GPU TipWriter) on Kaggle GPUs,
opening a public HTTPS URL (`https://....trycloudflare.com`) for browser play.

Usage inside a Kaggle GPU Notebook (Accelerator: GPU T4 x2, Internet: ON):
  !git pull origin main
  !pip install -q -U pymupdf python-pptx python-docx accelerate bitsandbytes peft qwen-vl-utils transformers
  !pip install -q -r Models/OralCoach/requirements_kaggle.txt
  !python Models/OralCoach/setup_models.py --whisper-model base
  !python kaggle_run_game.py
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

# Prevent CUDA VRAM fragmentation on Kaggle Tesla T4 GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from server import GameServerState, KnowledgeQuestHTTPHandler


def _start_cloudflare_tunnel(port: int = 8000):
    """
    Downloads the standalone `cloudflared` Linux x64 binary on Kaggle and creates
    a free instant HTTPS tunnel (`https://....trycloudflare.com`) with zero signup.
    """
    bin_path = "/tmp/cloudflared"
    if not os.path.exists(bin_path):
        print("  [Tunnel] Downloading cloudflared binary for Kaggle...")
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        urllib.request.urlretrieve(url, bin_path)
        os.chmod(bin_path, 0o755)

    proc = subprocess.Popen(
        [bin_path, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    public_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    def _reader():
        nonlocal public_url
        for line in proc.stdout:
            m = url_pattern.search(line)
            if m and not public_url:
                public_url = m.group(0)
                print("\n" + "=" * 88)
                print("  LIVE RPG URL (100% POWERED BY KAGGLE 2x T4 GPUs — CLICK TO PLAY IN BROWSER):")
                print(f"  >>>  {public_url}  <<<")
                print("=" * 88)
                print("  How to test PDF / PPTX / Word (.docx) ingestion in the game:")
                print("    1. Click the link above to open the RPG in your browser.")
                print("    2. Press ENTER -> select NEW COURSE (or JOURNAL -> Tab [3]).")
                print("    3. Select any .pdf, .pptx, or .docx file from your computer.")
                print("    4. Click INGEST DOCUMENT — Kaggle's Qwen3-VL-8B + LLMDet-Large")
                print("       will extract the photos & concepts and launch the Lab Monitor!")
                print("    5. Walk up to the PC desk in YOUR ROOM (top-left) and press Z")
                print("       to launch the live GPU Oral Presentation Coach in your browser!")
                print("=" * 88 + "\n", flush=True)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return proc


def main():
    parser = argparse.ArgumentParser(description="Run full 7-Model AI + RPG Server on Kaggle 2x T4 GPUs")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="Hugging Face VLM ID (fallback: Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--reasoner-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", help="Model #6 Pedagogical Reasoner LLM")
    parser.add_argument("--stylizer-model", default="NousResearch/Meta-Llama-3.1-8B-Instruct", help="Model #7 Persona Stylizer LLM (+ auto QLoRA)")
    parser.add_argument("--no-live-vlm", action="store_true", help="Skip loading live VLM weights")
    parser.add_argument("--no-live-llm", action="store_true", help="Skip loading live Stage 4 Reasoner + Stylizer LLMs")
    args = parser.parse_args()

    print("=" * 88)
    print("  BOOTING FULL AI + ORAL COACH RPG SERVER ON KAGGLE GPUs")
    print("=" * 88)
    state = GameServerState(
        default_course="vision_yolo_ocr",
        default_profile="average",
        vlm_mode="hybrid" if args.no_live_vlm else "live",
        vlm_model_id=args.model,
        llm_mode="fallback" if args.no_live_llm else "live",
        reasoner_model_id=args.reasoner_model,
        stylizer_model_id=args.stylizer_model,
    )
    KnowledgeQuestHTTPHandler.server_state = state

    eng = state.engine
    llm_r = eng.llm_info["reasoner"]
    llm_s = eng.llm_info["stylizer"]
    print(f"  [Model #1] {eng.dkt_info['name']:<26} | AUC={eng.dkt_info['auc']:.4f}")
    print(f"  [Model #2] {eng.rl_info['name']:<26} | Reward={eng.rl_info['reward']:.2f}")
    print(f"  [Model #3] {eng.grader_info['name']:<26} | Device={eng.grader_info['device']}")
    print(f"  [Model #4] {eng.dino_info['name']:<26} | Device={eng.dino_info['device']}")
    print(f"  [Model #5] {eng.vlm_info['name']:<26} | Mode={eng.vlm_info['mode'].upper()}")
    print(f"  [Model #6] {llm_r['name']:<26} | Live={llm_r['live']} ({llm_r['model_id']})")
    print(f"  [Model #7] {llm_s['name']:<26} | Live={llm_s['live']} | QLoRA={llm_s['qlora_active']}")
    if state.oral_coach is not None:
        oc = state.oral_coach.get_capabilities()
        oc_dev = oc.get("best_cuda_device") or "cpu"
        oc_vram = f"{oc['free_cuda_gib']:.2f}GiB free" if oc.get("free_cuda_gib") else "CPU mode"
        oc_tip = "Stage4 GPU LLM" if oc.get("stage4_llm_tip_writer") else "Template/Ollama"
        miss = oc.get("missing_packages") or []
        print(f"  [Model #8] {'Oral Presentation Coach':<26} | Target={oc_dev} ({oc_vram}) | TipWriter={oc_tip}")
        if miss:
            print(f"             Notice: Missing Kaggle pkgs {miss}")
            print(f"             Run: {oc.get('kaggle_install_cmd')}")
        elif not oc.get("face_model_ready"):
            print("             Tip: Cache models first -> !python Models/OralCoach/setup_models.py --whisper-model base")
    print("-" * 88)
    print("  Tip: Train custom RPG QLoRA adapter -> !python -u Models/LLM/scripts/train_persona_qlora.py")
    print("  Tip: Run Stage 4 LLM benchmark      -> !python -u Models/LLM/scripts/evaluate_stage4_llms.py")
    print("-" * 88)

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), KnowledgeQuestHTTPHandler)
    tunnel_proc = _start_cloudflare_tunnel(port=args.port)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Kaggle Game Server...")
        httpd.server_close()
        if tunnel_proc:
            tunnel_proc.terminate()


if __name__ == "__main__":
    main()
