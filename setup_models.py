"""Download the models once (about 2 GB in total) so the first live run starts fast.

  python setup_models.py           # everything
  python setup_models.py --light   # only what --light mode needs
"""
import argparse
import os
import urllib.request

FACE_LANDMARKER = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                   "face_landmarker/float16/1/face_landmarker.task")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--whisper-model", default="small")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    path = "models/face_landmarker.task"
    if not os.path.exists(path):
        print("MediaPipe face landmarker ...")
        urllib.request.urlretrieve(FACE_LANDMARKER, path)

    if not args.light:
        print("HSEmotion face emotions ...")
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        HSEmotionRecognizer("enet_b0_8_va_mtl")
        print("audEERING voice emotion (~1.3 GB) ...")
        from coach.config import Config
        from coach.voice import load_audeering
        load_audeering(Config())

    print(f"Whisper '{args.whisper_model}' ...")
    from faster_whisper import WhisperModel
    WhisperModel(args.whisper_model, device="cpu", compute_type="int8")

    print("Silero VAD ...")
    from silero_vad import load_silero_vad
    load_silero_vad()
    print("All models ready.")


if __name__ == "__main__":
    main()
