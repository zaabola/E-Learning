"""Oral coach: live signals + live recommendations.

  python run.py                 # webcam + mic, full stack, live camera window with scores
  python run.py --terminal      # old terminal dashboard instead of the camera window
  python run.py --light         # fallback: MediaPipe + prosody only (no emotion models)
  python run.py --video talk.mp4 --audio talk.mp4   # replay a recording in the live window
  python run.py --video talk.mp4 --out coached.mp4  # VIDEO MODE: write an MP4 with the overlay, tips and sounds
  python run.py --json          # print one JSON event per line instead of the dashboard (for the game)
  python run.py --llm ollama    # who words the tips: auto (default), ollama, claude, template
  python run.py --no-sound      # no game sound effects (--volume 0.3 for quieter ones)
  python run.py --pixel 8       # bigger camera pixels (1 = off); press p in the window to cycle
  python run.py --speak         # read each tip out loud (short); press v in the window to toggle

Every snapshot and coach event is also appended to logs/session-<time>.jsonl.
"""
import argparse
import json
import os
import sys
import threading
import time

from coach.channels import ChannelBuilder
from coach.config import Config
from coach.signals import Signals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--light", action="store_true", help="no HSEmotion / audEERING (CPU-only fallback)")
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-whisper", action="store_true", help="skip speech-to-text (no pace / fillers)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="video file instead of the webcam")
    ap.add_argument("--audio", help="audio (or video) file instead of the mic")
    ap.add_argument("--whisper-model", default=None, help="tiny / base / small / medium")
    ap.add_argument("--language", default=None, help="en, fr ... (default: auto-detect)")
    ap.add_argument("--calibration", type=float, default=None, help="seconds of speech used as your baseline")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--json", action="store_true", help="print JSON lines instead of the dashboard")
    ap.add_argument("--terminal", action="store_true", help="terminal dashboard instead of the camera window")
    ap.add_argument("--no-mirror", action="store_true", help="don't mirror the camera image")
    ap.add_argument("--scale", type=float, default=1.0, help="window zoom, e.g. 1.5")
    ap.add_argument("--llm", default="auto", choices=["auto", "ollama", "claude", "template"],
                    help="who words the tips (auto: Ollama, then Claude, then card wording)")
    ap.add_argument("--ollama-model", default="qwen2.5:3b")
    ap.add_argument("--claude-model", default="claude-opus-5-5")
    ap.add_argument("--tip-language", default="en", help="language of the tips (LLM only), e.g. fr")
    ap.add_argument("--no-coach", action="store_true", help="signals only, no tips")
    ap.add_argument("--no-sound", action="store_true", help="no game sound effects")
    ap.add_argument("--volume", type=float, default=0.6, help="sound effects volume, 0 to 1")
    ap.add_argument("--pixel", type=int, default=1, help="camera pixel size in the window (1 = off)")
    ap.add_argument("--speak", action="store_true", help="read tips out loud (offline voice); v toggles it")
    ap.add_argument("--out", help="video mode: process --video offline and write the annotated MP4 here")
    ap.add_argument("--mirror", action="store_true", help="video mode: mirror the video (off by default)")
    ap.add_argument("--no-landmarks", action="store_true", help="video mode: no face dots on the video")
    args = ap.parse_args()

    cfg = Config()
    cfg.camera_index = args.camera
    if args.whisper_model:
        cfg.whisper_model = args.whisper_model
    if args.language:
        cfg.language = args.language
    if args.calibration is not None:
        cfg.calibration_seconds = args.calibration
    if args.light:
        cfg.hsemotion_model = ""  # FaceSensor falls back to MediaPipe only

    if args.out:  # video mode (coach/offline.py); everything below is the live mode, unchanged
        if not args.video:
            ap.error("--out needs --video (the recording to process)")
        from coach.offline import run_video
        sys.exit(run_video(args, cfg))

    signals = Signals()
    stop = threading.Event()
    sensors = []
    face = None

    if not args.no_face:
        from coach.face import FaceSensor
        face = FaceSensor(cfg, signals, stop)
        if args.video:
            face.cfg.camera_index = args.video
        sensors.append(face)
    if not args.no_voice:
        from coach.voice import FileSource, VoiceSensor
        source = FileSource(args.audio, cfg) if args.audio else None
        sensors.append(VoiceSensor(cfg, signals, stop, source=source,
                                   use_emotion=not args.light, use_whisper=not args.no_whisper))
    for s in sensors:
        s.start()

    os.makedirs("logs", exist_ok=True)
    log_path = time.strftime("logs/session-%Y%m%d-%H%M%S.jsonl")
    log = open(log_path, "a", encoding="utf-8")
    builder = ChannelBuilder(cfg, signals)
    warnings: list[str] = []
    seen_warnings = set()

    recommender = writer = None
    if not args.no_coach:
        from coach.recommend import Coach, Knowledge, TipWriter
        here = os.path.dirname(os.path.abspath(__file__))
        writer = TipWriter(mode=args.llm, ollama_model=args.ollama_model, claude_model=args.claude_model,
                           language=args.tip_language)
        recommender = Coach(cfg, Knowledge(os.path.join(here, "knowledge")), writer,
                            on_tip=lambda tip: builder.restart(tip["restart"], tip["t"]))

    def collect_warnings():
        for s in sensors:
            if getattr(s, "error", None) and s.error not in seen_warnings:
                seen_warnings.add(s.error)
                warnings.append(s.error)
        extra = [writer.warning] if writer is not None and writer.warning else []
        if speaker is not None and speaker.warning:
            extra.append(speaker.warning)
        for w in signals.values("warning", 3600) + extra:
            if w not in seen_warnings:
                seen_warnings.add(w)
                warnings.append(w)

    # game feel: sound effects + XP / pop-ups, driven by the same events (window and terminal views only)
    feel = None
    if not args.json:
        from coach.juice import GameFeel
        from coach.sfx import Sfx
        sfx = Sfx(volume=args.volume, enabled=not args.no_sound)
        if sfx.warning:
            warnings.append(sfx.warning)
        feel = GameFeel(sfx)

    # spoken tips (off unless --speak or v): own thread, mic ignored and sounds ducked while it talks
    speaker = None
    if feel is not None and recommender is not None:
        from coach.speaker import Speaker
        speaker = Speaker(signals, feel.sfx, enabled=args.speak, language=args.tip_language)
        speaker.start()

    start = time.time()

    def take_snapshot():
        """One tick: channels snapshot + whatever the coach decided. Returns (snapshot, coach events)."""
        collect_warnings()
        snap = builder.tick()
        events = recommender.step(snap) if recommender is not None else []
        if speaker is not None:
            for e in events:
                if e["type"] == "tip":
                    speaker.say(e.get("say") or e["text"])
        for item in [snap] + events:
            log.write(json.dumps(item) + "\n")
        log.flush()
        if feel is not None:
            feel.update(snap, events, coach_view())
        return snap, events

    def snapshots():
        while not stop.is_set():
            time.sleep(cfg.tick_seconds)
            yield take_snapshot()
            if args.duration and time.time() - start >= args.duration:
                break

    def coach_view():
        return recommender.view(time.time()) if recommender is not None else None

    try:
        if args.json:
            for snap, events in snapshots():
                for item in [snap] + events:
                    print(json.dumps(item), flush=True)
        elif args.terminal:
            from rich.live import Live
            from coach.display import render
            feel.start()
            with Live(refresh_per_second=4, screen=False) as live:
                for snap, _ in snapshots():
                    live.update(render(snap, warnings, coach_view()))
        else:
            # camera window: draws every frame (~30 fps), snapshot once per tick as before
            from coach.overlay import Window
            win = Window(mirror=not args.no_mirror, scale=args.scale, feel=feel, pixel=args.pixel,
                         speaker=speaker)
            snap, next_tick = None, time.time() + cfg.tick_seconds
            try:
                while not stop.is_set():
                    if time.time() >= next_tick:
                        next_tick += cfg.tick_seconds
                        snap, _ = take_snapshot()
                        if args.duration and time.time() - start >= args.duration:
                            break
                    frame, face_view = face.view() if face is not None else (None, None)
                    if not win.show(frame, face_view, snap, warnings, coach_view()):
                        break
            finally:
                win.close()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        log.close()
        if feel is not None and feel.sfx is not None:
            feel.sfx.close()
        collect_warnings()
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)
        print(f"Session log: {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
