"""Oral coach: live signals + live recommendations.

  python run.py                       # webcam + mic, live camera window with scores and tips
  python run.py --whisper-model base  # lighter speech-to-text if "small" is too slow
  python run.py --speak               # read each tip out loud from the start (v in the window toggles it)

Keys in the window: q / Esc quit, s sound on/off, v spoken tips on/off.

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
from coach.face import FaceSensor
from coach.juice import GameFeel
from coach.overlay import Window
from coach.recommend import Coach, Knowledge, TipWriter
from coach.sfx import Sfx
from coach.signals import Signals
from coach.speaker import Speaker
from coach.voice import VoiceSensor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whisper-model", default=None, help="tiny / base / small / medium")
    ap.add_argument("--speak", action="store_true", help="read tips out loud (offline voice); v toggles it")
    args = ap.parse_args()

    cfg = Config()
    if args.whisper_model:
        cfg.whisper_model = args.whisper_model

    signals = Signals()
    stop = threading.Event()
    face = FaceSensor(cfg, signals, stop)
    sensors = [face, VoiceSensor(cfg, signals, stop)]
    for s in sensors:
        s.start()

    os.makedirs("logs", exist_ok=True)
    log_path = time.strftime("logs/session-%Y%m%d-%H%M%S.jsonl")
    log = open(log_path, "a", encoding="utf-8")
    builder = ChannelBuilder(cfg, signals)
    warnings: list[str] = []
    seen_warnings = set()

    # tips are worded by Ollama if it runs, else Claude (ANTHROPIC_API_KEY), else the card's own wording
    here = os.path.dirname(os.path.abspath(__file__))
    writer = TipWriter()
    recommender = Coach(cfg, Knowledge(os.path.join(here, "knowledge")), writer,
                        on_tip=lambda tip: builder.restart(tip["restart"], tip["t"]))

    def collect_warnings():
        for s in sensors:
            if getattr(s, "error", None) and s.error not in seen_warnings:
                seen_warnings.add(s.error)
                warnings.append(s.error)
        extra = [writer.warning] if writer.warning else []
        if speaker.warning:
            extra.append(speaker.warning)
        for w in signals.values("warning", 3600) + extra:
            if w not in seen_warnings:
                seen_warnings.add(w)
                warnings.append(w)

    # game feel: sound effects + XP / pop-ups, driven by the same events
    sfx = Sfx()
    if sfx.warning:
        warnings.append(sfx.warning)
    feel = GameFeel(sfx)

    # spoken tips (off unless --speak or v): own thread, mic ignored and sounds ducked while it talks
    speaker = Speaker(signals, feel.sfx, enabled=args.speak)
    speaker.start()

    def take_snapshot():
        """One tick: channels snapshot + whatever the coach decided. Returns (snapshot, coach events)."""
        collect_warnings()
        snap = builder.tick()
        events = recommender.step(snap)
        for e in events:
            if e["type"] == "tip":
                speaker.say(e.get("say") or e["text"])
        for item in [snap] + events:
            log.write(json.dumps(item) + "\n")
        log.flush()
        feel.update(snap, events, coach_view())
        return snap, events

    def coach_view():
        return recommender.view(time.time())

    try:
        # camera window: draws every frame (~30 fps), snapshot once per tick
        win = Window(feel=feel, speaker=speaker)
        snap, next_tick = None, time.time() + cfg.tick_seconds
        try:
            while not stop.is_set():
                if time.time() >= next_tick:
                    next_tick += cfg.tick_seconds
                    snap, _ = take_snapshot()
                frame, face_view = face.view()
                if not win.show(frame, face_view, snap, warnings, coach_view()):
                    break
        finally:
            win.close()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        log.close()
        feel.sfx.close()
        collect_warnings()
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)
        print(f"Session log: {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
