"""
Generates local offline 15-second spoken Podcast Hint `.wav` files for the
Computer Vision (CNNs, YOLO & OCR) course (`course_material/vision_yolo_ocr/audio/`).

Uses Windows built-in .NET `System.Speech.Synthesis.SpeechSynthesizer` (zero external
pip packages or cloud API keys required), with a synthesized chime `.wav` fallback.
"""

import json
import math
import os
import re
import struct
import subprocess
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
CURRICULUM_JSON = os.path.join(HERE, 'vision_yolo_ocr', 'course_curriculum.json')
AUDIO_DIR = os.path.join(HERE, 'vision_yolo_ocr', 'audio')


def clean_speech_text(raw_hint: str, title: str) -> str:
    # Strip "[Podcast Clip 00:45 - 01:00]" prefix and quotes for natural spoken narration
    body = re.sub(r'^\[[^\]]+\]\s*', '', raw_hint).strip().strip('"')
    return f"Prof Ada Vision Lab Note on {title}. {body}"


def write_fallback_chime_wav(out_path: str, duration_sec: float = 2.4, sr: int = 16000):
    notes = [523.25, 659.25, 783.99, 1046.50]
    n_samples = int(sr * duration_sec)
    seg = max(1, n_samples // len(notes))
    with wave.open(out_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n_samples):
            freq = notes[min(len(notes) - 1, i // seg)]
            local_t = (i % seg) / sr
            env = math.exp(-3.0 * local_t)
            val = int(12000 * env * math.sin(2.0 * math.pi * freq * (i / sr)))
            frames.extend(struct.pack('<h', max(-32767, min(32767, val))))
        wf.writeframes(bytes(frames))


def synthesize_wav_windows(text: str, out_path: str) -> bool:
    escaped_text = text.replace("'", "''")
    escaped_out = out_path.replace("'", "''")
    ps_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = 1; "
        f"$s.SetOutputToWaveFile('{escaped_out}'); "
        f"$s.Speak('{escaped_text}'); "
        "$s.Dispose();"
    )
    try:
        res = subprocess.run(
            ['powershell', '-NoProfile', '-Command', ps_script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024
    except Exception:
        return False


def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    with open(CURRICULUM_JSON, 'r', encoding='utf-8') as f:
        pkg = json.load(f)

    print("Generating local spoken Podcast Hint WAV clips...")
    for c in pkg['concepts']:
        cid = c['concept_id']
        out_wav = os.path.join(AUDIO_DIR, f'hint_c{cid}.wav')
        spoken = clean_speech_text(c['podcast_hint'], c['short_title'])
        ok = synthesize_wav_windows(spoken, out_wav)
        if not ok:
            write_fallback_chime_wav(out_wav)
            mode = 'fallback-chime'
        else:
            mode = 'spoken-tts'
        kb = os.path.getsize(out_wav) / 1024.0
        print(f"  [C{cid}] {out_wav} ({kb:.1f} KB, {mode})")


if __name__ == '__main__':
    main()
