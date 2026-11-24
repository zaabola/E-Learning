"""Retro game sound effects, synthesised in code (square waves, triangle, noise): no sound files to download.

    sfx = Sfx(volume=0.6)
    sfx.play("worked")              # non-blocking, sounds can overlap
    sfx.play("levelup", delay=0.3)

Playback: a small mixer on a `sounddevice` output stream (already installed for the microphone), so several
sounds can play at once with low delay. If that fails it falls back to Windows' built-in `winsound`
(one sound at a time), and if that fails too it stays silent and sets `sfx.warning`.

    python -m coach.sfx             # plays every sound once, with its name
    python -m coach.sfx --export sounds   # writes them all as .wav files (e.g. to reuse in the game)

Tip: use headphones during real sessions, so the microphone does not hear the coach's own sounds.
"""
import io
import math
import os
import random
import tempfile
import threading
import time
import wave

import numpy as np

SR = 22050


# --- tiny synth ---------------------------------------------------------------------------------
def hz(note: str) -> float:
    """'C5' -> 523.25 Hz, 'F#4' -> 369.99 Hz."""
    names = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    n = names[note[0]] + (1 if "#" in note else -1 if note[1] == "b" else 0)
    octave = int(note[-1])
    return 440.0 * 2 ** ((n + 12 * (octave + 1) - 69) / 12)


def _phase(freq, n):
    f = np.broadcast_to(np.asarray(freq, float), (n,))
    return np.cumsum(f) / SR  # in cycles


def wave_(kind: str, freq, dur: float, duty: float = 0.5) -> np.ndarray:
    n = max(1, int(dur * SR))
    ph = _phase(freq, n) % 1.0
    if kind == "square":
        return np.where(ph < duty, 1.0, -1.0)
    if kind == "tri":
        return 4 * np.abs(ph - 0.5) - 1
    if kind == "noise":
        # chunky NES-style noise: hold each random value for a few samples
        hold = max(1, int(SR / max(50.0, float(np.mean(freq)))))
        r = np.random.default_rng(1).uniform(-1, 1, n // hold + 1)
        return np.repeat(r, hold)[:n]
    raise ValueError(kind)


def env(n: int, attack: float = 0.003, decay: float = 4.0, sustain: float = 0.0) -> np.ndarray:
    """Fast attack, exponential decay towards `sustain`, tiny fade-out at the end (no clicks)."""
    t = np.arange(n) / SR
    e = sustain + (1 - sustain) * np.exp(-decay * t / max(t[-1], 1e-3))
    a = int(attack * SR)
    if a:
        e[:a] *= np.linspace(0, 1, a)
    r = min(n, int(0.006 * SR))
    e[-r:] *= np.linspace(1, 0, r)
    return e


def tone(note, dur, kind="square", duty=0.5, vol=1.0, decay=3.0, sustain=0.25, vibrato=0.0):
    f = hz(note) if isinstance(note, str) else note
    n = max(1, int(dur * SR))
    if vibrato:
        f = f * (1 + vibrato * np.sin(2 * math.pi * 6 * np.arange(n) / SR))
    return vol * wave_(kind, f, dur, duty) * env(n, decay=decay, sustain=sustain)


def slide(f0, f1, dur, kind="square", duty=0.5, vol=1.0, decay=1.5, sustain=0.3):
    n = int(dur * SR)
    f = np.geomspace(f0, f1, n)
    return vol * wave_(kind, f, dur, duty) * env(n, decay=decay, sustain=sustain)


def seq(*parts, gap=0.0):
    out = []
    for p in parts:
        out += [p, np.zeros(int(gap * SR))]
    return np.concatenate(out)


def mix(*parts, offsets=None):
    offsets = offsets or [0.0] * len(parts)
    n = max(int(o * SR) + len(p) for p, o in zip(parts, offsets))
    out = np.zeros(n)
    for p, o in zip(parts, offsets):
        s = int(o * SR)
        out[s: s + len(p)] += p
    return out


def echo(x, delay=0.09, feedback=0.35, repeats=3):
    d = int(delay * SR)
    out = np.concatenate([x, np.zeros(d * repeats)])
    for i in range(1, repeats + 1):
        out[d * i: d * i + len(x)] += x * feedback ** i
    return out


def crush(x, levels=24):
    """Coarse amplitude steps: the gritty 8-bit sound."""
    peak = np.max(np.abs(x)) or 1.0
    return np.round(x / peak * levels) / levels * peak


def arp(notes, step, **kw):
    return seq(*[tone(n, step, **kw) for n in notes])


# --- the sound set ------------------------------------------------------------------------------
def build_sounds() -> dict[str, np.ndarray]:
    s = {}
    # start / stop
    s["boot"] = mix(arp(["C5", "E5", "G5", "C6"], 0.07, duty=0.25, sustain=0.4),
                    tone("C6", 0.45, duty=0.25, decay=3, vibrato=0.006) * 0.8,
                    tone("C4", 0.6, kind="tri", decay=2, sustain=0.4) * 0.9, offsets=[0, 0.28, 0.28])
    s["quit"] = seq(arp(["G5", "E5", "C5"], 0.07, duty=0.25), slide(520, 90, 0.35, duty=0.5, vol=0.7))
    # typewriter text blips (two pitches, picked at random so long tips don't drone)
    s["blip"] = tone(1180, 0.022, duty=0.5, vol=0.35, decay=2, sustain=0.6)
    s["blip2"] = tone(1050, 0.022, duty=0.5, vol=0.35, decay=2, sustain=0.6)
    # a tip appears, by level: 1 friendly, 2 "try this now", 3 calm "pause" (not an alarm)
    s["tip1"] = seq(tone("G5", 0.06, duty=0.25), tone("C6", 0.16, duty=0.25, decay=4, sustain=0.2))
    s["tip2"] = seq(tone("C6", 0.05, duty=0.25), tone("C6", 0.05, duty=0.25), tone("G6", 0.2, duty=0.25,
                    decay=4, sustain=0.2), gap=0.03)
    s["tip3"] = mix(arp(["E5", "C5", "G4"], 0.2, kind="tri", decay=2.5, sustain=0.35),
                    tone("C3", 0.65, kind="tri", decay=1.5, sustain=0.5) * 0.7)
    # did it help?
    s["worked"] = mix(arp(["C5", "E5", "G5"], 0.075, duty=0.5, sustain=0.5),
                      tone("C6", 0.42, duty=0.5, decay=3, sustain=0.2, vibrato=0.008),
                      tone("E5", 0.42, duty=0.25, decay=3, sustain=0.2) * 0.5,
                      arp(["C3", "G3", "C4"], 0.075, kind="tri", sustain=0.6) * 0.9,
                      offsets=[0, 0.225, 0.225, 0])
    s["not_yet"] = seq(tone("E4", 0.11, kind="tri", sustain=0.5), tone("C4", 0.22, kind="tri", decay=3))
    # good streak / praise: sparkly arpeggio with an echo
    s["praise"] = echo(arp(["C6", "E6", "G6", "C7", "E7"], 0.04, duty=0.125, vol=0.6, sustain=0.4), 0.08, 0.4)
    # XP coin and level up fanfare
    s["xp"] = seq(tone("B5", 0.055, duty=0.5, sustain=0.9), tone("E6", 0.28, duty=0.5, decay=5, sustain=0.1))
    s["levelup"] = mix(seq(arp(["G4", "C5", "E5", "G5"], 0.07, duty=0.25, sustain=0.5),
                           arp(["E5", "G5"], 0.09, duty=0.25, sustain=0.5),
                           tone("C6", 0.55, duty=0.25, decay=2, sustain=0.35, vibrato=0.01)),
                       seq(arp(["C3", "E3", "G3", "C4"], 0.07, kind="tri", sustain=0.7),
                           tone("G3", 0.18, kind="tri", sustain=0.7), tone("C4", 0.55, kind="tri", decay=2,
                                                                          sustain=0.4)) * 0.9)
    # calibration finished: power-up sweep + ding
    s["calibrated"] = seq(slide(220, 1300, 0.32, duty=0.25, vol=0.6, decay=0.3, sustain=0.8),
                          tone("C7", 0.25, duty=0.5, vol=0.6, decay=5))
    # a bar went into the red zone, and back out
    s["warn"] = seq(tone("A3", 0.06, duty=0.5, vol=0.55, sustain=0.8), tone("A3", 0.09, duty=0.5, vol=0.55,
                    decay=4), gap=0.04)
    s["recover"] = seq(tone("E5", 0.04, duty=0.25, vol=0.5, sustain=0.8), tone("A5", 0.08, duty=0.25, vol=0.5))
    # the student froze: a soft "heartbeat" thump
    s["freeze"] = seq(mix(slide(140, 60, 0.12, kind="tri", decay=3), wave_("noise", 300, 0.03) * 0.25),
                      mix(slide(120, 50, 0.16, kind="tri", decay=3), wave_("noise", 300, 0.03) * 0.2), gap=0.07)
    # keys pressed in the window
    s["menu"] = tone(1500, 0.025, duty=0.25, vol=0.45, decay=3, sustain=0.5)
    return {k: crush(v / max(1.0, np.max(np.abs(v)))).astype(np.float32) for k, v in s.items()}


def to_wav_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32000).astype("<i2").tobytes())
    return buf.getvalue()


# --- playback -----------------------------------------------------------------------------------
class Sfx:
    def __init__(self, volume: float = 0.6, enabled: bool = True):
        self.volume = max(0.0, min(1.0, volume))
        self.sounds = build_sounds()
        self.warning = None
        self.backend = "off"
        self._voices = []           # [samples, position]
        self._lock = threading.Lock()
        self._stream = None
        self._win_files = {}
        self._win_busy_until = 0.0
        self.duck = 1.0             # lowered while spoken tips talk (coach/speaker.py)
        if not enabled or self.volume == 0:
            return
        try:
            import sounddevice as sd
            self._stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", blocksize=256,
                                           latency="low", callback=self._callback)
            self._stream.start()
            self.backend = "sounddevice"
            return
        except Exception as e:  # noqa: BLE001  (no output device, PortAudio missing...)
            first_error = e
        try:
            import winsound  # noqa: F401  (Windows only)
            folder = os.path.join(tempfile.gettempdir(), "oral-coach-sfx")
            os.makedirs(folder, exist_ok=True)
            for name, x in self.sounds.items():
                path = os.path.join(folder, f"{name}.wav")
                with open(path, "wb") as f:
                    f.write(to_wav_bytes(x * self.volume))
                self._win_files[name] = path
            self.backend = "winsound"
        except Exception:  # noqa: BLE001
            self.warning = f"Sound effects off (no audio output: {first_error})"

    def _callback(self, out, frames, _time, _status):
        buf = np.zeros(frames, np.float32)
        with self._lock:
            alive = []
            for v in self._voices:
                x, pos = v
                if pos < 0:  # still waiting (delay)
                    start = -pos
                    if start >= frames:
                        v[1] += frames
                        alive.append(v)
                        continue
                    chunk = x[: frames - start]
                    buf[start: start + len(chunk)] += chunk
                    v[1] = len(chunk)
                else:
                    chunk = x[pos: pos + frames]
                    buf[: len(chunk)] += chunk
                    v[1] = pos + frames
                if v[1] < len(x):
                    alive.append(v)
            self._voices = alive
        out[:, 0] = np.tanh(buf * self.volume * self.duck * 1.2)  # soft limiter: overlapping sounds never clip harshly

    def play(self, name: str, delay: float = 0.0, gain: float = 1.0):
        x = self.sounds.get(name)
        if x is None or self.backend == "off" or self.volume <= 0:
            return
        if self.backend == "sounddevice":
            with self._lock:
                if name.startswith("blip"):  # only one text blip at a time
                    self._voices = [v for v in self._voices if v[0] is not self.sounds["blip"]
                                    and v[0] is not self.sounds["blip2"]]
                self._voices.append([x * gain if gain != 1 else x, -int(delay * SR)])
        elif self.backend == "winsound":
            import winsound
            now = time.time()
            if name.startswith("blip") and (now < self._win_busy_until or self.duck < 1):
                return  # winsound plays one sound at a time: don't cut a jingle for a text blip
            if delay:
                threading.Timer(delay, self.play, (name,)).start()
                return
            self._win_busy_until = now + len(x) / SR
            winsound.PlaySound(self._win_files[name], winsound.SND_FILENAME | winsound.SND_ASYNC
                               | winsound.SND_NODEFAULT)

    def blip(self):
        self.play(random.choice(("blip", "blip2")))

    def close(self):
        if self._stream is not None:
            time.sleep(0.05)
            self._stream.stop()
            self._stream.close()
            self._stream = None


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Play or export the coach's sound effects")
    ap.add_argument("--export", metavar="FOLDER", help="write every sound as a .wav file")
    ap.add_argument("--volume", type=float, default=0.6)
    args = ap.parse_args()
    if args.export:
        os.makedirs(args.export, exist_ok=True)
        for name, x in build_sounds().items():
            with open(os.path.join(args.export, f"{name}.wav"), "wb") as f:
                f.write(to_wav_bytes(x * args.volume))
        print(f"Wrote {len(build_sounds())} sounds to {args.export}/")
        return
    sfx = Sfx(volume=args.volume)
    print(f"Sound backend: {sfx.backend}" + (f" ({sfx.warning})" if sfx.warning else ""))
    for name, x in sfx.sounds.items():
        print(f"  {name:<11} {len(x) / SR:.2f} s")
        if name.startswith("blip"):
            for _ in range(12):
                sfx.blip()
                time.sleep(0.045)
        else:
            sfx.play(name)
        time.sleep(len(x) / SR + 0.45)
    sfx.close()


if __name__ == "__main__":
    main()
