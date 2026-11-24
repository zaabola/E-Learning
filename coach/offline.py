"""Video mode: run the whole coach on a recorded talk and write an annotated MP4.

    python run.py --video talk.mp4 --out talk-coached.mp4

The recording goes through the same pipeline as a live session (face sensor, voice sensor, channels,
pattern detector, coach, LLM tip writer, game feel, retro window), but on the video's own clock instead
of the wall clock: frame N happens at N / fps seconds, audio block K at K * 32 ms. Nothing waits for real
time, so processing can be slower than real time (no dropped frames, no LLM timeouts) or faster.

Because it is not live, video mode uses the better settings by default:
  * every video frame is analysed (live: 10 per second)
  * a bigger Whisper (large-v3-turbo, beam search 5; live: small, greedy), on the same phrases as live
  * a voice emotion estimate every 0.5 s (live: every 1.5 s)
  * the LLM writes every tip, urgent ones too, and the coach waits for it (live: 6 s timeout, urgent tips
    use the card wording)

Output: the original video + the overlay (tips, why, bars, FEELING line, XP...) on every frame, with the
original sound and the coach's sound effects mixed in. Needs ffmpeg on the PATH.

Live mode does not use this file and is unchanged.
"""
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time as _time
import wave

import cv2
import numpy as np

from .channels import ChannelBuilder
from .signals import Signals
from .voice import BLOCK, _WORD, FileSource, WhisperWorker, load_audeering, pick_device

DISPLAY_MAX_H = 720            # the overlay is drawn on frames scaled down to this height (fonts stay readable)
VIDEO_WHISPER = "large-v3-turbo"
VIDEO_WHISPER_FALLBACK = "medium"
VIDEO_EMOTION_EVERY = 0.5      # seconds between two voice emotion estimates
VIDEO_LLM_TIMEOUT = 120.0      # the coach waits this long for the LLM before using the card wording
SFX_MIX = 0.7                  # loudness of the sound effects against your voice in the final mix


# --- video clock --------------------------------------------------------------------------------
class VideoClock:
    """Stands in for the `time` module inside the coach: time() is the video's time, the rest is real."""

    def __init__(self, t0: float):
        self.t0 = self.now = t0

    def time(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(_time, name)


def use_clock(clock: VideoClock):
    """Point every coach module that reads the time at the video clock."""
    from . import channels, face, juice, overlay, signals, voice
    for mod in (channels, face, juice, overlay, signals, voice):
        mod.time = clock


# --- audio fed to the voice sensor in lock step with the video -----------------------------------
class LockstepSource:
    """Audio source for VoiceSensor: blocks go in only when the video reaches them, and feed() returns once
    the sensor has processed all of them, so voice and face stay exactly in sync."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()

    def blocks(self, stop):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            yield item
            self.q.task_done()

    def feed(self, items, thread: threading.Thread):
        for it in items:
            self.q.put(it)
        while self.q.unfinished_tasks and thread.is_alive():
            _time.sleep(0.0005)

    def close(self):
        self.q.put(None)


# --- the LLM writes every tip -------------------------------------------------------------------
def make_writer(args):
    from .recommend import TipWriter

    class VideoTipWriter(TipWriter):
        """Live mode words urgent tips with the card (no time to wait). A video can wait: the LLM writes them."""

        def template(self, req, note=""):
            if note.startswith("urgent") and self.backend != "template":
                return self._write(req)
            return super().template(req, note)

    return VideoTipWriter(mode=args.llm, ollama_model=args.ollama_model, claude_model=args.claude_model,
                          language=args.tip_language, timeout=VIDEO_LLM_TIMEOUT)


# --- sound effects recorded on the video's timeline ----------------------------------------------
class SfxRecorder:
    """Same interface as coach.sfx.Sfx, but instead of playing, it notes when each sound starts."""

    def __init__(self, clock: VideoClock, volume: float):
        from .sfx import SR, build_sounds
        self.sr, self.sounds = SR, build_sounds()
        self.clock, self.volume = clock, max(0.0, min(1.0, volume))
        self.backend = "recorder" if self.volume > 0 else "off"
        self.warning = None
        self.events: list[tuple[float, str, float]] = []

    def play(self, name: str, delay: float = 0.0, gain: float = 1.0):
        if name in self.sounds and self.volume > 0:
            self.events.append((self.clock.now - self.clock.t0 + delay, name, gain))

    def blip(self):
        self.play(random.choice(("blip", "blip2")))

    def close(self):
        pass

    def render(self, seconds: float) -> np.ndarray:
        buf = np.zeros(int(seconds * self.sr) + 1, np.float32)
        events = sorted(self.events)
        blip_starts = [int(t * self.sr) for t, n, _ in events if n.startswith("blip")]
        for t, name, gain in events:
            x = self.sounds[name] * gain
            i = int(t * self.sr)
            if i >= len(buf):
                continue
            if name.startswith("blip"):  # like live: only one text blip at a time, a new one cuts the old
                later = [j for j in blip_starts if j > i]
                if later:
                    x = x[: later[0] - i]
            n = min(len(x), len(buf) - i)
            buf[i:i + n] += x[:n]
        return np.tanh(buf * self.volume * 1.2)  # same soft limiter as the live mixer


def write_wav(path: str, x: np.ndarray, sr: int):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32000).astype("<i2").tobytes())


def has_audio(path: str) -> bool:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                              "-of", "csv=p=0", path], capture_output=True, text=True).stdout
        return bool(out.strip())
    except FileNotFoundError:  # no ffprobe: try decoding anyway
        return True


def mmss(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


# --- speech-to-text on each phrase, done right away (live does it in a worker thread) ------------
class PhraseTranscriber:
    def __init__(self, cfg, sig: Signals, model_name: str):
        from faster_whisper import WhisperModel
        self.cfg, self.sig = cfg, sig
        device = pick_device(cfg)
        compute = "float16" if device == "cuda" else "int8"
        try:
            self.model = WhisperModel(model_name, device=device, compute_type=compute)
            self.name = model_name
        except Exception as e:  # noqa: BLE001  (older faster-whisper without large-v3-turbo, no memory...)
            if model_name == VIDEO_WHISPER_FALLBACK:
                raise
            print(f"warning: whisper {model_name} not loaded ({e}); using {VIDEO_WHISPER_FALLBACK}", file=sys.stderr)
            self.model = WhisperModel(VIDEO_WHISPER_FALLBACK, device=device, compute_type=compute)
            self.name = VIDEO_WHISPER_FALLBACK
        self.fillers, self.soft = set(cfg.fillers), set(cfg.soft_fillers)

    def __call__(self, seg_start: float, audio: np.ndarray) -> str:
        """Same word / filler / transcript signals as WhisperWorker, with beam search for accuracy."""
        segments, _ = self.model.transcribe(audio, language=self.cfg.language, word_timestamps=True,
                                            initial_prompt=WhisperWorker.PROMPT, condition_on_previous_text=False,
                                            vad_filter=False, beam_size=5)
        words = [w for s in segments for w in (s.words or [])]
        text, prev = [], ""
        for w in words:
            token = _WORD.sub("", w.word.lower())
            if not token:
                continue
            t = seg_start + w.end
            self.sig.push("word", token, t)
            text.append(w.word)
            if token in self.fillers:
                self.sig.push("filler", token, t)
            elif token in self.soft or f"{prev} {token}" in self.soft:
                self.sig.push("soft_filler", token if token in self.soft else f"{prev} {token}", t)
            prev = token
        if text:
            line = "".join(text).strip()
            self.sig.push("transcript", line, seg_start + words[-1].end)
            return line
        return ""


# --- main ---------------------------------------------------------------------------------------
def run_video(args, cfg) -> int:
    if not shutil.which("ffmpeg"):
        print("Video mode needs ffmpeg on your PATH (https://ffmpeg.org/download.html, or `winget install ffmpeg`).",
              file=sys.stderr)
        return 1
    src = args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open video {src}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not 1 <= fps <= 240:
        fps = 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_s = n_frames / fps if n_frames else None
    if args.duration:
        total_s = min(total_s or args.duration, args.duration)

    # quality settings for a recording (live mode keeps its own, see config.py)
    cfg.voice_emotion_every = VIDEO_EMOTION_EVERY
    whisper_name = args.whisper_model or VIDEO_WHISPER

    # every coach module now reads the video's clock
    clock = VideoClock(_time.time())
    use_clock(clock)
    sig = Signals()
    stop = threading.Event()
    warnings: list[str] = []

    def warn(msg):
        if msg and msg not in warnings:
            warnings.append(msg)
            print(f"warning: {msg}", file=sys.stderr)

    # --- sensors -----------------------------------------------------------------------------
    face = None
    if not args.no_face:
        from .face import FaceSensor
        face = FaceSensor(cfg, sig, stop)
        print("Loading face models ...", file=sys.stderr)
        try:
            face._load()
        except Exception as e:  # noqa: BLE001
            warn(f"face analysis off: {e}")
            face = None

    voice = source = audio = transcriber = emotion = None
    if not args.no_voice:
        if not has_audio(src):
            warn("the video has no sound: voice analysis off")
        else:
            from .voice import VoiceSensor
            print("Reading the audio ...", file=sys.stderr)
            audio = FileSource(src, cfg, realtime=False).audio
            source = LockstepSource()
            voice = VoiceSensor(cfg, sig, stop, source=source, use_emotion=False, use_whisper=False)
            voice.start()
            if not args.no_whisper:
                print(f"Loading speech-to-text (whisper {whisper_name}) ...", file=sys.stderr)
                try:
                    transcriber = PhraseTranscriber(cfg, sig, whisper_name)
                except Exception as e:  # noqa: BLE001
                    warn(f"whisper not loaded ({e}); pace and fillers disabled")
            if not args.light:
                print("Loading voice emotion model ...", file=sys.stderr)
                try:
                    emotion = load_audeering(cfg)
                except Exception as e:  # noqa: BLE001
                    warn(f"voice emotion model not loaded ({e}); voice arousal/dominance disabled")

    # --- coach + LLM ---------------------------------------------------------------------------
    os.makedirs("logs", exist_ok=True)
    stem = os.path.splitext(os.path.basename(src))[0]
    log_path = _time.strftime(f"logs/video-{stem}-%Y%m%d-%H%M%S.jsonl")
    log = open(log_path, "w", encoding="utf-8")
    builder = ChannelBuilder(cfg, sig)
    recommender = writer = None
    if not args.no_coach:
        from .recommend import Coach, Knowledge
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        writer = make_writer(args)
        recommender = Coach(cfg, Knowledge(os.path.join(here, "knowledge")), writer,
                            on_tip=lambda tip: builder.restart(tip["restart"], tip["t"]))
        if args.llm != "template":
            print("Looking for the LLM (Ollama / Claude) ...", file=sys.stderr)
        writer._ready.wait()
        print(f"Tips written by: {writer.status}", file=sys.stderr)
        warn(writer.warning)

    from .juice import GameFeel
    from .overlay import PANEL_W, compose, typed_chars
    sfx = SfxRecorder(clock, 0.0 if args.no_sound else args.volume)
    feel = GameFeel(sfx)
    shown: dict = {}
    mirror = args.mirror

    # --- output ------------------------------------------------------------------------------
    tmp = tempfile.mkdtemp(prefix="oral-coach-")
    silent_path = os.path.join(tmp, "video.mp4")
    encoder = None
    out_size = None

    def open_encoder(w, h):
        return subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", silent_path], stdin=subprocess.PIPE)

    # --- the loop: one video frame at a time, on the video's clock ---------------------------
    sr = cfg.sample_rate
    next_block = 0          # index of the next audio sample to feed
    next_emotion = VIDEO_EMOTION_EVERY
    next_tick = cfg.tick_seconds
    snap = None
    face_view = None
    last_ts = -1
    frame_i = 0
    started = _time.time()
    last_report = -1
    tips = 0
    print(f"Processing {src} ({mmss(total_s) if total_s else '?'} at {fps:.0f} fps) ...", file=sys.stderr)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vt = frame_i / fps
            if total_s and args.duration and vt >= total_s:
                break
            clock.now = clock.t0 + vt

            # 1. audio up to this frame -> VAD, volume, pitch, pauses (voice sensor), then phrases and emotion
            if voice is not None:
                end = min(len(audio), int(vt * sr))
                items = []
                while next_block + BLOCK <= end:
                    items.append((clock.t0 + (next_block + BLOCK) / sr, audio[next_block:next_block + BLOCK].copy()))
                    next_block += BLOCK
                if items:
                    source.feed(items, voice)
                if voice.error:
                    warn(voice.error)
                while True:
                    try:
                        seg_start, seg_audio = voice.segment_q.get_nowait()
                    except queue.Empty:
                        break
                    if transcriber is not None:
                        transcriber(seg_start, seg_audio)
                if emotion is not None and vt >= next_emotion:
                    next_emotion += VIDEO_EMOTION_EVERY
                    share = sig.mean("speaking", cfg.voice_emotion_window)
                    if share and share >= 0.5:
                        model, processor, torch, device = emotion
                        clip, t = voice.last_audio(cfg.voice_emotion_window)
                        x = processor(clip, sampling_rate=sr, return_tensors="pt").input_values.to(device)
                        with torch.no_grad():
                            _, out = model(x)
                        arousal, dominance, valence = (float(v) for v in out[0].cpu())
                        sig.push("voice_arousal", arousal, t)
                        sig.push("voice_dominance", dominance, t)
                        sig.push("voice_valence", valence, t)

            # 2. face: every frame
            if face is not None:
                ts = max(last_ts + 1, int(vt * 1000))
                last_ts = ts
                try:
                    face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), clock.now, ts)
                except Exception as e:  # noqa: BLE001
                    warn(f"face frame error: {e}")
                face_view = face._face if sig.latest("face_present", 0.5) == 1.0 else None

            # 3. once per second: snapshot -> coach (waits for the LLM) -> game feel
            if vt >= next_tick:
                next_tick += cfg.tick_seconds
                for w in sig.values("warning", 3600):
                    warn(w)
                snap = builder.tick(clock.now)
                events = []
                if recommender is not None:
                    events = recommender.step(snap)
                    if recommender.pending is not None:
                        try:
                            recommender.pending["future"].result(timeout=VIDEO_LLM_TIMEOUT)
                        except Exception:  # noqa: BLE001  (the coach falls back to the card wording)
                            pass
                        events += recommender._deliver_pending(clock.now)
                    warn(writer.warning)
                for item in [snap] + events:
                    log.write(json.dumps(dict(item, video_s=round(vt, 2))) + "\n")
                    if item.get("type") == "tip":
                        tips += 1
                        tag = "URGENT " if item.get("urgent") else ""
                        print(f"  [{mmss(vt)}] {tag}TIP: {item['text']}  (why: {item['why']})", file=sys.stderr)
                    elif item.get("type") == "tip_result":
                        print(f"  [{mmss(vt)}] {item.get('message', '')}", file=sys.stderr)
                feel.update(snap, events, recommender.view(clock.now) if recommender else None, clock.now)

            # 4. draw the overlay on this frame, exactly like the live window
            coach_view = recommender.view(clock.now) if recommender is not None else None
            feel.start()
            tip_id, n, full = typed_chars(coach_view)
            if tip_id and (coach_view.get("tip_age") or 0) < coach_view.get("tip_show", 12.0):
                feel.typed(tip_id, n, full)
            h, w = frame.shape[:2]
            shown_frame = frame
            if h > DISPLAY_MAX_H:
                shown_frame = cv2.resize(frame, (round(w * DISPLAY_MAX_H / h), DISPLAY_MAX_H),
                                         interpolation=cv2.INTER_AREA)
            img = compose(shown_frame, face_view, snap, warnings, mirror, not args.no_landmarks, None,
                          coach_view, feel, shown, args.pixel, clock.now, sfx.backend != "off")
            if out_size is None:
                out_size = (img.shape[1] // 2 * 2, img.shape[0] // 2 * 2)  # x264 wants even sizes
                encoder = open_encoder(*out_size)
            encoder.stdin.write(np.ascontiguousarray(img[:out_size[1], :out_size[0]]).tobytes())
            frame_i += 1

            if total_s:
                pct = int(100 * vt / total_s)
                if pct >= last_report + 5:
                    last_report = pct
                    speed = vt / max(1e-6, _time.time() - started)
                    print(f"  {pct:3d}%  {mmss(vt)} / {mmss(total_s)}  ({speed:.2f}x real time)", file=sys.stderr)
    except KeyboardInterrupt:
        print("Stopped: writing what was processed so far.", file=sys.stderr)
    finally:
        cap.release()
        stop.set()
        if face is not None:
            face.landmarker.close()
        if source is not None:
            source.close()
        log.close()

    if encoder is None:
        print("No frames could be read from the video.", file=sys.stderr)
        return 1
    encoder.stdin.close()
    encoder.wait()
    length = frame_i / fps

    # --- sound: original audio + coach sound effects -------------------------------------------
    print("Mixing the sound ...", file=sys.stderr)
    sfx_path = os.path.join(tmp, "sfx.wav")
    write_wav(sfx_path, sfx.render(length), sfx.sr)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", silent_path]
    if has_audio(src):
        cmd += ["-t", f"{length:.3f}", "-i", src, "-i", sfx_path, "-filter_complex",
                "[1:a:0]aresample=48000,aformat=channel_layouts=stereo[v];"
                f"[2:a]aresample=48000,aformat=channel_layouts=stereo,volume={SFX_MIX}[s];"
                "[v][s]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[a]"]
    else:
        cmd += ["-i", sfx_path, "-filter_complex",
                f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={SFX_MIX}[a]"]
    cmd += ["-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart", args.out]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        shutil.copy(silent_path, args.out)
        warn("could not mix the sound; the video was written without sound")
    shutil.rmtree(tmp, ignore_errors=True)

    took = _time.time() - started
    print(f"\nDone: {args.out}  ({mmss(length)} of video, {tips} tips, processed in {mmss(took)})", file=sys.stderr)
    print(f"Session log: {log_path}", file=sys.stderr)
    return 0
