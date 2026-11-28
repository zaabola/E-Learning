"""Backend bridge integrating `Models/OralCoach` (`origin/oral_coach`) into `server.py`.

Supports:
  1. Kaggle `2x Tesla T4` Multi-GPU execution via browser-streamed webcam JPEG
     frames + 16 kHz mono PCM audio:
       - Automatically picks the CUDA GPU with the most free VRAM (typically
         `cuda:1` on Kaggle `2x T4`), running `audEERING` (FP16) and
         `faster-whisper` (`float16`, `small` or `base`) on GPU with automatic
         CPU fallback on low VRAM or cuDNN errors.
       - Reuses Stage-4's loaded `Meta-Llama-3.1-8B-Instruct` via `Stage4TipWriter`
         (with `_base_model_ctx()` and `llm_orchestrator._gpu_lock`) so coach tips
         get rewritten on GPU with zero extra VRAM allocation.
  2. Optional localhost desktop OpenCV window launcher (`python run.py`).
  3. Clean graceful degradation when optional dependencies (`cv2`, `mediapipe`,
     `hsemotion_onnx`, `silero_vad`, `parselmouth`, `faster_whisper`, etc.) are
     not yet installed.
"""

from __future__ import annotations

import base64
from collections import deque
import copy
import ctypes
import gc
import glob
import importlib.util
import io
import json
import os
import queue
import re
import site
import subprocess
import sys
import threading
import time
import urllib.request

ORAL_COACH_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ORAL_COACH_ROOT, "models")
LOGS_DIR = os.path.join(ORAL_COACH_ROOT, "logs")
KNOWLEDGE_DIR = os.path.join(ORAL_COACH_ROOT, "knowledge")
FACE_LANDMARKER_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Browser-streamed Oral Coach packages (runs locally and over Kaggle HTTPS tunnels)
BROWSER_REQUIRED_MODULES = [
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("hsemotion_onnx", "hsemotion-onnx"),
    ("onnxruntime", "onnxruntime"),
    ("silero_vad", "silero-vad"),
    ("parselmouth", "praat-parselmouth"),
    ("faster_whisper", "faster-whisper"),
]

# Desktop-only OpenCV GUI packages (not required for browser / Kaggle streaming)
DESKTOP_ONLY_MODULES = [
    ("sounddevice", "sounddevice"),
    ("pyttsx3", "pyttsx3"),
]

VOICE_BLOCK = 512  # Silero VAD requires 512 samples at 16 kHz (32 ms)
_WHISPER_WORD_RE = re.compile(r"[^\w']+")


def _ensure_import_path() -> None:
    if ORAL_COACH_ROOT not in sys.path:
        sys.path.insert(0, ORAL_COACH_ROOT)


def _has_module(mod_name: str) -> bool:
    return importlib.util.find_spec(mod_name) is not None


def _decode_base64_payload(raw_b64: str) -> bytes:
    if not raw_b64:
        return b""
    if "," in raw_b64 and raw_b64.lstrip().startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]
    return base64.b64decode(raw_b64)


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _preload_ctranslate2_cuda_libs() -> bool:
    """Preload CUDA 12 cublas/cudnn shared libraries into global process symbols on Linux.

    `faster-whisper` (`ctranslate2`) dlopens `libcublas.so.12` without inspecting PyTorch's
    bundled `nvidia/*/lib` wheels on Kaggle/Colab unless already loaded with RTLD_GLOBAL.
    """
    if not sys.platform.startswith("linux"):
        return True

    rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)
    search_roots = []
    try:
        search_roots.extend(site.getsitepackages())
    except Exception:
        pass
    search_roots.extend(p for p in sys.path if p)

    lib_dirs = []
    for root in search_roots:
        for sub in ("nvidia/cublas/lib", "nvidia/cudnn/lib", "nvidia/cuda_runtime/lib"):
            cand = os.path.join(root, sub)
            if os.path.isdir(cand):
                lib_dirs.append(cand)
    for pattern in ("/usr/local/cuda*/lib64", "/usr/local/cuda*/targets/*/lib", "/usr/lib/x86_64-linux-gnu"):
        for cand in glob.glob(pattern):
            if os.path.isdir(cand):
                lib_dirs.append(cand)

    # Deduplicate preserving order
    seen_dirs = set()
    uniq_dirs = [d for d in lib_dirs if not (d in seen_dirs or seen_dirs.add(d))]

    target_libs = (
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
    )
    loaded_cublas = False
    for lib_name in target_libs:
        for lib_dir in uniq_dirs:
            cand = os.path.join(lib_dir, lib_name)
            if os.path.exists(cand):
                try:
                    ctypes.CDLL(cand, mode=rtld_global)
                    if lib_name == "libcublas.so.12":
                        loaded_cublas = True
                    break
                except OSError:
                    pass

    if not loaded_cublas:
        try:
            ctypes.CDLL("libcublas.so.12", mode=rtld_global)
            loaded_cublas = True
        except OSError:
            loaded_cublas = False
    return loaded_cublas


def _pick_best_cuda_device(min_free_gib: float = 1.8) -> tuple[int | None, float]:
    """Select the CUDA GPU index with the most free VRAM if >= `min_free_gib`."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None, 0.0
        count = torch.cuda.device_count()
        best_idx = None
        best_free = 0.0
        for idx in range(count):
            try:
                free_bytes, _ = torch.cuda.mem_get_info(idx)
                free_gib = free_bytes / (1024.0 ** 3)
            except Exception:
                free_gib = 0.0
            if best_idx is None or free_gib > best_free:
                best_idx = idx
                best_free = free_gib
        if best_idx is not None and best_free >= min_free_gib:
            return best_idx, best_free
        return None, best_free
    except Exception:
        return None, 0.0


def _build_stage4_tip_writer(
    llm_orchestrator=None,
    mode: str = "auto",
    language: str = "en",
):
    """Creates a `TipWriter` (or `Stage4TipWriter` sharing live Stage-4 GPU LLM)."""
    _ensure_import_path()
    from coach.recommend.writer import SYSTEM, TipWriter, _clean, _parse_json, build_prompt

    class Stage4TipWriter(TipWriter):
        def __init__(self, orch, lang: str = "en", fallback_mode: str = "auto"):
            stylizer = getattr(orch, "stylizer", None) if orch is not None else None
            use_live = bool(stylizer is not None and stylizer.is_live())
            if use_live:
                # Pass mode="template" first so base class skips slow Ollama/Claude probe,
                # then switch backend/mode to stage4_gpu so submit() dispatches to _write().
                super().__init__(mode="template", language=lang)
                self.mode = "stage4_gpu"
                self.backend = "stage4_gpu"
                self.status = f"Kaggle GPU: {stylizer.short_name}"
                self._llm_orch = orch
            else:
                super().__init__(mode=fallback_mode, language=lang)
                self._llm_orch = None

        def _write(self, req: dict) -> dict:
            if self._llm_orch is None or self.backend != "stage4_gpu":
                return super()._write(req)
            stylizer = getattr(self._llm_orch, "stylizer", None)
            gpu_lock = getattr(self._llm_orch, "_gpu_lock", None)
            if stylizer is None or not stylizer.is_live():
                return self.template(req, note="Stage-4 LLM offline")

            prompt = build_prompt(req)
            acquired = True
            if gpu_lock is not None:
                acquired = gpu_lock.acquire(timeout=4.0)
            if not acquired:
                return self.template(req, note="GPU busy")

            try:
                import torch

                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ]
                prompt_str = stylizer.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                first_dev = next(stylizer.model.parameters()).device
                inputs = stylizer.tokenizer(prompt_str, return_tensors="pt").to(first_dev)
                with stylizer._base_model_ctx():
                    with torch.no_grad():
                        out_ids = stylizer.model.generate(
                            **inputs,
                            max_new_tokens=95,
                            do_sample=True,
                            temperature=0.4,
                            top_p=0.90,
                            pad_token_id=stylizer.tokenizer.pad_token_id,
                            eos_token_id=stylizer.tokenizer.eos_token_id,
                        )
                gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
                raw_text = stylizer.tokenizer.decode(gen_ids, skip_special_tokens=True)
                parsed = _parse_json(raw_text)
                clean = parsed and _clean(parsed, req)
                if clean:
                    return dict(clean, backend=self.status)
            except Exception as exc:
                self.warning = f"Stage-4 tip rewrite fell back to card wording ({type(exc).__name__})"
            finally:
                if gpu_lock is not None and acquired:
                    gpu_lock.release()
            return self.template(req, note="GPU output fallback")

    return Stage4TipWriter(llm_orchestrator, lang=language, fallback_mode=mode)


class _GpuVoiceEmotionWorker(threading.Thread):
    """audEERING wav2vec2 in FP16 on best CUDA GPU with automatic CPU fallback."""

    def __init__(self, voice, gpu_idx: int | None):
        super().__init__(daemon=True, name="oral-voice-emotion")
        self.v = voice
        self.gpu_idx = gpu_idx
        self.device_label = f"cuda:{gpu_idx} (fp16)" if gpu_idx is not None else "cpu (fp32)"

    def run(self):
        cfg, sig = self.v.cfg, self.v.sig
        _ensure_import_path()
        from coach.voice import load_audeering

        model = processor = torch = None
        device = f"cuda:{self.gpu_idx}" if self.gpu_idx is not None else "cpu"
        use_fp16 = self.gpu_idx is not None
        try:
            cfg_dev = copy.copy(cfg)
            cfg_dev.device = device
            model, processor, torch, device = load_audeering(cfg_dev)
            if use_fp16 and str(device).startswith("cuda"):
                model = model.half().eval()
                self.device_label = f"{device} (fp16)"
            else:
                use_fp16 = False
                self.device_label = "cpu (fp32)"
        except Exception as exc:
            if self.gpu_idx is not None:
                try:
                    _release_cuda_cache()
                    cfg_cpu = copy.copy(cfg)
                    cfg_cpu.device = "cpu"
                    model, processor, torch, device = load_audeering(cfg_cpu)
                    use_fp16 = False
                    self.device_label = "cpu (fp32 fallback)"
                    sig.push("warning", f"audEERING moved to CPU ({exc})")
                except Exception as exc2:
                    self.device_label = "disabled"
                    sig.push("warning", f"voice emotion not loaded ({exc2}); voice arousal/dominance disabled")
                    return
            else:
                self.device_label = "disabled"
                sig.push("warning", f"voice emotion not loaded ({exc}); voice arousal/dominance disabled")
                return

        try:
            while not self.v.stop.is_set():
                time.sleep(cfg.voice_emotion_every)
                speech_share = sig.mean("speaking", cfg.voice_emotion_window)
                if not speech_share or speech_share < 0.5:
                    continue
                audio, t = self.v.last_audio(cfg.voice_emotion_window)
                try:
                    x = processor(audio, sampling_rate=cfg.sample_rate, return_tensors="pt").input_values
                    if use_fp16:
                        x = x.to(device=device, dtype=torch.float16)
                    else:
                        x = x.to(device=device)
                    with torch.no_grad():
                        _, out = model(x)
                    arousal, dominance, valence = (float(v) for v in out[0].float().cpu())
                    sig.push("voice_arousal", arousal, t)
                    sig.push("voice_dominance", dominance, t)
                    sig.push("voice_valence", valence, t)
                except RuntimeError as rt_err:
                    if use_fp16:
                        use_fp16 = False
                        device = "cpu"
                        model = model.float().to("cpu").eval()
                        self.device_label = "cpu (fp32 fallback)"
                        _release_cuda_cache()
                        sig.push("warning", f"audEERING GPU fallback to CPU: {rt_err}")
                    else:
                        sig.push("warning", f"audEERING inference error: {rt_err}")
        finally:
            model = None
            processor = None
            _release_cuda_cache()


class _GpuWhisperWorker(threading.Thread):
    """faster-whisper in FP16 on best CUDA GPU with automatic CPU int8 fallback."""

    PROMPT = "Umm, so, uh, I think, like, euh... you know, hmm, the main point is"

    def __init__(self, voice, gpu_idx: int | None, free_gib: float, explicit_model: bool = False):
        super().__init__(daemon=True, name="oral-whisper")
        self.v = voice
        self.gpu_idx = gpu_idx
        self.free_gib = free_gib
        self.explicit_model = explicit_model
        self.device_label = "loading"

    def _load(self, WhisperModel, model_size: str, use_gpu: bool):
        if use_gpu and self.gpu_idx is not None:
            m = WhisperModel(model_size, device="cuda", device_index=self.gpu_idx, compute_type="float16")
            self.device_label = f"cuda:{self.gpu_idx} {model_size} (fp16)"
            return m, True
        m = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.device_label = f"cpu {model_size} (int8)"
        return m, False

    def run(self):
        cfg, sig = self.v.cfg, self.v.sig
        can_gpu = (self.gpu_idx is not None) and _preload_ctranslate2_cuda_libs()
        if self.gpu_idx is not None and not can_gpu:
            print("[ORAL COACH] libcublas.so.12 unavailable; running Whisper on CPU int8", flush=True)

        if self.explicit_model:
            model_size = cfg.whisper_model
        else:
            model_size = "small" if (can_gpu and self.free_gib >= 3.0) else "base"

        model = None
        on_gpu = False
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            self.device_label = "disabled"
            sig.push("warning", f"faster-whisper import failed ({exc}); pace and fillers disabled")
            return

        try:
            model, on_gpu = self._load(WhisperModel, model_size, use_gpu=can_gpu)
        except Exception as exc:
            if can_gpu:
                try:
                    _release_cuda_cache()
                    cpu_size = cfg.whisper_model if self.explicit_model else "base"
                    model, on_gpu = self._load(WhisperModel, cpu_size, use_gpu=False)
                    model_size = cpu_size
                    print(f"[ORAL COACH] Whisper loaded on CPU int8 fallback ({exc})", flush=True)
                except Exception as exc2:
                    self.device_label = "disabled"
                    sig.push("warning", f"whisper not loaded ({exc2}); pace and fillers disabled")
                    return
            else:
                self.device_label = "disabled"
                sig.push("warning", f"whisper not loaded ({exc}); pace and fillers disabled")
                return

        fillers, soft = set(cfg.fillers), set(cfg.soft_fillers)
        try:
            while not self.v.stop.is_set():
                try:
                    seg_start, audio = self.v.segment_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    segments, _ = model.transcribe(
                        audio,
                        language=cfg.language,
                        word_timestamps=True,
                        initial_prompt=self.PROMPT,
                        condition_on_previous_text=False,
                        vad_filter=False,
                        beam_size=1,
                    )
                    words = [w for s in segments for w in (s.words or [])]
                except Exception as exc:
                    if on_gpu:
                        # Catch lazy cuDNN/CUDA initialization failures during first transcribe()
                        try:
                            model = None
                            _release_cuda_cache()
                            cpu_size = cfg.whisper_model if self.explicit_model else "base"
                            model, on_gpu = self._load(WhisperModel, cpu_size, use_gpu=False)
                            model_size = cpu_size
                            print(f"[ORAL COACH] Whisper switched to CPU int8 after CUDA/cuDNN error ({exc})", flush=True)
                            segments, _ = model.transcribe(
                                audio,
                                language=cfg.language,
                                word_timestamps=True,
                                initial_prompt=self.PROMPT,
                                condition_on_previous_text=False,
                                vad_filter=False,
                                beam_size=1,
                            )
                            words = [w for s in segments for w in (s.words or [])]
                        except Exception as exc2:
                            sig.push("warning", f"Whisper transcribe failed: {exc2}")
                            continue
                    else:
                        sig.push("warning", f"Whisper transcribe error: {exc}")
                        continue

                text = []
                prev = ""
                for w in words:
                    token = _WHISPER_WORD_RE.sub("", w.word.lower())
                    if not token:
                        continue
                    t = seg_start + w.end
                    sig.push("word", token, t)
                    text.append(w.word)
                    if token in fillers:
                        sig.push("filler", token, t)
                    elif token in soft or f"{prev} {token}" in soft:
                        sig.push("soft_filler", token if token in soft else f"{prev} {token}", t)
                    prev = token
                if text:
                    sig.push("transcript", "".join(text).strip(), seg_start + (words[-1].end if words else 0))
        finally:
            model = None
            _release_cuda_cache()


def _build_kaggle_voice_sensor(
    cfg,
    signals,
    stop: threading.Event,
    source,
    use_emotion: bool,
    use_whisper: bool,
    gpu_idx: int | None,
    free_gib: float,
    explicit_whisper: bool,
    has_pitch: bool,
):
    """Constructs a `VoiceSensor` using Kaggle-GPU-aware emotion & whisper workers."""
    _ensure_import_path()
    from coach.voice import VoiceSensor

    class _KaggleSafeVoiceSensor(VoiceSensor):
        def __init__(self):
            super().__init__(
                cfg,
                signals,
                stop,
                source=source,
                use_emotion=False,
                use_whisper=False,
            )
            self._want_emotion = use_emotion
            self._want_whisper = use_whisper
            self.emotion_worker = None
            self.whisper_worker = None

        def run(self):
            try:
                from silero_vad import load_silero_vad
                import torch

                self.torch = torch
                self.vad = load_silero_vad()
            except Exception as exc:
                self.error = f"voice setup failed: {exc}"
                return
            if self._want_emotion:
                self.emotion_worker = _GpuVoiceEmotionWorker(self, gpu_idx=gpu_idx)
                self.emotion_worker.start()
            if self._want_whisper:
                self.whisper_worker = _GpuWhisperWorker(
                    self, gpu_idx=gpu_idx, free_gib=free_gib, explicit_model=explicit_whisper
                )
                self.whisper_worker.start()
            try:
                self._loop()
            except Exception as exc:
                self.error = f"voice loop failed: {exc}"

    vs = _KaggleSafeVoiceSensor()
    if not has_pitch:
        vs._pitch = lambda _t: None
    return vs


class BrowserAudioSource:
    """Thread-safe audio source for `coach.voice.VoiceSensor(source=...)`."""

    def __init__(self, target_sr: int = 16000, block_size: int = VOICE_BLOCK):
        self.target_sr = int(target_sr)
        self.block_size = int(block_size)
        self._block_dt = self.block_size / float(self.target_sr)
        self._buf = deque()
        self._buf_len = 0
        self._cond = threading.Condition()
        self._last_block_t = 0.0
        self.last_rms_db: float | None = None

    def push_pcm_int16(self, pcm_b64: str, sample_rate: int = 16000) -> int:
        raw = _decode_base64_payload(pcm_b64)
        if not raw:
            return 0
        import numpy as np

        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if arr.size == 0:
            return 0
        rms = float(np.sqrt(np.mean(arr ** 2)) + 1e-9)
        self.last_rms_db = round(float(20.0 * np.log10(rms)), 2)

        in_sr = int(sample_rate) if sample_rate else self.target_sr
        if in_sr != self.target_sr and in_sr > 0 and arr.size > 1:
            out_len = max(1, int(round(arr.size * (self.target_sr / float(in_sr)))))
            xp = np.linspace(0.0, arr.size - 1.0, arr.size, dtype=np.float32)
            x_new = np.linspace(0.0, arr.size - 1.0, out_len, dtype=np.float32)
            arr = np.interp(x_new, xp, arr).astype(np.float32)

        with self._cond:
            self._buf.append(arr)
            self._buf_len += int(arr.size)
            self._cond.notify_all()
        return int(arr.size)

    def _pop_block(self):
        import numpy as np

        need = self.block_size
        parts = []
        while need > 0:
            head = self._buf[0]
            if head.size <= need:
                parts.append(self._buf.popleft())
                self._buf_len -= int(head.size)
                need -= int(head.size)
            else:
                parts.append(head[:need])
                self._buf[0] = head[need:]
                self._buf_len -= need
                need = 0
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def blocks(self, stop: threading.Event):
        while not stop.is_set():
            with self._cond:
                while not stop.is_set() and self._buf_len < self.block_size:
                    self._cond.wait(timeout=0.2)
                if stop.is_set():
                    break
                block = self._pop_block()
            now = time.time()
            t = max(now, self._last_block_t + self._block_dt * 0.5)
            self._last_block_t = t
            yield t, block


class BrowserFaceAdapter:
    """Wraps `coach.face.FaceSensor` for browser-uploaded JPEG frames without `VideoCapture`."""

    def __init__(self, cfg, signals, stop: threading.Event):
        self.cfg = cfg
        self.sig = signals
        self.stop = stop
        self._sensor = None
        self._ready = False
        self._failed = False
        self._lock = threading.Lock()
        self._start_t = time.time()
        self._last_ts_ms = -1
        self._last_process_t = 0.0
        self._min_period = 1.0 / max(1.0, float(getattr(cfg, "camera_fps", 10.0)))
        self._latest_face_ui = None
        self._latest_face_t = 0.0

    def start_async(self) -> None:
        threading.Thread(target=self._init_sensor, daemon=True, name="oral-face-init").start()

    def _ensure_landmarker_file(self) -> bool:
        path = self.cfg.face_landmarker_path
        if os.path.exists(path):
            return True
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            req = urllib.request.Request(
                FACE_LANDMARKER_URL,
                headers={"User-Agent": "KnowledgeQuest-OralCoach/1.0"},
            )
            with urllib.request.urlopen(req, timeout=12) as resp, open(path, "wb") as out:
                out.write(resp.read())
            return os.path.exists(path)
        except Exception as exc:
            self.sig.push(
                "warning",
                f"face_landmarker.task missing ({exc}); run python Models/OralCoach/setup_models.py",
            )
            return False

    def _init_sensor(self) -> None:
        try:
            if not _has_module("mediapipe") or not _has_module("cv2"):
                miss = [m for m in ("cv2", "mediapipe") if not _has_module(m)]
                self.sig.push(
                    "warning",
                    f"face tracking disabled (missing {', '.join(miss)}; install requirements_kaggle.txt)",
                )
                self._failed = True
                return
            if not self._ensure_landmarker_file():
                self._failed = True
                return
            _ensure_import_path()
            from coach.face import FaceSensor

            sensor = FaceSensor(self.cfg, self.sig, self.stop)
            sensor._load()
            if self.stop.is_set():
                return
            self._sensor = sensor
            self._ready = True
        except Exception as exc:
            self._failed = True
            self.sig.push("warning", f"face model init failed ({exc}); face metrics disabled")

    def ingest_jpeg(self, frame_b64: str, now: float | None = None) -> None:
        if not self._ready or self._sensor is None or self.stop.is_set():
            return
        now = time.time() if now is None else now
        if now - self._last_process_t < self._min_period * 0.85:
            return
        raw = _decode_base64_payload(frame_b64)
        if not raw:
            return
        from PIL import Image
        import numpy as np

        with Image.open(io.BytesIO(raw)) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

        with self._lock:
            self._last_process_t = now
            ts_ms = max(self._last_ts_ms + 1, int((now - self._start_t) * 1000.0))
            self._last_ts_ms = ts_ms
            self._sensor.process(rgb, now, ts_ms)

        _, face_view = self._sensor.view()
        if face_view and face_view.get("points"):
            pts = face_view["points"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            self._latest_face_ui = {
                "looking": bool(face_view.get("looking", True)),
                "yaw": round(float(face_view.get("yaw", 0.0)), 1),
                "pitch": round(float(face_view.get("pitch", 0.0)), 1),
                "bbox": [
                    round(max(0.0, min(xs)), 4),
                    round(max(0.0, min(ys)), 4),
                    round(min(1.0, max(xs)), 4),
                    round(min(1.0, max(ys)), 4),
                ],
            }
            self._latest_face_t = time.time()
        else:
            self._latest_face_ui = None

    def get_face_ui(self, now: float | None = None) -> dict | None:
        now = time.time() if now is None else now
        if self._latest_face_ui is not None and (now - self._latest_face_t) < 0.8:
            return self._latest_face_ui
        return None


class OralCoachSession:
    """Manages one live Oral Presentation Coach session (Kaggle Multi-GPU & local)."""

    def __init__(
        self,
        session_id: str,
        whisper_model: str | None = None,
        language: str | None = None,
        course_title: str = "",
        tip_writer_mode: str = "auto",
        llm_orchestrator=None,
    ):
        _ensure_import_path()
        from coach.channels import ChannelBuilder
        from coach.config import Config
        from coach.juice import GameFeel
        from coach.recommend import Coach, Knowledge
        from coach.signals import Signals

        self.session_id = session_id
        self.course_title = course_title or ""
        self.llm_orchestrator = llm_orchestrator
        self.stop_event = threading.Event()
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._warnings: list[str] = []
        self._seen_warnings: set[str] = set()
        self._event_buf: deque[dict] = deque(maxlen=200)
        self._seq = 0
        self._tips_shown = 0
        self._tips_worked = 0
        self._praises = 0
        self._total_xp_earned = 0
        self._last_snap: dict | None = None
        self._log_file = None
        self.log_path = ""

        # Pause background RPG encounter prefetching while Oral Coach is actively using GPU
        self._paused_vlm_prefetch = False
        if self.llm_orchestrator is not None and hasattr(self.llm_orchestrator, "set_vlm_busy"):
            try:
                self.llm_orchestrator.set_vlm_busy(True)
                self._paused_vlm_prefetch = True
            except Exception:
                pass

        best_gpu_idx, best_free_gib = _pick_best_cuda_device(min_free_gib=1.8)
        self.best_gpu_idx = best_gpu_idx
        self.best_free_gib = best_free_gib

        cfg = Config()
        cfg.face_landmarker_path = FACE_LANDMARKER_PATH
        explicit_whisper = bool(whisper_model)
        if whisper_model:
            cfg.whisper_model = whisper_model
        elif best_gpu_idx is not None and best_free_gib >= 3.0:
            cfg.whisper_model = "small"
        else:
            cfg.whisper_model = "base"
        cfg.device = f"cuda:{best_gpu_idx}" if best_gpu_idx is not None else "cpu"
        if language in ("en", "fr"):
            cfg.language = language
        self.cfg = cfg

        self.signals = Signals()
        self.builder = ChannelBuilder(cfg, self.signals)
        lang_for_writer = cfg.language or "en"
        self.writer = _build_stage4_tip_writer(
            llm_orchestrator=self.llm_orchestrator,
            mode=tip_writer_mode,
            language=lang_for_writer,
        )
        self.knowledge = Knowledge(KNOWLEDGE_DIR)
        self.recommender = Coach(
            cfg,
            self.knowledge,
            self.writer,
            on_tip=lambda tip: self.builder.restart(tip["restart"], tip["t"]),
        )
        self.feel = GameFeel(sfx=None)
        self.feel.start()

        os.makedirs(LOGS_DIR, exist_ok=True)
        self.log_path = os.path.join(
            LOGS_DIR, time.strftime("session-%Y%m%d-%H%M%S.jsonl")
        )
        self._log_file = open(self.log_path, "a", encoding="utf-8")

        self.has_vad = _has_module("silero_vad") and _has_module("numpy")
        self.has_whisper = _has_module("faster_whisper")
        self.has_voice_emotion = _has_module("transformers") and _has_module("torch")
        self.has_pitch = _has_module("parselmouth")
        self.has_face = _has_module("mediapipe") and _has_module("cv2")

        self.audio_source = BrowserAudioSource(target_sr=cfg.sample_rate)
        self.face_adapter = BrowserFaceAdapter(cfg, self.signals, self.stop_event)
        self.voice_sensor = None
        self._explicit_whisper = explicit_whisper

        self._start_workers()
        self._ticker_thread = threading.Thread(
            target=self._tick_loop, daemon=True, name="oral-coach-ticker"
        )
        self._ticker_thread.start()

    def _record_warning(self, w: str) -> None:
        if w and w not in self._seen_warnings:
            self._seen_warnings.add(w)
            self._warnings.append(w)

    def _start_workers(self) -> None:
        self.face_adapter.start_async()

        if self.has_vad:
            try:
                vs = _build_kaggle_voice_sensor(
                    cfg=self.cfg,
                    signals=self.signals,
                    stop=self.stop_event,
                    source=self.audio_source,
                    use_emotion=self.has_voice_emotion,
                    use_whisper=self.has_whisper,
                    gpu_idx=self.best_gpu_idx,
                    free_gib=self.best_free_gib,
                    explicit_whisper=self._explicit_whisper,
                    has_pitch=self.has_pitch,
                )
                if not self.has_pitch:
                    self._record_warning("praat-parselmouth not installed; pitch/intonation disabled")
                if not self.has_whisper:
                    self._record_warning(
                        "faster-whisper not installed; server pace/fillers disabled (using browser captions)"
                    )
                if not self.has_voice_emotion:
                    self._record_warning("transformers/torch not available; voice emotion disabled")
                vs.start()
                self.voice_sensor = vs
            except Exception as exc:
                self._record_warning(f"voice sensor failed to start: {exc}")
        else:
            self._record_warning(
                "silero-vad not installed; install Models/OralCoach/requirements_kaggle.txt for full speech/prosody scoring"
            )

    def _collect_warnings(self) -> None:
        if self.voice_sensor is not None and getattr(self.voice_sensor, "error", None):
            self._record_warning(self.voice_sensor.error)
        if getattr(self.writer, "warning", None):
            self._record_warning(self.writer.warning)
        for w in self.signals.values("warning", 3600):
            self._record_warning(w)

    def get_device_info(self) -> dict:
        ew = getattr(self.voice_sensor, "emotion_worker", None)
        ww = getattr(self.voice_sensor, "whisper_worker", None)
        return {
            "best_cuda_device": f"cuda:{self.best_gpu_idx}" if self.best_gpu_idx is not None else None,
            "free_cuda_gib": round(self.best_free_gib, 2) if self.best_free_gib else 0.0,
            "voice_emotion_device": getattr(ew, "device_label", "disabled"),
            "whisper_device": getattr(ww, "device_label", "disabled"),
            "tip_writer_backend": self.writer.status,
            "vlm_prefetch_paused": self._paused_vlm_prefetch,
        }

    def _emit_events(self, events: list[dict], snap: dict) -> None:
        xp_before = (self.feel.level, self.feel.xp)
        self.feel.update(snap, events, self.recommender.view(time.time()))
        lvl_diff = self.feel.level - xp_before[0]
        if lvl_diff == 0:
            self._total_xp_earned += max(0, self.feel.xp - xp_before[1])
        else:
            from coach.juice import xp_needed

            gained = self.feel.xp
            for lv in range(xp_before[0], self.feel.level):
                gained += xp_needed(lv)
            gained -= xp_before[1]
            self._total_xp_earned += max(0, gained)

        for ev in events:
            t = ev.get("type")
            if t == "tip":
                self._tips_shown += 1
            elif t == "tip_result" and ev.get("worked"):
                self._tips_worked += 1
            elif t == "praise":
                self._praises += 1
            self._seq += 1
            self._event_buf.append({"seq": self._seq, **ev})

        if self._log_file is not None:
            try:
                for item in [snap] + events:
                    self._log_file.write(json.dumps(item) + "\n")
                self._log_file.flush()
            except Exception:
                pass

    def step_once(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            self._collect_warnings()
            snap = self.builder.tick(now)
            events = self.recommender.step(snap)
            self._emit_events(events, snap)
            self._last_snap = snap
            return snap

    def _tick_loop(self) -> None:
        next_t = time.time() + self.cfg.tick_seconds
        while not self.stop_event.is_set():
            dt = next_t - time.time()
            if dt > 0:
                self.stop_event.wait(min(0.25, dt))
                continue
            next_t += self.cfg.tick_seconds
            if not self.stop_event.is_set():
                try:
                    self.step_once()
                except Exception as exc:
                    self._record_warning(f"coach tick error: {exc}")

    def mute_for(self, seconds: float) -> float:
        dur = max(0.0, min(30.0, float(seconds)))
        until = time.time() + dur
        self.signals.mute_until = max(getattr(self.signals, "mute_until", 0.0), until)
        return dur

    def ingest(
        self,
        frame_b64: str | None = None,
        pcm16_b64: str | None = None,
        sample_rate: int = 16000,
        transcript: str | None = None,
        since_seq: int = 0,
    ) -> dict:
        now = time.time()
        if frame_b64:
            try:
                self.face_adapter.ingest_jpeg(frame_b64, now=now)
            except Exception as exc:
                self._record_warning(f"frame decode error: {exc}")
        if pcm16_b64:
            try:
                self.audio_source.push_pcm_int16(pcm16_b64, sample_rate=sample_rate)
            except Exception as exc:
                self._record_warning(f"audio ingest error: {exc}")
        if transcript and not self.has_whisper:
            clean = str(transcript).strip()
            if clean:
                self.signals.push("transcript", clean, now)
        return self.get_state(since_seq=since_seq)

    def get_state(self, since_seq: int = 0) -> dict:
        now = time.time()
        with self._lock:
            self._collect_warnings()
            cv = self.recommender.view(now)
            feel = self.feel
            popups = [p for p in feel.popups if 0.0 <= (now - p["t"]) < 2.5]
            flash = None
            if feel.flash and 0.0 <= (now - feel.flash["t"]) < 0.45:
                flash = feel.flash
            result_banner = None
            if feel.result_banner and 0.0 <= (now - feel.result_banner["t"]) < 4.5:
                result_banner = feel.result_banner
            events = [ev for ev in self._event_buf if ev["seq"] > since_seq]
            return {
                "active": not self.stop_event.is_set(),
                "session_id": self.session_id,
                "course_title": self.course_title,
                "elapsed_s": round(now - self.started_at, 1),
                "snap": self._last_snap,
                "coach": cv,
                "device_info": self.get_device_info(),
                "face": self.face_adapter.get_face_ui(now=now),
                "feel": {
                    "level": feel.level,
                    "xp": feel.xp,
                    "xp_frac": round(feel.xp_frac(), 3),
                    "total_xp_earned": self._total_xp_earned,
                    "red_rows": sorted(feel.red_rows()),
                    "popups": popups,
                    "flash": flash,
                    "levelup": (now - feel.levelup_t) < 2.2,
                    "result_banner": result_banner,
                },
                "audio_rms_db": self.audio_source.last_rms_db,
                "warnings": list(self._warnings),
                "events": events,
                "seq": self._seq,
            }

    def stop(self) -> dict:
        self.stop_event.set()
        if self._ticker_thread.is_alive():
            self._ticker_thread.join(timeout=1.5)
        if self.voice_sensor is not None and self.voice_sensor.is_alive():
            self.voice_sensor.join(timeout=2.0)
        if self._paused_vlm_prefetch and self.llm_orchestrator is not None:
            try:
                self.llm_orchestrator.set_vlm_busy(False)
            except Exception:
                pass
            self._paused_vlm_prefetch = False
        _release_cuda_cache()
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        return {
            "session_id": self.session_id,
            "duration_s": round(time.time() - self.started_at, 1),
            "level": self.feel.level,
            "xp": self.feel.xp,
            "xp_earned": self._total_xp_earned,
            "tips_shown": self._tips_shown,
            "tips_worked": self._tips_worked,
            "praises": self._praises,
            "log_file": os.path.relpath(self.log_path, ORAL_COACH_ROOT).replace("\\", "/"),
        }


class OralCoachService:
    """Thread-safe coordinator owned by `GameServerState`."""

    def __init__(self, llm_orchestrator=None):
        self._lock = threading.Lock()
        self.llm_orchestrator = llm_orchestrator
        self._session: OralCoachSession | None = None
        self._counter = 0
        self._desktop_proc: subprocess.Popen | None = None

    def get_capabilities(self) -> dict:
        available = {}
        missing_browser = []
        missing_desktop = []
        for mod_name, pip_name in BROWSER_REQUIRED_MODULES:
            ok = _has_module(mod_name)
            available[mod_name] = ok
            if not ok:
                missing_browser.append(pip_name)
        for mod_name, pip_name in DESKTOP_ONLY_MODULES:
            ok = _has_module(mod_name)
            available[mod_name] = ok
            if not ok:
                missing_desktop.append(pip_name)

        best_idx, free_gib = _pick_best_cuda_device(min_free_gib=1.8)
        stylizer = getattr(self.llm_orchestrator, "stylizer", None) if self.llm_orchestrator else None
        with self._lock:
            sess = self._session
        return {
            "available_modules": available,
            "missing_packages": missing_browser,
            "desktop_missing_packages": missing_desktop,
            "face_model_ready": os.path.exists(FACE_LANDMARKER_PATH),
            "best_cuda_device": f"cuda:{best_idx}" if best_idx is not None else None,
            "free_cuda_gib": round(free_gib, 2) if free_gib else None,
            "stage4_llm_tip_writer": bool(stylizer is not None and stylizer.is_live()),
            "install_cmd": "pip install -r Models/OralCoach/requirements.txt && python Models/OralCoach/setup_models.py",
            "kaggle_install_cmd": "pip install -q -r Models/OralCoach/requirements_kaggle.txt && python Models/OralCoach/setup_models.py --whisper-model base",
            "desktop_supported": bool(os.name == "nt" or os.environ.get("DISPLAY")),
            "active": bool(sess is not None and not sess.stop_event.is_set()),
            "session_id": sess.session_id if sess is not None else None,
        }

    def start_session(
        self,
        whisper_model: str | None = None,
        language: str | None = None,
        course_title: str = "",
        tip_writer_mode: str = "auto",
    ) -> dict:
        with self._lock:
            if self._session is not None and not self._session.stop_event.is_set():
                self._session.stop()
            self._counter += 1
            sid = f"oc-{int(time.time())}-{self._counter}"
            self._session = OralCoachSession(
                session_id=sid,
                whisper_model=whisper_model,
                language=language,
                course_title=course_title,
                tip_writer_mode=tip_writer_mode,
                llm_orchestrator=self.llm_orchestrator,
            )
            state = self._session.get_state(since_seq=0)
        state["capabilities"] = self.get_capabilities()
        return state

    def get_status(self, since_seq: int = 0) -> dict:
        with self._lock:
            sess = self._session
        caps = self.get_capabilities()
        if sess is None or sess.stop_event.is_set():
            return {"active": False, "capabilities": caps, "seq": 0, "events": []}
        out = sess.get_state(since_seq=since_seq)
        out["capabilities"] = caps
        return out

    def ingest(
        self,
        frame_b64: str | None = None,
        pcm16_b64: str | None = None,
        sample_rate: int = 16000,
        transcript: str | None = None,
        since_seq: int = 0,
    ) -> dict:
        with self._lock:
            sess = self._session
        if sess is None or sess.stop_event.is_set():
            return {"active": False, "error": "No active Oral Coach session."}
        return sess.ingest(
            frame_b64=frame_b64,
            pcm16_b64=pcm16_b64,
            sample_rate=sample_rate,
            transcript=transcript,
            since_seq=since_seq,
        )

    def mute(self, seconds: float = 3.0) -> dict:
        with self._lock:
            sess = self._session
        if sess is None or sess.stop_event.is_set():
            return {"active": False, "muted_s": 0.0}
        return {"active": True, "muted_s": sess.mute_for(seconds)}

    def stop_session(self) -> dict:
        with self._lock:
            sess = self._session
            self._session = None
        if sess is None:
            return {"stopped": False, "summary": None}
        return {"stopped": True, "summary": sess.stop()}

    def launch_desktop(self, whisper_model: str | None = None, speak: bool = False) -> dict:
        caps = self.get_capabilities()
        if not caps["desktop_supported"]:
            return {"launched": False, "error": "Desktop GUI unavailable in headless environment."}
        missing_all = list(caps["missing_packages"]) + list(caps["desktop_missing_packages"])
        if missing_all:
            return {
                "launched": False,
                "error": f"Missing packages: {', '.join(missing_all)}",
                "install_cmd": caps["install_cmd"],
            }
        with self._lock:
            if self._desktop_proc is not None and self._desktop_proc.poll() is None:
                return {"launched": True, "pid": self._desktop_proc.pid, "already_running": True}
            cmd = [sys.executable, "run.py"]
            if whisper_model:
                cmd.extend(["--whisper-model", str(whisper_model)])
            if speak:
                cmd.append("--speak")
            self._desktop_proc = subprocess.Popen(cmd, cwd=ORAL_COACH_ROOT)
            return {"launched": True, "pid": self._desktop_proc.pid}
