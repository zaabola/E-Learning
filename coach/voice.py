"""Microphone sensor.

One capture thread feeds 32 ms blocks to:
  * Silero VAD          -> speaking / pauses
  * RMS + Parselmouth   -> volume, pitch level, pitch variation (monotone)
  * audEERING wav2vec2  -> voice arousal, dominance, valence   (worker thread)
  * faster-whisper      -> words with timestamps, pace, fillers (worker thread)
"""
import queue
import re
import threading
import time

import numpy as np

from .config import Config
from .signals import Signals

BLOCK = 512  # Silero VAD needs exactly 512 samples at 16 kHz (32 ms)


def pick_device(cfg: Config) -> str:
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------- audio sources
class MicSource:
    def __init__(self, cfg: Config):
        import sounddevice as sd
        self.q: queue.Queue = queue.Queue()
        self.stream = sd.InputStream(samplerate=cfg.sample_rate, channels=1, dtype="float32",
                                     blocksize=BLOCK, device=cfg.mic_device,
                                     callback=lambda data, *_: self.q.put(data[:, 0].copy()))
        self.stream.start()

    def blocks(self, stop: threading.Event):
        while not stop.is_set():
            try:
                yield time.time(), self.q.get(timeout=0.5)
            except queue.Empty:
                continue
        self.stream.stop()


# ---------------------------------------------------------------- main sensor
class VoiceSensor(threading.Thread):
    def __init__(self, cfg: Config, signals: Signals, stop: threading.Event,
                 source=None, use_emotion=True, use_whisper=True):
        super().__init__(daemon=True, name="voice")
        self.cfg, self.sig, self.stop = cfg, signals, stop
        self.source = source
        self.use_emotion, self.use_whisper = use_emotion, use_whisper
        self.error: str | None = None
        sr = cfg.sample_rate
        self.ring = np.zeros(int(sr * 30), dtype=np.float32)  # last 30 s of audio
        self.ring_end_t = 0.0
        self.ring_lock = threading.Lock()
        self.segment_q: queue.Queue = queue.Queue(maxsize=8)

    # -- ring buffer helpers
    def _append(self, t, block):
        with self.ring_lock:
            self.ring = np.roll(self.ring, -len(block))
            self.ring[-len(block):] = block
            self.ring_end_t = t

    def last_audio(self, seconds):
        n = int(seconds * self.cfg.sample_rate)
        with self.ring_lock:
            return self.ring[-n:].copy(), self.ring_end_t

    # -- thread body
    def run(self):
        try:
            from silero_vad import load_silero_vad
            import torch
            self.torch = torch
            self.vad = load_silero_vad()
            if self.source is None:
                self.source = MicSource(self.cfg)
        except Exception as e:
            self.error = f"voice setup failed: {e}"
            return
        if self.use_emotion:
            VoiceEmotionWorker(self).start()
        if self.use_whisper:
            WhisperWorker(self).start()
        try:
            self._loop()
        except Exception as e:
            self.error = f"voice loop failed: {e}"

    def _loop(self):
        c, sr = self.cfg, self.cfg.sample_rate
        speaking = False
        silence_start = None
        seg_audio, seg_start = [], None
        last_pitch = 0.0
        muted_at = None
        for t, block in self.source.blocks(self.stop):
            if t < getattr(self.sig, "mute_until", 0.0):  # spoken tip playing (--speak): not your voice
                if muted_at is None:
                    muted_at = t
                    if seg_start is not None and seg_audio:  # your phrase so far still counts: send it to
                        try:                                  # speech-to-text instead of throwing it away
                            self.segment_q.put_nowait((seg_start, np.concatenate(seg_audio)))
                        except queue.Full:
                            pass
                    seg_audio, seg_start = [], None
                if silence_start is not None:  # keep a running silence where it was (no fake "you froze")
                    self.sig.push("silence_now", muted_at - silence_start, t)
                continue
            if muted_at is not None:  # the coach's talking time does not count as your silence
                if silence_start is not None:
                    silence_start += t - muted_at
                muted_at = None
            self._append(t, block)
            prob = float(self.vad(self.torch.from_numpy(block), sr).item())
            is_speech = prob >= c.vad_threshold
            self.sig.push("speaking", 1.0 if is_speech else 0.0, t)

            if is_speech:
                rms = float(np.sqrt(np.mean(block ** 2)) + 1e-9)
                self.sig.push("volume_db", 20 * np.log10(rms), t)
                if silence_start is not None and speaking:
                    gap = t - silence_start
                    if gap >= c.min_pause:
                        self.sig.push("pause", gap, t)  # a finished pause and its length
                silence_start = None
                speaking = True
                if seg_start is None:
                    seg_start = t - BLOCK / sr
                seg_audio.append(block)
            else:
                if silence_start is None:
                    silence_start = t
                if seg_start is not None:
                    seg_audio.append(block)  # keep a little trailing silence for Whisper
            self.sig.push("silence_now", 0.0 if silence_start is None else t - silence_start, t)

            # close the phrase on a pause or when it gets long, send it to Whisper
            if seg_start is not None:
                seg_len = len(seg_audio) * BLOCK / sr
                if (silence_start is not None and t - silence_start >= c.segment_end_silence) or seg_len >= c.max_segment:
                    try:
                        self.segment_q.put_nowait((seg_start, np.concatenate(seg_audio)))
                    except queue.Full:
                        self.sig.push("warning", "speech-to-text is falling behind; try a smaller whisper model")
                    seg_audio, seg_start = [], None

            if t - last_pitch >= 0.5:
                last_pitch = t
                self._pitch(t)

    def _pitch(self, t):
        import parselmouth
        audio, _ = self.last_audio(self.cfg.pitch_window)
        if not any(self.sig.values("speaking", 1.0)):
            return
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=self.cfg.sample_rate)
        f0 = snd.to_pitch(time_step=0.01, pitch_floor=70, pitch_ceiling=400).selected_array["frequency"]
        f0 = f0[f0 > 0]
        if len(f0) < 20:
            return
        st = 12 * np.log2(f0 / 100.0)  # semitones re 100 Hz: comparable across voices
        self.sig.push("pitch_st", float(np.median(st)), t)
        self.sig.push("pitch_var", float(np.std(st)), t)  # low = monotone


class VoiceEmotionWorker(threading.Thread):
    """audEERING wav2vec2: continuous arousal / dominance / valence from the voice."""
    def __init__(self, voice: VoiceSensor):
        super().__init__(daemon=True, name="voice-emotion")
        self.v = voice

    def run(self):
        cfg, sig = self.v.cfg, self.v.sig
        try:
            model, processor, torch, device = load_audeering(cfg)
        except Exception as e:
            sig.push("warning", f"voice emotion model not loaded ({e}); voice arousal/dominance disabled")
            return
        while not self.v.stop.is_set():
            time.sleep(cfg.voice_emotion_every)
            speech_share = sig.mean("speaking", cfg.voice_emotion_window)
            if not speech_share or speech_share < 0.5:
                continue
            audio, t = self.v.last_audio(cfg.voice_emotion_window)
            x = processor(audio, sampling_rate=cfg.sample_rate, return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                _, out = model(x)
            arousal, dominance, valence = (float(v) for v in out[0].cpu())
            sig.push("voice_arousal", arousal, t)
            sig.push("voice_dominance", dominance, t)
            sig.push("voice_valence", valence, t)


def load_audeering(cfg: Config):
    """Model definition copied from the audEERING model card (it is not a stock transformers class)."""
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            x = self.dropout(x)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return hidden, self.classifier(hidden)

    device = pick_device(cfg)
    processor = Wav2Vec2Processor.from_pretrained(cfg.voice_emotion_model)
    model = EmotionModel.from_pretrained(cfg.voice_emotion_model).to(device).eval()
    return model, processor, torch, device


_WORD = re.compile(r"[^\w']+")


class WhisperWorker(threading.Thread):
    """faster-whisper on each finished phrase: words, pace and fillers."""
    # Whisper tends to delete fillers; a disfluent prompt nudges it to keep them.
    PROMPT = "Umm, so, uh, I think, like, euh... you know, hmm, the main point is"

    def __init__(self, voice: VoiceSensor):
        super().__init__(daemon=True, name="whisper")
        self.v = voice

    def run(self):
        cfg, sig = self.v.cfg, self.v.sig
        try:
            from faster_whisper import WhisperModel
            device = pick_device(cfg)
            model = WhisperModel(cfg.whisper_model, device=device,
                                 compute_type="float16" if device == "cuda" else "int8")
        except Exception as e:
            sig.push("warning", f"whisper not loaded ({e}); pace and fillers disabled")
            return
        fillers, soft = set(cfg.fillers), set(cfg.soft_fillers)
        while not self.v.stop.is_set():
            try:
                seg_start, audio = self.v.segment_q.get(timeout=0.5)
            except queue.Empty:
                continue
            segments, _ = model.transcribe(audio, language=cfg.language, word_timestamps=True,
                                           initial_prompt=self.PROMPT, condition_on_previous_text=False,
                                           vad_filter=False, beam_size=1)
            words = [w for s in segments for w in (s.words or [])]
            text = []
            prev = ""
            for w in words:
                token = _WORD.sub("", w.word.lower())
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
