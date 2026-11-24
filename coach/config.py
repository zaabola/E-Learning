"""All tunable numbers in one place. Pattern thresholds live at the top of coach/recommend/patterns.py."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- inputs ---
    camera_index: int = 0
    camera_fps: float = 10.0          # face analysis rate; 5-10 is plenty for coaching
    sample_rate: int = 16000          # all voice models expect 16 kHz mono
    mic_device: int | None = None     # None = system default mic

    # --- models ---
    face_landmarker_path: str = "models/face_landmarker.task"
    hsemotion_model: str = "enet_b0_8_va_mtl"   # 8 emotions + valence + arousal
    voice_emotion_model: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    whisper_model: str = "small"                 # "base" if slow, "medium" if your GPU is strong
    language: str | None = None                  # "en", "fr" ... None = auto-detect
    device: str = "auto"                         # "cuda", "cpu" or "auto"

    # --- timing ---
    tick_seconds: float = 1.0          # how often a snapshot is printed / emitted
    voice_emotion_window: float = 3.0  # seconds of audio per voice-emotion estimate
    voice_emotion_every: float = 1.5
    pitch_window: float = 3.0
    calibration_seconds: float = 30.0  # seconds of *speech* used as the personal baseline
    eye_contact_window: float = 5.0    # eye contact = share of the last N seconds looking at camera
    pace_window: float = 30.0          # words per minute over the last N seconds
    filler_window: float = 30.0

    # --- thresholds ---
    vad_threshold: float = 0.5         # Silero speech probability
    min_pause: float = 0.3             # silence shorter than this is not a pause
    segment_end_silence: float = 0.5   # silence that closes a phrase sent to Whisper
    max_segment: float = 8.0           # longest phrase sent to Whisper at once
    look_away_yaw_deg: float = 25.0
    look_away_pitch_deg: float = 20.0
    look_away_blendshape: float = 0.55
    blink_threshold: float = 0.5

    # --- coach (recommendations) ---
    coach_tip_cooldown: float = 25.0     # min seconds between two tips (a freeze may interrupt after 8 s)
    coach_check_delay: float = 6.0       # give the student a few seconds to act on a tip...
    coach_check_window: float = 20.0     # ...then judge it on the badness until 20 s after the tip
    coach_escalate_within: float = 90.0  # same problem again within this time after a failed tip -> next level
    coach_pattern_rest: float = 45.0     # after a tip worked, leave that problem alone for a while
    coach_give_up_rest: float = 90.0     # after a level-3 pause did not help, stop nagging about it
    coach_praise_after: float = 45.0     # seconds with no problem before a "steady" praise
    coach_tip_show: float = 12.0         # seconds a tip stays big on screen

    fillers: tuple = field(default_factory=lambda: (
        # English
        "um", "umm", "uh", "uhh", "erm", "er", "ah", "hmm", "mm",
        # French
        "euh", "heu", "ben", "bah", "hein",
    ))
    # multi-word / context-dependent fillers are counted separately (lower confidence)
    soft_fillers: tuple = field(default_factory=lambda: ("like", "you know", "basically", "genre", "du coup", "en fait"))
