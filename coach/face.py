"""Webcam sensor: MediaPipe Face Landmarker (gaze, head pose, blinks, smile)
+ HSEmotion (8 emotions, valence, arousal). Runs in its own thread."""
import math
import threading
import time
import urllib.request  # noqa: F401  (hsemotion_onnx forgets to import this before downloading)

import cv2
import numpy as np

from .config import Config
from .signals import Signals

EMOTIONS_8 = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]


def head_angles(matrix) -> tuple[float, float]:
    """Yaw (left/right) and pitch (up/down) in degrees from MediaPipe's 4x4 face transform."""
    r = np.asarray(matrix)[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    return yaw, pitch


def gaze_from_blendshapes(bs: dict) -> dict:
    """Average left/right eye gaze blendshapes into 4 directions (0..1)."""
    def avg(a, b):
        return (bs.get(a, 0.0) + bs.get(b, 0.0)) / 2
    return {
        "down": avg("eyeLookDownLeft", "eyeLookDownRight"),
        "up": avg("eyeLookUpLeft", "eyeLookUpRight"),
        # looking to one side = one eye "out", the other "in"
        "side": max(avg("eyeLookOutLeft", "eyeLookInRight"), avg("eyeLookInLeft", "eyeLookOutRight")),
    }


class FaceSensor(threading.Thread):
    def __init__(self, cfg: Config, signals: Signals, stop: threading.Event):
        super().__init__(daemon=True, name="face")
        self.cfg, self.sig, self.stop = cfg, signals, stop
        self.error: str | None = None
        self._blink_closed = False
        # latest camera frame + face geometry, read by the live window (main thread)
        self._view_lock = threading.Lock()
        self._frame = None      # BGR image
        self._face = None       # dict: points (normalised x,y), looking, yaw, pitch
        self._face_t = 0.0

    def view(self):
        """(frame, face) for display. face is None if no face was seen in the last 0.5 s."""
        with self._view_lock:
            face = self._face if time.time() - self._face_t < 0.5 else None
            return self._frame, face

    def _load(self):
        from mediapipe.tasks.python import BaseOptions, vision
        import mediapipe as mp
        import os
        self.mp = mp
        if not os.path.exists(self.cfg.face_landmarker_path):
            raise RuntimeError(f"{self.cfg.face_landmarker_path} is missing: run `python setup_models.py` "
                               "from the oral-coach folder first")
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.cfg.face_landmarker_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.emotion = None
        if not self.cfg.hsemotion_model:  # light mode
            return
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            self.emotion = HSEmotionRecognizer(model_name=self.cfg.hsemotion_model)
        except Exception as e:  # keep running with MediaPipe only (= light fallback)
            self.emotion = None
            self.sig.push("warning", f"HSEmotion not loaded ({e}); face emotions disabled")

    def run(self):
        try:
            self._load()
            cap = cv2.VideoCapture(self.cfg.camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open camera {self.cfg.camera_index}")
        except Exception as e:
            self.error = str(e)
            return
        period = 1.0 / self.cfg.camera_fps
        is_file = isinstance(self.cfg.camera_index, str)
        file_dt = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0.0
        start = time.time()
        last_analysis = 0.0
        while not self.stop.is_set():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._view_lock:
                self._frame = frame  # display runs at camera speed ...
            if t0 - last_analysis >= period:  # ... analysis at camera_fps (default 10)
                last_analysis = t0
                try:
                    self.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t0, int((t0 - start) * 1000))
                except Exception as e:
                    self.sig.push("warning", f"face frame error: {e}")
            if is_file:
                time.sleep(max(0.0, file_dt - (time.time() - t0)))
        cap.release()

    def process(self, rgb: np.ndarray, t: float, ts_ms: int):
        """Analyse one RGB frame and push measurements. Public so it can be tested on images."""
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect_for_video(image, ts_ms)
        if not res.face_landmarks:
            self.sig.push("face_present", 0.0, t)
            # no face found (out of frame, bad light, hand in front) is not "looking away": push None so eye
            # contact is measured on the frames with a face only, and is unknown (no tips) without any
            self.sig.push("looking", None, t)
            return
        self.sig.push("face_present", 1.0, t)

        bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
        yaw, pitch = head_angles(res.facial_transformation_matrixes[0])
        gaze = gaze_from_blendshapes(bs)
        c = self.cfg
        head_away = abs(yaw) > c.look_away_yaw_deg or abs(pitch) > c.look_away_pitch_deg
        eyes_away = max(gaze.values()) > c.look_away_blendshape
        looking = 0.0 if (head_away or eyes_away) else 1.0

        blink_val = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
        if blink_val > c.blink_threshold and not self._blink_closed:
            self.sig.push("blink", 1.0, t)
        self._blink_closed = blink_val > c.blink_threshold

        push = self.sig.push
        push("looking", looking, t)
        with self._view_lock:
            self._face = {"points": [(p.x, p.y) for p in res.face_landmarks[0]],
                          "looking": bool(looking), "yaw": yaw, "pitch": pitch}
            self._face_t = time.time()
        push("looking_down", 1.0 if (gaze["down"] > c.look_away_blendshape or pitch < -c.look_away_pitch_deg) else 0.0, t)
        push("head_yaw", yaw, t)
        push("head_pitch", pitch, t)
        push("smile", (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2, t)
        push("brow_down", (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2, t)
        push("lip_press", (bs.get("mouthPressLeft", 0) + bs.get("mouthPressRight", 0)) / 2, t)

        if self.emotion is not None:
            crop = self._crop(rgb, res.face_landmarks[0])
            if crop is not None:
                _, scores = self.emotion.predict_emotions(crop, logits=False)
                probs = dict(zip(EMOTIONS_8, map(float, scores[:8])))
                push("face_emotions", probs, t)
                if len(scores) >= 10:  # *_va_mtl models append valence, arousal
                    push("face_valence", float(scores[-2]), t)
                    push("face_arousal", float(scores[-1]), t)

    @staticmethod
    def _crop(rgb, landmarks, pad=0.12):
        h, w = rgb.shape[:2]
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        x0, x1 = int(max(0, (x0 - px) * w)), int(min(w, (x1 + px) * w))
        y0, y1 = int(max(0, (y0 - py) * h)), int(min(h, (y1 + py) * h))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        return rgb[y0:y1, x0:x1]
