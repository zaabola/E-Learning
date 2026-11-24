"""Thread-safe store of timestamped raw measurements.

Sensors (face, voice) push samples here from their own threads; the channel
builder reads windows out of it once per tick. Keeping raw samples (not just
the latest value) is what makes "for how long" and "dropped since" questions
answerable later.
"""
import threading
import time
from collections import defaultdict, deque


class Signals:
    def __init__(self, keep_seconds: float = 120.0):
        self._data = defaultdict(deque)
        self._lock = threading.Lock()
        self.keep = keep_seconds

    def push(self, name: str, value, t: float | None = None):
        t = time.time() if t is None else t
        with self._lock:
            q = self._data[name]
            q.append((t, value))
            while q and q[0][0] < t - self.keep:
                q.popleft()

    def window(self, name: str, seconds: float, now: float | None = None):
        now = time.time() if now is None else now
        with self._lock:
            return [(t, v) for t, v in self._data.get(name, ()) if t >= now - seconds]

    def values(self, name: str, seconds: float, now: float | None = None):
        return [v for _, v in self.window(name, seconds, now)]

    def mean(self, name: str, seconds: float, now: float | None = None):
        vals = [v for v in self.values(name, seconds, now) if v is not None]
        return sum(vals) / len(vals) if vals else None

    def latest(self, name: str, max_age: float = 5.0, now: float | None = None):
        now = time.time() if now is None else now
        with self._lock:
            q = self._data.get(name)
            if not q:
                return None
            t, v = q[-1]
        return v if now - t <= max_age else None
