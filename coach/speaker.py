"""Spoken tips: reads a short version of each tip out loud, so you can keep looking at the camera.

Off by default. Turn it on with `python run.py --speak`, or press v in the window.

  * Offline text-to-speech, nothing to download: on Windows the built-in SAPI voices (through comtypes,
    or pyttsx3), else pyttsx3 / `say` (macOS) / `espeak` (Linux), else Windows PowerShell's speech.
  * Runs in its own thread, so the window never waits for it. Only the newest tip is spoken: if two
    tips come while it talks, the older one is dropped.
  * While it talks (+ a short tail) the microphone analysis ignores the audio (signals.mute_until),
    so the coach's voice is never scored as yours: no fake pace, fillers, stress or "you froze".
  * Sound effects are ducked while it talks (sfx.duck).
"""
import queue
import shutil
import subprocess
import sys
import threading
import time

MIC_TAIL = 0.6        # seconds the mic stays ignored after the voice stops (room echo, audio latency)
AFTER_JINGLE = 0.6    # wait for the tip sound before speaking
DUCK = 0.25           # sound effects volume while the voice talks
LANG_NAMES = {"en": "english", "fr": "french", "es": "spanish", "de": "german", "it": "italian",
              "ar": "arabic", "pt": "portuguese", "nl": "dutch"}


class Speaker(threading.Thread):
    def __init__(self, signals=None, sfx=None, enabled: bool = False, language: str = "en", rate: int = 0):
        super().__init__(daemon=True, name="speaker")
        self.sig, self.sfx = signals, sfx
        self.enabled = enabled
        self.language = (language or "en").lower()
        self.rate = rate                  # SAPI rate, -10 (slow) .. 10 (fast)
        self.q: queue.Queue = queue.Queue()
        self.backend = None               # chosen on first use, in this thread
        self.warning = None
        self.talking = False
        self._say = None

    # --- called from the main thread ------------------------------------------------------------
    def say(self, text: str, force: bool = False):
        if (self.enabled or force) and text:
            while not self.q.empty():     # only the newest tip matters
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            self.q.put(text.strip().rstrip("."))

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        self.say("Voice tips on" if self.enabled else "Voice tips off", force=True)
        return self.enabled

    def stop(self):
        self.q.put(None)

    # --- the thread -------------------------------------------------------------------------------
    def run(self):
        while True:
            text = self.q.get()
            if text is None:
                return
            if self._say is None and not self._setup():
                continue
            time.sleep(AFTER_JINGLE)
            if not self.q.empty():
                continue                  # a newer tip arrived meanwhile: say that one instead
            self._mute(True)
            try:
                self._say(text)
            except Exception as e:  # noqa: BLE001
                self.warning = f"spoken tips failed ({e})"
            finally:
                self._mute(False)

    def _mute(self, on: bool):
        self.talking = on
        if self.sig is not None:
            self.sig.mute_until = float("inf") if on else time.time() + MIC_TAIL
            # remember when the mic was deaf, so fillers / pace don't count that time as "you said nothing"
            if on:
                self.sig.mute_since = time.time()
            elif getattr(self.sig, "mute_since", None) is not None:
                self.sig.push("muted", (self.sig.mute_since, time.time() + MIC_TAIL))
                self.sig.mute_since = None
        if self.sfx is not None:
            self.sfx.duck = DUCK if on else 1.0

    def _setup(self) -> bool:
        lang = LANG_NAMES.get(self.language[:2], self.language)
        if sys.platform == "win32":
            try:  # the Windows voices directly: synchronous and reliable from a thread
                import comtypes
                import comtypes.client
                comtypes.CoInitialize()
                voice = comtypes.client.CreateObject("SAPI.SpVoice")
                for v in voice.GetVoices():
                    if lang in v.GetDescription().lower():
                        voice.Voice = v
                        break
                voice.Rate = self.rate
                self._say, self.backend = (lambda s: voice.Speak(s)), "sapi"
                return True
            except Exception:  # noqa: BLE001
                pass
        try:
            import pyttsx3
            engine = pyttsx3.init()
            for v in engine.getProperty("voices"):
                if lang in (v.name or "").lower() or any(self.language[:2] in str(x).lower() for x in (v.languages or [])):
                    engine.setProperty("voice", v.id)
                    break

            def speak(s):
                engine.say(s)
                engine.runAndWait()
            self._say, self.backend = speak, "pyttsx3"
            return True
        except Exception:  # noqa: BLE001
            pass
        for cmd in (["say"], ["espeak", "-v", self.language[:2]]):
            if shutil.which(cmd[0]):
                self._say, self.backend = (lambda s, c=cmd: subprocess.run(c + [s], check=False)), cmd[0]
                return True
        if sys.platform == "win32" and shutil.which("powershell"):
            ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  "$s.Speak([Console]::In.ReadToEnd())")
            self._say = lambda s: subprocess.run(["powershell", "-NoProfile", "-Command", ps], input=s, text=True,
                                                 check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.backend = "powershell"
            return True
        self.warning = "spoken tips: no text-to-speech found (pip install pyttsx3)"
        self.enabled = False
        return False

