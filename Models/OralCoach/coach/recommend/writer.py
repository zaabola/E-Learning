"""Tip writer: turns (pattern + evidence + retrieved advice) into one short, personal tip.

Backends, tried in this order (mode "auto", the default):
  ollama     local model through Ollama (free, offline, runs on your GPU). Default model qwen2.5:3b.
  claude     Anthropic API (needs `pip install anthropic` and ANTHROPIC_API_KEY).
  template   no LLM: uses the card's own wording. Always available, instant; also the fallback
             whenever an LLM is slow (> timeout) or returns something unusable.

The LLM only rewrites retrieved advice for this moment. It never chooses the strategy (the coach does)
and never invents the source: the citation always comes from the card.
"""
import concurrent.futures as cf
import json
import os
import re
import threading
import urllib.error
import urllib.request

SYSTEM = """You are a calm, friendly presentation coach. A student is practising a talk alone, live, in front
of a webcam. Your tip appears on screen while they keep talking, so they read it in one glance.

Rules:
- Base the tip ONLY on the advice passages given. Do not add new techniques.
- "tip": imperative, at most 10 words, no emojis, no quotes. Level 1 = gentle nudge, level 2 = one concrete
  action to do now, level 3 = ask them to pause briefly and do the action.
- "why": at most 18 words. Say what you noticed in their signals (use the evidence) and why the tip helps.
- Never diagnose, never mention anxiety disorders, never shame. Be encouraging.
- Write in the requested language.
Answer with JSON only: {"tip": "...", "why": "..."}"""

LEVEL_NAMES = {1: "hint", 2: "action", 3: "pause"}


def build_prompt(req: dict) -> str:
    lines = [
        f"Language: {req.get('language', 'en')}",
        f"Detected pattern: {req['label']} ({req['pattern']})",
        "Evidence from the sensors: " + "; ".join(req["evidence"]),
        f"Level: {req['level']} ({LEVEL_NAMES[req['level']]})",
        f"Strategy chosen: {req['strategy']} ({req['choice']})",
    ]
    if req.get("previous"):
        lines.append(f"Previous tip for this problem did not help enough: \"{req['previous']}\"")
    if req.get("transcript"):
        lines.append(f"What they just said: \"{req['transcript'][-160:]}\"")
    lines.append(f"Advice passage (from {req['source']}): {req['passage']}")
    lines.append(f"Suggested wording for this level: \"{req['template_tip']}\"")
    for p in req.get("extra", []):
        lines.append(f"Extra passage (from {p['citation']}): {p['text']}")
    return "\n".join(lines)


def _clean(out: dict, req: dict) -> dict | None:
    tip = str(out.get("tip", "")).strip().strip('"').replace("\n", " ")
    why = str(out.get("why", "")).strip().strip('"').replace("\n", " ")
    if not tip or len(tip.split()) > 14 or len(why.split()) > 30:
        return None
    return {"text": tip, "why": why or req["template_why"]}


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


class TipWriter:
    def __init__(self, mode: str = "auto", ollama_model: str = "qwen2.5:3b", claude_model: str = "claude-opus-5-5",
                 ollama_url: str = "http://localhost:11434", timeout: float = 6.0, language: str = "en"):
        self.mode, self.ollama_model, self.claude_model = mode, ollama_model, claude_model
        self.ollama_url, self.timeout, self.language = ollama_url.rstrip("/"), timeout, language
        self.backend = "template"
        self.status = "template (no LLM)"
        self.warning = None
        self._claude = None
        self._pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tipwriter")
        self._ready = threading.Event()
        if mode == "template":
            self._ready.set()
        else:
            threading.Thread(target=self._detect, daemon=True, name="tipwriter-detect").start()

    # --- backend detection (runs in the background during calibration) --------------------------
    def _detect(self):
        try:
            order = {"auto": ["ollama", "claude"], "ollama": ["ollama"], "claude": ["claude"]}.get(self.mode, [])
            problems = []
            for b in order:
                ok, why = self._try_ollama() if b == "ollama" else self._try_claude()
                if ok:
                    self.backend = b
                    self.status = f"{b}: {self.ollama_model if b == 'ollama' else self.claude_model}"
                    return
                problems.append(why)
            self.backend = "template"
            self.status = "template (no LLM found)"
            if self.mode != "auto" or problems:
                self.warning = "Tips use card wording, no LLM: " + "; ".join(problems)
        finally:
            self._ready.set()

    def _try_ollama(self):
        try:
            with urllib.request.urlopen(self.ollama_url + "/api/tags", timeout=2) as r:
                names = [m["name"] for m in json.load(r).get("models", [])]
        except (OSError, ValueError):
            return False, "Ollama not running (install from ollama.com)"
        if not any(n == self.ollama_model or n.split(":")[0] == self.ollama_model for n in names):
            return False, f"Ollama has no {self.ollama_model} (run: ollama pull {self.ollama_model})"
        try:  # warm up: loads the model into GPU memory now, not at the first tip
            self._ollama_call("Say OK as JSON: {\"tip\": \"OK\", \"why\": \"OK\"}", timeout=60)
        except Exception as e:  # noqa: BLE001
            return False, f"Ollama error: {e}"
        return True, ""

    def _try_claude(self):
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return False, "no ANTHROPIC_API_KEY set"
        try:
            import anthropic
        except ImportError:
            return False, "pip install anthropic"
        self._claude = anthropic.Anthropic(timeout=self.timeout + 4, max_retries=1)
        return True, ""

    # --- calls ---------------------------------------------------------------------------------
    def _ollama_call(self, prompt: str, timeout: float) -> dict | None:
        body = json.dumps({
            "model": self.ollama_model, "stream": False, "format": "json", "keep_alive": "30m",
            "options": {"temperature": 0.4, "num_predict": 120},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(self.ollama_url + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _parse_json(json.load(r)["message"]["content"])

    def _claude_call(self, prompt: str) -> dict | None:
        resp = self._claude.beta.messages.create(
            model=self.claude_model,
            max_tokens=2000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            return None
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _parse_json(text)

    def _write(self, req: dict) -> dict:
        prompt = build_prompt(req)
        try:
            out = self._ollama_call(prompt, self.timeout) if self.backend == "ollama" else self._claude_call(prompt)
            clean = out and _clean(out, req)
            if clean:
                return dict(clean, backend=self.status)
        except Exception as e:  # noqa: BLE001  any LLM failure -> card wording
            self.warning = f"LLM call failed, used card wording: {type(e).__name__}"
        return self.template(req, note="LLM gave no usable tip")

    def template(self, req: dict, note: str = "") -> dict:
        return {"text": req["template_tip"], "why": req["template_why"],
                "backend": "template" + (f" ({note})" if note else "")}

    def submit(self, req: dict) -> cf.Future:
        """Start writing a tip. Returns a Future with {text, why, backend}."""
        req.setdefault("language", self.language)
        if not self._ready.is_set() or self.backend == "template" or self.mode == "template":
            f = cf.Future()
            f.set_result(self.template(req, "" if self.backend == "template" else "LLM still loading"))
            return f
        return self._pool.submit(self._write, req)
