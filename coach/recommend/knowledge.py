"""The coach's library: tip cards + optional book passages, searched with BM25.

Retrieval is "filter, then rank": only cards tagged for the detected pattern are candidates
(metadata filter), then BM25 ranks them against a query built from the pattern and its evidence.
Pure Python on purpose: ~30 cards and a few chapters do not need a vector database, and BM25 scores
are easy to show to a teacher. Swap `search` for Chroma/embeddings later without touching the coach.
"""
import math
import os
import re
from dataclasses import dataclass, field

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "knowledge")
TOKEN = re.compile(r"[a-zà-ÿ]+")
STOP = set("""a an the and or of to in on for with at by from is are was were be been it its this that as
your you i we they he she not no but if so than then too very can will just do does into about over""".split())
LEVEL_KEYS = {1: "hint", 2: "action", 3: "pause"}
# how each card strategy is named in tips and messages
STRATEGY_NAMES = {
    "breathing": "slow breathing", "slow_down": "slowing down", "silent_pause": "silent pauses",
    "eye_anchor": "looking at the camera", "restart_point": "restating your last point",
    "voice_vary": "stressing key words", "energy_up": "projecting your voice", "smile_relax": "relaxing your face",
    "reframe": "reframing (\"I'm excited\")", "ground_posture": "grounding your stance",
    "relax_voice": "relaxing shoulders and throat",
    "audience_focus": "focusing on the message",
    "glance_speak": "glance, look up, then speak", "gesture": "using your hands",
    "finish_strong": "finishing strong",
}


def tokens(text: str) -> list[str]:
    out = []
    for w in TOKEN.findall(text.lower()):
        if w in STOP or len(w) < 2:
            continue
        for suf in ("ing", "ed", "es", "s"):  # tiny stemmer: breathing/breathe, pauses/pause
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


@dataclass
class Card:
    id: str
    strategy: str
    patterns: list
    hint: str
    action: str
    pause: str
    why: str
    source: str
    passage: str
    file: str = ""
    tokens: list = field(default_factory=list)

    def text_for(self, level: int) -> str:
        return getattr(self, LEVEL_KEYS[max(1, min(3, level))]) or self.action or self.hint


@dataclass
class Passage:
    id: str
    citation: str
    text: str
    tokens: list = field(default_factory=list)


def parse_cards(path: str) -> list[Card]:
    cards = []
    with open(path, encoding="utf-8") as f:
        blocks = re.split(r"^## ", f.read(), flags=re.M)[1:]
    for b in blocks:
        head, _, body = b.partition("\n")
        meta, passage = {}, []
        lines = body.strip().splitlines()
        i = 0
        while i < len(lines) and re.match(r"^[a-z_]+:\s", lines[i]):
            k, _, v = lines[i].partition(":")
            meta[k.strip()] = v.strip()
            i += 1
        passage = " ".join(l.strip() for l in lines[i:] if l.strip())
        c = Card(id=head.strip(), strategy=meta.get("strategy", head.strip()),
                 patterns=[p.strip() for p in meta.get("patterns", "").split(",") if p.strip()],
                 hint=meta.get("hint", ""), action=meta.get("action", ""), pause=meta.get("pause", ""),
                 why=meta.get("why", ""), source=meta.get("source", ""), passage=passage,
                 file=os.path.basename(path))
        cards.append(c)
    return cards


def chunk_book(path: str, words: int = 120) -> list[Passage]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    if not lines:
        return []
    citation = lines[0].strip().lstrip("# ") or os.path.basename(path)
    ws = " ".join(lines[1:]).split()
    out = []
    for n, i in enumerate(range(0, len(ws), words - 20)):  # 20-word overlap
        text = " ".join(ws[i:i + words])
        if len(text.split()) >= 20:
            out.append(Passage(f"{os.path.basename(path)}#{n}", citation, text))
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1=1.4, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = sum(map(len, docs)) / max(1, len(docs))
        df = {}
        for d in docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def score(self, query: list[str], i: int) -> float:
        d = self.docs[i]
        tf = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in set(query):
            if w in tf:
                f = tf[w]
                s += self.idf.get(w, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
        return s


class Knowledge:
    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = root
        self.cards: list[Card] = []
        self.passages: list[Passage] = []
        cards_dir, books_dir = os.path.join(root, "cards"), os.path.join(root, "books")
        if os.path.isdir(cards_dir):
            for fn in sorted(os.listdir(cards_dir)):
                if fn.endswith(".md"):
                    self.cards += parse_cards(os.path.join(cards_dir, fn))
        if os.path.isdir(books_dir):
            for fn in sorted(os.listdir(books_dir)):
                if fn.lower().endswith((".txt", ".md")) and fn.lower() != "readme.txt":
                    self.passages += chunk_book(os.path.join(books_dir, fn))
        if not self.cards:
            raise FileNotFoundError(f"No tip cards found in {cards_dir}")
        for c in self.cards:
            c.tokens = tokens(" ".join([c.strategy.replace("_", " "), " ".join(c.patterns).replace("_", " "),
                                        c.hint, c.action, c.pause, c.why, c.passage]))
        for p in self.passages:
            p.tokens = tokens(p.text)
        self._card_bm25 = BM25([c.tokens for c in self.cards])
        self._book_bm25 = BM25([p.tokens for p in self.passages]) if self.passages else None
        self.sources = self._load_sources()

    def _load_sources(self) -> dict:
        """Short source key -> full reference line, from knowledge/sources.md."""
        out = {}
        path = os.path.join(self.root, "sources.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for block in re.split(r"\n- ", f.read()):
                    m = re.match(r"\*\*(.+?)\*\*:\s*(.+)", block.strip(), flags=re.S)
                    if m:
                        out[m.group(1)] = " ".join(m.group(2).split())
        return out

    def search(self, pattern: str, query: str, k: int = 4) -> list[tuple[Card, float]]:
        """Cards tagged for `pattern`, ranked by BM25 against `query`, scores scaled to 0..1."""
        q = tokens(query)
        cand = [(i, c) for i, c in enumerate(self.cards) if pattern in c.patterns]
        scored = [(c, self._card_bm25.score(q, i)) for i, c in cand]
        top = max([s for _, s in scored] + [1e-9])
        scored = [(c, s / top) for c, s in scored]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def search_books(self, query: str, k: int = 2) -> list[tuple[Passage, float]]:
        if not self._book_bm25:
            return []
        q = tokens(query)
        scored = [(p, self._book_bm25.score(q, i)) for i, p in enumerate(self.passages)]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
