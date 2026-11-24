"""A tiny hand-made 5x8 pixel font for the retro camera window (no font files needed).

Each glyph is 8 rows of '#' (ink) and '.' (paper). Letters are trimmed to their ink, so the font is
proportional ("i" is narrower than "m"). Accented letters (é, è, ê, à, ç, ï...) are built from the plain
letter plus a small accent drawn in the two empty rows above it, so French tips keep their accents.

    mask = text_mask("Hello", px=3)       # bool array, every font pixel is a 3x3 block
    draw_text(img, "Hello", (x, y), px=3, colour=(255, 255, 255), shadow=(0, 0, 0))
"""
import unicodedata
from functools import lru_cache

import numpy as np

H = 8  # rows per glyph (row 7 is for descenders: g, j, p, q, y)

_RAW = {
    "A": ".###. #...# #...# ##### #...# #...# #...#",
    "B": "####. #...# #...# ####. #...# #...# ####.",
    "C": ".###. #...# #.... #.... #.... #...# .###.",
    "D": "####. #...# #...# #...# #...# #...# ####.",
    "E": "##### #.... #.... ####. #.... #.... #####",
    "F": "##### #.... #.... ####. #.... #.... #....",
    "G": ".###. #...# #.... #.### #...# #...# .####",
    "H": "#...# #...# #...# ##### #...# #...# #...#",
    "I": "### .#. .#. .#. .#. .#. ###",
    "J": "..### ...#. ...#. ...#. #..#. #..#. .##..",
    "K": "#...# #..#. #.#.. ##... #.#.. #..#. #...#",
    "L": "#.... #.... #.... #.... #.... #.... #####",
    "M": "#...# ##.## #.#.# #.#.# #...# #...# #...#",
    "N": "#...# #...# ##..# #.#.# #..## #...# #...#",
    "O": ".###. #...# #...# #...# #...# #...# .###.",
    "P": "####. #...# #...# ####. #.... #.... #....",
    "Q": ".###. #...# #...# #...# #.#.# #..#. .##.#",
    "R": "####. #...# #...# ####. #.#.. #..#. #...#",
    "S": ".#### #.... #.... .###. ....# ....# ####.",
    "T": "##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
    "U": "#...# #...# #...# #...# #...# #...# .###.",
    "V": "#...# #...# #...# #...# #...# .#.#. ..#..",
    "W": "#...# #...# #...# #.#.# #.#.# #.#.# .#.#.",
    "X": "#...# #...# .#.#. ..#.. .#.#. #...# #...#",
    "Y": "#...# #...# .#.#. ..#.. ..#.. ..#.. ..#..",
    "Z": "##### ....# ...#. ..#.. .#... #.... #####",
    "a": "..... ..... .###. ....# .#### #...# .####",
    "b": "#.... #.... ####. #...# #...# #...# ####.",
    "c": "..... ..... .###. #.... #.... #.... .###.",
    "d": "....# ....# .#### #...# #...# #...# .####",
    "e": "..... ..... .###. #...# ##### #.... .###.",
    "f": "..##. .#... .#... ###.. .#... .#... .#...",
    "g": "..... ..... .#### #...# #...# .#### ....# .###.",
    "h": "#.... #.... ####. #...# #...# #...# #...#",
    "i": ".#. ... ##. .#. .#. .#. ###",
    "j": "...# .... ..## ...# ...# ...# #..# .##.",
    "k": "#... #... #..# #.#. ##.. #.#. #..#",
    "l": "##. .#. .#. .#. .#. .#. ###",
    "m": "..... ..... ##.#. #.#.# #.#.# #.#.# #...#",
    "n": "..... ..... ####. #...# #...# #...# #...#",
    "o": "..... ..... .###. #...# #...# #...# .###.",
    "p": "..... ..... ####. #...# #...# ####. #.... #....",
    "q": "..... ..... .#### #...# #...# .#### ....# ....#",
    "r": "..... ..... #.##. ##..# #.... #.... #....",
    "s": "..... ..... .#### #.... .###. ....# ####.",
    "t": ".#... .#... ####. .#... .#... .#..# ..##.",
    "u": "..... ..... #...# #...# #...# #..## .##.#",
    "v": "..... ..... #...# #...# #...# .#.#. ..#..",
    "w": "..... ..... #...# #...# #.#.# #.#.# .#.#.",
    "x": "..... ..... #...# .#.#. ..#.. .#.#. #...#",
    "y": "..... ..... #...# #...# #...# .#### ....# .###.",
    "z": "..... ..... ##### ...#. ..#.. .#... #####",
    "0": ".###. #...# #..## #.#.# ##..# #...# .###.",
    "1": ".#. ##. .#. .#. .#. .#. ###",
    "2": ".###. #...# ....# ...#. ..#.. .#... #####",
    "3": "####. ....# ....# .###. ....# ....# ####.",
    "4": "...#. ..##. .#.#. #..#. ##### ...#. ...#.",
    "5": "##### #.... ####. ....# ....# #...# .###.",
    "6": "..##. .#... #.... ####. #...# #...# .###.",
    "7": "##### ....# ...#. ..#.. .#... .#... .#...",
    "8": ".###. #...# #...# .###. #...# #...# .###.",
    "9": ".###. #...# #...# .#### ....# ...#. .##..",
    "!": "# # # # # . #",
    '"': "#.# #.#",
    "#": ".#.#. .#.#. ##### .#.#. ##### .#.#. .#.#.",
    "$": "..#.. .#### #.#.. .###. ..#.# ####. ..#..",
    "%": "##... ##..# ...#. ..#.. .#... #..## ...##",
    "&": ".##.. #..#. #.#.. .#... #.#.# #..#. .##.#",
    "'": "# #",
    "(": "..# .#. #.. #.. #.. .#. ..#",
    ")": "#.. .#. ..# ..# ..# .#. #..",
    "*": "..... ..#.. #.#.# .###. #.#.# ..#..",
    "+": "..... ..#.. ..#.. ##### ..#.. ..#..",
    ",": ".. .. .. .. .. .# .# #.",
    "-": ".... .... .... ####",
    ".": ". . . . . . #",
    "/": "....# ....# ...#. ..#.. .#... #.... #....",
    ":": ". . # . . # .",
    ";": ".. .. .# .. .. .# .# #.",
    "<": "...# ..#. .#.. #... .#.. ..#. ...#",
    "=": "..... ..... ##### ..... #####",
    ">": "#... .#.. ..#. ...# ..#. .#.. #...",
    "?": ".###. #...# ....# ...#. ..#.. ..... ..#..",
    "@": ".###. #...# #.### #.#.# #.### #.... .###.",
    "[": "## #. #. #. #. #. ##",
    "\\": "#.... #.... .#... ..#.. ...#. ....# ....#",
    "]": "## .# .# .# .# .# ##",
    "^": "..#.. .#.#. #...#",
    "_": "..... ..... ..... ..... ..... ..... #####",
    "`": "#. .#",
    "{": "..# .#. .#. #.. .#. .#. ..#",
    "|": "# # # # # # #",
    "}": "#.. .#. .#. ..# .#. .#. #..",
    "~": "..... ..... .#... #.#.# ...#.",
    # little icons for the retro look
    "▼": "##### .###. ..#..",
    "▶": "#... ##.. ###. #### ###. ##.. #...",
    "♥": "..... .#.#. ##### ##### .###. ..#..",
    "★": "..#.. ..#.. ##### .###. .#.#. #...#",
    "●": "..... .###. ##### ##### ##### .###.",
    "•": ". . . # .",
}

# accents drawn in rows 0-1 above lowercase letters (cedilla goes in row 7)
_ACCENTS = {
    "́": ("...#.", "..#.."),   # acute é
    "̀": (".#...", "..#.."),   # grave è à
    "̂": ("..#..", ".#.#."),   # circumflex ê
    "̈": (".#.#.", "....."),   # diaeresis ë ï
}


def _parse(raw: str) -> np.ndarray:
    rows = raw.split(" ")
    w = max(len(r) for r in rows)
    g = np.zeros((H, w), bool)
    for i, r in enumerate(rows):
        g[i, : len(r)] = [c == "#" for c in r]
    return g


GLYPHS = {ch: _parse(raw) for ch, raw in _RAW.items()}
SPACE_W = 3


def _trim(g: np.ndarray) -> np.ndarray:
    cols = np.where(g.any(0))[0]
    return g[:, cols[0]: cols[-1] + 1] if len(cols) else np.zeros((H, SPACE_W), bool)


@lru_cache(maxsize=512)
def glyph(ch: str) -> np.ndarray:
    if ch == " ":
        return np.zeros((H, SPACE_W), bool)
    if ch in GLYPHS:
        return _trim(GLYPHS[ch])
    base, *marks = unicodedata.normalize("NFD", ch)
    if base in GLYPHS:
        g = GLYPHS[base].copy()
        if base.islower():  # caps have no room above them: they just lose the accent
            w = g.shape[1]
            if base in "ij" and any(m in _ACCENTS for m in marks):
                g[0:2, :] = False  # the accent replaces the dot of i / j
            for m in marks:
                if m in _ACCENTS:
                    acc = np.array([[c == "#" for c in row] for row in _ACCENTS[m]])
                    g[0:2, :] |= acc[:, :w] if w >= 5 else acc[:, 1: 1 + w]
                elif m == "̧":  # cedilla ç
                    g[7, 1:3] = True
        return _trim(g)
    plain = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
    if plain and plain[0] in GLYPHS:
        return _trim(GLYPHS[plain[0]])
    return _trim(GLYPHS["?"])


@lru_cache(maxsize=4096)
def _line_mask(text: str) -> np.ndarray:
    parts = []
    for ch in text:
        parts += [glyph(ch), np.zeros((H, 1), bool)]
    if not parts:
        return np.zeros((H, 0), bool)
    return np.hstack(parts[:-1])


@lru_cache(maxsize=4096)
def text_mask(text: str, px: int = 2) -> np.ndarray:
    """Bool mask of one line of text, every font pixel blown up to a px x px block."""
    m = _line_mask(text)
    return np.repeat(np.repeat(m, px, 0), px, 1) if px > 1 else m


def text_width(text: str, px: int = 2) -> int:
    return _line_mask(text).shape[1] * px


def line_height(px: int = 2) -> int:
    return H * px


def _blit(img, mask, x, y, colour):
    h, w = mask.shape
    ih, iw = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(iw, x + w), min(ih, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sub = mask[y0 - y: y1 - y, x0 - x: x1 - x]
    img[y0:y1, x0:x1][sub] = colour


def draw_text(img, text: str, org, px: int = 2, colour=(255, 255, 255), shadow=(0, 0, 0)):
    """Draw text with its top-left corner at org. The shadow is a hard 1-font-pixel drop shadow."""
    x, y = int(org[0]), int(org[1])
    mask = text_mask(str(text), px)
    if shadow is not None:
        _blit(img, mask, x + px, y + px, shadow)
    _blit(img, mask, x, y, colour)
    return x + mask.shape[1]


def wrap(text: str, max_px: int, px: int = 2) -> list[str]:
    """Greedy word wrap by real pixel width."""
    lines, cur = [], ""
    for word in str(text).split():
        cand = f"{cur} {word}".strip()
        if cur and text_width(cand, px) > max_px:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    return lines + ([cur] if cur else [])
