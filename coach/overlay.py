"""Live camera window (OpenCV) in a chunky retro handheld-RPG style.

Pixelated webcam feed + face brackets, a status screen with HP-style bars, the coach's tips in a dialog box
that types itself out (with text blips), and pop-ups / flashes when a tip works, on praise and on level up.

Runs on the main thread (OpenCV windows must, especially on Windows).
Only draws: it reads the same snapshot dict the terminal view and the game use, plus the GameFeel state.
Keys: q / Esc quit, l landmarks, m mirror, p pixel size, s sound on/off.
"""
import math
import time

import cv2
import numpy as np

from .juice import ROWS, SHORT, GameFeel, row_value, zone
from .pixelfont import draw_text, line_height, text_width, wrap

WINDOW = "Oral coach"
CANVAS = (540, 960)  # h, w used when there is no camera (--no-face)
PANEL_W = 420
PIXEL_SIZES = [1, 3, 5, 8]  # camera pixel block size, cycled with "p" (1 = off)
TYPE_SPEED = 42.0           # characters per second for the dialog typewriter


def rgb(r, g, b):
    return (b, g, r)  # OpenCV wants BGR


INK = rgb(20, 18, 34)
PAPER = rgb(250, 248, 238)
PAPER_EDGE = rgb(176, 184, 208)
BG, BG2 = rgb(30, 34, 62), rgb(36, 41, 72)
WIN = rgb(18, 22, 44)
FRAME_HI, FRAME_LO = rgb(232, 232, 244), rgb(100, 110, 172)
WHITE, GREY, DIM = rgb(248, 248, 248), rgb(150, 158, 196), rgb(92, 98, 140)
TEXT_DARK, TEXT_MID = rgb(40, 40, 64), rgb(96, 100, 132)
GOLD = rgb(255, 206, 60)
GREEN, GREEN_HI = rgb(72, 208, 88), rgb(170, 248, 150)
YELLOW, YELLOW_HI = rgb(248, 196, 40), rgb(255, 240, 150)
RED, RED_HI = rgb(232, 60, 60), rgb(255, 150, 140)
CYAN, CYAN_HI = rgb(80, 196, 248), rgb(180, 236, 255)
ORANGE = rgb(248, 140, 40)
PURPLE, PURPLE_HI = rgb(168, 104, 248), rgb(214, 180, 255)
TROUGH = rgb(52, 56, 84)

ZONE_COLOURS = {"good": (GREEN, GREEN_HI), "warn": (YELLOW, YELLOW_HI), "bad": (RED, RED_HI)}
LEVEL_COLOURS = {1: CYAN, 2: YELLOW, 3: ORANGE}
LEVEL_WORDS = {1: "TIP", 2: "TRY THIS NOW", 3: "PAUSE"}


# --- pixel-art drawing helpers ------------------------------------------------------------------
def rect(img, x0, y0, x1, y1, colour):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = colour


def box(img, x0, y0, x1, y1, fill=WIN, style="dark", px=3):
    """RPG window: ink outline, light frame line, coloured inner line, notched (pixel-rounded) corners."""
    if style == "light":
        layers = [INK, PAPER, PAPER_EDGE]
    elif isinstance(style, tuple):  # custom accent colour
        layers = [INK, style, INK]
    else:
        layers = [INK, FRAME_HI, FRAME_LO]
    rect(img, x0, y0, x1, y1, layers[0])
    rect(img, x0 + px, y0 + px, x1 - px, y1 - px, layers[1])
    rect(img, x0 + 2 * px, y0 + 2 * px, x1 - 2 * px, y1 - 2 * px, layers[2])
    rect(img, x0 + 3 * px, y0 + 3 * px, x1 - 3 * px, y1 - 3 * px, fill)
    # notch the corners so the frame looks rounded in pixels
    for cx, cy in ((x0, y0), (x1 - px, y0), (x0, y1 - px), (x1 - px, y1 - px)):
        _clear_corner(img, cx, cy, px)


def _clear_corner(img, x, y, px):
    # we can't know what was behind the box: leave it ink, which reads as a rounded corner on any background
    rect(img, x, y, x + px, y + px, INK)


def bar(img, x0, y0, x1, h, v, colours, px=2, blink=False):
    """HP-style bar: ink border, dark trough, coloured fill with a lighter top row."""
    rect(img, x0, y0, x1, y0 + h, INK)
    rect(img, x0 + px, y0 + px, x1 - px, y0 + h - px, TROUGH)
    if v is None:
        return
    fill, hi = colours
    if blink:
        fill, hi = hi, WHITE
    inner = x1 - x0 - 2 * px
    fw = int(round(inner * max(0.0, min(1.0, v)) / px)) * px  # snap to the pixel grid
    if fw > 0:
        rect(img, x0 + px, y0 + px, x0 + px + fw, y0 + h - px, fill)
        rect(img, x0 + px, y0 + px, x0 + px + fw, y0 + px + max(px, (h - 2 * px) // 3), hi)


def text(img, s, x, y, px=2, colour=WHITE, shadow=INK):
    return draw_text(img, s, (x, y), px, colour, shadow)


def text_right(img, s, x_right, y, px=2, colour=WHITE, shadow=INK):
    return draw_text(img, s, (x_right - text_width(s, px), y), px, colour, shadow)


def text_center(img, s, cx, y, px=2, colour=WHITE, shadow=INK):
    return draw_text(img, s, (cx - text_width(s, px) // 2, y), px, colour, shadow)


def fit(s: str, max_px: int, px: int = 2) -> str:
    if text_width(s, px) <= max_px:
        return s
    while s and text_width(s + "..", px) > max_px:
        s = s[:-1]
    return s.rstrip() + ".."


def tint(img, colour, amount):
    if amount <= 0:
        return
    overlay = np.empty_like(img)
    overlay[:] = colour
    cv2.addWeighted(overlay, amount, img, 1 - amount, 0, img)


def pixelate(img, block: int):
    """Chunky pixels + fewer colours, like an old handheld screen."""
    if block <= 1:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    small = (small // 40) * 40 + 20  # posterize: ~7 levels per channel
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


_STAGE = {}


def stage(h, w, msg):
    """Pixel-art stage drawn in code, shown when there is no camera (--no-face)."""
    key = (h, w, msg)
    if key in _STAGE:
        return _STAGE[key].copy()
    p = 4  # one art pixel = 4 screen pixels
    lh, lw = h // p, w // p
    s = np.zeros((lh, lw, 3), np.uint8)
    s[:] = rgb(26, 22, 46)
    floor_y = int(lh * 0.72)
    for y in range(floor_y):  # back wall: subtle vertical stripes
        s[y, :] = rgb(38, 30, 66) if (y // 3) % 2 else rgb(34, 27, 60)
    s[floor_y:, :] = rgb(92, 56, 40)  # wooden floor with planks
    for y in range(floor_y, lh, 3):
        s[y, :] = rgb(74, 44, 32)
    for x in range(0, lw, 13):
        s[floor_y:, x] = rgb(74, 44, 32)
    s[floor_y, :] = rgb(140, 92, 60)
    # spotlight: lighter ellipse on the floor and a soft cone
    cx = lw // 2
    yy, xx = np.mgrid[0:lh, 0:lw]
    spot = ((xx - cx) / (lw * 0.2)) ** 2 + ((yy - (floor_y + (lh - floor_y) * 0.5)) / ((lh - floor_y) * 0.35)) ** 2
    s[(spot < 1) & (yy > floor_y)] = rgb(176, 124, 76)
    s[(spot < 0.45) & (yy > floor_y)] = rgb(208, 156, 96)
    cone = (np.abs(xx - cx) < (yy + 4) * 0.28) & (yy < floor_y) & (((xx + yy) % 2) == 0)
    s[cone] = np.minimum(255, s[cone].astype(int) + 22).astype(np.uint8)
    # curtains, left and right, with folds
    cw = int(lw * 0.14)
    for i in range(cw):
        shade = rgb(150, 30, 52) if (i // 3) % 2 else rgb(120, 22, 42)
        sag = int(3 * math.sin(i / cw * math.pi))
        s[: floor_y + 2, i] = shade
        s[: floor_y + 2, lw - 1 - i] = shade
        s[: 6 + sag, :] = rgb(150, 30, 52)
    s[: 6, :] = rgb(150, 30, 52)
    s[5:7, :] = rgb(232, 180, 64)  # gold trim
    # lectern in the spotlight
    lx0, lx1 = cx - 7, cx + 7
    s[floor_y - 16: floor_y + 4, lx0 + 2: lx1 - 2] = rgb(110, 68, 44)
    s[floor_y - 18: floor_y - 15, lx0: lx1] = rgb(150, 98, 62)
    s[floor_y - 13: floor_y + 4, lx0 + 2] = rgb(80, 48, 32)
    img = cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)
    text_center(img, msg, w // 2, int(h * 0.36), 3, GREY)
    _STAGE[key] = img
    return img.copy()


# --- camera side --------------------------------------------------------------------------------
def draw_face(img, face: dict, mirror: bool, landmarks: bool, t: float):
    h, w = img.shape[:2]
    pts = np.array([((1 - x) if mirror else x, y) for x, y in face["points"]]) * (w, h)
    colour = GREEN if face["looking"] else YELLOW
    q = 3
    if landmarks:
        for x, y in pts[::4].astype(int):
            x, y = x // q * q, y // q * q
            rect(img, x, y, x + q, y + q, colour)
    x0, y0 = (pts.min(0) // q * q).astype(int)
    x1, y1 = (pts.max(0) // q * q).astype(int)
    pad = 6 + int(2 * math.sin(t * 4)) // 2 * 2  # brackets "breathe" a little
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    arm, th = max(12, (x1 - x0) // 5), 6
    for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        for col, grow in ((INK, 3), (colour, 0)):
            ax = sorted((cx - sx * grow, cx + sx * (arm + grow)))
            ay = sorted((cy - sy * grow, cy + sy * (th + grow)))
            rect(img, ax[0], ay[0], ax[1], ay[1], col)
            bx = sorted((cx - sx * grow, cx + sx * (th + grow)))
            by = sorted((cy - sy * grow, cy + sy * (arm + grow)))
            rect(img, bx[0], by[0], bx[1], by[1], col)
    label = "LOOKING AT YOU" if face["looking"] else "LOOKING AWAY"
    sub = f"yaw {face['yaw']:+.0f} pitch {face['pitch']:+.0f}"
    lw = max(text_width(label, 2), text_width(sub, 1)) + 16
    ly = max(0, y0 - 44)
    box(img, x0, ly, x0 + lw, ly + 38, WIN, (colour if face["looking"] else YELLOW), 2)
    text(img, label, x0 + 8, ly + 7, 2, colour)
    text(img, sub, x0 + 8, ly + 26, 1, GREY, None)


def typed_chars(coach: dict | None) -> tuple[str | None, int, str]:
    """(tip id, characters visible so far, full text) for the dialog typewriter."""
    tip = coach.get("tip") if coach else None
    if not tip:
        return None, 0, ""
    age = coach.get("tip_age") or 0.0
    return tip["id"], min(len(tip["text"]), int(age * TYPE_SPEED)), tip["text"]


def draw_tip(img, coach: dict, t: float, subtitle_room: list):
    """The tip as an RPG dialog box at the bottom, typed out letter by letter; later a small reminder.
    subtitle_room gets the dialog's top edge while the big dialog is on screen (subtitles hide meanwhile)."""
    tip = coach.get("tip")
    if not tip:
        return
    h, w = img.shape[:2]
    colour = LEVEL_COLOURS.get(tip["level"], CYAN)
    age = coach.get("tip_age") or 0.0
    show = coach.get("tip_show", 12.0)
    if age > show:  # compact reminder, top-left, with a draining "did it help?" timer
        line = fit(f"{LEVEL_WORDS[tip['level']]}: {tip['text']}", w - 80, 2)
        bw = text_width(line, 2) + 30
        box(img, 8, 8, 8 + bw, 52, WIN, colour, 2)
        text(img, line, 20, 18, 2, colour)
        chk = coach.get("checking")
        if chk is not None:
            bar(img, 20, 38, 8 + bw - 12, 8, chk / coach.get("check_total", 20.0), (colour, WHITE), 2)
        return
    m = 12
    big_px = 3 if w >= 700 else 2
    inner = w - 2 * m - 40
    _, n, full = typed_chars(coach)
    lines = wrap(full, inner, big_px)[:3]
    why = wrap("Why: " + tip["why"], inner, 2)[:2]
    meta = fit(f"Source: {tip['source']}  -  {tip['choice']}", inner, 1)
    bh = 30 + len(lines) * (line_height(big_px) + 8) + len(why) * 20 + 22
    # slide in from the bottom, slide out at the end
    slide = max(0.0, 1 - age / 0.18) + max(0.0, (age - (show - 0.3)) / 0.3)
    y0 = h - m - bh + int(min(1.0, slide) * (bh + m + 30))
    subtitle_room.append(y0 - 34)
    box(img, m, y0, w - m, h - m + int(min(1.0, slide) * (bh + m + 30)), PAPER, "light", 3)
    # name plate
    plate = f"{'URGENT' if tip.get('urgent') else 'COACH'}  {LEVEL_WORDS[tip['level']]}"
    pw_ = text_width(plate, 2) + 28
    box(img, m + 14, y0 - 22, m + 14 + pw_, y0 + 12, colour, "dark", 2)
    text(img, plate, m + 28, y0 - 10, 2, INK, None)
    lbl = fit(tip["label"], w - 2 * m - pw_ - 60, 2)
    text(img, lbl, m + 28 + pw_, y0 - 20, 2, WHITE)
    # typed text
    y = y0 + 24
    shown = n
    for line in lines:
        part = line[: max(0, shown)]
        draw_text(img, part, (m + 20, y), big_px, TEXT_DARK, PAPER_EDGE)
        shown -= len(line) + 1
        y += line_height(big_px) + 8
    if n >= len(full):
        for line in why:
            draw_text(img, line, (m + 20, y), 2, TEXT_MID, None)
            y += 20
        draw_text(img, meta, (m + 20, y + 4), 1, TEXT_MID, None)
        if int(t * 2.5) % 2 == 0:  # blinking "next" arrow
            text(img, "▼", w - m - 30, h - m - 26, 2, colour, INK)


def draw_toast(img, coach: dict, feel: GameFeel | None, now: float):
    """Result pop-up: drops in with a bounce from the top."""
    toast = coach.get("toast")
    if not toast:
        return
    h, w = img.shape[:2]
    age = max(0.0, now - toast["t"])
    rb = feel.result_banner if feel else None
    from_result = rb is not None and abs(rb["t"] - toast["t"]) < 2.5
    if toast["kind"] != "good":
        head, colour = "NOT YET...", GREY
    elif from_result:
        head, colour = "IT WORKED!", GREEN
    else:
        head, colour = "★ NICE! ★", GOLD
    msg = fit(toast["text"], w - 120, 2)
    bw = max(text_width(head, 4), text_width(msg, 2)) + 48
    bh = 84
    drop = int(-120 * math.exp(-age * 9) * math.cos(age * 16)) if age < 0.8 else 0
    shake = int(6 * math.sin(age * 60) * math.exp(-age * 6)) if toast["kind"] != "good" else 0
    x0 = (w - bw) // 2 + shake
    y0 = 70 + drop
    box(img, x0, y0, x0 + bw, y0 + bh, WIN, colour, 3)
    text_center(img, head, x0 + bw // 2, y0 + 16, 4, colour)
    text_center(img, msg, x0 + bw // 2, y0 + 56, 2, WHITE)


def draw_subtitle(img, transcript: str, bottom: int):
    h, w = img.shape[:2]
    lines = wrap(transcript[-140:], w - 80, 2)[-2:]
    y = bottom - 22 * len(lines)
    for line in lines:
        text_center(img, line, w // 2, y, 2, WHITE)
        y += 22


def draw_levelup(img, feel: GameFeel, now: float):
    age = now - feel.levelup_t
    if not 0 <= age < 2.2:
        return
    h, w = img.shape[:2]
    if int(age * 8) % 2 == 0 or age > 0.6:
        bob = int(4 * math.sin(age * 10))
        cy = h - 110 + bob  # low on the screen, so it doesn't sit on your face
        box(img, w // 2 - 190, cy - 50, w // 2 + 190, cy + 50, WIN, GOLD, 4)
        text_center(img, "LEVEL UP!", w // 2, cy - 30, 5, GOLD)
        text_center(img, f"★ LV {feel.level} ★", w // 2, cy + 18, 3, WHITE)


# --- right panel --------------------------------------------------------------------------------
def draw_panel(img, snap: dict | None, warnings: list[str], coach: dict | None = None,
               feel: GameFeel | None = None, shown: dict | None = None, now: float = 0.0):
    """Status screen, drawn on its own image placed right of the camera (never hides your face)."""
    h, pw = img.shape[:2]
    img[:] = BG
    img[::8, :] = BG2  # faint scanline texture
    m = 8
    if snap is None:
        box(img, m, m, pw - m, 70)
        text(img, "LOADING...", m + 20, m + 22, 3, WHITE)
        return
    ch = snap["channels"]

    # header: title, timer, level / XP, calibration
    hb = 106
    box(img, m, m, pw - m, m + hb)
    text(img, "★ ORAL COACH", m + 18, m + 16, 2, GOLD)
    s = int(snap["session_s"])
    text_right(img, f"{s // 60:02d}:{s % 60:02d}", pw - m - 18, m + 16, 2, WHITE)
    if feel is not None:
        text(img, f"LV{feel.level}", m + 18, m + 42, 2, PURPLE_HI)
        bar(img, m + 72, m + 42, pw - m - 18, 16, feel.xp_frac(), (PURPLE, PURPLE_HI), 2)
        text(img, "XP", m + 78, m + 46, 1, WHITE, None)
    if snap["calibrated"]:
        text(img, "♥ Calibrated to you", m + 18, m + 72, 2, GREEN)
    else:
        text(img, "Calibrating: talk normally", m + 18, m + 66, 2, YELLOW)
        bar(img, m + 18, m + 86, pw - m - 18, 12, snap["calibration_progress"], (YELLOW, YELLOW_HI), 2)

    # status bars
    y0 = m + hb + 8
    warn_lines = []
    for wmsg in warnings[-2:]:
        warn_lines += wrap("! " + wmsg, pw - 2 * m - 36, 1)[:3]
    warn_h = (len(warn_lines) * 12 + 30) if warn_lines else 0
    coach_min = 150 if coach is not None else 40
    row_h = max(22, min(30, (h - y0 - coach_min - warn_h - 40) // len(ROWS)))
    sb = 36 + row_h * (len(ROWS) - 1) + 30
    box(img, m, y0, pw - m, y0 + sb)
    text(img, "STATUS", m + 18, y0 + 14, 2, CYAN)
    reds = feel.red_rows() if feel else set()
    blink_on = int(now * 4) % 2 == 0
    y = y0 + 36
    bx0, bx1 = m + 146, pw - m - 100
    for key, label, high_bad in ROWS:
        v, val = row_value(key, ch)
        if shown is not None and v is not None:  # bars slide towards their new value
            prev = shown.get(key, v)
            shown[key] = prev + (v - prev) * 0.18
            dv = shown[key]
        else:
            dv = v
        z = zone(v, high_bad) if v is not None else "good"
        lab_col = RED if key in reds and blink_on else WHITE
        text(img, SHORT[key], m + 18, y + 1, 2, lab_col)
        bar(img, bx0, y - 2, bx1, 16, dv, ZONE_COLOURS[z], 2, blink=key in reds and not blink_on)
        if key in ("pace", "stress", "confidence"):  # 0.5 = your normal: tick mark
            mx = (bx0 + bx1) // 2 // 2 * 2
            rect(img, mx, y - 6, mx + 2, y + 18, WHITE)
        val = fit(val, pw - m - 14 - bx1 - 8, 2) if v is not None else "--"
        text(img, val, bx1 + 8, y + 1, 2, WHITE if v is not None else DIM)
        y += row_h

    # coach box
    cy0 = y0 + sb + 8
    cy1 = h - m - (warn_h + 8 if warn_h else 0)
    if coach is not None and cy1 - cy0 > 40:
        box(img, m, cy0, pw - m, cy1)
        text(img, "COACH", m + 18, cy0 + 14, 2, CYAN)
        y = cy0 + 36
        lines = []
        sil = ch.get("silence_now_s")
        if sil is not None and sil >= 1.0:
            lines.append((f"Silent for {sil:.1f}s", YELLOW, 2))
        if coach.get("writing"):
            dots = "." * (1 + int(now * 3) % 3)
            lines.append((f"Writing a tip{dots}", GREY, 2))
        if coach.get("checking") is not None:
            lines.append((f"Did it help? {coach['checking']:.0f}s", YELLOW, 2))
        if coach.get("active"):
            for i, ln in enumerate(wrap(", ".join(coach["active"]), pw - 2 * m - 70, 2)[:2]):
                lines.append(((("▶ " if i == 0 else "  ") + ln), WHITE, 2))
        if snap.get("face_top_emotion"):
            e, p = snap["face_top_emotion"]
            lines.insert(0, (f"FEELING: {e.upper()} {p:.0%}", YELLOW, 2))
        for i, (s_, col, px) in enumerate(lines):
            if y + 8 * px > cy1 - 18:
                break
            text(img, s_, m + 18, y, px, col)
            y += 20 if px == 2 else 14
    if warn_lines:
        wy0 = h - m - warn_h
        box(img, m, wy0, pw - m, h - m, WIN, RED, 2)
        y = wy0 + 14
        for ln in warn_lines:
            text(img, ln, m + 16, y, 1, RED_HI, None)
            y += 12

    # floating "+50 XP" pop-ups rise from the XP bar
    if feel is not None:
        for p in feel.popups:
            age = now - p["t"]
            if age < 0 or (age > 1.8 and int(age * 10) % 2):
                continue
            yy = m + 40 - int(age * 18) + 70
            col = GOLD if p["kind"] == "xp" else GREEN
            text_right(img, p["text"], pw - m - 18, yy, 2, col)


# --- whole frame --------------------------------------------------------------------------------
def compose(frame, face, snap, warnings, mirror=True, landmarks=True, fps=None, coach=None,
            feel=None, shown=None, pixel=5, now=None, sound_on=True, voice_on=None):
    now = time.time() if now is None else now
    if frame is None:
        cam = stage(*CANVAS, "NO CAMERA")
    else:
        cam = cv2.flip(frame, 1) if mirror else frame.copy()
        cam = pixelate(cam, pixel)
        if face is not None:
            draw_face(cam, face, mirror, landmarks, now)
        else:
            box(cam, 8, 8, 230, 48, WIN, YELLOW, 2)
            text(cam, "NO FACE FOUND", 22, 20, 2, YELLOW)
    # flashes
    if feel is not None and feel.flash:
        age = now - feel.flash["t"]
        if 0 <= age < 0.35:
            col = {"good": GREEN_HI, "praise": GOLD, "level": WHITE}.get(feel.flash["kind"], WHITE)
            tint(cam, col, 0.45 * (1 - age / 0.35))
    room = []
    if coach:
        draw_tip(cam, coach, now, room)
    if snap and snap.get("transcript") and not room:  # the big dialog box has priority over subtitles
        draw_subtitle(cam, snap["transcript"], cam.shape[0] - 14)
    if coach:
        draw_toast(cam, coach, feel, now)
    if feel is not None:
        draw_levelup(cam, feel, now)
    if fps and not (coach and coach.get("tip")):
        hint = f"{fps:.0f}fps  q quit  l landmarks  m mirror  p pixels  s sound {'on' if sound_on else 'off'}"
        if voice_on is not None:
            hint += f"  v voice {'on' if voice_on else 'off'}"
        text_right(cam, hint, cam.shape[1] - 10, 10, 1, WHITE)
    panel = np.empty((cam.shape[0], PANEL_W, 3), np.uint8)
    draw_panel(panel, snap, warnings, coach, feel, shown, now)
    return np.hstack([cam, panel])


class Window:
    """Call show() in a loop on the main thread. Returns False when the user quits."""

    def __init__(self, mirror=True, landmarks=True, scale=1.0, feel: GameFeel | None = None,
                 pixel: int = 5, speaker=None):
        self.speaker = speaker
        self.mirror, self.landmarks, self.scale = mirror, landmarks, scale
        self.feel = feel
        self.pixel = pixel
        self.shown = {}
        self._t, self._fps = time.time(), None
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def show(self, frame, face, snap, warnings, coach=None) -> bool:
        now = time.time()
        dt, self._t = now - self._t, now
        self._fps = 1 / dt if self._fps is None else 0.9 * self._fps + 0.1 / max(dt, 1e-3)
        sfx = self.feel.sfx if self.feel else None
        sound_on = bool(sfx and sfx.backend != "off" and sfx.volume > 0)
        if self.feel is not None:
            self.feel.start()
            tip_id, n, full = typed_chars(coach)
            if tip_id and (coach.get("tip_age") or 0) < coach.get("tip_show", 12.0):
                self.feel.typed(tip_id, n, full)
        img = compose(frame, face, snap, warnings, self.mirror, self.landmarks, self._fps, coach,
                      self.feel, self.shown, self.pixel, now, sound_on,
                      None if self.speaker is None else self.speaker.enabled)
        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(15) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord("v") and self.speaker is not None:
            self.speaker.toggle()
        if key in (ord("l"), ord("m"), ord("p"), ord("s"), ord("v")) and self.feel:
            if key == ord("s") and sfx:
                sfx.volume, sfx.saved_volume = (0.0, sfx.volume) if sfx.volume > 0 else (
                    getattr(sfx, "saved_volume", 0.6) or 0.6, 0.0)
            self.feel.key()
        if key == ord("l"):
            self.landmarks = not self.landmarks
        if key == ord("m"):
            self.mirror = not self.mirror
        if key == ord("p"):
            self.pixel = PIXEL_SIZES[(PIXEL_SIZES.index(self.pixel) + 1) % len(PIXEL_SIZES)
                                     ] if self.pixel in PIXEL_SIZES else PIXEL_SIZES[0]
        try:  # closed with the X button
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except cv2.error:
            return False
        return True

    def close(self):
        if self.feel is not None:
            self.feel.stop()
            time.sleep(0.6)  # let the power-down sound finish
        cv2.destroyAllWindows()
