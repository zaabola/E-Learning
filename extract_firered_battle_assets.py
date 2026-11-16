"""
Deterministic extractor for authentic GBA Pokemon FireRed battle sprites,
arenas, sliding platforms, trainer back-throw animations, Poke Balls,
stat-stage wave textures, HUD icons, and move effect strips from NewAssets/.
"""
import os
from collections import deque
from PIL import Image
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
NEW_DIR = os.path.join(ROOT, "NewAssets")
ASSETS = os.path.join(ROOT, "assets")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_png(img, rel_path):
    full = os.path.join(ASSETS, rel_path)
    ensure_dir(os.path.dirname(full))
    img.save(full, "PNG")
    with open(full, "rb") as f:
        sig = f.read(8)
    assert sig == b"\x89PNG\r\n\x1a\n", f"Invalid PNG header: {rel_path}"
    return full


def flood_bg_alpha(rgb_arr, extra_bg_colors=None, remove_enclosed_bg=False):
    """Remove background color seeded from all 4 border edges (4-connected)."""
    H, W, _ = rgb_arr.shape
    corners = [
        tuple(int(v) for v in rgb_arr[0, 0]),
        tuple(int(v) for v in rgb_arr[0, W - 1]),
        tuple(int(v) for v in rgb_arr[H - 1, 0]),
        tuple(int(v) for v in rgb_arr[H - 1, W - 1]),
    ]
    # Dominant corner color
    bg_set = {max(set(corners), key=corners.count)}
    if extra_bg_colors:
        for c in extra_bg_colors:
            bg_set.add(tuple(c))

    match = np.zeros((H, W), dtype=bool)
    for r, g, b in bg_set:
        match |= (rgb_arr[:, :, 0] == r) & (rgb_arr[:, :, 1] == g) & (rgb_arr[:, :, 2] == b)

    if remove_enclosed_bg:
        bg_mask = match
    else:
        bg_mask = np.zeros((H, W), dtype=bool)
        q = deque()
        for x in range(W):
            for y in (0, H - 1):
                if match[y, x] and not bg_mask[y, x]:
                    bg_mask[y, x] = True
                    q.append((x, y))
        for y in range(H):
            for x in (0, W - 1):
                if match[y, x] and not bg_mask[y, x]:
                    bg_mask[y, x] = True
                    q.append((x, y))
        while q:
            cx, cy = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H and match[ny, nx] and not bg_mask[ny, nx]:
                    bg_mask[ny, nx] = True
                    q.append((nx, ny))

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb_arr
    rgba[:, :, 3] = np.where(bg_mask, 0, 255)
    return Image.fromarray(rgba, "RGBA")


def extract_backgrounds():
    bg_path = os.path.join(
        NEW_DIR,
        "Game Boy Advance - Pokemon FireRed _ LeafGreen - Battle Effects - Battle Backgrounds.png",
    )
    arr = np.array(Image.open(bg_path).convert("RGB"))
    arenas = {
        "lab": (6, 6),
        "grass": (249, 6),
        "gym": (6, 121),
    }
    for key, (x0, y0) in arenas.items():
        crop = arr[y0 : y0 + 112, x0 : x0 + 240]
        save_png(Image.fromarray(crop, "RGB"), f"battle/back_{key}.png")

        # Pure scanline sky/floor column (x=0..7)
        sky = crop[:, 0:8, :]
        save_png(Image.fromarray(sky, "RGB"), f"battle/sky_{key}.png")

        # Exact platform alpha cutouts relative to pure scanline col 0
        ref = crop[:, 0:1, :]
        diff = np.any(crop != ref, axis=2)

        # Enemy oval platform: x in [104..240), y in [32..92)
        ew, eh = 136, 60
        epad = np.zeros((eh, ew, 4), dtype=np.uint8)
        sub_c = crop[32:92, 104:240]
        sub_m = diff[32:92, 104:240]
        epad[:, :, :3] = sub_c
        epad[:, :, 3] = np.where(sub_m, 255, 0)
        save_png(Image.fromarray(epad, "RGBA"), f"battle/pad_{key}_enemy.png")

        # Player foreground platform: x in [0..124), y in [84..112)
        pw, ph = 124, 28
        ppad = np.zeros((ph, pw, 4), dtype=np.uint8)
        psub_c = crop[84:112, 0:124]
        psub_m = diff[84:112, 0:124]
        ppad[:, :, :3] = psub_c
        ppad[:, :, 3] = np.where(psub_m, 255, 0)
        save_png(Image.fromarray(ppad, "RGBA"), f"battle/pad_{key}_player.png")
    print("[OK] Extracted 3 FireRed arenas + sky strips + 6 sliding platforms.")


def extract_trainers():
    tr_path = os.path.join(
        NEW_DIR,
        "Game Boy Advance - Pokemon FireRed _ LeafGreen - Trainers & Non-Playable Characters - Trainers.png",
    )
    arr = np.array(Image.open(tr_path).convert("RGB"))
    ORANGE = (255, 127, 39)

    def get_trainer_cell(cx, cy):
        box = arr[cy : cy + 64, cx : cx + 64]
        return flood_bg_alpha(box, extra_bg_colors=[ORANGE], remove_enclosed_bg=True)

    # 5-frame Red Back-Sprite Throw Animation: y=967, xs=[8, 73, 138, 203, 268]
    red_xs = [8, 73, 138, 203, 268]
    strip = Image.new("RGBA", (64 * 5, 64), (0, 0, 0, 0))
    for i, rx in enumerate(red_xs):
        frame = get_trainer_cell(rx, 967)
        strip.paste(frame, (i * 64, 0))
        if i == 0:
            save_png(frame, "trainers/player_back.png")
    save_png(strip, "trainers/player_back_throw.png")

    # All 5 trainers in Wildvale / KnowledgeQuest
    trainer_coords = {
        "rival_early": (8, 142),   # tr_rex (Rival)
        "scout": (268, 242),       # tr_cal (Camper / Scout)
        "picnic": (333, 242),      # tr_mira (Picnicker)
        "hiker": (8, 307),         # tr_rocco (Hiker)
        "leader_mason": (8, 667),  # tr_mason (Gym Leader)
    }
    for name, (tx, ty) in trainer_coords.items():
        save_png(get_trainer_cell(tx, ty), f"trainers/{name}.png")
    print("[OK] Extracted 5-frame Red throw strip + all 5 trainer front sprites.")


def extract_pokemon():
    pk_path = os.path.join(
        NEW_DIR,
        "Game Boy Advance - Pokemon FireRed _ LeafGreen - Pokemon - Pokemon.png",
    )
    arr = np.array(Image.open(pk_path).convert("RGB"))

    def clean_poke_cell(box):
        # Corner-seeded flood fill first, PLUS remove enclosed holes that match exact corner color
        # UNLESS corner color is light grey (216,216,216) or (232,232,232), where border-flood +
        # enclosed-bg check (connected components of exact corner color >= 6px not inside eyes) is safer.
        corner = tuple(int(v) for v in box[0, 0])
        img = flood_bg_alpha(box)
        rgba = np.array(img)
        # Check for enclosed holes of exact corner color (e.g. between Charizard wings/tail, Beedrill legs)
        c_match = (
            (box[:, :, 0] == corner[0])
            & (box[:, :, 1] == corner[1])
            & (box[:, :, 2] == corner[2])
            & (rgba[:, :, 3] > 0)
        )
        if np.any(c_match):
            # Only clear components of exact corner color with >= 4 pixels (preserves 1-2px shading highlights if any)
            H, W = c_match.shape
            vis = np.zeros((H, W), dtype=bool)
            for y in range(H):
                for x in range(W):
                    if c_match[y, x] and not vis[y, x]:
                        q = deque([(x, y)])
                        vis[y, x] = True
                        pts = [(x, y)]
                        while q:
                            cx, cy = q.popleft()
                            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                nx, ny = cx + dx, cy + dy
                                if 0 <= nx < W and 0 <= ny < H and c_match[ny, nx] and not vis[ny, nx]:
                                    vis[ny, nx] = True
                                    q.append((nx, ny))
                                    pts.append((nx, ny))
                        if len(pts) >= 4 and corner not in ((232, 232, 232),):
                            for px, py in pts:
                                rgba[py, px, 3] = 0
        return Image.fromarray(rgba, "RGBA")

    for dex in range(1, 21):
        row = (dex - 1) // 15
        col = (dex - 1) % 15
        xb = 10 + col * 130
        yf = 46 if row == 0 else 211
        yb = 111 if row == 0 else 276

        front = clean_poke_cell(arr[yf : yf + 64, xb + 1 : xb + 65])
        front_s = clean_poke_cell(arr[yf : yf + 64, xb + 66 : xb + 130])
        back = clean_poke_cell(arr[yb : yb + 64, xb + 1 : xb + 65])
        back_s = clean_poke_cell(arr[yb : yb + 64, xb + 66 : xb + 130])

        save_png(front, f"sprites/front/{dex}.png")
        save_png(front_s, f"sprites/front-shiny/{dex}.png")
        save_png(back, f"sprites/back/{dex}.png")
        save_png(back_s, f"sprites/back-shiny/{dex}.png")
    print("[OK] Extracted 80 authentic GBA FireRed 64x64 Pokemon sprites (#001-#020).")


def extract_pokeballs_and_hud():
    pb_path = os.path.join(
        NEW_DIR,
        "Game Boy Advance - Pokemon FireRed _ LeafGreen - Battle Effects - Poke Balls.png",
    )
    arr = np.array(Image.open(pb_path).convert("RGB"))
    PINK = (255, 166, 166)

    def crop_clean(x, y, w, h, cell_w=24, cell_h=24):
        sub = arr[y : y + h, x : x + w]
        mask = (sub[:, :, 0] == PINK[0]) & (sub[:, :, 1] == PINK[1]) & (sub[:, :, 2] == PINK[2])
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = sub
        rgba[:, :, 3] = np.where(mask, 0, 255)
        # Find tight non-transparent bounds to center cleanly inside cell_w x cell_h
        ys, xs = np.where(~mask)
        cell = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
        if len(xs) > 0:
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            piece = Image.fromarray(rgba[y0:y1, x0:x1], "RGBA")
            dx = (cell_w - (x1 - x0)) // 2
            dy = (cell_h - (y1 - y0)) // 2
            cell.paste(piece, (dx, dy))
        return cell

    # Atlas: 4 cols x 3 rows of 24x24 cells (96 x 72)
    # Row 0: Poke Ball (closed, half, open), Ultra Ball closed (col 3 bonus)
    # Row 1: Great Ball (closed, half, open), Master Ball closed (col 3 bonus)
    # Row 2: Catch Star, Energy Orb, Green Sparkle Big, Diag Streak
    atlas = Image.new("RGBA", (96, 72), (0, 0, 0, 0))
    # Poke Ball col x=32..48; Great Ball col x=48..64
    ball_rows_y = [(18, 16), (34, 16), (52, 20)]
    for fi, (by, bh) in enumerate(ball_rows_y):
        atlas.paste(crop_clean(32, by, 16, bh), (fi * 24, 0))
        atlas.paste(crop_clean(48, by, 16, bh), (fi * 24, 24))
    atlas.paste(crop_clean(0, 18, 16, 16), (72, 0))   # Ultra Ball closed
    atlas.paste(crop_clean(96, 18, 16, 16), (72, 24)) # Master Ball closed

    # Row 2 FX: star (88,80,9,16), blue orb (96,80,9,16), green sparkle (112,80,10,16), diag beam (64,80,9,16)
    atlas.paste(crop_clean(87, 80, 10, 16), (0, 48))
    atlas.paste(crop_clean(96, 80, 10, 16), (24, 48))
    atlas.paste(crop_clean(112, 80, 10, 16), (48, 48))
    atlas.paste(crop_clean(64, 80, 9, 16), (72, 48))
    save_png(atlas, "battle_fx/pokeballs.png")

    # Copy Type & Status Icons sheet (already crisp RGBA!)
    ts_path = os.path.join(
        NEW_DIR,
        "Game Boy Advance - Pokemon FireRed _ LeafGreen - Menu Elements - Type _ Status Icons.png",
    )
    ts_img = Image.open(ts_path).convert("RGBA")
    save_png(ts_img, "battle_fx/type_status_icons.png")

    # Extract individual crisp 32x12 Type icons for move-panel & UI!
    # Let's also slice party Pokeball HUD icons from HP Bars & In-battle Menu.png
    hud_path = os.path.join(
        NEW_DIR,
        "Game Boy Advance - Pokemon FireRed _ LeafGreen - Battle Effects - HP Bars & In-battle Menu.png",
    )
    hud_arr = np.array(Image.open(hud_path).convert("RGB"))
    # Party ball icons at y=58..70, x=122..166 (empty, healthy, statused, fainted)
    balls_sub = hud_arr[58:70, 122:166]
    bmask = (balls_sub[:, :, 0] == 255) & (balls_sub[:, :, 1] == 255) & (balls_sub[:, :, 2] == 255)
    brgba = np.zeros((12, 44, 4), dtype=np.uint8)
    brgba[:, :, :3] = balls_sub
    # Only outer white is transparent (seed from edges so inner white half of ball stays opaque!)
    save_png(flood_bg_alpha(balls_sub), "battle_fx/hud_party_balls.png")
    print("[OK] Extracted Poke Balls 3-state atlas, Type/Status icons, and HUD party balls.")


def extract_stat_waves():
    st_path = os.path.join(
        NEW_DIR,
        "Game Boy Advance - Pokemon FireRed _ LeafGreen - Battle Effects - Stat Effects.png",
    )
    arr = np.array(Image.open(st_path).convert("RGB"))
    # Exact measured 256x256 boxes:
    # 0:atk (204,160), 1:def (502,160), 2:acc (59,441), 3:eva (357,441),
    # 4:spe (655,441), 5:spa (204,725), 6:spd (502,725), 7:mix (357,1013)
    boxes = [
        (204, 160),
        (502, 160),
        (59, 441),
        (357, 441),
        (655, 441),
        (204, 725),
        (502, 725),
        (357, 1013),
    ]
    atlas = Image.new("RGBA", (64 * 8, 128), (0, 0, 0, 0))
    for i, (bx, by) in enumerate(boxes):
        # Take a 64x128 vertical slice from interior (bx+32, by+32) so it tiles seamlessly
        sub = arr[by + 32 : by + 160, bx + 32 : bx + 96]
        rgba = np.zeros((128, 64, 4), dtype=np.uint8)
        rgba[:, :, :3] = sub
        rgba[:, :, 3] = 255
        atlas.paste(Image.fromarray(rgba, "RGBA"), (i * 64, 0))
    save_png(atlas, "battle_fx/stat_waves.png")
    print("[OK] Extracted 8-stat scrolling FireRed chevron wave atlas (512x128).")


def extract_attack_fx():
    ae_path = os.path.join(
        NEW_DIR,
        "Game Boy Advance - Pokemon FireRed _ LeafGreen - Battle Effects - Attack Effects.png",
    )
    arr = np.array(Image.open(ae_path).convert("RGB"))
    GREEN = (34, 177, 76)
    SUB_GREY = (160, 144, 136)

    def crop_centered(rect, cell_w, cell_h, extra_bg=None):
        x, y, w, h = rect
        sub = arr[y : y + h, x : x + w]
        mask = (sub[:, :, 0] == GREEN[0]) & (sub[:, :, 1] == GREEN[1]) & (sub[:, :, 2] == GREEN[2])
        if extra_bg:
            for r, g, b in extra_bg:
                mask |= (sub[:, :, 0] == r) & (sub[:, :, 1] == g) & (sub[:, :, 2] == b)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = sub
        rgba[:, :, 3] = np.where(mask, 0, 255)
        ys, xs = np.where(~mask)
        cell = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
        if len(xs) > 0:
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            pw, ph = x1 - x0, y1 - y0
            piece = Image.fromarray(rgba[y0:y1, x0:x1], "RGBA")
            if pw > cell_w or ph > cell_h:
                piece.thumbnail((cell_w, cell_h), Image.NEAREST)
                pw, ph = piece.size
            dx = (cell_w - pw) // 2
            dy = (cell_h - ph) // 2
            cell.paste(piece, (dx, dy))
        return cell

    def make_strip(rects, cell_w, cell_h, rel_path, extra_bg=None):
        strip = Image.new("RGBA", (cell_w * len(rects), cell_h), (0, 0, 0, 0))
        for i, r in enumerate(rects):
            strip.paste(crop_centered(r, cell_w, cell_h, extra_bg=extra_bg), (i * cell_w, 0))
        save_png(strip, rel_path)
        return strip

    # 1. Classic FireRed Normal Hit Starburst (4 frames, 48x48)
    make_strip(
        [
            (488, 411, 16, 16),  # initial orange/white spark
            (480, 437, 32, 32),  # 8-point FireRed hit starburst
            (480, 469, 32, 33),  # expanding ring burst
            (480, 501, 32, 32),  # dissipating shock ring
        ],
        48, 48, "battle_fx/fx_hit.png",
    )

    # 2. Diagonal Red/White Claw Tear (Scratch / Fury Attack / Slash, 5 frames, 48x48)
    make_strip(
        [
            (426, 144, 10, 10),
            (427, 175, 16, 17),
            (426, 207, 27, 26),
            (426, 240, 28, 28),
            (426, 272, 30, 30),
        ],
        48, 48, "battle_fx/fx_slash.png",
    )

    # 3. Golden 3-Prong Slash Claws (Slash / Metal Claw, 5 frames, 48x48)
    claw_strip = make_strip(
        [
            (746, 190, 28, 22),
            (748, 214, 28, 32),
            (750, 314, 24, 28),
            (742, 342, 30, 30),
            (746, 376, 18, 20),
        ],
        48, 48, "battle_fx/fx_claw.png",
    )
    # Create cool steel-silver recolor for Metal Claw!
    mc = np.array(claw_strip).astype(np.float32)
    lum = 0.299 * mc[:, :, 0] + 0.587 * mc[:, :, 1] + 0.114 * mc[:, :, 2]
    mc_out = np.zeros_like(mc, dtype=np.uint8)
    mc_out[:, :, 0] = np.clip(lum * 0.92 + 25, 0, 255).astype(np.uint8)
    mc_out[:, :, 1] = np.clip(lum * 1.00 + 32, 0, 255).astype(np.uint8)
    mc_out[:, :, 2] = np.clip(lum * 1.12 + 45, 0, 255).astype(np.uint8)
    mc_out[:, :, 3] = np.array(claw_strip)[:, :, 3]
    save_png(Image.fromarray(mc_out, "RGBA"), "battle_fx/fx_claw_metal.png")

    # 4. Bite / Hyper Fang / Super Fang Jaws (2 frames: upper jaw, lower jaw, 64x24 each)
    make_strip(
        [
            (385, 90, 62, 16),   # upper teeth
            (385, 110, 62, 16),  # lower teeth
        ],
        64, 24, "battle_fx/fx_fangs.png",
    )

    # 5. Fire Plume (Ember / Flamethrower / Burn, 6 frames, 48x48) + Blazing Fireball Meteor (64x64)
    make_strip(
        [
            (7, 218, 28, 26),
            (7, 247, 28, 32),
            (7, 280, 28, 31),
            (7, 311, 28, 30),
            (7, 341, 28, 34),
            (7, 377, 28, 29),
        ],
        48, 48, "battle_fx/fx_fire.png",
    )
    make_strip([(53, 206, 50, 61)], 64, 64, "battle_fx/fx_fireball.png")

    # 6. Water Gun & Splash (6 frames, 48x48) + Bubbles (4 frames, 32x32)
    make_strip(
        [
            (5, 3, 16, 16),     # water droplet
            (35, 16, 10, 12),   # fast water drop
            (9, 62, 10, 31),    # water jet column 1
            (6, 99, 14, 32),    # water jet column 2
            (24, 67, 32, 14),   # water splash crown 1
            (27, 85, 26, 14),   # water splash crown 2
        ],
        48, 48, "battle_fx/fx_water.png",
    )
    make_strip(
        [
            (63, 7, 24, 24),    # iridescent bubble 1
            (60, 36, 27, 30),   # iridescent bubble 2
            (65, 72, 22, 22),   # popping bubble ring
            (65, 170, 27, 27),  # water burst ring
        ],
        32, 32, "battle_fx/fx_bubble.png",
    )

    # 7. Grass: Vine Whip (6 frames, 48x48), Razor Leaf (8 frames, 32x32), Leech Seed (3 frames, 32x32)
    make_strip(
        [
            (125, 31, 23, 15),
            (125, 55, 30, 14),
            (125, 78, 32, 16),
            (125, 103, 32, 15),
            (126, 158, 30, 31),
            (125, 229, 32, 24),
        ],
        48, 48, "battle_fx/fx_vine.png",
    )
    make_strip(
        [
            (928, 95, 6, 13),
            (924, 113, 9, 11),
            (921, 130, 12, 9),
            (921, 150, 14, 7),
            (922, 163, 12, 9),
            (920, 176, 9, 11),
            (922, 191, 6, 13),
            (920, 224, 12, 12),
        ],
        24, 24, "battle_fx/fx_leaf.png",
    )
    make_strip(
        [
            (108, 6, 12, 11),   # seed
            (107, 25, 14, 11),  # small sprout
            (106, 42, 16, 10),  # twin leaf sprout
        ],
        24, 24, "battle_fx/fx_seed.png",
    )

    # 8. Flying / Dragon: Tornado (2 frames: normal + horizontally flipped so spinning animates!, 48x64)
    t0 = crop_centered((341, 4, 32, 64), 48, 64, extra_bg=[SUB_GREY])
    t1 = t0.transpose(Image.FLIP_LEFT_RIGHT)
    t_strip = Image.new("RGBA", (96, 64), (0, 0, 0, 0))
    t_strip.paste(t0, (0, 0))
    t_strip.paste(t1, (48, 0))
    save_png(t_strip, "battle_fx/fx_tornado.png")

    # Crescent Wind Blades + Feathers (4 frames, 48x48)
    # Also grab white feather from Move Effects.png!
    me_path = os.path.join(
        NEW_DIR,
        "Game Boy Advance - Pokemon FireRed _ LeafGreen - Battle Effects - Move Effects.png",
    )
    me_arr = np.array(Image.open(me_path).convert("RGB"))
    # White feathers in Move Effects.png around y=915..965, x=440..473 on black (0,0,0)
    wind_strip = Image.new("RGBA", (48 * 4, 48), (0, 0, 0, 0))
    wind_strip.paste(crop_centered((317, 330, 30, 12), 48, 48), (0, 0))
    wind_strip.paste(crop_centered((317, 360, 30, 14), 48, 48), (48, 0))
    # Feather from Move Effects or Attack Effects petal/wing
    # Check Attack Effects (170, 218..252) pink petals + crisp white feather built from Move Effects
    f_sub = me_arr[914:968, 441:472]
    f_black = (f_sub[:, :, 0] < 16) & (f_sub[:, :, 1] < 16) & (f_sub[:, :, 2] < 16)
    ys, xs = np.where(~f_black)
    if len(xs) > 0:
        # Split top feather and bottom feather
        mid_y = (ys.min() + ys.max()) // 2
        for fi, (fy0, fy1) in enumerate([(0, mid_y), (mid_y, f_sub.shape[0])]):
            part = f_sub[fy0:fy1, :]
            pm = (part[:, :, 0] < 16) & (part[:, :, 1] < 16) & (part[:, :, 2] < 16)
            prgba = np.zeros((part.shape[0], part.shape[1], 4), dtype=np.uint8)
            prgba[:, :, :3] = part
            prgba[:, :, 3] = np.where(pm, 0, 255)
            fcell = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
            pimg = Image.fromarray(prgba, "RGBA")
            fcell.paste(pimg, ((48 - pimg.width) // 2, (48 - pimg.height) // 2))
            wind_strip.paste(fcell, ((2 + fi) * 48, 0))
    save_png(wind_strip, "battle_fx/fx_wind.png")

    # 9. Bug / Poison: Spider Web (64x64), Stinger Needles (4 frames, 36x20), Poison Flames/Drops (5 frames, 36x36)
    web = crop_centered((388, 14, 64, 64), 64, 64, extra_bg=[SUB_GREY])
    save_png(web, "battle_fx/fx_web.png")

    make_strip(
        [
            (298, 64, 32, 11),
            (298, 80, 24, 11),
            (298, 96, 17, 11),
            (298, 112, 8, 11),
        ],
        36, 36, "battle_fx/fx_needle.png",
    )

    make_strip(
        [
            (698, 235, 12, 21),  # purple flame 1
            (698, 260, 12, 27),  # purple flame 2
            (697, 292, 12, 27),  # purple flame 3
            (383, 473, 14, 14),  # poison droplet
            (381, 489, 16, 30),  # poison splash
        ],
        36, 36, "battle_fx/fx_poison.png",
    )

    # 10. Psychic / Sound: Eye + 4 Concentric Rings (5 frames, 48x48)
    make_strip(
        [
            (314, 440, 38, 34),  # Psychic eye!
            (218, 159, 32, 32),  # Ring 1
            (218, 191, 32, 32),  # Ring 2
            (218, 223, 32, 32),  # Ring 3
            (218, 255, 32, 32),  # Ring 4
        ],
        48, 48, "battle_fx/fx_psychic.png",
    )

    # 11. Status & Utility Icons (9 frames, 32x32):
    # 0,1: Confusion yellow duck; 2,3: Musical notes (Growl/Supersonic);
    # 4: Fluffy Smoke/Sand cloud; 5: Crit/Electric spark; 6: Pink heart (Sweet Scent);
    # 7: Angry vein mark; 8: GBA Sleep Z!
    st_strip = make_strip(
        [
            (133, 298, 16, 14),  # 0: Confusion duck 1
            (135, 314, 13, 14),  # 1: Confusion duck 2
            (242, 686, 10, 13),  # 2: Musical note 1
            (241, 717, 15, 15),  # 3: Musical note 2
            (618, 551, 28, 24),  # 4: Smoke / Sand cloud puff
            (329, 120, 22, 22),  # 5: Electric / Crit spark
            (431, 530, 14, 12),  # 6: Pink heart (Sweet Scent)
            (196, 27, 18, 18),   # 7: Angry vein mark
        ],
        32, 32, "battle_fx/fx_status.png",
    )
    # Append Frame 8: authentic GBA Sleep Z from Move Effects.png (357, 71, 17, 14)
    z_sub = me_arr[71:85, 357:374]
    z_img = flood_bg_alpha(z_sub)
    st_full = Image.new("RGBA", (32 * 9, 32), (0, 0, 0, 0))
    st_full.paste(st_strip, (0, 0))
    st_full.paste(z_img, (8 * 32 + (32 - z_img.width) // 2, (32 - z_img.height) // 2), z_img)
    save_png(st_full, "battle_fx/fx_status.png")

    # 12. Scary Face (112x76 composite: glowing yellow eyes + jagged mouth!)
    sf = Image.new("RGBA", (112, 76), (0, 0, 0, 0))
    eye_l = crop_centered((376, 691, 38, 30), 42, 32)
    eye_r = crop_centered((429, 692, 38, 30), 42, 32)
    mouth = crop_centered((372, 739, 102, 30), 104, 34)
    sf.paste(eye_l, (10, 4))
    sf.paste(eye_r, (60, 4))
    sf.paste(mouth, (4, 38))
    save_png(sf, "battle_fx/fx_scaryface.png")

    print("[OK] Extracted 12 multi-frame FireRed attack & status FX strips.")


if __name__ == "__main__":
    extract_backgrounds()
    extract_trainers()
    extract_pokemon()
    extract_pokeballs_and_hud()
    extract_stat_waves()
    extract_attack_fx()
    print("=== ALL FIRERED BATTLE ASSETS EXTRACTED & PNG-VALIDATED ===")
