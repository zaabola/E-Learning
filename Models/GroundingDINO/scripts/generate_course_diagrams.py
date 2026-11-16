"""
Course Technical Diagram Generator for Model #3: Grounding DINO
===============================================================
Generates 5 high-contrast, crisp 720x420 technical architecture diagrams for
the `vision_yolo_ocr` course (`course_material/vision_yolo_ocr/diagrams/`)
and exports structured Grounding DINO puzzle definitions to:
  `Models/GroundingDINO/data/diagram_puzzles.json`
"""

import json
import os
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DINO_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(DINO_ROOT))

DIAGRAM_DIR = os.path.join(REPO_ROOT, "course_material", "vision_yolo_ocr", "diagrams")
DATA_DIR = os.path.join(DINO_ROOT, "data")

W, H = 720, 420


def get_font(size=14, bold=False):
    candidates = [
        "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_header(draw, title, subtitle):
    draw.rectangle([0, 0, W, 52], fill=(14, 22, 36))
    draw.line([0, 52, W, 52], fill=(56, 128, 200), width=2)
    f_title = get_font(16, bold=True)
    f_sub = get_font(12, bold=False)
    draw.text((18, 10), title, fill=(248, 216, 72), font=f_title)
    draw.text((18, 31), subtitle, fill=(160, 200, 235), font=f_sub)


def draw_arrow(draw, x1, y1, x2, y2, color=(120, 200, 255), width=3):
    draw.line([x1, y1, x2, y2], fill=color, width=width)
    dx = x2 - x1
    dy = y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head_len = 10
    head_w = 6
    p1 = (x2, y2)
    p2 = (int(x2 - ux * head_len + px * head_w), int(y2 - uy * head_len + py * head_w))
    p3 = (int(x2 - ux * head_len - px * head_w), int(y2 - uy * head_len - py * head_w))
    draw.polygon([p1, p2, p3], fill=color)


def draw_box_block(draw, box, bg, border, title, lines, title_color=(255, 255, 255)):
    x1, y1, x2, y2 = box
    draw.rectangle([x1 + 3, y1 + 3, x2 + 3, y2 + 3], fill=(8, 12, 20))
    draw.rectangle(box, fill=bg, outline=border, width=3)
    draw.rectangle([x1, y1, x2, y1 + 28], fill=border)
    f_t = get_font(13, bold=True)
    f_b = get_font(12, bold=False)
    draw.text((x1 + 8, y1 + 6), title, fill=title_color, font=f_t)
    cy = y1 + 38
    for ln in lines:
        draw.text((x1 + 10, cy), ln, fill=(230, 240, 250), font=f_b)
        cy += 19


def make_c0_cnn_fpn():
    img = Image.new("RGB", (W, H), (20, 28, 44))
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "LAB SCHEMATIC C0: CNN BACKBONE, STRIDE & FEATURE PYRAMID (FPN)",
        "Multi-scale feature extraction with 3x3 sliding filter, 1x1 lateral bridge, and P3/P4/P5 heads",
    )

    # Component 1: 3x3 Convolution Filter Kernel on Input Image
    b_kernel = [28, 95, 215, 365]
    draw_box_block(
        draw,
        b_kernel,
        bg=(28, 44, 68),
        border=(240, 140, 40),
        title="3x3 CONV KERNEL SCAN",
        lines=[
            "Input RGB: 640x640x3",
            "Shared Weights W (3x3)",
            "Stride = 2 halves H,W",
        ],
        title_color=(20, 16, 10),
    )
    # Draw visual 5x5 pixel grid + highlighted 3x3 orange filter window inside b_kernel
    gx0, gy0, cell = 48, 210, 26
    for r in range(5):
        for c in range(5):
            cx1 = gx0 + c * cell
            cy1 = gy0 + r * cell
            draw.rectangle([cx1, cy1, cx1 + cell, cy1 + cell], fill=(42, 66, 98), outline=(90, 130, 175), width=1)
    # Highlight 3x3 kernel
    draw.rectangle([gx0 + cell, gy0 + cell, gx0 + 4 * cell, gy0 + 4 * cell], outline=(255, 180, 40), width=4)
    draw.text((gx0 + cell + 8, gy0 + 2 * cell + 5), "3x3 W", fill=(255, 220, 110), font=get_font(13, True))

    # Component 2: Deep Stride-32 Semantic Pyramid Top
    b_deep = [250, 82, 435, 212]
    draw_box_block(
        draw,
        b_deep,
        bg=(44, 32, 72),
        border=(180, 110, 250),
        title="DEEP STRIDE-32 C5",
        lines=[
            "20x20 Low-Res Grid",
            "High Semantic Context",
            "2x Upsample Top-Down",
        ],
        title_color=(20, 12, 32),
    )

    # Shallow C3 block below it
    b_shallow = [250, 248, 435, 372]
    draw_box_block(
        draw,
        b_shallow,
        bg=(26, 54, 66),
        border=(72, 190, 220),
        title="SHALLOW STRIDE-8 C3",
        lines=[
            "80x80 High-Res Grid",
            "Sharp Edges & Corners",
            "Small Object Detail",
        ],
        title_color=(12, 24, 30),
    )

    # Component 3: 1x1 Lateral Connection Fusion Bridge
    b_lateral = [462, 165, 572, 295]
    draw_box_block(
        draw,
        b_lateral,
        bg=(24, 62, 52),
        border=(64, 235, 160),
        title="1x1 LATERAL",
        lines=[
            "Channel Align",
            "Element-Wise",
            "Add (+) Fuse",
        ],
        title_color=(10, 28, 20),
    )

    # Component 4: Multi-Scale FPN Output Pyramid
    b_fpn = [596, 88, 700, 372]
    draw_box_block(
        draw,
        b_fpn,
        bg=(56, 44, 24),
        border=(248, 208, 64),
        title="FPN HEADS",
        lines=[
            "P5: Large Obj",
            "",
            "P4: Med Obj",
            "",
            "P3: Tiny Obj",
            "(80x80 Fused)",
        ],
        title_color=(24, 20, 8),
    )

    draw_arrow(draw, 215, 155, 250, 145, (255, 180, 70))
    draw_arrow(draw, 215, 295, 250, 305, (80, 210, 240))
    draw_arrow(draw, 342, 248, 342, 212, (200, 140, 255))
    draw_arrow(draw, 435, 150, 515, 165, (190, 130, 255))
    draw_arrow(draw, 435, 305, 462, 245, (64, 235, 160))
    draw_arrow(draw, 572, 230, 596, 230, (248, 208, 64))

    out_path = os.path.join(DIAGRAM_DIR, "c0_cnn_fpn.png")
    img.save(out_path)
    return [
        {
            "puzzle_id": "c0_p1",
            "concept_id": 0,
            "concept_title": "CNN Backbones, Stride & Feature Pyramids (FPN)",
            "diagram_file": "course_material/vision_yolo_ocr/diagrams/c0_cnn_fpn.png",
            "diagram_width": W,
            "diagram_height": H,
            "target_label": "3x3 Convolution Filter Kernel",
            "dino_prompt": "orange 3x3 convolution kernel grid box.",
            "prompt": "LAB CALIBRATION C0-A: Click or drag a bounding box around the 3x3 Convolution Filter Kernel scanning the input image.",
            "target_box": b_kernel,
            "all_regions": [
                {"label": "3x3 Convolution Filter Kernel", "box": b_kernel},
                {"label": "Deep Stride-32 Semantic Layer", "box": b_deep},
                {"label": "Shallow Stride-8 High-Res Layer", "box": b_shallow},
                {"label": "1x1 Lateral Connection Bridge", "box": b_lateral},
                {"label": "Multi-Scale FPN Output Heads", "box": b_fpn},
            ],
            "explanation": "The 3x3 Convolution Kernel shares weights across all (H, W) positions, while the 1x1 Lateral Connection fuses shallow high-res features with upsampled deep semantic maps.",
        },
        {
            "puzzle_id": "c0_p2",
            "concept_id": 0,
            "concept_title": "CNN Backbones, Stride & Feature Pyramids (FPN)",
            "diagram_file": "course_material/vision_yolo_ocr/diagrams/c0_cnn_fpn.png",
            "diagram_width": W,
            "diagram_height": H,
            "target_label": "1x1 Lateral Connection Bridge",
            "dino_prompt": "green 1x1 lateral connection box.",
            "prompt": "LAB CALIBRATION C0-B: Point to the 1x1 Lateral Connection Bridge that aligns channels and fuses top-down and shallow feature maps.",
            "target_box": b_lateral,
            "all_regions": [
                {"label": "3x3 Convolution Filter Kernel", "box": b_kernel},
                {"label": "Deep Stride-32 Semantic Layer", "box": b_deep},
                {"label": "Shallow Stride-8 High-Res Layer", "box": b_shallow},
                {"label": "1x1 Lateral Connection Bridge", "box": b_lateral},
                {"label": "Multi-Scale FPN Output Heads", "box": b_fpn},
            ],
            "explanation": "1x1 lateral connections match channel dimensions between shallow C3/C4 layers and upsampled deep layers before element-wise addition.",
        },
    ]


def make_c1_iou_nms():
    img = Image.new("RGB", (W, H), (20, 28, 44))
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "LAB SCHEMATIC C1: BOUNDING BOX IoU & NON-MAXIMUM SUPPRESSION (NMS)",
        "Left: Intersection over Union (A n B / A u B) | Right: Suppressing duplicate ghost boxes via NMS",
    )

    # Left panel container
    draw.rectangle([24, 72, 348, 392], fill=(26, 38, 58), outline=(80, 120, 170), width=2)
    draw.text((38, 82), "IoU GEOMETRY: Area(A n B) / Area(A u B)", fill=(190, 220, 250), font=get_font(12, True))

    # Ground truth box A (blue) and Predicted box B (amber)
    box_a = [52, 120, 232, 305]
    box_b = [125, 165, 315, 360]
    inter_box = [125, 165, 232, 305]

    draw.rectangle(box_a, fill=(35, 75, 130), outline=(80, 170, 255), width=3)
    draw.text((62, 130), "Ground Truth A", fill=(160, 215, 255), font=get_font(12, True))

    draw.rectangle(box_b, fill=(110, 75, 30), outline=(255, 180, 60), width=3)
    draw.text((195, 330), "Predicted Box B", fill=(255, 215, 130), font=get_font(12, True))

    # Intersection Overlap Region (bright emerald highlight)
    draw.rectangle(inter_box, fill=(38, 165, 105), outline=(120, 255, 180), width=4)
    draw.text((135, 215), "INTERSECTION", fill=(255, 255, 255), font=get_font(12, True))
    draw.text((142, 235), "OVERLAP A n B", fill=(230, 255, 240), font=get_font(11, True))

    # Right panel: NMS before vs after
    b_ghosts = [378, 82, 535, 382]
    draw_box_block(
        draw,
        b_ghosts,
        bg=(58, 28, 36),
        border=(245, 95, 95),
        title="DUPLICATE GHOSTS",
        lines=[
            "Box 1: Score 0.96",
            "Box 2: Score 0.81 (IoU 0.78)",
            "Box 3: Score 0.64 (IoU 0.72)",
            "High Overlap -> Suppress!",
        ],
        title_color=(255, 240, 240),
    )
    # Draw overlapping red dashed rectangles inside b_ghosts
    draw.rectangle([402, 225, 500, 345], outline=(255, 100, 100), width=2)
    draw.rectangle([414, 235, 512, 355], outline=(255, 140, 90), width=2)
    draw.rectangle([395, 218, 492, 338], outline=(255, 220, 80), width=3)

    b_nms_winner = [568, 82, 698, 382]
    draw_box_block(
        draw,
        b_nms_winner,
        bg=(24, 58, 44),
        border=(72, 240, 145),
        title="SURVIVING NMS BOX",
        lines=[
            "Kept: Score 0.96",
            "Duplicates Removed",
            "1 Clean Detection!",
        ],
        title_color=(12, 28, 18),
    )
    draw.rectangle([590, 222, 678, 345], fill=(34, 110, 74), outline=(100, 255, 170), width=4)
    draw.text((602, 272), "CONF: 0.96", fill=(255, 255, 255), font=get_font(12, True))

    draw_arrow(draw, 535, 232, 568, 232, (90, 245, 160))

    out_path = os.path.join(DIAGRAM_DIR, "c1_iou_nms.png")
    img.save(out_path)
    return [
        {
            "puzzle_id": "c1_p1",
            "concept_id": 1,
            "concept_title": "Bounding Boxes, IoU & Non-Maximum Suppression (NMS)",
            "diagram_file": "course_material/vision_yolo_ocr/diagrams/c1_iou_nms.png",
            "diagram_width": W,
            "diagram_height": H,
            "target_label": "Intersection Overlap Area (A n B)",
            "dino_prompt": "green intersection overlap box.",
            "prompt": "LAB CALIBRATION C1-A: Click or box the green Intersection Overlap Area (A n B) that forms the numerator of the IoU metric.",
            "target_box": inter_box,
            "all_regions": [
                {"label": "Intersection Overlap Area (A n B)", "box": inter_box},
                {"label": "Duplicate Overlapping Ghost Boxes", "box": b_ghosts},
                {"label": "Surviving NMS Best Bounding Box", "box": b_nms_winner},
            ],
            "explanation": "Intersection over Union (IoU) divides the area of the overlapping intersection (A n B) by the total union area (A u B).",
        },
        {
            "puzzle_id": "c1_p2",
            "concept_id": 1,
            "concept_title": "Bounding Boxes, IoU & Non-Maximum Suppression (NMS)",
            "diagram_file": "course_material/vision_yolo_ocr/diagrams/c1_iou_nms.png",
            "diagram_width": W,
            "diagram_height": H,
            "target_label": "Surviving NMS Best Bounding Box",
            "dino_prompt": "surviving green NMS bounding box.",
            "prompt": "LAB CALIBRATION C1-B: Point to the Surviving NMS Best Bounding Box (Score 0.96) after duplicate ghost detections are suppressed.",
            "target_box": b_nms_winner,
            "all_regions": [
                {"label": "Intersection Overlap Area (A n B)", "box": inter_box},
                {"label": "Duplicate Overlapping Ghost Boxes", "box": b_ghosts},
                {"label": "Surviving NMS Best Bounding Box", "box": b_nms_winner},
            ],
            "explanation": "Non-Maximum Suppression (NMS) sorts boxes by confidence, keeps the highest-scoring box (0.96), and suppresses lower-scoring neighbors with high IoU.",
        },
    ]


def make_c2_yolo_grid():
    img = Image.new("RGB", (W, H), (20, 28, 44))
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "LAB SCHEMATIC C2: SINGLE-STAGE REAL-TIME YOLO DETECTION ARCHITECTURE",
        "S x S Grid Center Responsibility -> CSPBackbone & PANet Neck -> Decoupled Regression/Class Head",
    )

    # Component 1: S x S Grid with glowing center cell
    b_grid = [28, 86, 268, 376]
    draw_box_block(
        draw,
        b_grid,
        bg=(26, 40, 64),
        border=(248, 196, 56),
        title="S x S SPATIAL GRID",
        lines=["Center (cx, cy) cell owns", "the object prediction!"],
        title_color=(24, 18, 6),
    )
    gx0, gy0, cell = 48, 165, 38
    for r in range(5):
        for c in range(5):
            x1 = gx0 + c * cell
            y1 = gy0 + r * cell
            is_center = (r == 2 and c == 2)
            fill_c = (210, 60, 60) if is_center else (38, 58, 88)
            out_c = (255, 230, 90) if is_center else (85, 125, 170)
            draw.rectangle([x1, y1, x1 + cell, y1 + cell], fill=fill_c, outline=out_c, width=2 if is_center else 1)
    # Target object bounding box around the center cell
    draw.rectangle([gx0 + 28, gy0 + 28, gx0 + 4 * cell + 10, gy0 + 4 * cell + 8], outline=(80, 245, 150), width=3)
    draw.ellipse([gx0 + 2 * cell + 14, gy0 + 2 * cell + 14, gx0 + 2 * cell + 24, gy0 + 2 * cell + 24], fill=(255, 255, 100))

    # Component 2: CSPDarknet + PANet Neck
    b_neck = [304, 112, 482, 352]
    draw_box_block(
        draw,
        b_neck,
        bg=(36, 48, 76),
        border=(90, 185, 250),
        title="BACKBONE + PANET NECK",
        lines=[
            "CSPDarknet Features",
            "Top-Down FPN Path",
            "Bottom-Up PANet Path",
            "Single Forward Pass!",
            "(Zero RPN Stage)",
        ],
        title_color=(12, 22, 36),
    )

    # Component 3: Decoupled Detection Head
    b_head = [518, 86, 696, 376]
    draw_box_block(
        draw,
        b_head,
        bg=(48, 32, 68),
        border=(215, 115, 255),
        title="DECOUPLED YOLO HEAD",
        lines=[
            "Branch 1: Box Regression",
            " -> [dx, dy, w, h] (CIoU)",
            "",
            "Branch 2: Class Logits",
            " -> Sigmoid [C1..Ck] (BCE)",
            "",
            "Real-Time > 60 FPS!",
        ],
        title_color=(24, 12, 36),
    )

    draw_arrow(draw, 268, 230, 304, 230, (248, 196, 56))
    draw_arrow(draw, 482, 230, 518, 230, (215, 115, 255))

    out_path = os.path.join(DIAGRAM_DIR, "c2_yolo_grid.png")
    img.save(out_path)
    return [
        {
            "puzzle_id": "c2_p1",
            "concept_id": 2,
            "concept_title": "Single-Stage Real-Time Detection & YOLO Architecture",
            "diagram_file": "course_material/vision_yolo_ocr/diagrams/c2_yolo_grid.png",
            "diagram_width": W,
            "diagram_height": H,
            "target_label": "S x S Spatial Grid & Target Center Cell",
            "dino_prompt": "yellow S x S spatial grid box.",
            "prompt": "LAB CALIBRATION C2-A: Click or box the S x S Spatial Grid where the cell containing the object's center (cx, cy) predicts the bounding box.",
            "target_box": b_grid,
            "all_regions": [
                {"label": "S x S Spatial Grid & Target Center Cell", "box": b_grid},
                {"label": "CSPDarknet Backbone & PANet Neck", "box": b_neck},
                {"label": "Decoupled Detection Head (Box + Class)", "box": b_head},
            ],
            "explanation": "YOLO divides the image into an S x S grid; the unique cell containing the object's center point (cx, cy) is responsible for predicting its bounding box.",
        },
        {
            "puzzle_id": "c2_p2",
            "concept_id": 2,
            "concept_title": "Single-Stage Real-Time Detection & YOLO Architecture",
            "diagram_file": "course_material/vision_yolo_ocr/diagrams/c2_yolo_grid.png",
            "diagram_width": W,
            "diagram_height": H,
            "target_label": "Decoupled Detection Head (Box + Class)",
            "dino_prompt": "purple decoupled detection head box.",
            "prompt": "LAB CALIBRATION C2-B: Point to the Decoupled Detection Head that separates bounding-box coordinate regression from class probability prediction.",
            "target_box": b_head,
            "all_regions": [
                {"label": "S x S Spatial Grid & Target Center Cell", "box": b_grid},
                {"label": "CSPDarknet Backbone & PANet Neck", "box": b_neck},
                {"label": "Decoupled Detection Head (Box + Class)", "box": b_head},
            ],
            "explanation": "Modern YOLO detectors use a decoupled head with separate convolutional branches for spatial box localization (CIoU) and semantic classification (BCE).",
        },
    ]


def make_c3_crnn_ctc():
    img = Image.new("RGB", (W, H), (20, 28, 44))
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "LAB SCHEMATIC C3: SCENE TEXT OCR — CRNN & CTC ALIGNMENT LOSS",
        "CNN Vertical Feature Slices -> Bidirectional LSTM Sequence Modeling -> CTC Blank Collapse Decoder",
    )

    # Component 1: Scene Text Image & Vertical CNN Feature Columns
    b_cnn_slices = [26, 92, 232, 372]
    draw_box_block(
        draw,
        b_cnn_slices,
        bg=(28, 48, 68),
        border=(80, 200, 245),
        title="CNN VERTICAL SLICES",
        lines=[
            "Input Word Crop: 'YOLO'",
            "Sliced into T columns",
            "No char boxes needed!",
        ],
        title_color=(10, 24, 34),
    )
    # Draw plate with vertical slice columns
    draw.rectangle([46, 225, 212, 335], fill=(240, 235, 205), outline=(255, 190, 60), width=3)
    draw.text((64, 260), "Y  O  L  O", fill=(20, 30, 48), font=get_font(22, True))
    for sx in range(68, 205, 24):
        draw.line([sx, 225, sx, 335], fill=(220, 70, 70), width=2)

    # Component 2: Bidirectional LSTM Sequence Encoder
    b_bilstm = [266, 92, 480, 372]
    draw_box_block(
        draw,
        b_bilstm,
        bg=(44, 36, 74),
        border=(190, 125, 255),
        title="BIDIRECTIONAL LSTM",
        lines=[
            "Forward RNN (Left->Right)",
            "Backward RNN (Right->Left)",
            "Captures Context Across",
            "Time Steps t1 .. tT",
            "Disambiguates '0' vs 'O'!",
        ],
        title_color=(20, 12, 36),
    )
    # Draw recurrent nodes inside BiLSTM
    for i in range(4):
        nx = 292 + i * 46
        draw.ellipse([nx, 275, nx + 32, 307], fill=(95, 70, 165), outline=(220, 185, 255), width=2)
        draw.text((nx + 6, 284), f"h{i+1}", fill=(255, 255, 255), font=get_font(11, True))

    # Component 3: CTC Blank Collapse Decoder
    b_ctc = [514, 92, 696, 372]
    draw_box_block(
        draw,
        b_ctc,
        bg=(24, 58, 46),
        border=(72, 240, 155),
        title="CTC COLLAPSE DECODER",
        lines=[
            "Raw Frame Emissions:",
            " 'Y - Y - _ - O - L - O'",
            "",
            "1. Merge repeats (Y-Y->Y)",
            "2. Drop blank token (_)",
            "",
            "Output String: 'YOLO'",
        ],
        title_color=(10, 28, 20),
    )

    draw_arrow(draw, 232, 232, 266, 232, (80, 200, 245))
    draw_arrow(draw, 480, 232, 514, 232, (72, 240, 155))

    out_path = os.path.join(DIAGRAM_DIR, "c3_crnn_ctc.png")
    img.save(out_path)
    return [
        {
            "puzzle_id": "c3_p1",
            "concept_id": 3,
            "concept_title": "OCR: Scene Text Detection, CRNN & CTC Loss",
            "diagram_file": "course_material/vision_yolo_ocr/diagrams/c3_crnn_ctc.png",
            "diagram_width": W,
            "diagram_height": H,
            "target_label": "Bidirectional LSTM (BiLSTM) Sequence Encoder",
            "dino_prompt": "purple bidirectional LSTM sequence box.",
            "prompt": "LAB CALIBRATION C3-A: Click or box the Bidirectional LSTM (BiLSTM) Sequence Encoder that models left-to-right and right-to-left context.",
            "target_box": b_bilstm,
            "all_regions": [
                {"label": "CNN Vertical Feature Slices", "box": b_cnn_slices},
                {"label": "Bidirectional LSTM (BiLSTM) Sequence Encoder", "box": b_bilstm},
                {"label": "CTC Blank-Token Collapse Decoder", "box": b_ctc},
            ],
            "explanation": "The Bidirectional LSTM propagates context in both directions across feature columns, helping distinguish look-alike glyphs like '0' and 'O'.",
        },
        {
            "puzzle_id": "c3_p2",
            "concept_id": 3,
            "concept_title": "OCR: Scene Text Detection, CRNN & CTC Loss",
            "diagram_file": "course_material/vision_yolo_ocr/diagrams/c3_crnn_ctc.png",
            "diagram_width": W,
            "diagram_height": H,
            "target_label": "CTC Blank-Token Collapse Decoder",
            "dino_prompt": "green CTC collapse decoder box.",
            "prompt": "LAB CALIBRATION C3-B: Point to the CTC Blank-Token Collapse Decoder that merges repeated consecutive characters and strips blank tokens.",
            "target_box": b_ctc,
            "all_regions": [
                {"label": "CNN Vertical Feature Slices", "box": b_cnn_slices},
                {"label": "Bidirectional LSTM (BiLSTM) Sequence Encoder", "box": b_bilstm},
                {"label": "CTC Blank-Token Collapse Decoder", "box": b_ctc},
            ],
            "explanation": "CTC (Connectionist Temporal Classification) collapses repeated emissions unless separated by a blank token (_), turning 'Y-Y-_-O-L-O' into 'YOLO'.",
        },
    ]


def make_c4_vision_pipeline():
    img = Image.new("RGB", (W, H), (20, 28, 44))
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "LAB SCHEMATIC C4: END-TO-END TWO-STAGE YOLO + CRNN OCR VISION PIPELINE",
        "Stage 1: YOLO Localizes Text ROI -> Spatial Rectification & Crop -> Stage 2: CRNN + CTC Reads Text",
    )

    # Component 1: Stage 1 YOLO Detector
    b_stage1 = [24, 92, 228, 372]
    draw_box_block(
        draw,
        b_stage1,
        bg=(48, 36, 26),
        border=(250, 165, 55),
        title="STAGE 1: YOLO DETECTOR",
        lines=[
            "Full 1080p Camera Frame",
            "98% Background Filtered",
            "Localizes Box + Confidence",
        ],
        title_color=(24, 16, 6),
    )
    draw.rectangle([48, 220, 204, 340], fill=(32, 48, 66), outline=(110, 150, 190), width=2)
    draw.rectangle([78, 255, 178, 305], fill=(60, 110, 80), outline=(90, 255, 140), width=3)
    draw.text((88, 272), "PLATE 0.95", fill=(255, 255, 255), font=get_font(11, True))

    # Component 2: Perspective Rectification & ROI Crop
    b_rectify = [262, 92, 474, 372]
    draw_box_block(
        draw,
        b_rectify,
        bg=(26, 52, 68),
        border=(75, 210, 250),
        title="ROI CROP & RECTIFY",
        lines=[
            "1. Crop Bounding Box",
            "   (with 8% Margin Pad)",
            "2. Spatial Transformer",
            "   Deskews Tilted Text",
            "3. Resize H=32 Fixed",
            "   (Aspect Ratio Kept!)",
        ],
        title_color=(10, 24, 34),
    )

    # Component 3: Stage 2 CRNN + CTC Reader
    b_stage2 = [508, 92, 696, 372]
    draw_box_block(
        draw,
        b_stage2,
        bg=(26, 60, 44),
        border=(80, 245, 150),
        title="STAGE 2: CRNN + CTC OCR",
        lines=[
            "Reads Rectified Strip",
            "Decodes: 'DL-2026'",
            "",
            "Confidence Gate:",
            " Score_total =",
            "  Conf_YOLO x Conf_OCR",
        ],
        title_color=(10, 26, 18),
    )

    draw_arrow(draw, 228, 232, 262, 232, (250, 165, 55))
    draw_arrow(draw, 474, 232, 508, 232, (80, 245, 150))

    out_path = os.path.join(DIAGRAM_DIR, "c4_vision_pipeline.png")
    img.save(out_path)
    return [
        {
            "puzzle_id": "c4_p1",
            "concept_id": 4,
            "concept_title": "End-to-End Vision Pipelines: Combining YOLO + OCR",
            "diagram_file": "course_material/vision_yolo_ocr/diagrams/c4_vision_pipeline.png",
            "diagram_width": W,
            "diagram_height": H,
            "target_label": "ROI Crop & Perspective Rectification Stage",
            "dino_prompt": "blue ROI crop and rectify box.",
            "prompt": "LAB CALIBRATION C4-A: Click or box the ROI Crop & Perspective Rectification module that deskews tilted text between YOLO and OCR.",
            "target_box": b_rectify,
            "all_regions": [
                {"label": "Stage 1: Real-Time YOLO Detector", "box": b_stage1},
                {"label": "ROI Crop & Perspective Rectification Stage", "box": b_rectify},
                {"label": "Stage 2: CRNN + CTC OCR Reader", "box": b_stage2},
            ],
            "explanation": "Between Stage 1 (YOLO detection) and Stage 2 (CRNN OCR), the ROI Crop & Rectify module deskews perspective tilt and normalizes height to H=32 without squashing aspect ratio.",
        },
        {
            "puzzle_id": "c4_p2",
            "concept_id": 4,
            "concept_title": "End-to-End Vision Pipelines: Combining YOLO + OCR",
            "diagram_file": "course_material/vision_yolo_ocr/diagrams/c4_vision_pipeline.png",
            "diagram_width": W,
            "diagram_height": H,
            "target_label": "Stage 2: CRNN + CTC OCR Reader",
            "dino_prompt": "green stage 2 CRNN CTC OCR box.",
            "prompt": "LAB CALIBRATION C4-B: Point to Stage 2: CRNN + CTC OCR Reader, which decodes the character sequence and multiplies YOLO x OCR confidence.",
            "target_box": b_stage2,
            "all_regions": [
                {"label": "Stage 1: Real-Time YOLO Detector", "box": b_stage1},
                {"label": "ROI Crop & Perspective Rectification Stage", "box": b_rectify},
                {"label": "Stage 2: CRNN + CTC OCR Reader", "box": b_stage2},
            ],
            "explanation": "Decoupling Stage 1 (YOLO localization) from Stage 2 (CRNN sequence decoding) prevents running expensive OCR over 98% background pixels.",
        },
    ]


def generate_all():
    os.makedirs(DIAGRAM_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    puzzles = []
    puzzles.extend(make_c0_cnn_fpn())
    puzzles.extend(make_c1_iou_nms())
    puzzles.extend(make_c2_yolo_grid())
    puzzles.extend(make_c3_crnn_ctc())
    puzzles.extend(make_c4_vision_pipeline())

    out_json = os.path.join(DATA_DIR, "diagram_puzzles.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"course_id": "vision_yolo_ocr", "puzzles": puzzles}, f, indent=2)

    print(f"Generated 5 course diagrams in: {DIAGRAM_DIR}")
    print(f"Saved {len(puzzles)} Grounding DINO diagram puzzles to: {out_json}")
    return puzzles


if __name__ == "__main__":
    generate_all()
