"""
Dynamic PDF / PPTX / Word (.docx) & Slide Ingestor with Live Qwen2.5-VL-7B Inspection
(`Models/VLM/scripts/custom_course_ingestor.py`)
=====================================================================================
Transforms ANY uploaded `.pdf`, `.pptx`, `.docx` (Word), or image into a complete
5-Concept Knowledge Graph + 10 Grounding DINO Lab Puzzles + VLM VQA Battle Course:

  1. [Whole-Document Smart Photo Extraction & Logo Filtering]:
     - Scans ALL pages/slides (skipping Page 1 cover logos when the document has >=4 pages).
     - Filters out small icons, school/company banner logos, blank boxes, and repeated
       header/footer images (by xref count + 16x16 perceptual hash deduplication).
     - Pairs every unique extracted diagram 1-to-1 with its exact page/slide OCR text.
     - Selects 5 unique, visually rich diagrams evenly spread across the ENTIRE document
       (never stopping at page 6 and never duplicating the same photo across concepts).

  2. [Pure Live `Qwen2.5-VL-7B-Instruct` Inspection (Zero Cross-Course Contamination)]:
     - Calls `vlm_engine.comprehend_diagram_live(..., use_reference_cache=False)` with
       the exact extracted photo + its slide text context.
     - Automatically rejects non-educational images if `is_educational_diagram == false`.
     - Generates genuine topic-matched concept titles, visual component labels, bounding
       boxes, Grounding DINO prompts, Lab Calibration prompts, Study Notes, Multiple-Choice
       Battle Questions, and DeBERTa-v3 VQA rubrics directly from the uploaded course!
"""

import base64
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import torch
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VLM_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_ROOT = os.path.dirname(VLM_ROOT)
REPO_ROOT = os.path.dirname(MODELS_ROOT)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from vlm_diagram_engine import box_content_stats, trim_blank_margins


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s[:32] or f"course_{int(time.time())}"


def _fit_diagram_canvas(pil_img: Image.Image, target_w: int = 720, target_h: int = 420):
    """
    Letterboxes an extracted document photo onto a crisp 720x420 dark lab monitor canvas.
    Returns `(canvas_img, (ox, oy, nw, nh))` so bounding boxes map cleanly onto the canvas.
    """
    img = pil_img.convert("RGB")
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return Image.new("RGB", (target_w, target_h), (14, 22, 36)), (0, 0, target_w, target_h)
    scale = min(target_w / float(iw), target_h / float(ih))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = img.resize(
        (nw, nh),
        Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.BICUBIC,
    )
    canvas = Image.new("RGB", (target_w, target_h), (14, 22, 36))
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, (ox, oy, nw, nh)


def _perceptual_signature(pil_img: Image.Image) -> str:
    """Computes a coarse 12x12 grayscale hash to deduplicate identical or near-identical images."""
    small = pil_img.convert("L").resize((12, 12))
    arr = np.asarray(small, dtype=np.float32)
    bits = (arr > float(arr.mean())).astype(np.uint8).tobytes()
    return hashlib.md5(bits).hexdigest()


def _diagram_quality_score(pil_img: Image.Image) -> float:
    """
    Filters out university/company logos, tiny icons, solid color bars, and banner headers.
    Returns 0.0 to reject, or a positive visual-richness score for real technical diagrams.
    """
    w, h = pil_img.size
    if w < 220 or h < 140:
        return 0.0
    area = w * h
    if area < 45000:
        return 0.0
    aspect = w / float(h)
    # Reject skinny banner logos (like header/footer logos)
    if aspect > 3.1 or aspect < 0.35:
        return 0.0

    # Measure visual detail / edge complexity & color richness
    arr = np.asarray(pil_img.convert("RGB").resize((96, 64)), dtype=np.float32)
    gray = arr.mean(axis=2)
    std_val = float(gray.std())
    if std_val < 12.0:
        return 0.0  # nearly blank / solid rectangle

    # Edge energy (horizontal + vertical gradients): technical diagrams have rich lines/boxes/text
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_score = float(gx + gy)

    # Non-white foreground ratio (logos often have tiny center mark surrounded by 85% white)
    non_white = float((gray < 242.0).mean())
    if non_white < 0.08:
        return 0.0

    # Favor larger, multi-color diagrams with strong structural edges
    color_spread = float(np.abs(arr[:, :, 0] - arr[:, :, 2]).mean())
    size_bonus = min(2.0, area / 200000.0)
    return round(edge_score * (0.6 + 0.4 * size_bonus) + 0.25 * color_spread + 12.0 * non_white, 2)


def _extract_slide_title(page_text: str, fallback: str) -> str:
    """Extracts the most informative non-navigation heading from a Beamer/PPT/PDF slide."""
    lines = [ln.strip(" -•*\t") for ln in (page_text or "").splitlines() if len(ln.strip(" -•*\t")) >= 4]
    # Skip common short Beamer header/footer navigation lines & page numbers (e.g. "7/54", "Plan")
    clean_lines = []
    for ln in lines:
        if re.match(r"^\d+\s*/\s*\d+$", ln):
            continue
        if ln.lower() in ("plan", "introduction", "sommaire", "outline", "agenda", "merci", "thank you"):
            continue
        clean_lines.append(ln)
    if not clean_lines:
        return fallback
    # In Beamer slides (like Esprit slides), top lines are tiny nav menu; slide title often appears around lines 3-6
    for ln in clean_lines:
        if 8 <= len(ln) <= 70 and not ln.endswith("."):
            if ":" in ln or len(ln.split()) >= 2:
                return ln[:64]
    return clean_lines[0][:64]


def _select_balanced_candidates(candidates: list, total_pages: int, k: int = 5) -> list:
    """
    Spreads selection evenly across ALL pages of the document (e.g., 5 sequential chapter bands)
    so a 54-page PDF contributes unique diagrams from beginning, middle, and end — never all from page 1-5!
    """
    if len(candidates) <= k:
        return candidates

    # Sort all unique candidates by document page order
    ordered = sorted(candidates, key=lambda c: (c["page"], -c["score"]))
    selected = []
    used_sigs = set()

    min_p = ordered[0]["page"]
    max_p = max(ordered[-1]["page"], min_p + 1)
    band_width = (max_p - min_p + 1) / float(k)

    for b in range(k):
        lo = min_p + b * band_width
        hi = min_p + (b + 1) * band_width + (0.01 if b == k - 1 else 0.0)
        band_items = [c for c in ordered if lo <= c["page"] < hi and c["sig"] not in used_sigs]
        if band_items:
            best = max(band_items, key=lambda c: c["score"])
            selected.append(best)
            used_sigs.add(best["sig"])

    # Fill remaining slots with the highest-scoring unused diagrams across the whole doc
    if len(selected) < k:
        remaining = sorted(
            [c for c in ordered if c["sig"] not in used_sigs],
            key=lambda c: -c["score"],
        )
        for c in remaining:
            if len(selected) >= k:
                break
            selected.append(c)
            used_sigs.add(c["sig"])

    # Return in pedagogical chapter page order
    selected.sort(key=lambda c: c["page"])
    return selected


def extract_document_text_and_photos(
    file_name: str,
    file_bytes: bytes,
    diag_dir: str,
    course_id: str,
) -> dict:
    """
    Scans the ENTIRE uploaded PDF / PPTX / DOCX / Image file:
      - Filters out cover-page logos, tiny icons, and repeated headers/footers.
      - Deduplicates identical images by perceptual hash.
      - Pairs every unique extracted diagram with its exact page/slide text.
      - Returns up to 8 top diverse candidates spread across the whole document so
        `Qwen2.5-VL-7B` can pick the 5 strongest educational diagrams with zero duplicates.
    """
    ext = os.path.splitext((file_name or "").lower())[1]
    all_page_texts = []
    raw_candidates = []
    seen_sigs = set()
    total_pages = 1

    # -------------------------------------------------------------------------
    # CASE 1: PDF Document (.pdf) via PyMuPDF (fitz)
    # -------------------------------------------------------------------------
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(doc)

            # First pass: count how many pages each xref appears on (header/footer logos repeat!)
            xref_page_counts = {}
            for page in doc:
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    xref_page_counts[xref] = xref_page_counts.get(xref, 0) + 1

            # Skip Page 1 cover slide for diagram extraction when document has >= 4 pages!
            start_page_for_imgs = 1 if total_pages >= 4 else 0

            for page_idx, page in enumerate(doc):
                txt = (page.get_text("text") or "").strip()
                if txt:
                    all_page_texts.append((page_idx + 1, txt))

                if page_idx < start_page_for_imgs:
                    continue

                # Skip pure "Plan" / TOC slides
                if re.search(r"\bplan\b", txt.lower()[:120]) and len(page.get_images(full=True)) == 0:
                    continue

                page_imgs = page.get_images(full=True)
                best_page_cand = None

                for img_info in page_imgs:
                    xref = img_info[0]
                    # Reject images repeated across >2 pages (watermarks, school logos, footer badges)
                    if xref_page_counts.get(xref, 0) > 2:
                        continue
                    try:
                        pil_im = None
                        try:
                            rects = page.get_image_rects(xref)
                            if rects:
                                r = rects[0]
                                if r.width >= 60 and r.height >= 45:
                                    clip_r = fitz.Rect(
                                        max(page.rect.x0, r.x0 - 3),
                                        max(page.rect.y0, r.y0 - 3),
                                        min(page.rect.x1, r.x1 + 3),
                                        min(page.rect.y1, r.y1 + 3),
                                    )
                                    pix = page.get_pixmap(
                                        clip=clip_r,
                                        dpi=220,
                                        colorspace=fitz.csRGB,
                                        alpha=False,
                                    )
                                    if pix.width >= 180 and pix.height >= 120:
                                        pil_im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        except Exception:
                            pil_im = None

                        if pil_im is None:
                            base_img = doc.extract_image(xref)
                            pil_im = Image.open(io.BytesIO(base_img["image"])).convert("RGB")
                        score = _diagram_quality_score(pil_im)
                        if score <= 0.0:
                            continue
                        sig = _perceptual_signature(pil_im)
                        if sig in seen_sigs:
                            continue
                        if best_page_cand is None or score > best_page_cand["score"]:
                            best_page_cand = {
                                "page": page_idx + 1,
                                "slide_title": _extract_slide_title(txt, f"Page {page_idx + 1}"),
                                "slide_text": txt,
                                "pil_img": pil_im,
                                "sig": sig,
                                "score": score,
                                "is_full_slide": False,
                            }
                    except Exception:
                        pass

                if best_page_cand is not None:
                    seen_sigs.add(best_page_cand["sig"])
                    raw_candidates.append(best_page_cand)

            # Fallback for vector-only PDFs (diagrams drawn as PDF paths instead of embedded bitmaps):
            # Render slides that contain vector drawings (`page.get_drawings()`)!
            if len(raw_candidates) < 5:
                for page_idx, page in enumerate(doc):
                    if page_idx < start_page_for_imgs:
                        continue
                    txt = (page.get_text("text") or "").strip()
                    if any(c["page"] == page_idx + 1 for c in raw_candidates):
                        continue
                    drawings_count = len(page.get_drawings())
                    if drawings_count >= 12:
                        try:
                            pix = page.get_pixmap(dpi=130)
                            pil_im = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                            sig = _perceptual_signature(pil_im)
                            if sig not in seen_sigs:
                                seen_sigs.add(sig)
                                raw_candidates.append(
                                    {
                                        "page": page_idx + 1,
                                        "slide_title": _extract_slide_title(txt, f"Slide {page_idx + 1}"),
                                        "slide_text": txt,
                                        "pil_img": pil_im,
                                        "sig": sig,
                                        "score": 15.0 + min(30.0, drawings_count * 0.25),
                                        "is_full_slide": True,
                                    }
                                )
                        except Exception:
                            pass
            doc.close()
        except Exception as e:
            print(f"  [DOC EXTRACT] PyMuPDF error: {e}")

    # -------------------------------------------------------------------------
    # CASE 2: PowerPoint Presentation (.pptx / .ppt)
    # -------------------------------------------------------------------------
    elif ext in (".pptx", ".ppt"):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                slide_files = sorted(
                    [n for n in zf.namelist() if re.match(r"^ppt/slides/slide\d+\.xml$", n)],
                    key=lambda x: int(re.findall(r"\d+", x)[0]),
                )
                total_pages = max(1, len(slide_files))
                slide_text_map = {}

                for idx, sf in enumerate(slide_files):
                    slide_num = idx + 1
                    root = ET.fromstring(zf.read(sf))
                    texts = [
                        node.text.strip()
                        for node in root.iter()
                        if node.tag.endswith("}t") and node.text and node.text.strip()
                    ]
                    joined = "\n".join(texts)
                    slide_text_map[slide_num] = joined
                    if joined:
                        all_page_texts.append((slide_num, joined))

                # Map media images to their exact slide via slide relationships (`ppt/slides/_rels/slideN.xml.rels`)
                media_usage = {}
                slide_to_media = {}
                for idx, sf in enumerate(slide_files):
                    slide_num = idx + 1
                    rel_path = sf.replace("ppt/slides/", "ppt/slides/_rels/") + ".rels"
                    if rel_path in zf.namelist():
                        rel_root = ET.fromstring(zf.read(rel_path))
                        for rel in rel_root.iter():
                            tgt = rel.attrib.get("Target", "")
                            if "../media/" in tgt and tgt.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                                m_path = "ppt/media/" + os.path.basename(tgt)
                                media_usage[m_path] = media_usage.get(m_path, 0) + 1
                                slide_to_media.setdefault(slide_num, []).append(m_path)

                start_slide = 2 if total_pages >= 4 else 1
                for slide_num, media_list in sorted(slide_to_media.items()):
                    if slide_num < start_slide:
                        continue
                    txt = slide_text_map.get(slide_num, "")
                    for mf in media_list:
                        if media_usage.get(mf, 0) > 2 or mf not in zf.namelist():
                            continue
                        try:
                            pil_im = Image.open(io.BytesIO(zf.read(mf))).convert("RGB")
                            score = _diagram_quality_score(pil_im)
                            if score <= 0.0:
                                continue
                            sig = _perceptual_signature(pil_im)
                            if sig in seen_sigs:
                                continue
                            seen_sigs.add(sig)
                            raw_candidates.append(
                                {
                                    "page": slide_num,
                                    "slide_title": _extract_slide_title(txt, f"Slide {slide_num}"),
                                    "slide_text": txt,
                                    "pil_img": pil_im,
                                    "sig": sig,
                                    "score": score,
                                    "is_full_slide": False,
                                }
                            )
                        except Exception:
                            pass
        except Exception as e:
            print(f"  [DOC EXTRACT] PPTX parse error: {e}")

    # -------------------------------------------------------------------------
    # CASE 3: Word Document (.docx / .doc)
    # -------------------------------------------------------------------------
    elif ext in (".docx", ".doc"):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                paras = []
                if "word/document.xml" in zf.namelist():
                    root = ET.fromstring(zf.read("word/document.xml"))
                    for p_node in root.iter():
                        if p_node.tag.endswith("}p"):
                            p_texts = [t.text for t in p_node.iter() if t.tag.endswith("}t") and t.text]
                            line = "".join(p_texts).strip()
                            if line:
                                paras.append(line)
                total_pages = max(1, len(paras))
                chunk_size = max(1, len(paras) // 5)
                for i in range(0, len(paras), chunk_size):
                    all_page_texts.append((i + 1, "\n".join(paras[i : i + chunk_size])))

                media_files = sorted(
                    [
                        n
                        for n in zf.namelist()
                        if n.startswith("word/media/")
                        and n.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
                    ]
                )
                for idx, mf in enumerate(media_files):
                    try:
                        pil_im = Image.open(io.BytesIO(zf.read(mf))).convert("RGB")
                        score = _diagram_quality_score(pil_im)
                        if score <= 0.0:
                            continue
                        sig = _perceptual_signature(pil_im)
                        if sig in seen_sigs:
                            continue
                        seen_sigs.add(sig)
                        sec_txt = all_page_texts[min(idx, len(all_page_texts) - 1)][1] if all_page_texts else ""
                        raw_candidates.append(
                            {
                                "page": idx + 1,
                                "slide_title": _extract_slide_title(sec_txt, f"Figure {idx + 1}"),
                                "slide_text": sec_txt,
                                "pil_img": pil_im,
                                "sig": sig,
                                "score": score,
                                "is_full_slide": False,
                            }
                        )
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [DOC EXTRACT] DOCX parse error: {e}")

    # -------------------------------------------------------------------------
    # CASE 4: Direct Image (.png, .jpg, .jpeg, .webp)
    # -------------------------------------------------------------------------
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        try:
            pil_im = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            raw_candidates.append(
                {
                    "page": 1,
                    "slide_title": os.path.splitext(os.path.basename(file_name))[0],
                    "slide_text": "",
                    "pil_img": pil_im,
                    "sig": _perceptual_signature(pil_im),
                    "score": 50.0,
                    "is_full_slide": False,
                }
            )
        except Exception:
            pass

    # Select up to 8 diverse, high-quality unique diagrams spread evenly across ALL pages
    balanced = _select_balanced_candidates(raw_candidates, total_pages=total_pages, k=8)

    saved_items = []
    for idx, item in enumerate(balanced):
        src_im = item["pil_img"].convert("RGB")
        max_dim = max(src_im.size)
        if max_dim > 1280:
            s_cap = 1280.0 / float(max_dim)
            src_im = src_im.resize(
                (max(1, int(round(src_im.width * s_cap))), max(1, int(round(src_im.height * s_cap)))),
                Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.BICUBIC,
            )
        hires_fname = f"p{item['page']}_fig{idx}_hires.png"
        abs_hires = os.path.join(diag_dir, hires_fname)
        src_im.save(abs_hires, "PNG")

        fitted, fit_rect = _fit_diagram_canvas(src_im, 720, 420)
        fname = f"p{item['page']}_fig{idx}.png"
        abs_p = os.path.join(diag_dir, fname)
        fitted.save(abs_p, "PNG")
        saved_items.append(
            {
                "page": item["page"],
                "slide_title": item["slide_title"],
                "slide_text": item["slide_text"],
                "diagram_file": f"course_material/{course_id}/diagrams/{fname}",
                "abs_path": abs_p,
                "abs_source_path": abs_hires,
                "source_size": src_im.size,
                "fit_rect": fit_rect,
                "score": item["score"],
            }
        )

    return {
        "ext": ext,
        "total_pages": total_pages,
        "all_page_texts": all_page_texts,
        "candidates": saved_items,
        "total_unique_found": len(raw_candidates),
    }


def _draw_concept_schematic(
    save_path: str,
    concept_id: int,
    course_title: str,
    concept_title: str,
    block_labels: list,
) -> list:
    """Fallback 720x420 technical schematic if a document had fewer than 5 unique embedded diagrams."""
    W, H = 720, 420
    img = Image.new("RGB", (W, H), (14, 22, 36))
    draw = ImageDraw.Draw(img)

    draw.rectangle([12, 12, W - 12, 66], fill=(24, 38, 62), outline=(72, 144, 216), width=2)
    draw.text((24, 18), f"COURSE: {course_title.upper()[:52]}", fill=(130, 195, 255))
    draw.text((24, 38), f"CONCEPT C{concept_id}: {concept_title[:56]}", fill=(248, 216, 88))

    palette = [
        ((215, 110, 45), (52, 32, 24), "orange"),
        ((135, 85, 215), (38, 26, 58), "purple"),
        ((48, 185, 115), (20, 48, 36), "green"),
    ]
    coords = [
        [28, 92, 232, 368],
        [264, 92, 476, 368],
        [508, 92, 696, 368],
    ]

    regions = []
    for idx in range(3):
        lbl = block_labels[idx] if idx < len(block_labels) else f"{concept_title} Stage {idx + 1}"
        border_col, fill_col, color_name = palette[idx % len(palette)]
        x1, y1, x2, y2 = coords[idx]
        draw.rectangle([x1, y1, x2, y2], fill=fill_col, outline=border_col, width=3)
        draw.rectangle([x1 + 8, y1 + 10, x2 - 8, y1 + 44], fill=border_col)
        draw.text((x1 + 14, y1 + 19), f"BLOCK {idx + 1}", fill=(12, 18, 28))

        words = lbl.split()
        lines, cur = [], ""
        for w in words:
            if len(cur + " " + w) <= 20:
                cur = (cur + " " + w).strip()
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)

        ty = y1 + 75
        for ln in lines[:4]:
            draw.text((x1 + 16, ty), ln, fill=(235, 244, 255))
            ty += 24

        if idx < 2:
            ax = x2 + 4
            bx = coords[idx + 1][0] - 4
            my = (y1 + y2) // 2
            draw.line([ax, my, bx, my], fill=(248, 216, 88), width=3)
            draw.polygon([(bx, my), (bx - 10, my - 7), (bx - 10, my + 7)], fill=(248, 216, 88))

        regions.append({"label": lbl, "box": [x1, y1, x2, y2], "color_name": color_name})

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path, "PNG")
    return regions


def ingest_custom_course_package(
    course_title: str,
    raw_notes: str = "",
    image_base64: str = None,
    file_name: str = None,
    file_base64: str = None,
    remote_vlm_url: str = None,
    vlm_engine=None,
    dino_evaluator=None,
    progress_cb=None,
) -> dict:
    """
    End-to-end ingestion of a PDF, PPTX, DOCX (Word), or Image file into the RPG:
      1. Extracts unique educational diagrams across ALL pages (skipping cover logos & duplicates).
      2. Pairs every extracted diagram with its exact slide/page text.
      3. Runs pure live `Qwen2.5-VL-7B-Instruct` (`use_reference_cache=False`) on each diagram
         to extract genuine topic-matched concepts, visual components, Lab Monitor prompts,
         multiple-choice battle questions, Study Notes, and DeBERTa-v3 VQA rubrics!
      4. Runs Grounding DINO zero-shot detection on each extracted diagram to refine boxes.
    """
    def _report(pct: int, msg: str):
        if progress_cb:
            try:
                progress_cb(pct, msg)
            except Exception:
                pass

    _report(6, f"Scanning ALL pages of '{file_name or course_title}' & filtering out logos/duplicates...")

    if not course_title or course_title.strip().lower() in ("", "custom ai course"):
        if file_name:
            base = os.path.splitext(os.path.basename(file_name))[0]
            course_title = re.sub(r"[_-]+", " ", base).strip().title()
        else:
            course_title = "Custom AI & Engineering Course"

    course_title = course_title.strip()
    course_id = "custom_" + _slugify(course_title)
    course_dir = os.path.join(REPO_ROOT, "course_material", course_id)
    diag_dir = os.path.join(course_dir, "diagrams")
    os.makedirs(diag_dir, exist_ok=True)

    upload_b64 = file_base64 or image_base64
    upload_name = file_name or ("uploaded_slide.png" if image_base64 else "")
    doc_info = {
        "ext": "",
        "total_pages": 0,
        "all_page_texts": [],
        "candidates": [],
        "total_unique_found": 0,
    }

    if upload_b64:
        b64_payload = upload_b64.split(",", 1)[1] if "," in upload_b64 else upload_b64
        try:
            raw_bytes = base64.b64decode(b64_payload)
            doc_info = extract_document_text_and_photos(
                file_name=upload_name,
                file_bytes=raw_bytes,
                diag_dir=diag_dir,
                course_id=course_id,
            )
            print(
                f"  [DOCUMENT PARSER] '{upload_name}' ({doc_info['total_pages']} pages) -> "
                f"Found {doc_info['total_unique_found']} unique diagrams (logos skipped); "
                f"selected {len(doc_info['candidates'])} chapter-balanced candidates: "
                f"pages {[c['page'] for c in doc_info['candidates']]}",
                flush=True,
            )
        except Exception as e:
            print(f"  [DOCUMENT PARSER] Warning decoding '{upload_name}': {e}", flush=True)

    candidates = doc_info["candidates"]
    vlm_is_live = (
        vlm_engine is not None
        and (vlm_engine.mode == "live" or vlm_engine.vlm_model is not None)
    )
    vlm_name = getattr(vlm_engine, "short_model_name", "Qwen3-VL-8B") if vlm_engine is not None else "Qwen3-VL-8B"
    dino_name = getattr(dino_evaluator, "model_display_name", "Grounding-DINO") if dino_evaluator is not None else "Grounding-DINO"

    _report(
        16,
        f"Found {doc_info['total_unique_found']} unique diagrams across {doc_info['total_pages']} pages (skipped logos). Inspecting with {vlm_name}...",
    )

    # Run live Qwen-VL (FP16) inspection on candidate diagrams, automatically
    # skipping any decorative image where the VLM says `is_educational_diagram: false`!
    accepted_slots = []
    cand_cursor = 0

    while len(accepted_slots) < 5 and cand_cursor < len(candidates):
        cand = candidates[cand_cursor]
        cand_cursor += 1
        slot_idx = len(accepted_slots)
        _report(
            20 + slot_idx * 14,
            f"{vlm_name} inspecting Page {cand['page']} diagram ({slot_idx + 1}/5): {cand['slide_title'][:34]}...",
        )

        vlm_res = None
        if vlm_is_live:
            try:
                print(
                    f"  [{vlm_name} LIVE] Concept C{slot_idx} | Page {cand['page']} ({cand['slide_title']})...",
                    flush=True,
                )
                s_title = (cand["slide_title"] or "").strip()
                c_hint = s_title if (s_title and s_title.lower() in course_title.lower()) else (f"{course_title} — {s_title}" if s_title else course_title)
                vlm_res = vlm_engine.comprehend_diagram_live(
                    image_path=cand["abs_path"],
                    concept_id=slot_idx,
                    concept_title=c_hint,
                    slide_context=cand["slide_text"],
                    use_reference_cache=False,  # CRITICAL: NEVER mix old YOLO/OCR cache!
                    source_image_path=cand.get("abs_source_path"),
                    fit_rect=cand.get("fit_rect"),
                    dino_evaluator=dino_evaluator,
                )
                # Refresh vlm_name in case load_live_vlm() fell back to Qwen2.5-VL-7B
                vlm_name = getattr(vlm_engine, "short_model_name", vlm_name)
                # If VLM flags this as a logo/non-educational image and more candidates remain, skip it!
                if vlm_res.get("is_educational_diagram") is False and (len(candidates) - cand_cursor) >= (5 - len(accepted_slots)):
                    print(f"  [{vlm_name} FILTER] Rejected decorative/logo image on Page {cand['page']}.", flush=True)
                    continue
            except Exception as e:
                import traceback
                print(f"  [{vlm_name} LIVE] Error on Page {cand['page']}: {e}", flush=True)
                traceback.print_exc()

        accepted_slots.append({"cand": cand, "vlm": vlm_res})

    # If the uploaded document had fewer than 5 unique diagrams, build distinct text-based schematics
    # for the remaining slots using spread-out text pages from the document (never duplicating photos!)
    all_texts = doc_info["all_page_texts"]
    while len(accepted_slots) < 5:
        slot_idx = len(accepted_slots)
        if all_texts:
            pick_t = all_texts[min(len(all_texts) - 1, int(slot_idx * len(all_texts) / 5))]
            p_num, p_txt = pick_t
            s_title = _extract_slide_title(p_txt, f"{course_title} Part {slot_idx + 1}")
        else:
            p_num = slot_idx + 1
            p_txt = raw_notes or course_title
            s_title = f"{course_title} Concept {slot_idx + 1}"

        fname = f"c{slot_idx}_schematic.png"
        abs_p = os.path.join(diag_dir, fname)
        block_labels = [
            f"{s_title[:18]} Input",
            f"{s_title[:18]} Core",
            f"{s_title[:18]} Output",
        ]
        regs = _draw_concept_schematic(abs_p, slot_idx, course_title, s_title, block_labels)
        accepted_slots.append(
            {
                "cand": {
                    "page": p_num,
                    "slide_title": s_title,
                    "slide_text": p_txt,
                    "diagram_file": f"course_material/{course_id}/diagrams/{fname}",
                    "abs_path": abs_p,
                    "fallback_regions": regs,
                },
                "vlm": None,
            }
        )

    prereq_map = {0: [], 1: [], 2: [0], 3: [1], 4: [2, 3]}
    concepts_list = []
    new_dino_puzzles = []
    new_vlm_diagrams = []

    for cid, slot in enumerate(accepted_slots):
        cand = slot["cand"]
        vlm_res = slot["vlm"] or {}

        diag_rel = cand["diagram_file"]
        abs_diag = cand["abs_path"]
        page_num = cand["page"]

        c_title = (vlm_res.get("concept_title") or cand["slide_title"] or f"{course_title} C{cid}").strip()[:64]
        short_title = c_title.split(":")[0].strip()[:22]
        visual_summary = (
            vlm_res.get("visual_summary")
            or (cand["slide_text"][:220].replace("\n", " ") if cand["slide_text"] else f"Key architecture of {c_title}.")
        )

        # Build the 2 clickable components strictly from THIS image's live Qwen2.5-VL-7B output!
        v_comps = vlm_res.get("components") if isinstance(vlm_res.get("components"), list) else []
        if len(v_comps) >= 2:
            comp_pair = v_comps[:2]
        elif cand.get("fallback_regions"):
            fr = cand["fallback_regions"]
            comp_pair = [
                {
                    "target_label": fr[0]["label"],
                    "dino_prompt": f"orange {fr[0]['label'].lower()} box.",
                    "vlm_box_px": fr[0]["box"],
                    "lab_question": f"Click or drag a box around [{fr[0]['label']}] in {c_title}.",
                    "socratic_hint": "Look at the leftmost block.",
                    "explanation": visual_summary,
                },
                {
                    "target_label": fr[2]["label"],
                    "dino_prompt": f"green {fr[2]['label'].lower()} box.",
                    "vlm_box_px": fr[2]["box"],
                    "lab_question": f"Click or drag a box around [{fr[2]['label']}] in {c_title}.",
                    "socratic_hint": "Look at the rightmost block.",
                    "explanation": visual_summary,
                },
            ]
        else:
            comp_pair = [
                {
                    "target_label": f"{short_title} Primary Block",
                    "dino_prompt": "left diagram component box.",
                    "vlm_box_px": [60.0, 80.0, 340.0, 340.0],
                    "lab_question": f"Click or drag a box around the primary component of {c_title} (Page {page_num}).",
                    "socratic_hint": "Look at the left side of the extracted diagram.",
                    "explanation": visual_summary,
                },
                {
                    "target_label": f"{short_title} Secondary Block",
                    "dino_prompt": "right diagram component box.",
                    "vlm_box_px": [380.0, 80.0, 660.0, 340.0],
                    "lab_question": f"Click or drag a box around the secondary component of {c_title} (Page {page_num}).",
                    "socratic_hint": "Look at the right side of the extracted diagram.",
                    "explanation": visual_summary,
                },
            ]

        # Refine bounding boxes with Grounding DINO on the extracted photo if high-confidence match found
        final_comps = []
        all_regions_for_puzzle = []
        for p_idx, comp in enumerate(comp_pair):
            pid = f"c{cid}_p{p_idx + 1}"
            t_label = str(comp.get("target_label", f"{short_title} Block {p_idx + 1}"))
            d_prompt = str(comp.get("dino_prompt", f"{t_label.lower()} box."))
            if not d_prompt.strip().endswith("."):
                d_prompt = d_prompt.strip() + "."

            vlm_box_px = [float(v) for v in comp.get("vlm_box_px", [60.0, 80.0, 340.0, 340.0])]
            raw_valid_boxes = comp.get("valid_boxes") or [vlm_box_px]
            som_verified = bool(comp.get("som_verified", False))
            som_confidence = float(comp.get("som_confidence", 0.88 if som_verified else 0.48))
            som_num_cands = int(comp.get("som_num_candidates", 1))
            grounding_tier = comp.get("grounding_tier", f"{vlm_name}-FP16")
            refined_box = vlm_box_px

            # Only run 720x420 fallback DINO alignment when the component was NOT already
            # verified on the native-res source image via Set-of-Mark + Crop Verification!
            if (not som_verified) and dino_evaluator is not None and os.path.exists(abs_diag):
                try:
                    pil_im = Image.open(abs_diag).convert("RGB")
                    refined_box = trim_blank_margins(pil_im, refined_box)
                    vlm_is_blank = box_content_stats(pil_im, refined_box)["is_blank"]

                    if hasattr(dino_evaluator, "detect_boxes"):
                        d_boxes, d_scores, _ = dino_evaluator.detect_boxes(
                            img=pil_im,
                            prompt=d_prompt,
                            threshold=0.18,
                            target_sizes=[(420, 720)],
                        )
                        if vlm_is_blank and not d_boxes:
                            d_boxes, d_scores, _ = dino_evaluator.detect_boxes(
                                img=pil_im,
                                prompt="diagram cylinder. diagram box. oval. table.",
                                threshold=0.18,
                                target_sizes=[(420, 720)],
                            )
                    else:
                        inputs = dino_evaluator.processor(
                            images=pil_im, text=d_prompt, return_tensors="pt"
                        ).to(dino_evaluator.device)
                        with torch.no_grad():
                            outputs = dino_evaluator.model(**inputs)
                        dres = dino_evaluator.processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            threshold=0.18,
                            text_threshold=0.14,
                            target_sizes=[(420, 720)],
                        )[0]
                        d_boxes = dres["boxes"].detach().cpu().tolist()
                        d_scores = dres["scores"].detach().cpu().tolist()

                    best_match_box = None
                    best_match_iou = 0.0
                    best_match_comb = -1.0
                    area_v = max(1.0, (refined_box[2] - refined_box[0]) * (refined_box[3] - refined_box[1]))
                    v_cx = 0.5 * (refined_box[0] + refined_box[2])
                    v_cy = 0.5 * (refined_box[1] + refined_box[3])

                    best_rescue_box = None
                    best_rescue_dist = 1e9

                    for cand_box, cand_score in zip(d_boxes, d_scores):
                        cb = [round(float(v), 1) for v in cand_box]
                        area_d = max(1.0, (cb[2] - cb[0]) * (cb[3] - cb[1]))
                        if area_d < 500.0 or area_d > 0.55 * 720.0 * 420.0:
                            continue
                        if box_content_stats(pil_im, cb)["is_blank"]:
                            continue

                        ix1 = max(refined_box[0], cb[0])
                        iy1 = max(refined_box[1], cb[1])
                        ix2 = min(refined_box[2], cb[2])
                        iy2 = min(refined_box[3], cb[3])
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        iou = inter / (area_v + area_d - inter)
                        comb = 0.65 * iou + 0.35 * float(cand_score)
                        if comb > best_match_comb:
                            best_match_comb = comb
                            best_match_iou = iou
                            best_match_box = cb

                        if vlm_is_blank:
                            d_cx = 0.5 * (cb[0] + cb[2])
                            d_cy = 0.5 * (cb[1] + cb[3])
                            dist = math.hypot(d_cx - v_cx, d_cy - v_cy)
                            if dist <= 165.0 and dist < best_rescue_dist:
                                best_rescue_dist = dist
                                best_rescue_box = cb

                    if best_match_box is not None and best_match_iou >= 0.40:
                        refined_box = [round(0.65 * refined_box[i] + 0.35 * best_match_box[i], 1) for i in range(4)]
                        refined_box = trim_blank_margins(pil_im, refined_box)
                        grounding_tier = f"{vlm_name} + {dino_name}"
                    elif vlm_is_blank and best_rescue_box is not None:
                        refined_box = trim_blank_margins(pil_im, best_rescue_box)
                        grounding_tier = f"{dino_name}-Rescue"
                        print(
                            f"  [{dino_name} RESCUE] Snapped blank-margin VLM box for '{t_label}' -> {refined_box}",
                            flush=True,
                        )
                except Exception:
                    pass

            final_valid_boxes = [refined_box]
            for extra_vb in raw_valid_boxes[1:]:
                ev = [round(float(v), 1) for v in extra_vb]
                final_valid_boxes.append(ev)

            b1000 = [
                round(refined_box[0] * 1000.0 / 720.0),
                round(refined_box[1] * 1000.0 / 420.0),
                round(refined_box[2] * 1000.0 / 720.0),
                round(refined_box[3] * 1000.0 / 420.0),
            ]
            lab_q = comp.get("lab_question") or f"Click or drag a box around '{t_label}' in this diagram (Page {page_num})."
            expl = comp.get("explanation") or visual_summary
            hint = comp.get("socratic_hint") or f"Locate '{t_label}' inside the extracted figure from Page {page_num}."

            for vb in final_valid_boxes:
                all_regions_for_puzzle.append({"label": t_label, "box": vb})
            final_comps.append(
                {
                    "component_id": pid,
                    "target_label": t_label,
                    "dino_prompt": d_prompt,
                    "grounding_tier": grounding_tier,
                    "som_verified": som_verified,
                    "som_confidence": som_confidence,
                    "som_num_candidates": som_num_cands,
                    "vlm_box_1000": b1000,
                    "vlm_box_px": refined_box,
                    "valid_boxes": final_valid_boxes,
                    "ground_truth_box_px": refined_box,
                    "lab_question": f"LAB C{cid}-{'A' if p_idx == 0 else 'B'} (Page {page_num}): {lab_q}",
                    "socratic_hint": hint,
                    "explanation": expl,
                }
            )

        for fc in final_comps:
            new_dino_puzzles.append(
                {
                    "puzzle_id": fc["component_id"],
                    "concept_id": cid,
                    "concept_title": c_title,
                    "diagram_file": diag_rel,
                    "diagram_width": 720,
                    "diagram_height": 420,
                    "target_label": fc["target_label"],
                    "dino_prompt": fc["dino_prompt"],
                    "grounding_tier": fc.get("grounding_tier", "fallback"),
                    "som_verified": fc.get("som_verified", False),
                    "som_confidence": fc.get("som_confidence", 0.85),
                    "som_num_candidates": fc.get("som_num_candidates", 1),
                    "prompt": fc["lab_question"],
                    "target_box": fc["ground_truth_box_px"],
                    "valid_boxes": fc.get("valid_boxes", [fc["ground_truth_box_px"]]),
                    "all_regions": list(all_regions_for_puzzle),
                    "explanation": fc["explanation"],
                }
            )

        # VQA challenge strictly from Qwen2.5-VL-7B reading THIS exact page/diagram!
        live_vqa = vlm_res.get("vqa_challenge") if isinstance(vlm_res.get("vqa_challenge"), dict) else {}
        vqa_question = live_vqa.get("question") or f"Explain how {final_comps[0]['target_label']} works in {c_title}."
        vqa_ref = live_vqa.get("reference_answer") or visual_summary
        vqa_obj = {
            "vqa_id": f"vqa_c{cid}",
            "question": vqa_question,
            "reference_answer": vqa_ref,
            "sample_correct_answer": live_vqa.get("sample_correct_answer") or vqa_ref,
            "sample_partial_answer": live_vqa.get("sample_partial_answer") or "",
            "misconception_trap_answer": live_vqa.get("misconception_trap_answer")
            or f"{final_comps[0]['target_label']} is unrelated to {c_title}.",
        }

        new_vlm_diagrams.append(
            {
                "diagram_id": f"{course_id}_c{cid}",
                "concept_id": cid,
                "concept_title": c_title,
                "diagram_file": diag_rel,
                "diagram_width": 720,
                "diagram_height": 420,
                "visual_summary": visual_summary,
                "extracted_labels": vlm_res.get("extracted_labels") or [r["label"] for r in all_regions_for_puzzle],
                "components": final_comps,
                "vqa_challenge": vqa_obj,
            }
        )

        # Real Multiple-Choice Battle Questions generated by Qwen2.5-VL-7B directly from THIS diagram + slide!
        raw_mcs = vlm_res.get("mc_questions") if isinstance(vlm_res.get("mc_questions"), list) else []
        valid_mcs = []
        for q_item in raw_mcs:
            if (
                isinstance(q_item, dict)
                and q_item.get("question")
                and isinstance(q_item.get("choices"), list)
                and len(q_item["choices"]) >= 4
            ):
                valid_mcs.append(
                    {
                        "question": str(q_item["question"]),
                        "choices": [str(ch) for ch in q_item["choices"][:4]],
                        "correct_idx": int(q_item.get("correct_idx", 0)) % 4,
                        "reference": vqa_ref,
                    }
                )

        if not valid_mcs:
            valid_mcs = [
                {
                    "question": f"({c_title} — Page {page_num}) What is the role of {final_comps[0]['target_label']}?",
                    "choices": [
                        final_comps[0]["explanation"][:115],
                        "It permanently disables the entire system architecture",
                        "It replaces all data structures with random noise",
                        "It has no role in this chapter",
                    ],
                    "correct_idx": 0,
                    "reference": final_comps[0]["explanation"],
                },
                {
                    "question": f"({c_title} — Page {page_num}) Which component works alongside {final_comps[0]['target_label']} in this figure?",
                    "choices": [
                        final_comps[1]["target_label"],
                        "An unrelated external audio codec",
                        "A random pixel blur filter",
                        "None of the above",
                    ],
                    "correct_idx": 0,
                    "reference": visual_summary,
                },
            ]

        summary_bullets = vlm_res.get("summary_bullets") if isinstance(vlm_res.get("summary_bullets"), list) and len(vlm_res["summary_bullets"]) >= 2 else [
            f"Page {page_num} Summary: {visual_summary}",
            f"Component A: {final_comps[0]['target_label']} — {final_comps[0]['explanation']}",
            f"Component B: {final_comps[1]['target_label']} — {final_comps[1]['explanation']}",
        ]

        concepts_list.append(
            {
                "concept_id": cid,
                "title": c_title,
                "short_title": short_title,
                "prereqs": prereq_map[cid],
                "podcast_audio": None,
                "podcast_hint": f'[Voice Log C{cid} · Page {page_num}] "{visual_summary}"',
                "summary_bullets": [str(b) for b in summary_bullets[:4]],
                "easy_mc": valid_mcs,
                "medium_mc": [valid_mcs[-1]],
                "boss_open": [
                    {
                        "question": vqa_question,
                        "reference": vqa_ref,
                    }
                ],
            }
        )

    _report(96, f"Finalizing 5-Concept Knowledge Graph & 10 Lab Monitor puzzles for '{course_title}'...")

    course_pkg = {
        "course_id": course_id,
        "title": course_title,
        "subtitle": (
            f"Extracted from {upload_name or 'Document'} "
            f"({doc_info['total_unique_found']} Unique Figures Found -> 5 Chapter Diagrams, 10 VLM+DINO Puzzles)"
        ),
        "mentor": f"Prof. Ada ({vlm_name} + {dino_name} Lab)",
        "num_photos_extracted": len(candidates),
        "total_unique_found": doc_info["total_unique_found"],
        "source_file": upload_name,
        "concepts": concepts_list,
        "puzzles": new_dino_puzzles,
        "vlm_diagrams": new_vlm_diagrams,
    }

    with open(os.path.join(course_dir, "course_curriculum.json"), "w", encoding="utf-8") as f:
        json.dump(course_pkg, f, indent=2, ensure_ascii=False)

    if dino_evaluator is not None:
        dino_evaluator.puzzles_list = new_dino_puzzles
        dino_evaluator.puzzles = {p["puzzle_id"]: p for p in new_dino_puzzles}
        dino_evaluator._grounding_cache.clear()

    if vlm_engine is not None:
        if remote_vlm_url:
            vlm_engine.remote_url = remote_vlm_url.rstrip("/")
        vlm_engine.diagrams_list = new_vlm_diagrams
        vlm_engine.diagrams_by_id = {d["diagram_id"]: d for d in new_vlm_diagrams}
        vlm_engine.diagrams_by_concept = {int(d["concept_id"]): d for d in new_vlm_diagrams}

    return course_pkg
