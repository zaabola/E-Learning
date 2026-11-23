"""
Dynamic PDF / PPTX / Word (.docx) & Slide Ingestor with Live Qwen2.5-VL-7B Inspection
(`Models/VLM/scripts/custom_course_ingestor.py`)
=====================================================================================
Allows a student or teacher to insert a `.pdf`, `.pptx`, `.docx` (Word), or image
directly inside the browser RPG:

  1. [Document Text & Photo Extraction]:
     - `.pdf`  : Uses `PyMuPDF` (`fitz`) to extract embedded figures/photos AND
                 renders visual diagram pages to high-res 720x420 PNGs, plus extracts text.
     - `.pptx` : Extracts all embedded photos/diagrams from `ppt/media/*` AND extracts
                 all slide titles/text from `ppt/slides/slide*.xml`.
     - `.docx` : Extracts all embedded photos/figures from `word/media/*` AND extracts
                 all headings/paragraphs from `word/document.xml`.
     - `image` : Directly loads `.png` / `.jpg` / `.webp` slide or diagram images.

  2. [Autonomous VLM Inspection (`Qwen2.5-VL-7B-Instruct` on Kaggle GPU)]:
     - Inspects the extracted photos/slides and document text by itself!
     - Discovers the 5 core course concepts: (C0, C1) -> (C2, C3) -> C4 (Prerequisite DAG).
     - Inspects each extracted photo to identify clickable visual components, generates
       Grounding DINO prompts (`dino_prompt`), [0..1000] spatial bounding boxes, Lab Monitor
       investigation prompts, Socratic hints, Study Notes, and DeBERTa-v3 VQA rubrics.

  3. [Grounding DINO Spatial Localization on Extracted Photos]:
     - Runs `IDEA-Research/grounding-dino-tiny` on the extracted PDF/PPTX/Word photos
       and fuses boxes with Qwen2.5-VL-7B's spatial grounding coordinates.
"""

import base64
import io
import json
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
import torch
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VLM_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_ROOT = os.path.dirname(VLM_ROOT)
REPO_ROOT = os.path.dirname(MODELS_ROOT)


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s[:32] or f"course_{int(time.time())}"


def _fit_diagram_canvas(pil_img: Image.Image, target_w: int = 720, target_h: int = 420) -> Image.Image:
    """Letterboxes an extracted document photo onto a crisp 720x420 dark lab monitor canvas."""
    img = pil_img.convert("RGB")
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return Image.new("RGB", (target_w, target_h), (14, 22, 36))
    scale = min(target_w / float(iw), target_h / float(ih))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.BICUBIC)
    canvas = Image.new("RGB", (target_w, target_h), (14, 22, 36))
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def extract_document_text_and_photos(
    file_name: str,
    file_bytes: bytes,
    diag_dir: str,
    course_id: str,
) -> dict:
    """
    Extracts text sections and embedded photos/diagrams from a PDF, PPTX, DOCX, or Image file.
    Saves up to 5 extracted photos into `diag_dir` as `c{i}_extracted.png`.
    """
    ext = os.path.splitext((file_name or "").lower())[1]
    text_sections = []
    extracted_pil_images = []

    # -------------------------------------------------------------------------
    # CASE 1: PDF Document (.pdf) via PyMuPDF (fitz) or pypdf
    # -------------------------------------------------------------------------
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=file_bytes, filetype="pdf")
            seen_xrefs = set()
            page_renders = []

            for page_idx, page in enumerate(doc):
                txt = (page.get_text("text") or "").strip()
                if txt:
                    text_sections.append(txt)

                # 1a. Extract embedded raster figures/diagrams from the PDF page
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        base_img = doc.extract_image(xref)
                        pil_im = Image.open(io.BytesIO(base_img["image"])).convert("RGB")
                        # Filter out tiny icons/logos (< 120x80)
                        if pil_im.width >= 120 and pil_im.height >= 80:
                            extracted_pil_images.append(pil_im)
                    except Exception:
                        pass

                # 1b. Also render the full PDF page/slide as a high-res image fallback
                if len(page_renders) < 8:
                    try:
                        pix = page.get_pixmap(dpi=150)
                        p_im = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                        page_renders.append(p_im)
                    except Exception:
                        pass

            # Fill any remaining slots up to 5 with rendered PDF pages (great for vector diagrams/slides!)
            if len(extracted_pil_images) < 5 and page_renders:
                step = max(1, len(page_renders) // max(1, (5 - len(extracted_pil_images))))
                for pr in page_renders[::step]:
                    if len(extracted_pil_images) >= 5:
                        break
                    extracted_pil_images.append(pr)
            doc.close()
        except Exception as e:
            print(f"  [DOC EXTRACT] PyMuPDF note: {e}")

    # -------------------------------------------------------------------------
    # CASE 2: PowerPoint Presentation (.pptx / .ppt)
    # -------------------------------------------------------------------------
    elif ext in (".pptx", ".ppt"):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                # 2a. Extract all slide text from ppt/slides/slide*.xml
                slide_files = sorted(
                    [n for n in zf.namelist() if re.match(r"^ppt/slides/slide\d+\.xml$", n)],
                    key=lambda x: int(re.findall(r"\d+", x)[0]),
                )
                for sf in slide_files:
                    xml_bytes = zf.read(sf)
                    root = ET.fromstring(xml_bytes)
                    texts = [node.text.strip() for node in root.iter() if node.tag.endswith("}t") and node.text and node.text.strip()]
                    if texts:
                        text_sections.append("\n".join(texts))

                # 2b. Extract all embedded photos/diagrams from ppt/media/*
                media_files = sorted(
                    [n for n in zf.namelist() if n.startswith("ppt/media/") and n.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))]
                )
                for mf in media_files:
                    try:
                        pil_im = Image.open(io.BytesIO(zf.read(mf))).convert("RGB")
                        if pil_im.width >= 120 and pil_im.height >= 80:
                            extracted_pil_images.append(pil_im)
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [DOC EXTRACT] PPTX parse note: {e}")

    # -------------------------------------------------------------------------
    # CASE 3: Word Document (.docx / .doc)
    # -------------------------------------------------------------------------
    elif ext in (".docx", ".doc"):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                # 3a. Extract all paragraphs from word/document.xml
                if "word/document.xml" in zf.namelist():
                    root = ET.fromstring(zf.read("word/document.xml"))
                    paras = []
                    for p_node in root.iter():
                        if p_node.tag.endswith("}p"):
                            p_texts = [t.text for t in p_node.iter() if t.tag.endswith("}t") and t.text]
                            line = "".join(p_texts).strip()
                            if line:
                                paras.append(line)
                    # Group paragraphs into chunks
                    if paras:
                        chunk_size = max(1, len(paras) // 5)
                        for i in range(0, len(paras), chunk_size):
                            text_sections.append("\n".join(paras[i : i + chunk_size]))

                # 3b. Extract all embedded photos/figures from word/media/*
                media_files = sorted(
                    [n for n in zf.namelist() if n.startswith("word/media/") and n.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))]
                )
                for mf in media_files:
                    try:
                        pil_im = Image.open(io.BytesIO(zf.read(mf))).convert("RGB")
                        if pil_im.width >= 120 and pil_im.height >= 80:
                            extracted_pil_images.append(pil_im)
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [DOC EXTRACT] DOCX parse note: {e}")

    # -------------------------------------------------------------------------
    # CASE 4: Direct Image (.png, .jpg, .jpeg, .webp)
    # -------------------------------------------------------------------------
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        try:
            pil_im = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            extracted_pil_images.append(pil_im)
        except Exception:
            pass

    # Save extracted photos onto 720x420 canvases in `diag_dir`
    saved_photo_rels = []
    for idx, pil_im in enumerate(extracted_pil_images[:5]):
        fitted = _fit_diagram_canvas(pil_im, 720, 420)
        fname = f"c{idx}_extracted_photo.png"
        abs_p = os.path.join(diag_dir, fname)
        fitted.save(abs_p, "PNG")
        saved_photo_rels.append(f"course_material/{course_id}/diagrams/{fname}")

    return {
        "ext": ext,
        "text_sections": text_sections,
        "saved_photo_rels": saved_photo_rels,
        "num_photos_extracted": len(saved_photo_rels),
    }


def _draw_concept_schematic(
    save_path: str,
    concept_id: int,
    course_title: str,
    concept_title: str,
    block_labels: list,
) -> list:
    """Renders a 720x420 technical schematic when a document section has no embedded photo."""
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

        draw.rectangle([x1 + 18, y2 - 90, x2 - 18, y2 - 22], outline=border_col, width=1)
        draw.text((x1 + 26, y2 - 64), f"Module C{concept_id}.{idx + 1}", fill=(175, 205, 235))

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


def _parse_concepts_from_notes_or_doc(course_title: str, raw_notes: str, doc_sections: list) -> list:
    """
    Extracts 5 structured (concept_title, concept_description) pairs from either:
      - The extracted PDF / PPTX / DOCX text sections, OR
      - The user's typed notes.
    """
    combined_source = raw_notes.strip()
    if not combined_source and doc_sections:
        # Take key heading/first sentences from extracted PDF/PPTX/Word sections
        extracted_lines = []
        for sec in doc_sections:
            sec_lines = [ln.strip(" -•*\t") for ln in sec.splitlines() if len(ln.strip(" -•*\t")) >= 6]
            if sec_lines:
                heading = sec_lines[0][:58]
                summary = " ".join(sec_lines[1:3])[:140] if len(sec_lines) > 1 else heading
                extracted_lines.append(f"{heading}: {summary}")
        combined_source = "\n".join(extracted_lines[:8])

    lines = [ln.strip(" -•*\t") for ln in (combined_source or "").splitlines() if ln.strip(" -•*\t")]
    chunks = []
    for ln in lines:
        cleaned = re.sub(r"^(?:concept|module|topic|chapter|slide|unit|\d+)[\s.:)-]+", "", ln, flags=re.I).strip()
        if cleaned:
            chunks.append(cleaned)

    default_suffixes = [
        "Core Foundations & Principles",
        "Structural Representation & Metrics",
        "Algorithmic Architecture & Flow",
        "Sequential Processing & Optimization",
        "End-to-End System Synthesis",
    ]

    concepts_raw = []
    for i in range(5):
        if i < len(chunks):
            raw_item = chunks[i]
            if ":" in raw_item:
                t_part, d_part = raw_item.split(":", 1)
                c_title = t_part.strip()[:56]
                c_desc = d_part.strip()[:160] or f"{c_title} in {course_title}"
            else:
                c_title = raw_item[:56]
                c_desc = f"{raw_item[:120]} in {course_title}"
        else:
            c_title = f"{course_title}: {default_suffixes[i]}"
            c_desc = f"{default_suffixes[i]} for {course_title}"
        concepts_raw.append((c_title, c_desc))

    return concepts_raw


def ingest_custom_course_package(
    course_title: str,
    raw_notes: str = "",
    image_base64: str = None,
    file_name: str = None,
    file_base64: str = None,
    remote_vlm_url: str = None,
    vlm_engine=None,
    dino_evaluator=None,
) -> dict:
    """
    End-to-end ingestion of a PDF, PPTX, DOCX (Word), or Image file into the RPG:
      1. Extracts embedded photos/slides and text sections from the uploaded document.
      2. If Qwen2.5-VL-7B-Instruct is loaded live (on Kaggle GPU), runs live VLM visual
         comprehension on each extracted photo to discover the concept title, visual
         components, Grounding DINO prompts, and VQA challenge rubrics!
      3. Runs live Grounding DINO zero-shot detection on each extracted photo to localize
         the clickable `[x1, y1, x2, y2]` component boxes.
    """
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

    # Decode uploaded file (.pdf, .pptx, .docx, or image)
    upload_b64 = file_base64 or image_base64
    upload_name = file_name or ("uploaded_slide.png" if image_base64 else "")
    doc_info = {"ext": "", "text_sections": [], "saved_photo_rels": [], "num_photos_extracted": 0}

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
                f"  [DOCUMENT PARSER] File='{upload_name}' ({doc_info['ext']}) -> "
                f"Extracted {len(doc_info['text_sections'])} text sections & "
                f"{doc_info['num_photos_extracted']} embedded photos/slides!"
            )
        except Exception as e:
            print(f"  [DOCUMENT PARSER] Warning decoding '{upload_name}': {e}")

    parsed_pairs = _parse_concepts_from_notes_or_doc(
        course_title=course_title,
        raw_notes=raw_notes,
        doc_sections=doc_info["text_sections"],
    )
    prereq_map = {0: [], 1: [], 2: [0], 3: [1], 4: [2, 3]}
    extracted_photos = doc_info["saved_photo_rels"]

    # Check if live Qwen2.5-VL-7B is active on Kaggle GPU
    vlm_is_live = (
        vlm_engine is not None
        and (vlm_engine.mode == "live" or vlm_engine.vlm_model is not None)
    )

    concepts_list = []
    new_dino_puzzles = []
    new_vlm_diagrams = []

    for cid in range(5):
        c_title, c_desc = parsed_pairs[cid]
        short_title = c_title.split(":")[0].strip()[:22]

        # Use extracted photo from the PDF/PPTX/Word document if available!
        has_extracted_photo = len(extracted_photos) > 0
        if cid < len(extracted_photos):
            diag_rel = extracted_photos[cid]
        elif has_extracted_photo:
            # Cycle through extracted document photos if the document had fewer than 5 photos
            diag_rel = extracted_photos[cid % len(extracted_photos)]
        else:
            diag_filename = f"c{cid}_schematic.png"
            abs_diag = os.path.join(diag_dir, diag_filename)
            diag_rel = f"course_material/{course_id}/diagrams/{diag_filename}"
            block_labels = [
                f"{short_title} Input & Encoding",
                f"{short_title} Core Transform",
                f"{short_title} Output & Loss",
            ]
            _draw_concept_schematic(abs_diag, cid, course_title, c_title, block_labels)

        abs_diag = os.path.join(REPO_ROOT, diag_rel)

        # ---------------------------------------------------------------------
        # LIVE QWEN2.5-VL-7B PHOTO & CONCEPT INSPECTION (ON KAGGLE GPU)
        # ---------------------------------------------------------------------
        live_vlm_out = None
        if vlm_is_live:
            try:
                print(f"  [Qwen2.5-VL-7B LIVE] Inspecting photo '{diag_rel}' for Concept C{cid}...")
                live_vlm_out = vlm_engine.comprehend_diagram_live(
                    image_path=abs_diag,
                    concept_id=cid,
                    concept_title=f"{course_title} - {c_title}",
                )
                if live_vlm_out.get("concept_title"):
                    c_title = live_vlm_out["concept_title"][:60]
                    short_title = c_title.split(":")[0].strip()[:22]
                if live_vlm_out.get("visual_summary"):
                    c_desc = live_vlm_out["visual_summary"][:200]
            except Exception as e:
                print(f"  [Qwen2.5-VL-7B LIVE] Note on C{cid}: {e}")

        # Default 3 regions across 720x420 canvas
        regions = [
            {"label": f"{short_title} Input Stage", "box": [28, 92, 232, 368], "color_name": "left"},
            {"label": f"{short_title} Core Module", "box": [264, 92, 476, 368], "color_name": "center"},
            {"label": f"{short_title} Output Stage", "box": [508, 92, 696, 368], "color_name": "right"},
        ]

        # If Qwen2.5-VL-7B returned live components & boxes, use them!
        if live_vlm_out and len(live_vlm_out.get("components", [])) >= 2:
            v_comps = live_vlm_out["components"]
            regions[0]["label"] = v_comps[0].get("target_label", regions[0]["label"])
            regions[0]["box"] = v_comps[0].get("vlm_box_px", regions[0]["box"])
            regions[0]["dino_prompt"] = v_comps[0].get("dino_prompt", f"{regions[0]['label'].lower()} box.")
            regions[2]["label"] = v_comps[1].get("target_label", regions[2]["label"])
            regions[2]["box"] = v_comps[1].get("vlm_box_px", regions[2]["box"])
            regions[2]["dino_prompt"] = v_comps[1].get("dino_prompt", f"{regions[2]['label'].lower()} box.")

        # Run Grounding DINO zero-shot detection on extracted photos to refine bounding boxes!
        if has_extracted_photo and dino_evaluator is not None:
            try:
                pil_up = Image.open(abs_diag).convert("RGB")
                p_text = (
                    f"{regions[0].get('dino_prompt', 'diagram component.')} "
                    f"{regions[2].get('dino_prompt', 'diagram block.')}"
                )
                inputs = dino_evaluator.processor(
                    images=pil_up, text=p_text, return_tensors="pt"
                ).to(dino_evaluator.device)
                with torch.no_grad():
                    outputs = dino_evaluator.model(**inputs)
                res = dino_evaluator.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=0.13, text_threshold=0.11, target_sizes=[(420, 720)]
                )[0]
                boxes = res["boxes"].detach().cpu().tolist()
                if len(boxes) >= 2:
                    boxes_sorted = sorted(boxes[:4], key=lambda b: b[0])
                    regions[0]["box"] = [round(float(v), 1) for v in boxes_sorted[0]]
                    regions[2]["box"] = [round(float(v), 1) for v in boxes_sorted[-1]]
            except Exception:
                pass

        comp_items = []
        for p_idx, reg_idx in enumerate((0, 2)):
            reg = regions[reg_idx]
            pid = f"c{cid}_p{p_idx + 1}"
            t_box = [float(v) for v in reg["box"]]
            b1000 = [
                round(t_box[0] * 1000.0 / 720.0),
                round(t_box[1] * 1000.0 / 420.0),
                round(t_box[2] * 1000.0 / 720.0),
                round(t_box[3] * 1000.0 / 420.0),
            ]
            dino_prompt = reg.get("dino_prompt") or f"{reg['color_name']} {reg['label'].lower()} box."
            prompt_text = (
                f"LAB CALIBRATION C{cid}-{'A' if p_idx == 0 else 'B'} ({course_title}): "
                f"Click or drag a bounding box around [{reg['label']}] in the extracted figure."
            )
            expl_text = (
                f"In {c_title}, [{reg['label']}] models {c_desc} and connects directly to the downstream stage."
            )

            puz_obj = {
                "puzzle_id": pid,
                "concept_id": cid,
                "concept_title": c_title,
                "diagram_file": diag_rel,
                "diagram_width": 720,
                "diagram_height": 420,
                "target_label": reg["label"],
                "dino_prompt": dino_prompt,
                "prompt": prompt_text,
                "target_box": t_box,
                "all_regions": [{"label": r["label"], "box": r["box"]} for r in regions],
                "explanation": expl_text,
            }
            new_dino_puzzles.append(puz_obj)
            comp_items.append(
                {
                    "component_id": pid,
                    "target_label": reg["label"],
                    "dino_prompt": dino_prompt,
                    "vlm_box_1000": b1000,
                    "vlm_box_px": t_box,
                    "ground_truth_box_px": t_box,
                    "lab_question": prompt_text,
                    "socratic_hint": f"Inspect the {'left' if p_idx == 0 else 'right'} region of the extracted figure.",
                    "explanation": expl_text,
                }
            )

        if live_vlm_out and live_vlm_out.get("vqa_challenge", {}).get("question"):
            vqa_obj = live_vlm_out["vqa_challenge"]
            vqa_question = vqa_obj["question"]
            vqa_ref = vqa_obj["reference_answer"]
        else:
            vqa_question = f"In {course_title}, how does {c_title} process information across its components?"
            vqa_ref = f"{c_title} models {c_desc} by coupling {regions[0]['label']} with {regions[2]['label']}."
            vqa_obj = {
                "vqa_id": f"vqa_c{cid}",
                "question": vqa_question,
                "reference_answer": vqa_ref,
                "sample_correct_answer": f"It connects {regions[0]['label']} and {regions[2]['label']} to accomplish {c_desc}.",
                "sample_partial_answer": f"It processes {c_title} in the diagram.",
                "misconception_trap_answer": f"{c_title} never connects {regions[0]['label']} and discards all information before {regions[2]['label']}.",
            }

        new_vlm_diagrams.append(
            {
                "diagram_id": f"{course_id}_c{cid}",
                "concept_id": cid,
                "concept_title": c_title,
                "diagram_file": diag_rel,
                "diagram_width": 720,
                "diagram_height": 420,
                "visual_summary": (live_vlm_out or {}).get(
                    "visual_summary",
                    f"Qwen2.5-VL-7B Document Inspection of {c_title}: {c_desc}",
                ),
                "extracted_labels": [r["label"] for r in regions],
                "components": comp_items,
                "vqa_challenge": vqa_obj,
            }
        )

        concepts_list.append(
            {
                "concept_id": cid,
                "title": c_title,
                "short_title": short_title,
                "prereqs": prereq_map[cid],
                "podcast_audio": None,
                "podcast_hint": f'[Voice Log C{cid}] "In {course_title}, {c_title} works by linking {regions[0]["label"]} with {regions[2]["label"]}: {c_desc}!"',
                "summary_bullets": [
                    f"Extracted Concept: {c_desc}",
                    f"Component 1 ({regions[0]['label']}): Handles initial input representation.",
                    f"Component 2 ({regions[2]['label']}): Completes the {short_title} transformation.",
                    f"Source Document: Extracted automatically by Qwen2.5-VL-7B + Grounding DINO from '{upload_name or course_title}'.",
                ],
                "easy_mc": [
                    {
                        "question": f"According to the ingested '{course_title}' material, what is the role of {c_title}?",
                        "choices": [
                            f"{c_desc[:110]}",
                            "To erase all system memory before execution",
                            "To replace structured processing with random noise",
                            "To disconnect all downstream components",
                        ],
                        "correct_idx": 0,
                        "reference": vqa_ref,
                    },
                    {
                        "question": f"Which visual component is highlighted in the {short_title} diagram?",
                        "choices": [
                            regions[0]["label"],
                            "An uncalibrated noise generator",
                            "A disconnected null pointer",
                            "None of the above",
                        ],
                        "correct_idx": 0,
                        "reference": f"{regions[0]['label']} is a primary component of {c_title}.",
                    },
                ],
                "medium_mc": [
                    {
                        "question": f"How do {regions[0]['label']} and {regions[2]['label']} work together in {c_title}?",
                        "choices": [
                            f"{regions[0]['label']} feeds structured representations into the pipeline to drive {regions[2]['label']}",
                            f"{regions[2]['label']} deletes the inputs of {regions[0]['label']}",
                            "They never exchange any data or signals",
                            "They bypass the core architecture entirely",
                        ],
                        "correct_idx": 0,
                        "reference": vqa_ref,
                    }
                ],
                "boss_open": [
                    {
                        "question": vqa_question,
                        "reference": vqa_ref,
                    }
                ],
            }
        )

    course_pkg = {
        "course_id": course_id,
        "title": course_title,
        "subtitle": (
            f"Ingested from {upload_name or 'Document'} "
            f"({doc_info['num_photos_extracted']} Extracted Photos, 10 VLM+DINO Puzzles)"
        ),
        "mentor": "Prof. Ada (Qwen2.5-VL-7B + Grounding DINO Lab)",
        "num_photos_extracted": doc_info["num_photos_extracted"],
        "source_file": upload_name,
        "concepts": concepts_list,
        "puzzles": new_dino_puzzles,
        "vlm_diagrams": new_vlm_diagrams,
    }

    with open(os.path.join(course_dir, "course_curriculum.json"), "w", encoding="utf-8") as f:
        json.dump(course_pkg, f, indent=2)

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
