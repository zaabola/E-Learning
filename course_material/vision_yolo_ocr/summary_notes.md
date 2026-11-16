# Deep Learning for Computer Vision: CNNs, YOLO Object Detection & OCR
**Willowbrook Vision Research Lab — Field Summary Notes (Mentor: Prof. Ada)**

---

## Concept 0 (C0): CNN Backbones, Stride & Feature Pyramid Networks (FPN)
* **Prerequisites:** None *(Foundation Node A)*
* **Core Idea:** Convolutional Neural Networks (CNNs) process images using small learnable **3×3 or 1×1 filters** that slide across spatial dimensions `(H, W)`. Unlike dense layers, convolutions enforce **local connectivity** and **weight sharing**, giving translation equivariance with drastically fewer parameters.
* **Stride, Padding & Receptive Field:**
  * Output spatial size: `floor((H + 2P - K) / S) + 1`.
  * A `stride = 2` convolution halves width and height (`4x` fewer pixels) while expanding the **receptive field** of deeper neurons.
* **The Multi-Scale Dilemma & Feature Pyramid Networks (FPN / PANet):**
  * **Shallow layers** (e.g., stride 4 or 8) have **high spatial resolution** (sharp edges, corners, tiny text strokes) but weak semantic context.
  * **Deep layers** (e.g., stride 32) have **strong semantic meaning** (whole vehicles, documents) but coarse spatial resolution where small objects vanish.
  * **FPN** adds a **top-down pathway with lateral 1×1 skip connections**, upsampling deep semantic maps and fusing them with high-resolution shallow maps so detectors spot both tiny distant objects and large foreground targets.

---

## Concept 1 (C1): Bounding Boxes, IoU & Non-Maximum Suppression (NMS)
* **Prerequisites:** None *(Foundation Node B)*
* **Bounding Box Representations:**
  * **Center-Size `(cx, cy, w, h)`:** Used by YOLO heads for scale-normalized regression.
  * **Corner `(x1, y1, x2, y2)`:** Convenient for computing overlap rectangles.
* **Intersection over Union (IoU):**
  * `IoU = Area(Box_pred ∩ Box_true) / Area(Box_pred ∪ Box_true)`, bounded in `[0, 1]`.
  * Standard `IoU = 0.50` is a common positive match threshold; COCO primary `mAP@0.50:0.95` averages Precision-Recall area across IoU thresholds from `0.50` to `0.95` in `0.05` steps, rewarding tight localization.
* **GIoU / CIoU Box Losses:**
  * Vanilla `1 - IoU` has **zero gradient** when two boxes do not overlap (`IoU = 0`).
  * **Complete IoU (CIoU)** adds penalties for **center-point distance** and **aspect-ratio mismatch**, pulling non-overlapping predictions smoothly toward ground truth.
* **Non-Maximum Suppression (NMS):**
  1. Discard boxes below confidence threshold `conf_thresh`.
  2. Sort remaining boxes by descending confidence score.
  3. Keep the highest-scoring box `B*`, and suppress any same-class box `B_i` whose `IoU(B*, B_i) > iou_thresh`. Repeat until all boxes are processed.

---

## Concept 2 (C2): Single-Stage Real-Time Detection & YOLO
* **Prerequisites:** `C0` (CNN/FPN) and `C1` (IoU/NMS)
* **Two-Stage vs. Single-Stage Detectors:**
  * **Two-Stage (Faster R-CNN):** Stage 1 runs a Region Proposal Network (RPN); Stage 2 pools features (`RoIAlign`) per proposal to classify and refine boxes. Accurate, but slower.
  * **Single-Stage (YOLO — "You Only Look Once"):** Frames detection as a **single dense forward pass** over multi-scale feature grids (typically strides `8`, `16`, and `32`, producing e.g. `80×80`, `40×40`, and `20×20` cells on a `640×640` image), achieving real-time 60–200+ FPS.
* **Modern YOLO Architecture (YOLOv8 / YOLOv10 / YOLOv11):**
  * **Backbone + Neck:** CSPDarknet / C2f blocks + SPPF (Spatial Pyramid Pooling Fast) + PAN-FPN multi-scale feature fusion.
  * **Anchor-Free Decoupled Head:** Instead of hand-tuned K-means anchor priors, each grid cell directly predicts distances `(left, top, right, bottom)` from the cell center using **separate classification and regression branches** (Binary Cross-Entropy for classes + CIoU & Distribution Focal Loss for boxes).
  * **Mosaic Augmentation:** Stitches 4 training images together at random scales, forcing the detector to learn small-scale objects and varied context.

---

## Concept 3 (C3): Optical Character Recognition (OCR) — Text Detection, CRNN & CTC Loss
* **Prerequisites:** `C0` (CNN Feature Extraction)
* **Two-Stage Scene OCR Architecture:**
  1. **Scene Text Detection (Where is the text?):** Models like **DBNet (Differentiable Binarization)** or **CRAFT** predict pixel-level probability & threshold maps to isolate tilted, curved, or multi-line text polygons in natural photos.
  2. **Scene Text Recognition (What does the cropped line say?):** Reads the cropped text strip into a character string without needing character-by-character bounding boxes!
* **CRNN (Convolutional Recurrent Neural Network):**
  * **CNN stem** converts a cropped word image (e.g., `32 × 128`) into a horizontal sequence of vertical feature columns `T`.
  * **BiLSTM / Transformer encoder** models left-to-right and right-to-left visual context across columns (distinguishing `l` vs `1` or `O` vs `0` from surrounding letters).
* **CTC Loss (Connectionist Temporal Classification):**
  * Because `T` visual columns do not align 1-to-1 with `L` characters (`L <= T`), CTC introduces a special **`<blank>` (`-`)** token.
  * During decoding, CTC **merges consecutive duplicate characters** and then **removes `<blank>` tokens**:
    * `h - e e l - l l o` -> `hello` (Notice how `<blank>` between the two `l`s preserves the double letter in `"hello"`!).
  * During training, CTC sums probabilities over all valid column alignments via dynamic programming so only whole-word text labels are needed.

---

## Concept 4 (C4): End-to-End YOLO + OCR Vision Pipelines & Edge Cases
* **Prerequisites:** `C2` (YOLO Detection) and `C3` (OCR Recognition)
* **Chained Industrial Vision Pipeline (ALPR, Smart Documents, Street Signs):**
  1. **Global Detection (YOLO):** Locates semantic regions of interest (e.g., `license_plate`, `invoice_total`, `hazard_label`) in full-resolution video frames in `<8 ms`.
  2. **Geometric Rectification:** Applies keypoint / corner detection + a **4-point perspective homography (`cv2.warpPerspective`)** to un-tilt skewed plates or labels into a flat horizontal strip.
  3. **Sequence Recognition (CRNN / TrOCR):** Decodes characters from the rectified crop with confidence filtering and multi-frame voting across video frames.
* **Real-World Failure Modes & Engineering Fixes:**
  * **Tiny Text Vanishing:** Running YOLO at low input resolution (`320px`) drops 8-pixel text below stride-8 Nyquist detail -> use higher input resolution, **SAHI tiled inference**, or P2 (stride-4) detection heads.
  * **NMS Crowd Suppression:** Two adjacent plates or dense text boxes with high overlap get wrongly deleted by greedy NMS -> tune IoU thresholds, use **Soft-NMS**, or polygon-based DBNet for dense lines.
  * **Perspective Distortion:** Feeding a 45°-tilted crop directly into horizontal-column CRNN scrambles vertical slices -> apply **Spatial Transformer Network (STN)** or 4-corner perspective warping first.
  * **Edge Deployment:** Export to **ONNX / TensorRT FP16** (or calibrated INT8) to cut latency and memory 2–3× while preserving mAP and Character Error Rate (CER).
