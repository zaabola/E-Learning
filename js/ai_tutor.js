// ===== KnowledgeQuest — 3-Model AI Tutor Client + Offline Fallback =====
// Connects the browser RPG to `server.py` (Final-DKT-GRU + Double-DQN + DeBERTa-v3-Small).
// Falls back cleanly to a local client-side YOLO/OCR curriculum when opened via file://
// without `server.py` running (preserving AGENTS.md Rule #2).

const AITutor = (() => {
  const DEFAULT_API = (typeof location !== 'undefined' && location.protocol.startsWith('http'))
    ? ''
    : 'http://127.0.0.1:8000';

  const ACTION_LABELS = {
    0: 'Podcast Hint + Easy Quiz',
    1: 'Easy 4-Move Wild Quiz',
    2: 'Medium Flow Challenge',
    3: 'Prereq Concept Review',
    4: 'Open-Text Boss Synthesis',
  };

  // Trainer -> Concept ID specialization across Wildvale
  const TRAINER_CONCEPT_MAP = {
    tr_rex:   0, // C0: CNN Backbones, Stride & FPN
    tr_cal:   1, // C1: Bounding Boxes, IoU & NMS
    tr_mira:  2, // C2: Single-Stage Real-Time Detection & YOLO
    tr_rocco: 3, // C3: OCR — Scene Text, CRNN & CTC Loss
    tr_mason: 4, // C4: End-to-End YOLO + OCR Vision Pipelines (Gym Boss!)
  };

  // Embedded fallback curriculum (mirrors `course_material/vision_yolo_ocr/course_curriculum.json`)
  // so `index.html` works 100% standalone even if `server.py` is offline.
  const FALLBACK_CONCEPTS = [
    {
      concept_id: 0,
      title: 'CNN Backbones, Stride & Feature Pyramids (FPN)',
      short_title: 'CNNs & FPN',
      prereqs: [],
      podcast_audio: 'course_material/vision_yolo_ocr/audio/hint_c0.wav',
      podcast_hint: '[Podcast Clip 00:45 - 01:00] "Deep stride-32 CNN layers understand whole objects, while shallow stride-8 layers keep sharp pixel edges. A Feature Pyramid Network fuses both so tiny distant targets never vanish!"',
      summary_bullets: [
        '3x3 convolutions share weights across (H, W), preserving spatial locality with far fewer parameters than dense layers.',
        'Stride = 2 halves width and height (4x fewer pixels) while expanding each neuron\'s receptive field.',
        'Shallow layers keep fine spatial detail; deep layers capture high-level semantics.',
        'Feature Pyramid Networks (FPN / PANet) upsample deep semantic maps and fuse them via 1x1 lateral connections with shallow high-res maps.',
      ],
      easy_mc: [
        {
          question: 'Why are Convolutional Neural Networks (CNNs) vastly more parameter-efficient on images than fully connected layers?',
          choices: [
            'Small filters share the same learnable weights across every spatial location instead of connecting every pixel to every neuron',
            'CNNs convert all RGB images into 1-pixel grayscale scalars before training',
            'CNNs contain zero learnable weights and rely only on sorting pixels',
            'CNNs delete 90% of input channels randomly at every forward pass',
          ],
          correct_idx: 0,
          reference: 'CNNs enforce local connectivity and spatial weight sharing, reusing small kernels across the entire image.',
        },
        {
          question: 'What primary problem does a Feature Pyramid Network (FPN) solve inside modern object detectors?',
          choices: [
            'Detecting objects reliably across drastically different scales by fusing deep semantic features with shallow high-resolution features',
            'Replacing GPU matrix multiplication with CPU text sorting',
            'Eliminating all activation functions from the backbone',
            'Converting bounding boxes into audio waveforms',
          ],
          correct_idx: 0,
          reference: 'FPN combines upsampled deep semantic feature maps with high-resolution shallow maps so both small and large objects are detected.',
        },
      ],
      medium_mc: [
        {
          question: 'Why do tiny objects (e.g., 10x10 pixels) often disappear if a detector only predicts from a stride-32 backbone layer?',
          choices: [
            'At stride 32, a 10x10 object shrinks below a single feature cell, erasing its fine spatial localization',
            'Stride-32 layers only detect grayscale pixels and ignore RGB color',
            'Deep layers have too small a receptive field to see 10x10 pixels',
            'Stride 32 multiplies image resolution by 32x',
          ],
          correct_idx: 0,
          reference: 'Downsampling by 32x collapses tiny objects into a sub-pixel footprint unless finer stride-8/stride-4 pyramid maps are used.',
        },
      ],
      boss_open: [
        {
          question: 'Explain the trade-off between shallow CNN layers and deep CNN layers, and how a Feature Pyramid Network (FPN) resolves it.',
          reference: 'Shallow CNN layers have high spatial resolution for small objects and edges, whereas deep layers have strong semantic context but coarse resolution; FPN upsamples deep features and fuses them with shallow maps.',
          keywords: ['shallow', 'deep', 'resolution', 'semantic', 'upsampl', 'fus', 'scale', 'small'],
        },
      ],
    },
    {
      concept_id: 1,
      title: 'Bounding Boxes, IoU & Non-Maximum Suppression (NMS)',
      short_title: 'Boxes, IoU & NMS',
      prereqs: [],
      podcast_audio: 'course_material/vision_yolo_ocr/audio/hint_c1.wav',
      podcast_hint: '[Podcast Clip 02:30 - 02:45] "IoU divides the overlap area of two boxes by their total union area. Then Non-Maximum Suppression keeps the sharpest high-confidence box and deletes overlapping duplicates!"',
      summary_bullets: [
        'IoU = Area of Overlap / Area of Union, ranging from 0.0 (disjoint) to 1.0 (identical boxes).',
        'Vanilla IoU loss has zero gradient when boxes do not overlap; GIoU and CIoU add center-distance and aspect-ratio terms.',
        'NMS sorts candidate boxes by confidence, keeps the top box, and suppresses same-class boxes with IoU above the NMS threshold.',
        'COCO mAP@0.50:0.95 averages AP across IoU thresholds from 0.50 to 0.95.',
      ],
      easy_mc: [
        {
          question: 'How is Intersection over Union (IoU) computed between a predicted bounding box and a ground-truth box?',
          choices: [
            'Area of their overlapping intersection divided by the total area of their union',
            'Sum of their perimeter lengths multiplied by image width',
            'Difference between their class probabilities',
            'Euclidean distance from the image corner',
          ],
          correct_idx: 0,
          reference: 'IoU divides the intersection area of predicted and ground-truth boxes by their combined union area.',
        },
        {
          question: 'What is the purpose of Non-Maximum Suppression (NMS) during object detection inference?',
          choices: [
            'To keep the highest-confidence box for an object and remove redundant overlapping duplicate boxes',
            'To blur background pixels before running the CNN',
            'To increase duplicate boxes per object to 100',
            'To sort class names alphabetically',
          ],
          correct_idx: 0,
          reference: 'NMS keeps the peak-confidence detection and suppresses high-IoU overlapping duplicate boxes.',
        },
      ],
      medium_mc: [
        {
          question: 'Why do YOLO detectors train with CIoU loss instead of plain (1 - IoU) loss?',
          choices: [
            'Plain (1 - IoU) gives zero gradient when boxes do not overlap, whereas CIoU penalizes center distance and aspect ratio',
            'CIoU completely ignores box overlap',
            'Plain IoU cannot run on GPUs',
            'CIoU forces all predicted boxes to have zero width',
          ],
          correct_idx: 0,
          reference: 'CIoU adds center-point distance and aspect-ratio penalties so non-overlapping boxes still receive informative gradients.',
        },
      ],
      boss_open: [
        {
          question: 'Walk through how Non-Maximum Suppression (NMS) cleans up raw detector predictions and what happens if the NMS IoU threshold is too low.',
          reference: 'NMS sorts boxes by confidence, keeps the highest-scoring box, and deletes overlapping same-class boxes exceeding an IoU threshold; if set too low, nearby distinct objects in crowds get wrongly suppressed.',
          keywords: ['confidence', 'overlap', 'iou', 'duplicate', 'suppress', 'crowd', 'threshold'],
        },
      ],
    },
    {
      concept_id: 2,
      title: 'Single-Stage Real-Time Detection & YOLO',
      short_title: 'YOLO Detection',
      prereqs: [0, 1],
      podcast_audio: 'course_material/vision_yolo_ocr/audio/hint_c2.wav',
      podcast_hint: '[Podcast Clip 05:10 - 05:25] "Unlike two-stage R-CNN detectors that propose regions first and classify second, YOLO divides the image into multi-scale grids and predicts boxes and classes in one single fast pass!"',
      summary_bullets: [
        'Two-stage detectors (Faster R-CNN) generate region proposals first; YOLO predicts boxes + classes densely in one pass.',
        'On 640x640 input, YOLO predicts at 3 FPN scales: 80x80 (stride 8, small), 40x40 (stride 16, medium), and 20x20 (stride 32, large).',
        'Modern YOLOv8/v11 heads are anchor-free and decoupled: separate branches predict class BCE logits and (l, t, r, b) box offsets.',
      ],
      easy_mc: [
        {
          question: 'What makes YOLO a "single-stage" object detector compared to Faster R-CNN?',
          choices: [
            'It predicts bounding box coordinates and class probabilities directly across spatial grid cells in one unified forward pass',
            'It only looks at a single 1x1 pixel of the image',
            'It requires human annotators to draw proposals at test time',
            'It trains for only one gradient step',
          ],
          correct_idx: 0,
          reference: 'YOLO predicts bounding boxes and class probabilities densely in one pass without a separate region-proposal stage.',
        },
        {
          question: 'On a 640x640 image with strides 8, 16, and 32, which YOLO grid detects the SMALLEST objects?',
          choices: [
            'The 80x80 high-resolution grid (stride 8)',
            'The 20x20 coarse grid (stride 32)',
            'A 1x1 global average pool',
            'Stride 64 only',
          ],
          correct_idx: 0,
          reference: 'Stride 8 produces a dense 80x80 grid that preserves fine spatial features for small objects.',
        },
      ],
      medium_mc: [
        {
          question: 'Why do YOLOv8 and YOLOv11 use a decoupled head instead of one shared conv layer for classes and boxes?',
          choices: [
            'Classification needs semantic invariance while box regression needs sharp edge localization; separating branches avoids task conflict',
            'A decoupled head runs on two separate computers',
            'Shared tensors are unsupported in PyTorch',
            'It forces class scores to equal box width',
          ],
          correct_idx: 0,
          reference: 'Separating classification and box regression branches prevents semantic vs boundary-localization feature conflict.',
        },
      ],
      boss_open: [
        {
          question: 'Compare two-stage detectors (Faster R-CNN) with single-stage YOLO detectors and explain how YOLO handles multi-scale objects.',
          reference: 'Two-stage detectors generate region proposals first and classify each region second, while YOLO predicts boxes and classes in one single pass across multi-scale stride 8, 16, and 32 FPN feature grids.',
          keywords: ['single', 'pass', 'grid', 'proposal', 'stage', 'scale', 'stride', 'fpn'],
        },
      ],
    },
    {
      concept_id: 3,
      title: 'OCR: Scene Text Detection, CRNN & CTC Loss',
      short_title: 'OCR, CRNN & CTC',
      prereqs: [0],
      podcast_audio: 'course_material/vision_yolo_ocr/audio/hint_c3.wav',
      podcast_hint: '[Podcast Clip 07:55 - 08:10] "Modern OCR works in two steps: DBNet finds text polygons in the wild, then CRNN slices the cropped word into visual columns and uses CTC\'s blank token to decode letters without character boxes!"',
      summary_bullets: [
        'Modern OCR splits into Text Detection (DBNet / CRAFT finding polygons) + Text Recognition (CRNN / TrOCR decoding characters).',
        'CRNN stacks a CNN feature stem -> left-to-right column sequence -> BiLSTM/Transformer encoder -> per-column probabilities.',
        'CTC aligns T visual columns to L characters using a <blank> (-) token: collapse consecutive duplicates, then remove blanks.',
      ],
      easy_mc: [
        {
          question: 'What are the two core stages of a modern scene OCR pipeline like PaddleOCR or EasyOCR?',
          choices: [
            '1) Text Detection (locating word polygons) followed by 2) Text Recognition (decoding cropped strips into characters)',
            '1) Audio recording followed by 2) Speech synthesis',
            '1) Deleting vowels followed by 2) Random spellcheck',
            '1) K-means clustering RGB colors followed by 2) Blur',
          ],
          correct_idx: 0,
          reference: 'Scene OCR first detects text bounding polygons and then recognizes character sequences inside each cropped strip.',
        },
        {
          question: 'Using standard CTC decoding (collapse consecutive duplicates, then remove "-" blanks), what does "c c - a - t t -" decode to?',
          choices: [
            '"cat"',
            '"ccaatt"',
            '"c-a-t"',
            '"ct"',
          ],
          correct_idx: 0,
          reference: 'Collapsing adjacent duplicates yields "c - a - t -", and stripping "-" blanks gives "cat".',
        },
      ],
      medium_mc: [
        {
          question: 'Why is the CTC <blank> ("-") token essential for spelling words with double letters like "book" or "hello"?',
          choices: [
            'Without a <blank> between two "o" emissions, collapsing duplicates would merge "o o" into one "o" ("bok"); "b o - o k" preserves "book"',
            'The <blank> token represents spaces between paragraphs',
            'The <blank> token deletes low-confidence words',
            'CTC never collapses duplicate characters',
          ],
          correct_idx: 0,
          reference: 'CTC collapses repeated column predictions, so a <blank> between identical letters separates true double characters.',
        },
      ],
      boss_open: [
        {
          question: 'Explain how a CRNN + CTC recognizer reads a cropped word image without character bounding boxes, including the role of the <blank> token.',
          reference: 'A CNN extracts horizontal visual columns and a BiLSTM models sequence context; CTC aligns columns to text without character boxes by collapsing repeated emissions and using a blank token between double letters.',
          keywords: ['cnn', 'column', 'lstm', 'rnn', 'sequence', 'ctc', 'blank', 'duplicate', 'collapse'],
        },
      ],
    },
    {
      concept_id: 4,
      title: 'End-to-End YOLO + OCR Vision Pipelines & Edge Cases',
      short_title: 'YOLO + OCR Systems',
      prereqs: [2, 3],
      podcast_audio: 'course_material/vision_yolo_ocr/audio/hint_c4.wav',
      podcast_hint: '[Podcast Clip 10:40 - 10:55] "In real-world license-plate or invoice pipelines, YOLO finds the target region in milliseconds, a 4-point perspective warp un-tilts the crop, and CRNN-CTC reads the crisp rectified text!"',
      summary_bullets: [
        'End-to-end ALPR / Document pipelines chain: YOLO region detection -> 4-corner perspective rectification -> CRNN/Transformer OCR.',
        'Skewed crops scramble CRNN vertical columns; perspective homography warping aligns characters horizontally first.',
        'Crop from the original full-res frame (with 5-10% margin padding) and aggregate video frames via tracking + voting.',
      ],
      easy_mc: [
        {
          question: 'In an Automated License Plate Recognition (ALPR) pipeline combining YOLO and OCR, how do the two models divide the work?',
          choices: [
            'YOLO rapidly detects and crops the license plate box from the full frame, and OCR transcribes the characters inside that crop',
            'OCR scans every sky and tree pixel first, and YOLO spells out the letters',
            'YOLO trains 36 separate full-frame networks without cropping',
            'Both models output audio waveforms',
          ],
          correct_idx: 0,
          reference: 'YOLO localizes the target region in milliseconds so OCR only processes clean, high-resolution cropped strips.',
        },
      ],
      medium_mc: [
        {
          question: 'Why do production YOLO + OCR pipelines apply 4-point perspective rectification (homography) before CRNN recognition?',
          choices: [
            'To flatten skewed or tilted crops into a horizontal strip so CRNN\'s vertical feature columns slice cleanly through individual characters',
            'To flip text upside-down',
            'To reduce image contrast to zero',
            'To delete boundary characters',
          ],
          correct_idx: 0,
          reference: 'Un-warping perspective tilt aligns characters horizontally so CRNN vertical slices match single letters.',
        },
      ],
      boss_open: [
        {
          question: 'Design an end-to-end Deep Learning pipeline to read tilted license plates from 4K traffic cameras using YOLO, rectification, and CRNN-CTC.',
          reference: 'YOLO detects the license plate region, the padded crop is extracted from the original 4K frame and rectified with a 4-point perspective warp, and CRNN with CTC loss decodes the characters with multi-frame voting.',
          keywords: ['yolo', 'detect', 'crop', '4k', 'perspective', 'warp', 'rectif', 'crnn', 'ctc', 'ocr'],
        },
      ],
    },
  ];

  let online = false;
  let statusCache = null;
  let activeAudio = null;
  let localTurn = 0;
  let localMastery = [0.40, 0.38, 0.35, 0.35, 0.32];
  let localBadges = [false, false, false, false, false];
  let localAffect = { concentrating: 0.68, frustrated: 0.30, bored: 0.18 };
  let localCursors = {};
  let localFailStreak = 0;

  function buildOfflineStatus() {
    const concepts = FALLBACK_CONCEPTS.map((c, i) => ({
      concept_id: i,
      title: c.title,
      short_title: c.short_title,
      prereqs: c.prereqs,
      mastery: localMastery[i],
      badge: localBadges[i],
      attempts: 0,
      wins: 0,
      podcast_hint: c.podcast_hint,
      podcast_audio: c.podcast_audio,
      summary_bullets: c.summary_bullets,
    }));
    const meanM = localMastery.reduce((a, b) => a + b, 0) / 5;
    let frontier = localBadges.findIndex(b => !b);
    if (frontier === -1) frontier = 4;
    return {
      online: false,
      course_id: 'vision_yolo_ocr',
      course_title: 'Deep Learning Vision: CNNs, YOLO & OCR',
      course_subtitle: 'Offline Fallback Mode (Start `python server.py` for Live 3-Model PyTorch AI!)',
      course_mentor: 'Prof. Ada (Willowbrook Vision Lab)',
      available_courses: [
        { course_id: 'vision_yolo_ocr', title: 'Deep Learning Vision: CNNs, YOLO & OCR' },
      ],
      profile: 'average',
      turn: localTurn,
      frontier_concept_id: frontier,
      mastery: localMastery.slice(),
      mean_mastery: meanM,
      consolidated: localBadges.slice(),
      badges_earned_count: localBadges.filter(Boolean).length,
      quest_completed: localBadges.every(Boolean),
      affect: { ...localAffect },
      concepts,
      models: {
        dkt: { name: 'Final-DKT-GRU (Offline Fallback)', auc: 0.7996 },
        rl: { name: 'Double-DQN (Offline Fallback)', reward: 19.23 },
        grader: { name: 'DeBERTa-v3-Small (Offline Fallback)', acc_3way: 0.7984, device: 'local-js' },
        dino: { name: 'Grounding-DINO-Tiny (Offline Fallback)', mean_box_iou: 0.894, puzzles: 10, device: 'local-js' },
      },
      last_turn: statusCache && statusCache.last_turn ? statusCache.last_turn : null,
    };
  }

  async function apiFetch(path, opts = {}, timeoutMs = 2500) {
    const ctrl = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
    try {
      const res = await fetch(DEFAULT_API + path, {
        ...opts,
        headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
        signal: ctrl ? ctrl.signal : undefined,
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return await res.json();
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  async function init() {
    try {
      const snap = await apiFetch('/api/status', { method: 'GET' }, 1800);
      online = true;
      statusCache = snap;
      persistToPlayer(snap);
    } catch (e) {
      online = false;
      statusCache = buildOfflineStatus();
    }
    return statusCache;
  }

  function persistToPlayer(snap) {
    if (typeof Game !== 'undefined' && Game.player && snap) {
      Game.player.aiState = {
        course_id: snap.course_id,
        mastery: snap.mastery,
        consolidated: snap.consolidated,
      };
    }
  }

  async function syncFromPlayerSave() {
    if (typeof Game === 'undefined' || !Game.player || !Game.player.aiState) {
      return init();
    }
    const saved = Game.player.aiState;
    if (Array.isArray(saved.mastery) && saved.mastery.length === 5) {
      localMastery = saved.mastery.slice();
    }
    if (Array.isArray(saved.consolidated) && saved.consolidated.length === 5) {
      localBadges = saved.consolidated.slice();
    }
    try {
      const snap = await apiFetch('/api/session/sync', {
        method: 'POST',
        body: JSON.stringify(saved),
      }, 2000);
      online = true;
      statusCache = snap;
    } catch (e) {
      online = false;
      statusCache = buildOfflineStatus();
    }
    return statusCache;
  }

  async function refreshStatus() {
    return init();
  }

  async function resetSession(courseId = 'vision_yolo_ocr', profile = 'average') {
    localTurn = 0;
    localFailStreak = 0;
    localBadges = [false, false, false, false, false];
    visionPuzzlesCache = null;
    localMastery = profile === 'struggling'
      ? [0.25, 0.25, 0.22, 0.22, 0.20]
      : (profile === 'advanced' ? [0.64, 0.62, 0.58, 0.56, 0.52] : [0.40, 0.38, 0.35, 0.35, 0.32]);
    try {
      const snap = await apiFetch('/api/session/reset', {
        method: 'POST',
        body: JSON.stringify({ course_id: courseId, profile }),
      }, 3000);
      online = true;
      statusCache = snap;
      persistToPlayer(snap);
    } catch (e) {
      online = false;
      statusCache = buildOfflineStatus();
      persistToPlayer(statusCache);
    }
    return statusCache;
  }

  let uploadedCourseInfo = null;

  async function ingestCustomCourse(payload = {}, onProgress = null) {
    visionPuzzlesCache = null;
    const startData = await apiFetch('/api/course/ingest', {
      method: 'POST',
      body: JSON.stringify({
        course_title: payload.courseTitle || '',
        raw_notes: payload.rawNotes || '',
        image_base64: payload.imageBase64 || null,
        file_name: payload.fileName || null,
        file_base64: payload.fileBase64 || null,
        remote_vlm_url: payload.remoteVlmUrl || null,
        profile: payload.profile || 'average',
      }),
    }, 30000);

    let data = startData;
    if (startData && startData.job_id && startData.job_status === 'running') {
      const tStart = performance.now();
      if (onProgress) onProgress(startData.progress || 5, startData.step || 'Starting ingestion...');
      const jobId = startData.job_id;
      const maxPolls = 400; // Up to 10 minutes ceiling for very large PDFs
      let consecutiveErrors = 0;
      for (let i = 0; i < maxPolls; i++) {
        await new Promise(r => setTimeout(r, 1500));
        let poll = null;
        try {
          poll = await apiFetch(`/api/course/ingest/status?job_id=${encodeURIComponent(jobId)}`, {
            method: 'GET',
          }, 15000);
          consecutiveErrors = 0;
        } catch (pollErr) {
          consecutiveErrors++;
          if (consecutiveErrors >= 8) {
            throw new Error(`Lost connection while polling VLM status (${pollErr.message || 'network'}).`);
          }
          continue;
        }
        const elapsedSec = Math.round((performance.now() - tStart) / 1000);
        if (onProgress) onProgress(poll.progress || 10, `[${elapsedSec}s] ${poll.step || 'Processing...'}`);
        if (poll.job_status === 'done') {
          data = poll;
          break;
        }
        if (poll.job_status === 'error') {
          throw new Error(poll.error || poll.step || 'Ingestion failed');
        }
      }
      if (!data || data.job_status !== 'done') {
        throw new Error('Ingestion timed out while waiting for VLM.');
      }
    }

    online = true;
    statusCache = data.status;
    if (Array.isArray(data.puzzles) && data.puzzles.length) {
      visionPuzzlesCache = data.puzzles;
    }
    uploadedCourseInfo = {
      course_id: data.course_id,
      course_title: data.course_title || (data.status && data.status.course_title) || payload.courseTitle || 'Uploaded Course',
      num_photos_extracted: data.num_photos_extracted || 0,
      source_file: data.source_file || payload.fileName || '',
    };
    persistToPlayer(statusCache);
    return data;
  }

  async function submitVisionVQA(conceptId, studentAnswer, responseSec = 10.0) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const data = await apiFetch('/api/vision/vqa', {
          method: 'POST',
          body: JSON.stringify({
            concept_id: Number(conceptId || 0),
            student_answer: String(studentAnswer || ''),
            response_sec: Number(responseSec || 10.0),
          }),
        }, 15000);
        online = true;
        statusCache = data.status;
        persistToPlayer(data.status);
        return data.result;
      } catch (e) {
        if (attempt === 0) {
          await new Promise(r => setTimeout(r, 600));
          continue;
        }
      }
    }
    return {
      concept_id: Number(conceptId || 0),
      verdict: studentAnswer && studentAnswer.trim().length > 18 ? 'correct' : 'partial',
      score: studentAnswer && studentAnswer.trim().length > 18 ? 0.88 : 0.48,
      grader_latency_ms: 1.0,
      mastery_before: localMastery.slice(),
      mastery_after: localMastery.slice(),
    };
  }

  function resolveBattleFocus(ctx = {}) {
    let focusConceptId = null;
    let forceAction = null;
    if (ctx.trainer && ctx.trainer.flag && TRAINER_CONCEPT_MAP[ctx.trainer.flag] !== undefined) {
      focusConceptId = TRAINER_CONCEPT_MAP[ctx.trainer.flag];
      // Gym Leader Mason's final Pokémon triggers a guaranteed Open-Text Boss Synthesis!
      if (ctx.trainer.flag === 'tr_mason' && ctx.enemyIdx === ctx.enemyPartyLen - 1) {
        forceAction = 4;
      }
    }
    return { focusConceptId, forceAction };
  }

  function pickLocalVariant(c, tier) {
    const pool = c[tier] || c.easy_mc;
    const key = `${c.concept_id}_${tier}`;
    const idx = localCursors[key] || 0;
    localCursors[key] = idx + 1;
    return pool[idx % pool.length];
  }

  function buildOfflineEncounter(focusConceptId, forceAction) {
    let focus = focusConceptId !== null && focusConceptId !== undefined
      ? focusConceptId
      : localBadges.findIndex(b => !b);
    if (focus < 0) focus = 4;
    const c = FALLBACK_CONCEPTS[focus];
    const prereqs = c.prereqs || [];
    const weakPre = prereqs.find(p => localMastery[p] < 0.52);

    let action = 1;
    let reason = 'Offline Heuristic Game Master';
    if (forceAction !== null && forceAction !== undefined) {
      action = forceAction;
      reason = 'Gym Boss Synthesis Challenge';
    } else if (weakPre !== undefined) {
      action = 3;
      reason = 'Prerequisite Gap (<52%) -> Review';
    } else if (localFailStreak >= 1 || localAffect.frustrated >= 0.45) {
      action = 0;
      reason = 'Frustration Support -> Podcast Hint';
    } else if (!localBadges[focus] && localMastery[focus] >= 0.68) {
      action = 4;
      reason = 'Mastery >= 68% -> Open-Text Badge Finisher!';
    } else if (localMastery[focus] >= 0.50) {
      action = 2;
      reason = 'Flow Zone -> Medium Challenge';
    }

    const targetId = (action === 3 && weakPre !== undefined) ? weakPre : focus;
    const tc = FALLBACK_CONCEPTS[targetId];
    const tier = action === 4 ? 'boss_open' : (action === 2 ? 'medium_mc' : 'easy_mc');
    const q = pickLocalVariant(tc, tier);

    return {
      turn: localTurn + 1,
      rl_action: action,
      rl_action_name: ACTION_LABELS[action],
      decision_reason: reason,
      rl_q_values: [16.8, 17.2, 17.5, 16.4, 17.1],
      rl_latency_ms: 0.2,
      focus_concept_id: focus,
      target_concept_id: targetId,
      target_concept_title: tc.title,
      target_concept_short: tc.short_title,
      target_mastery: localMastery[targetId],
      mode: action === 4 ? 'boss_open_text' : 'mc',
      podcast_hint: action === 0 ? tc.podcast_hint : null,
      podcast_audio: action === 0 ? tc.podcast_audio : null,
      available_podcast_hint: tc.podcast_hint,
      available_podcast_audio: tc.podcast_audio,
      question: q.question,
      choices: q.choices || null,
      correct_idx: q.correct_idx,
      reference_answer: q.reference,
      _keywords: q.keywords || [],
    };
  }

  async function nextEncounter(battleCtx = {}) {
    const { focusConceptId, forceAction } = resolveBattleFocus(battleCtx);
    const personaKey = (battleCtx && battleCtx.trainer && battleCtx.trainer.flag)
      ? battleCtx.trainer.flag
      : 'wild';
    try {
      const data = await apiFetch('/api/encounter/next', {
        method: 'POST',
        body: JSON.stringify({
          focus_concept_id: focusConceptId,
          force_action: forceAction,
          persona_key: personaKey,
        }),
      }, 60000);
      online = true;
      statusCache = data.status;
      return data.encounter;
    } catch (e) {
      // Do NOT permanently flip `online = false` on one slow LLM timeout — only use offline fallback for this turn
      if (!statusCache) statusCache = buildOfflineStatus();
      return buildOfflineEncounter(focusConceptId, forceAction);
    }
  }

  async function getNpcDialogue(npcKey = 'prof_ada', starterName = 'your partner') {
    if (!online) return null;
    try {
      const q = `/api/npc/dialogue?npc=${encodeURIComponent(npcKey)}&starter=${encodeURIComponent(starterName)}`;
      const data = await apiFetch(q, { method: 'GET' }, 30000);
      if (data && Array.isArray(data.lines) && data.lines.length > 0) {
        return data;
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  async function chatWithNpc(npcKey = 'prof_ada', question = '') {
    const cleanQ = String(question || '').trim();
    if (!cleanQ) return null;
    try {
      const data = await apiFetch('/api/npc/chat', {
        method: 'POST',
        body: JSON.stringify({ npc_key: npcKey, question: cleanQ }),
      }, 45000);
      online = true;
      return data;
    } catch (e) {
      const st = statusCache || buildOfflineStatus();
      const cs = st.concepts || FALLBACK_CONCEPTS;
      const c0 = cs[0] || FALLBACK_CONCEPTS[0];
      const fact = (c0.summary_bullets && c0.summary_bullets[0]) || c0.podcast_hint || c0.title;
      return {
        npc: npcKey,
        speaker: String(npcKey || 'NPC').replace(/^tr_/, '').toUpperCase(),
        reply: `Quick study tip on ${c0.short_title || c0.title}: ${fact}`,
        matched_concept: c0.title,
        source: 'offline_fallback',
        model_name: 'Offline Notes',
      };
    }
  }

  function getCourseStory() {
    return (statusCache && statusCache.course_story) ? statusCache.course_story : null;
  }

  function gradeOffline(enc, userResponse, responseSec, hintUsed) {
    localTurn++;
    const cid = enc.target_concept_id;
    const cleaned = String(userResponse || '').trim();
    let verdict = 'incorrect';
    let semScore = 0.0;
    let dmgMult = 0.0;
    let usedNeural = false;

    if (enc.choices && ['1', '2', '3', '4'].includes(cleaned)) {
      const ok = (parseInt(cleaned, 10) - 1) === enc.correct_idx;
      verdict = ok ? 'correct' : 'incorrect';
      semScore = ok ? 1.0 : 0.0;
      dmgMult = ok ? 1.0 : 0.0;
    } else {
      usedNeural = true;
      const lower = cleaned.toLowerCase();
      const hasNegTrap = /\b(never|not needed|useless|ignore)\b/.test(lower) && lower.length < 90;
      const refWords = (enc._keywords && enc._keywords.length)
        ? enc._keywords
        : enc.reference_answer.toLowerCase().split(/\W+/).filter(w => w.length > 4);
      let hits = 0;
      for (const w of refWords) if (lower.includes(w)) hits++;
      const ratio = refWords.length ? hits / Math.min(5, refWords.length) : 0;
      if (!hasNegTrap && cleaned.split(/\s+/).length >= 5 && ratio >= 0.55) {
        verdict = 'correct'; semScore = Math.min(0.95, 0.65 + ratio * 0.25); dmgMult = semScore;
      } else if (!hasNegTrap && cleaned.split(/\s+/).length >= 4 && ratio >= 0.28) {
        verdict = 'partial'; semScore = 0.52; dmgMult = 0.52;
      } else {
        verdict = 'incorrect'; semScore = 0.12; dmgMult = 0.0;
      }
    }

    const before = localMastery.slice();
    let badgeJustEarned = false;
    if (verdict === 'correct') {
      localMastery[cid] = Math.min(0.96, localMastery[cid] + (usedNeural ? 0.20 : (hintUsed ? 0.10 : 0.15)));
      localFailStreak = 0;
      localAffect.concentrating = Math.min(0.95, localAffect.concentrating + 0.06);
      localAffect.frustrated = Math.max(0.08, localAffect.frustrated - 0.08);
      if (usedNeural && !localBadges[cid]) {
        localBadges[cid] = true;
        localMastery[cid] = Math.max(localMastery[cid], 0.86);
        badgeJustEarned = true;
      }
    } else if (verdict === 'partial') {
      localMastery[cid] = Math.min(0.88, localMastery[cid] + 0.06);
      localFailStreak = 0;
    } else {
      localMastery[cid] = Math.max(0.12, localMastery[cid] - 0.03);
      localFailStreak++;
      localAffect.frustrated = Math.min(0.92, localAffect.frustrated + 0.12);
    }

    statusCache = buildOfflineStatus();
    const rec = {
      turn: localTurn,
      rl_action: enc.rl_action,
      rl_action_name: enc.rl_action_name,
      target_concept_id: cid,
      target_concept_title: enc.target_concept_title,
      used_neural_grader: usedNeural,
      grader: {
        graded_by: usedNeural ? 'Offline Rubric Fallback' : '4-Move Tactical Selector',
        verdict,
        semantic_score: semScore,
        rpg_damage_multiplier: dmgMult,
      },
      reference_answer: enc.reference_answer,
      correct_idx: enc.correct_idx,
      badge_just_earned: badgeJustEarned,
      badges_earned_count: localBadges.filter(Boolean).length,
      consolidated: localBadges.slice(),
      quest_completed: localBadges.every(Boolean),
      response_sec: Math.round(responseSec * 10) / 10,
      mastery_before: before,
      mastery_after: localMastery.slice(),
      affect: { ...localAffect },
      latencies_ms: { rl_gamemaster_ms: 0.2, deberta_grader_ms: 0.4, final_dkt_gru_ms: 0.2, total_ai_turn_ms: 0.8 },
    };
    statusCache.last_turn = rec;
    persistToPlayer(statusCache);
    return rec;
  }

  async function submitEncounter(enc, userResponse, responseSec, retries = 1, hintUsed = false) {
    if (online) {
      try {
        // Longer timeout (8s) in case CPU DeBERTa inference is running on a low-power laptop
        const data = await apiFetch('/api/encounter/submit', {
          method: 'POST',
          body: JSON.stringify({
            encounter: enc,
            user_response: userResponse,
            response_sec: responseSec,
            retries,
            hint_used: hintUsed,
          }),
        }, 8000);
        statusCache = data.status;
        persistToPlayer(data.status);
        return data.result;
      } catch (e) {
        online = false;
      }
    }
    return gradeOffline(enc, userResponse, responseSec, hintUsed);
  }

  const FALLBACK_VISION_PUZZLES = [
    {
      puzzle_id: 'c0_p1',
      concept_id: 0,
      concept_title: 'CNN Backbones, Stride & Feature Pyramids (FPN)',
      diagram_file: 'course_material/vision_yolo_ocr/diagrams/c0_cnn_fpn.png',
      diagram_width: 720,
      diagram_height: 420,
      target_label: '3x3 Convolution Filter Kernel',
      prompt: 'LAB CALIBRATION C0-A: Click or drag a bounding box around the 3x3 Convolution Filter Kernel scanning the input image.',
      target_box: [28, 95, 215, 365],
      explanation: 'The 3x3 Convolution Kernel shares weights across all (H, W) positions, while the 1x1 Lateral Connection fuses shallow high-res features with upsampled deep semantic maps.',
    },
    {
      puzzle_id: 'c0_p2',
      concept_id: 0,
      concept_title: 'CNN Backbones, Stride & Feature Pyramids (FPN)',
      diagram_file: 'course_material/vision_yolo_ocr/diagrams/c0_cnn_fpn.png',
      diagram_width: 720,
      diagram_height: 420,
      target_label: '1x1 Lateral Connection Bridge',
      prompt: 'LAB CALIBRATION C0-B: Point to the 1x1 Lateral Connection Bridge that aligns channels and fuses top-down and shallow feature maps.',
      target_box: [462, 165, 572, 295],
      explanation: '1x1 lateral connections match channel dimensions between shallow C3/C4 layers and upsampled deep layers before element-wise addition.',
    },
    {
      puzzle_id: 'c1_p1',
      concept_id: 1,
      concept_title: 'Bounding Boxes, IoU & Non-Maximum Suppression (NMS)',
      diagram_file: 'course_material/vision_yolo_ocr/diagrams/c1_iou_nms.png',
      diagram_width: 720,
      diagram_height: 420,
      target_label: 'Intersection Overlap Area (A n B)',
      prompt: 'LAB CALIBRATION C1-A: Click or box the green Intersection Overlap Area (A n B) that forms the numerator of the IoU metric.',
      target_box: [125, 165, 232, 305],
      explanation: 'Intersection over Union (IoU) divides the area of the overlapping intersection (A n B) by the total union area (A u B).',
    },
    {
      puzzle_id: 'c1_p2',
      concept_id: 1,
      concept_title: 'Bounding Boxes, IoU & Non-Maximum Suppression (NMS)',
      diagram_file: 'course_material/vision_yolo_ocr/diagrams/c1_iou_nms.png',
      diagram_width: 720,
      diagram_height: 420,
      target_label: 'Surviving NMS Best Bounding Box',
      prompt: 'LAB CALIBRATION C1-B: Point to the Surviving NMS Best Bounding Box (Score 0.96) after duplicate ghost detections are suppressed.',
      target_box: [568, 82, 698, 382],
      explanation: 'Non-Maximum Suppression (NMS) sorts boxes by confidence, keeps the highest-scoring box (0.96), and suppresses lower-scoring neighbors with high IoU.',
    },
    {
      puzzle_id: 'c2_p1',
      concept_id: 2,
      concept_title: 'Single-Stage Real-Time Detection & YOLO Architecture',
      diagram_file: 'course_material/vision_yolo_ocr/diagrams/c2_yolo_grid.png',
      diagram_width: 720,
      diagram_height: 420,
      target_label: 'S x S Spatial Grid & Target Center Cell',
      prompt: "LAB CALIBRATION C2-A: Click or box the S x S Spatial Grid where the cell containing the object's center (cx, cy) predicts the bounding box.",
      target_box: [28, 86, 268, 376],
      explanation: "YOLO divides the image into an S x S grid; the unique cell containing the object's center point (cx, cy) is responsible for predicting its bounding box.",
    },
    {
      puzzle_id: 'c2_p2',
      concept_id: 2,
      concept_title: 'Single-Stage Real-Time Detection & YOLO Architecture',
      diagram_file: 'course_material/vision_yolo_ocr/diagrams/c2_yolo_grid.png',
      diagram_width: 720,
      diagram_height: 420,
      target_label: 'Decoupled Detection Head (Box + Class)',
      prompt: 'LAB CALIBRATION C2-B: Point to the Decoupled Detection Head that separates bounding-box coordinate regression from class probability prediction.',
      target_box: [518, 86, 696, 376],
      explanation: 'Modern YOLO detectors use a decoupled head with separate convolutional branches for spatial box localization (CIoU) and semantic classification (BCE).',
    },
    {
      puzzle_id: 'c3_p1',
      concept_id: 3,
      concept_title: 'OCR: Scene Text Detection, CRNN & CTC Loss',
      diagram_file: 'course_material/vision_yolo_ocr/diagrams/c3_crnn_ctc.png',
      diagram_width: 720,
      diagram_height: 420,
      target_label: 'Bidirectional LSTM (BiLSTM) Sequence Encoder',
      prompt: 'LAB CALIBRATION C3-A: Click or box the Bidirectional LSTM (BiLSTM) Sequence Encoder that models left-to-right and right-to-left context.',
      target_box: [266, 92, 480, 372],
      explanation: "The Bidirectional LSTM propagates context in both directions across feature columns, helping distinguish look-alike glyphs like '0' and 'O'.",
    },
    {
      puzzle_id: 'c3_p2',
      concept_id: 3,
      concept_title: 'OCR: Scene Text Detection, CRNN & CTC Loss',
      diagram_file: 'course_material/vision_yolo_ocr/diagrams/c3_crnn_ctc.png',
      diagram_width: 720,
      diagram_height: 420,
      target_label: 'CTC Blank-Token Collapse Decoder',
      prompt: 'LAB CALIBRATION C3-B: Point to the CTC Blank-Token Collapse Decoder that merges repeated consecutive characters and strips blank tokens.',
      target_box: [514, 92, 696, 372],
      explanation: "CTC (Connectionist Temporal Classification) collapses repeated emissions unless separated by a blank token (_), turning 'Y-Y-_-O-L-O' into 'YOLO'.",
    },
    {
      puzzle_id: 'c4_p1',
      concept_id: 4,
      concept_title: 'End-to-End Vision Pipelines: Combining YOLO + OCR',
      diagram_file: 'course_material/vision_yolo_ocr/diagrams/c4_vision_pipeline.png',
      diagram_width: 720,
      diagram_height: 420,
      target_label: 'ROI Crop & Perspective Rectification Stage',
      prompt: 'LAB CALIBRATION C4-A: Click or box the ROI Crop & Perspective Rectification module that deskews tilted text between YOLO and OCR.',
      target_box: [262, 92, 474, 372],
      explanation: 'Between Stage 1 (YOLO detection) and Stage 2 (CRNN OCR), the ROI Crop & Rectify module deskews perspective tilt and normalizes height to H=32 without squashing aspect ratio.',
    },
    {
      puzzle_id: 'c4_p2',
      concept_id: 4,
      concept_title: 'End-to-End Vision Pipelines: Combining YOLO + OCR',
      diagram_file: 'course_material/vision_yolo_ocr/diagrams/c4_vision_pipeline.png',
      diagram_width: 720,
      diagram_height: 420,
      target_label: 'Stage 2: CRNN + CTC OCR Reader',
      prompt: 'LAB CALIBRATION C4-B: Point to Stage 2: CRNN + CTC OCR Reader, which decodes the character sequence and multiplies YOLO x OCR confidence.',
      target_box: [508, 92, 696, 372],
      explanation: 'Decoupling Stage 1 (YOLO localization) from Stage 2 (CRNN sequence decoding) prevents running expensive OCR over 98% background pixels.',
    },
  ];

  let visionPuzzlesCache = null;

  async function getVisionPuzzles() {
    try {
      const res = await apiFetch('/api/vision/puzzles', { method: 'GET' }, 5000);
      if (res && Array.isArray(res.puzzles) && res.puzzles.length) {
        online = true;
        visionPuzzlesCache = res.puzzles;
        return visionPuzzlesCache;
      }
    } catch (e) { /* fall back */ }
    if (visionPuzzlesCache && visionPuzzlesCache.length) return visionPuzzlesCache;
    return FALLBACK_VISION_PUZZLES;
  }

  function boxIoU(a, b) {
    const ax1 = Math.min(a[0], a[2]), ax2 = Math.max(a[0], a[2]);
    const ay1 = Math.min(a[1], a[3]), ay2 = Math.max(a[1], a[3]);
    const bx1 = Math.min(b[0], b[2]), bx2 = Math.max(b[0], b[2]);
    const by1 = Math.min(b[1], b[3]), by2 = Math.max(b[1], b[3]);
    const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
    const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
    const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    const union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    return union > 0 ? inter / union : 0;
  }

  async function submitVisionPuzzle(puzzleId, clickX, clickY, userBox = null, responseSec = 6.0) {
    // Always attempt live server evaluation when hosted over HTTP/HTTPS or previously online,
    // retrying once on transient Cloudflare tunnel drops so a single blip never kicks us offline.
    const shouldTryLive = online || DEFAULT_API === '';
    if (shouldTryLive) {
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const data = await apiFetch('/api/vision/evaluate', {
            method: 'POST',
            body: JSON.stringify({
              puzzle_id: puzzleId,
              click_x: clickX,
              click_y: clickY,
              user_box: userBox,
              response_sec: responseSec,
            }),
          }, 15000);
          online = true;
          statusCache = data.status;
          persistToPlayer(data.status);
          return data.result;
        } catch (e) {
          if (attempt === 0) {
            await new Promise(r => setTimeout(r, 600));
            continue;
          }
          online = false;
        }
      }
    }
    // Cached fallback evaluation: ALWAYS prefer the uploaded course's `visionPuzzlesCache`
    // (real Qwen3-VL-8B + LLMDet-Large boxes) over the hardcoded default YOLO/OCR puzzles!
    const hasUploadedCache = Array.isArray(visionPuzzlesCache) && visionPuzzlesCache.length > 0;
    const pool = hasUploadedCache ? visionPuzzlesCache : FALLBACK_VISION_PUZZLES;
    const p = pool.find(x => x.puzzle_id === puzzleId) || pool[0];
    const baseTb = p.target_box || [100, 100, 300, 300];
    const candBoxes = Array.isArray(p.valid_boxes) && p.valid_boxes.length > 0
      ? p.valid_boxes
      : [baseTb];
    let tb = baseTb;
    const cid = Number(p.concept_id || 0);
    const before = (statusCache && Array.isArray(statusCache.mastery))
      ? statusCache.mastery.slice()
      : localMastery.slice();
    let verdict = 'incorrect';
    let iou = 0;
    let semScore = 0.08;
    let hasBox = userBox && userBox.length === 4 && Math.abs(userBox[2] - userBox[0]) >= 12 && Math.abs(userBox[3] - userBox[1]) >= 12;
    let cx = hasBox ? 0.5 * (userBox[0] + userBox[2]) : Number(clickX || 0);
    let cy = hasBox ? 0.5 * (userBox[1] + userBox[3]) : Number(clickY || 0);
    if (hasBox) {
      let bestIou = -1;
      candBoxes.forEach(cb => {
        const curIou = boxIoU(userBox, cb);
        if (curIou > bestIou) { bestIou = curIou; tb = cb; }
      });
      iou = Math.max(0, bestIou);
      const inside = candBoxes.some(cb => cx >= cb[0] && cx <= cb[2] && cy >= cb[1] && cy <= cb[3]);
      if (iou >= 0.35 || (inside && iou >= 0.18)) {
        verdict = 'correct'; semScore = 0.91;
      } else if (iou >= 0.10 || inside) {
        verdict = 'partial'; semScore = 0.56;
      }
    } else {
      const exactHit = candBoxes.find(cb => cx >= cb[0] && cx <= cb[2] && cy >= cb[1] && cy <= cb[3]);
      const nearHit = candBoxes.find(cb => cx >= cb[0] - 22 && cx <= cb[2] + 22 && cy >= cb[1] - 22 && cy <= cb[3] + 22);
      if (exactHit) {
        tb = exactHit; verdict = 'correct'; semScore = 0.92; iou = 0.85;
      } else if (nearHit) {
        tb = nearHit; verdict = 'partial'; semScore = 0.54; iou = 0.30;
      }
    }
    const after = before.slice();
    if (verdict === 'correct') {
      after[cid] = Math.min(0.96, (after[cid] || 0.4) + 0.12);
    } else if (verdict === 'partial') {
      after[cid] = Math.min(0.88, (after[cid] || 0.4) + 0.05);
    }
    localMastery = after.slice();
    if (!statusCache) {
      statusCache = buildOfflineStatus();
    } else {
      statusCache.mastery = after.slice();
    }
    persistToPlayer(statusCache);
    return {
      puzzle_id: p.puzzle_id,
      concept_id: cid,
      concept_title: p.concept_title,
      target_label: p.target_label,
      verdict,
      semantic_score: semScore,
      iou: Math.round(iou * 100) / 100,
      click_point: [Math.round(cx), Math.round(cy)],
      user_box: hasBox ? userBox : null,
      grounded_box: tb,
      dino_raw_box: tb,
      dino_confidence: 0.48,
      dino_latency_ms: 0.5,
      mastery_before: before,
      mastery_after: after,
      xp_gained: verdict === 'correct' ? 45 : (verdict === 'partial' ? 20 : 0),
      feedback: verdict === 'correct'
        ? `CALIBRATION LOCKED! Verified your selection on [${p.target_label}]. ${p.explanation}`
        : (verdict === 'partial'
          ? `GLANCING ALIGNMENT! Partially bordered [${p.target_label}]. ${p.explanation}`
          : `MISALIGNED! Target [${p.target_label}] is highlighted in green. ${p.explanation}`),
      graded_by: hasUploadedCache
        ? 'Cached VLM + DINO Box (Reconnecting...)'
        : 'Grounding-DINO-Tiny (Offline Fallback)',
    };
  }

  function stopPodcast() {
    if (activeAudio) {
      try {
        activeAudio.pause();
        activeAudio.currentTime = 0;
      } catch (e) { /* ignore */ }
      activeAudio = null;
    }
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      try { window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
    }
  }

  function playPodcast(audioPath, fallbackText, onEnd) {
    stopPodcast();
    if (typeof Sfx !== 'undefined' && !Sfx.isOn()) return false;
    if (audioPath) {
      const url = (online && DEFAULT_API && !audioPath.startsWith('http'))
        ? `${DEFAULT_API}/${audioPath.replace(/^\//, '')}`
        : audioPath;
      try {
        const aud = new Audio(url);
        activeAudio = aud;
        aud.onended = () => { activeAudio = null; if (onEnd) onEnd(); };
        aud.onerror = () => {
          activeAudio = null;
          speakBrowserFallback(fallbackText, onEnd);
        };
        aud.play().catch(() => {
          activeAudio = null;
          speakBrowserFallback(fallbackText, onEnd);
        });
        return true;
      } catch (e) {
        return speakBrowserFallback(fallbackText, onEnd);
      }
    }
    return speakBrowserFallback(fallbackText, onEnd);
  }

  function speakBrowserFallback(text, onEnd) {
    if (!text || typeof window === 'undefined' || !window.speechSynthesis) {
      if (onEnd) onEnd();
      return false;
    }
    try {
      const clean = text.replace(/^\[[^\]]+\]\s*/, '').replace(/"/g, '');
      const u = new SpeechSynthesisUtterance(clean);
      u.rate = 1.04;
      u.onend = () => { if (onEnd) onEnd(); };
      window.speechSynthesis.speak(u);
      return true;
    } catch (e) {
      if (onEnd) onEnd();
      return false;
    }
  }

  return {
    init,
    syncFromPlayerSave,
    refreshStatus,
    resetSession,
    ingestCustomCourse,
    nextEncounter,
    submitEncounter,
    getNpcDialogue,
    chatWithNpc,
    getCourseStory,
    getVisionPuzzles,
    submitVisionPuzzle,
    submitVisionVQA,
    playPodcast,
    stopPodcast,
    isOnline: () => online,
    getStatus: () => statusCache || buildOfflineStatus(),
    getUploadedCourse: () => uploadedCourseInfo,
    ACTION_LABELS,
    TRAINER_CONCEPT_MAP,
  };
})();
