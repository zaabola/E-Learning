// Dev-only data validation (node validate.js). Not loaded by the game.
const fs = require('fs');
const path = require('path');
global.document = { createElement: () => ({ getContext: () => ({ fillRect(){}, drawImage(){}, strokeRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, fill(){}, ellipse(){}, arc(){}, save(){}, restore(){}, translate(){}, rotate(){}, fillText(){} }), width: 0, height: 0 }) };
global.window = {};
global.Image = function () {};

let src = '';
for (const f of ['data/types.js', 'data/moves.js', 'data/species.js', 'data/items.js', 'data/maps.js', 'data/sprites.js', 'data/assets.js', 'ai_tutor.js']) {
  src += fs.readFileSync(path.join(__dirname, 'js', f), 'utf8') + '\n';
}
// Run data files + validation body in one scope so consts are visible.
eval(src + '\n;(' + validate.toString() + ')();');

function validate() {
let errors = 0;
const err = m => { console.log('ERROR:', m); errors++; };

// 1. Map rows: consistent width, known tile chars
const KNOWN = new Set(['.', ':', 't', 'T', 'W', 'F', 'f', 'r', 'w', 'D', 's', '_', 'k', 'm', 'b', 'O', 'B', 'h',
  'R', 'M', 'c', 'g', 'H', 'G', 'n', '1', '2', '3', 'x']);
for (const [id, m] of Object.entries(MAPS)) {
  const w = m.rows[0].length;
  m.rows.forEach((r, y) => {
    if (r.length !== w) err(`${id} row ${y} width ${r.length} != ${w}: "${r}"`);
    for (const ch of r) if (!KNOWN.has(ch)) err(`${id} row ${y} unknown tile '${ch}'`);
  });
  // warps reference valid maps and land on walkable, in-bounds tiles
  for (const [key, wp] of Object.entries(m.warps || {})) {
    const [x, y] = key.split(',').map(Number);
    if (!['D', 'm', 'h'].includes(m.rows[y]?.[x])) err(`${id} warp at ${key} is on '${m.rows[y]?.[x]}'`);
    const t = MAPS[wp.map];
    if (!t) { err(`${id} warp ${key} -> missing map ${wp.map}`); continue; }
    const ch = t.rows[wp.y]?.[wp.x];
    if (ch === undefined) err(`${id} warp ${key} -> out of bounds ${wp.map} ${wp.x},${wp.y}`);
    else if (SOLID_TILES.has(ch)) err(`${id} warp ${key} -> lands on solid '${ch}' in ${wp.map}`);
  }
  for (const [dir, target] of Object.entries(m.links || {})) {
    if (!MAPS[target]) err(`${id} link ${dir} -> missing map ${target}`);
  }
  // signs on solid sign tiles
  for (const key of Object.keys(m.signs || {})) {
    const [x, y] = key.split(',').map(Number);
    if (m.rows[y]?.[x] !== 's') err(`${id} sign at ${key} is on '${m.rows[y]?.[x]}'`);
  }
  // npcs stand on walkable tiles
  for (const n of (m.npcs || [])) {
    const ch = m.rows[n.y]?.[n.x];
    if (ch === undefined || SOLID_TILES.has(ch)) err(`${id} npc ${n.id} on solid/missing tile '${ch}'`);
    if (n.trainer) for (const [sp] of n.trainer.party) if (!SPECIES[sp]) err(`${id} trainer ${n.id} unknown species ${sp}`);
  }
  // encounter tables
  if (m.encounters) for (const [sp] of m.encounters.table) if (!SPECIES[sp]) err(`${id} encounter unknown species ${sp}`);
}

// 2. Species: learnset moves exist, evolutions exist, sprite art exists, types exist
for (const [id, sp] of Object.entries(SPECIES)) {
  for (const e of sp.learnset) if (!MOVES[e.move]) err(`${id} learnset unknown move ${e.move}`);
  if (sp.evolve && !SPECIES[sp.evolve.to]) err(`${id} evolves to unknown ${sp.evolve.to}`);
  for (const t of sp.types) if (!TYPES[t]) err(`${id} unknown type ${t}`);
  if (!CREATURE_ART[id]) err(`${id} has no sprite art`);
  if (sp.learnset.filter(e => e.lv === 1).length === 0) err(`${id} has no level-1 moves`);
}
for (const id of Object.keys(CREATURE_ART)) if (!SPECIES[id]) err(`art for unknown species ${id}`);

// 3. Sprite art rows: 16 rows of 8 chars (or 16 chars when wide: true)
for (const [id, a] of Object.entries(CREATURE_ART)) {
  const want = a.wide ? 16 : 8;
  if (a.rows.length !== 16) err(`${id} art has ${a.rows.length} rows`);
  a.rows.forEach((r, i) => { if (r.length !== want) err(`${id} art row ${i} length ${r.length} != ${want}: "${r}"`); });
}

// 4. Moves reference valid types; type chart rows valid
for (const [id, mv] of Object.entries(MOVES)) {
  if (!TYPES[mv.type]) err(`move ${id} unknown type ${mv.type}`);
  if (mv.effect && mv.effect.status && !['PSN','BRN','PAR','SLP','FRZ'].includes(mv.effect.status)) err(`move ${id} bad status`);
}
for (const [atk, row] of Object.entries(TYPE_CHART)) {
  if (!TYPES[atk]) err(`chart unknown attacker ${atk}`);
  for (const d of Object.keys(row)) if (!TYPES[d]) err(`chart unknown defender ${d} (under ${atk})`);
}

// 5. Shop stock items exist
for (const id of SHOP_STOCK) if (!ITEMS[id]) err(`shop stock unknown item ${id}`);

// 6. makeCreature smoke test for every species
for (const id of Object.keys(SPECIES)) {
  const c = makeCreature(id, 10);
  if (!(c.maxHp > 10 && c.moves.length >= 1 && c.moves.length <= 4)) err(`makeCreature(${id}) bad: hp=${c.maxHp} moves=${c.moves.length}`);
}

// 7. KnowledgeQuest Computer Vision (YOLO & OCR) course + AITutor integrity
const courseFile = path.join(__dirname, 'course_material', 'vision_yolo_ocr', 'course_curriculum.json');
if (!fs.existsSync(courseFile)) {
  err('missing course_material/vision_yolo_ocr/course_curriculum.json');
} else {
  const pkg = JSON.parse(fs.readFileSync(courseFile, 'utf8'));
  if (!Array.isArray(pkg.concepts) || pkg.concepts.length !== 5) err('course_curriculum.json must have 5 concepts');
  pkg.concepts.forEach((c, idx) => {
    if (c.concept_id !== idx) err(`concept ${idx} bad concept_id ${c.concept_id}`);
    for (const p of (c.prereqs || [])) if (p < 0 || p >= 5 || p === idx) err(`concept ${idx} bad prereq ${p}`);
    if (!c.podcast_audio || !fs.existsSync(path.join(__dirname, c.podcast_audio))) {
      err(`concept ${idx} missing podcast audio ${c.podcast_audio}`);
    }
    for (const tier of ['easy_mc', 'medium_mc']) {
      if (!Array.isArray(c[tier]) || c[tier].length === 0) err(`concept ${idx} empty ${tier}`);
      for (const q of c[tier]) {
        if (!q.question || !q.reference) err(`concept ${idx} ${tier} missing question/reference`);
        if (!Array.isArray(q.choices) || q.choices.length !== 4) err(`concept ${idx} ${tier} must have 4 choices`);
        if (typeof q.correct_idx !== 'number' || q.correct_idx < 0 || q.correct_idx >= 4) err(`concept ${idx} ${tier} bad correct_idx`);
      }
    }
    if (!Array.isArray(c.boss_open) || c.boss_open.length === 0) err(`concept ${idx} empty boss_open`);
    for (const q of c.boss_open) {
      if (!q.question || !q.reference) err(`concept ${idx} boss_open missing question/reference`);
    }
  });
}
if (typeof AITutor === 'undefined' || typeof AITutor.getStatus !== 'function') {
  err('AITutor global missing or invalid');
} else {
  const st = AITutor.getStatus();
  if (!st || !Array.isArray(st.concepts) || st.concepts.length !== 5) err('AITutor offline fallback concepts invalid');
}

// 8. GBA FireRed battle sprites, arenas, trainers & FX PNG magic-byte integrity
const PNG_MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
function checkPng(rel) {
  const full = path.join(__dirname, rel);
  if (!fs.existsSync(full)) { err(`missing PNG asset ${rel}`); return; }
  const buf = fs.readFileSync(full);
  if (buf.length < 24 || !buf.subarray(0, 8).equals(PNG_MAGIC)) err(`invalid PNG header ${rel}`);
}
for (const arena of ['grass', 'lab', 'gym']) {
  checkPng(`assets/battle/back_${arena}.png`);
  checkPng(`assets/battle/sky_${arena}.png`);
  checkPng(`assets/battle/pad_${arena}_enemy.png`);
  checkPng(`assets/battle/pad_${arena}_player.png`);
}
checkPng('assets/trainers/player_back.png');
checkPng('assets/trainers/player_back_throw.png');
for (const tf of Object.values(GameAssets.TRAINER_SPRITE_FILE)) {
  checkPng(`assets/trainers/${tf}.png`);
}
for (const fn of GameAssets.BATTLE_FX_FILES) {
  checkPng(`assets/battle_fx/${fn}.png`);
}
for (const n of Object.values(DEX_NUM)) {
  for (const dir of ['front', 'front-shiny', 'back', 'back-shiny']) {
    checkPng(`assets/sprites/${dir}/${n}.png`);
  }
}

console.log(errors === 0 ? 'ALL CHECKS PASSED' : `${errors} ERRORS`);
process.exit(errors ? 1 : 0);
}
