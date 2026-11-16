// ===== Local image assets (downloaded by download-assets.js / download-overworld.js) =====
// The game must run from file:// — everything is local, no hotlinking.
// Every accessor degrades gracefully to the code-drawn sprites if an image
// is missing or fails to load.

const DEX_NUM = {
  bulbasaur: 1, ivysaur: 2, venusaur: 3,
  charmander: 4, charmeleon: 5, charizard: 6,
  squirtle: 7, wartortle: 8, blastoise: 9,
  caterpie: 10, metapod: 11, butterfree: 12,
  weedle: 13, kakuna: 14, beedrill: 15,
  pidgey: 16, pidgeotto: 17, pidgeot: 18,
  rattata: 19, raticate: 20,
};

const ITEM_ICON_FILE = {
  pokeball: 'poke-ball', greatball: 'great-ball',
  potion: 'potion', superpotion: 'super-potion',
  antidote: 'antidote', paralyzeheal: 'paralyze-heal',
  awakening: 'awakening', burnheal: 'burn-heal',
  iceheal: 'ice-heal', fullheal: 'full-heal',
  lab_ball: 'lab_ball',
};

// ---------- Overworld tileset slicing (Pokemon-Obsidian sheets) ----------
// Source tiles are 32px, drawn onto the 16px logical grid (2:1 downscale).
// Entry: s = sheet key, x/y/w/h = source rect, base = tile drawn underneath
// (for transparent / tall art), ball = pedestal ball-icon overlay.
// Entries taller than 32px extend upward over the tile above (trees, shelves).
const TILE_SRC_OUT = {
  '.': { s: 'outdoor', x: 0, y: 0, w: 32, h: 32 },
  ':': { s: 'outdoor', x: 32, y: 96, w: 32, h: 32 },
  't': { s: 'outdoor', x: 0, y: 2880, w: 32, h: 32, base: '.' },
  'T': { s: 'outdoor', x: 192, y: 32, w: 32, h: 64, base: '.' },
  'W': { s: 'outdoor', x: 32, y: 1056, w: 32, h: 32 },
  'F': { s: 'flowers', x: 0, y: 0, w: 32, h: 32, base: '.' },
  'f': { s: 'outdoor', x: 32, y: 768, w: 32, h: 32, base: '.' },
  's': { s: 'outdoor', x: 128, y: 96, w: 32, h: 32, base: '.' },
};
const TILE_SRC_IN = {
  '_': { s: 'indoor', x: 0, y: 1056, w: 32, h: 32 },
  'w': { s: 'indoor', x: 0, y: 64, w: 32, h: 32 },
  'k': { s: 'indoor', x: 96, y: 1192, w: 32, h: 32 },
  'm': { s: 'indoor', x: 8, y: 928, w: 32, h: 32, base: '_' },
  'B': { s: 'indoor', x: 96, y: 480, w: 32, h: 48, base: '_' },
  'h': { s: 'indoor', x: 208, y: 680, w: 32, h: 32, base: '_' },
  'b': { s: 'indoor', x: 0, y: 608, w: 32, h: 64, base: '_' },
  'G': { s: 'indoor', x: 32, y: 608, w: 32, h: 64, base: '_' },
  'H': { s: 'indoor', x: 176, y: 1232, w: 32, h: 48, base: '_' },
  'n': { s: 'indoor', x: 32, y: 1280, w: 32, h: 32, base: '_' },
  'O': { s: 'outdoor', x: 224, y: 1088, w: 32, h: 32, base: '_' },
  '1': { s: 'indoor', x: 0, y: 400, w: 32, h: 32, base: '_', ball: true },
  '2': { s: 'indoor', x: 0, y: 400, w: 32, h: 32, base: '_', ball: true },
  '3': { s: 'indoor', x: 0, y: 400, w: 32, h: 32, base: '_', ball: true },
};

// Opaque body of each building image (measured; the files carry transparent
// shadow padding right/bottom). All anchoring and collision use the BODY so
// buildings sit flush on their door row with no invisible walls.
const BUILDING_BODY = {
  house:        { x: 0, y: 9,  w: 80, h: 89 },
  house_player: { x: 0, y: 0,  w: 80, h: 72 },
  house_rival:  { x: 0, y: 0,  w: 80, h: 72 },
  lab:          { x: 0, y: 0,  w: 112, h: 72 },
  center:       { x: 0, y: 12, w: 64, h: 87 },
  mart:         { x: 0, y: 4,  w: 64, h: 79 },
};

// Whole-building images drawn over grassed footprints (tile rects).
// The body is anchored bottom-center on the footprint; every tile the body
// covers by >=8px becomes solid (except the 'D' door tile).
const MAP_DECALS = {
  hometown: [
    { img: 'house_player', x0: 5, y0: 4, x1: 9, y1: 7 },    // player's house
    { img: 'house_rival',  x0: 14, y0: 4, x1: 18, y1: 7 },  // Rex's house
    { img: 'lab',          x0: 13, y0: 9, x1: 19, y1: 13 }, // Ada's lab
  ],
  stonegate: [
    { img: 'house', x0: 5, y0: 5, x1: 11, y1: 7 },     // gym
    { img: 'center', x0: 2, y0: 11, x1: 6, y1: 13 },   // healing center
    { img: 'mart', x0: 13, y0: 11, x1: 17, y1: 13 },   // mart
  ],
};

// RMXP-style charsets: 4 frames per row, rows = down / left / right / up.
const CHAR_ROW = { down: 0, left: 1, right: 2, up: 3 };
const CHAR_FILES = ['player', 'parent', 'prof', 'rival', 'leader', 'hiker',
  'clerk', 'nurse', 'scout', 'picnic', 'villager', 'villager2', 'rival_mom'];

const GameAssets = {
  // Lazily populated caches — an Image is created the first time something
  // actually needs it (current map's sheets, on-screen NPC kinds, the
  // species in the active battle...), never all at once at boot.
  front: {}, back: {}, items: {},
  sheets: {}, chars: {}, buildings: {}, fx: {}, battleBg: null,

  TRAINER_SPRITE_FILE: {
    tr_rex:   'rival_early',
    tr_cal:   'scout',
    tr_mira:  'picnic',
    tr_rocco: 'hiker',
    tr_mason: 'leader_mason',
  },

  STAT_WAVE_INDEX: {
    atk: 0, def: 1, acc: 2, eva: 3, spe: 4, spa: 5, spd: 6, mix: 7,
  },

  TYPE_ICON_RECT: {
    NORMAL:   { x: 0,  y: 2,  w: 32, h: 12 },
    FIGHTING: { x: 32, y: 2,  w: 32, h: 12 },
    FLYING:   { x: 64, y: 2,  w: 32, h: 12 },
    POISON:   { x: 96, y: 2,  w: 32, h: 12 },
    GROUND:   { x: 0,  y: 18, w: 32, h: 12 },
    ROCK:     { x: 32, y: 18, w: 32, h: 12 },
    BUG:      { x: 64, y: 18, w: 32, h: 12 },
    GHOST:    { x: 96, y: 18, w: 32, h: 12 },
    STEEL:    { x: 0,  y: 34, w: 32, h: 12 },
    FIRE:     { x: 64, y: 34, w: 32, h: 12 },
    WATER:    { x: 96, y: 34, w: 32, h: 12 },
    GRASS:    { x: 0,  y: 50, w: 32, h: 12 },
    ELECTRIC: { x: 32, y: 50, w: 32, h: 12 },
    PSYCHIC:  { x: 64, y: 50, w: 32, h: 12 },
    ICE:      { x: 96, y: 50, w: 32, h: 12 },
    DRAGON:   { x: 0,  y: 66, w: 32, h: 12 },
    DARK:     { x: 32, y: 66, w: 32, h: 12 },
  },

  STATUS_ICON_RECT: {
    PSN: { x: 6,   y: 80, w: 20, h: 8 },
    PAR: { x: 38,  y: 80, w: 20, h: 8 },
    SLP: { x: 70,  y: 80, w: 20, h: 8 },
    FRZ: { x: 102, y: 80, w: 20, h: 8 },
    BRN: { x: 6,   y: 88, w: 20, h: 8 },
    FNT: { x: 70,  y: 88, w: 20, h: 8 },
  },

  BATTLE_FX_FILES: [
    'pokeballs', 'stat_waves', 'type_status_icons', 'hud_party_balls',
    'fx_hit', 'fx_slash', 'fx_claw', 'fx_claw_metal', 'fx_fangs',
    'fx_fire', 'fx_fireball', 'fx_water', 'fx_bubble',
    'fx_vine', 'fx_leaf', 'fx_seed', 'fx_tornado', 'fx_wind',
    'fx_web', 'fx_needle', 'fx_poison', 'fx_psychic', 'fx_status', 'fx_scaryface',
  ],

  init() { /* nothing eager — everything loads on demand */ },

  _lazy(store, key, src) {
    let img = store[key];
    if (!img) {
      img = new Image();
      img.src = src;
      store[key] = img;
    }
    return img;
  },

  _ready(img) { return !!(img && img.complete && img.naturalWidth > 0); },

  // Image or null (caller falls back to the code-drawn sprite).
  // Pass shiny=true for the shiny palette variant.
  frontFor(species, shiny) {
    const n = DEX_NUM[species];
    if (!n) return null;
    const key = shiny ? species + '_s' : species;
    const dir = shiny ? 'front-shiny' : 'front';
    const i = this._lazy(this.front, key, `assets/sprites/${dir}/${n}.png`);
    return this._ready(i) ? i : null;
  },
  backFor(species, shiny) {
    const n = DEX_NUM[species];
    if (!n) return null;
    const key = shiny ? species + '_s' : species;
    const dir = shiny ? 'back-shiny' : 'back';
    const i = this._lazy(this.back, key, `assets/sprites/${dir}/${n}.png`);
    return this._ready(i) ? i : null;
  },
  itemIconUrl(itemId) {
    const file = ITEM_ICON_FILE[itemId];
    return file ? `assets/items/${file}.png` : null;
  },
  itemIconImg(itemId) {
    const file = ITEM_ICON_FILE[itemId];
    if (!file) return null;
    const i = this._lazy(this.items, itemId, `assets/items/${file}.png`);
    return this._ready(i) ? i : null;
  },
  labBallImg() {
    return this.itemIconImg('lab_ball') || this.itemIconImg('pokeball');
  },
  battleArenaKey(mapId) {
    const m = mapId || ((typeof Game !== 'undefined' && Game.player) ? Game.player.map : null);
    if (m === 'gym') return 'gym';
    if (m === 'lab' || m === 'bedroom' || m === 'house' || m === 'rexhouse' || m === 'healstone' || m === 'shop1') {
      return 'lab';
    }
    return 'grass';
  },
  battleBackground(mapId) {
    const key = this.battleArenaKey(mapId);
    const i = this._lazy(this.sheets, 'bg_back_' + key, `assets/battle/back_${key}.png`);
    return this._ready(i) ? i : null;
  },
  battleSky(mapId) {
    const key = this.battleArenaKey(mapId);
    const i = this._lazy(this.sheets, 'bg_sky_' + key, `assets/battle/sky_${key}.png`);
    return this._ready(i) ? i : null;
  },
  battleEnemyPad(mapId) {
    const key = this.battleArenaKey(mapId);
    const i = this._lazy(this.sheets, 'bg_ep_' + key, `assets/battle/pad_${key}_enemy.png`);
    return this._ready(i) ? i : null;
  },
  battlePlayerPad(mapId) {
    const key = this.battleArenaKey(mapId);
    const i = this._lazy(this.sheets, 'bg_pp_' + key, `assets/battle/pad_${key}_player.png`);
    return this._ready(i) ? i : null;
  },
  battleFx(name) {
    const i = this._lazy(this.fx, name, `assets/battle_fx/${name}.png`);
    return this._ready(i) ? i : null;
  },
  mapBackground(mapId) {
    const supported = ['hometown', 'bedroom', 'house', 'rexhouse', 'lab'];
    if (!supported.includes(mapId)) return null;
    const i = this._lazy(this.sheets, 'map_' + mapId, `assets/maps/${mapId}.png`);
    return this._ready(i) ? i : null;
  },
  trainerFront(id) {
    const file = this.TRAINER_SPRITE_FILE[id] || null;
    if (!file) return null;
    const i = this._lazy(this.sheets, 'tr_' + file, `assets/trainers/${file}.png`);
    return this._ready(i) ? i : null;
  },
  trainerBack() {
    const i = this._lazy(this.sheets, 'tr_player_back', 'assets/trainers/player_back.png');
    return this._ready(i) ? i : null;
  },
  trainerBackThrow() {
    const i = this._lazy(this.sheets, 'tr_player_back_throw', 'assets/trainers/player_back_throw.png');
    return this._ready(i) ? i : null;
  },
  _sheet(key) { return this._lazy(this.sheets, key, `assets/overworld/${key}.png`); },
  _building(key) { return this._lazy(this.buildings, key, `assets/overworld/${key}.png`); },

  // ----- tileset tiles -----
  // Draws ch at (dx,dy) on the 16px grid. Returns false to request the
  // code-drawn fallback (sheet not ready or char unmapped).
  drawTile(ctx, ch, dx, dy, indoor) {
    const e = (indoor ? TILE_SRC_IN : TILE_SRC_OUT)[ch];
    if (!e) return false;
    const sheet = this._sheet(e.s);
    if (!this._ready(sheet)) return false;
    if (e.base && !this.drawTile(ctx, e.base, dx, dy, indoor)) return false;
    const dh = e.h / 2;
    ctx.drawImage(sheet, e.x, e.y, e.w, e.h, dx, dy + 16 - dh, 16, dh);
    if (e.ball) {
      const ball = this.itemIconImg('pokeball');
      if (ball) ctx.drawImage(ball, dx + 3, dy - 4, 10, 10);
    }
    return true;
  },

  // ----- building decals -----
  _decalAt(mapId, x, y) {
    return (MAP_DECALS[mapId] || []).find(d =>
      x >= d.x0 && x <= d.x1 && y >= d.y0 && y <= d.y1) || null;
  },

  // True when a ready building image hides this tile (renderer draws grass).
  tileCovered(mapId, x, y) {
    const d = this._decalAt(mapId, x, y);
    return !!(d && this._ready(this._building(d.img)));
  },

  // Tile span the building BODY occupies (measured opaque bounds — the
  // files carry transparent shadow padding that must not block tiles).
  // A tile counts only when the body covers >=8px of it.
  _visualSpan(d) {
    if (!this._ready(this._building(d.img))) return null;
    if (!d._span) {
      const b = BUILDING_BODY[d.img];
      const cx = ((d.x0 + d.x1 + 1) / 2) * 16;
      d._span = {
        x0: Math.floor((cx - b.w / 2 + 8) / 16),
        x1: Math.floor((cx + b.w / 2 - 8) / 16),
        y0: d.y1 - Math.ceil((b.h - 8) / 16) + 1,
        y1: d.y1,
      };
    }
    return d._span;
  },

  // Collision: any tile under a building body is solid (doors are exempted
  // by the caller so warps stay enterable).
  decalSolid(mapId, x, y) {
    for (const d of (MAP_DECALS[mapId] || [])) {
      const s = this._visualSpan(d);
      if (s && x >= s.x0 && x <= s.x1 && y >= s.y0 && y <= s.y1) return true;
    }
    return false;
  },

  drawDecals(ctx, mapId, camX, camY) {
    for (const d of (MAP_DECALS[mapId] || [])) {
      const img = this._building(d.img);
      if (!this._ready(img)) continue;
      const b = BUILDING_BODY[d.img];
      const cx = ((d.x0 + d.x1 + 1) / 2) * 16;
      const bottom = (d.y1 + 1) * 16;
      // body bottom-center sits on the footprint's bottom edge
      ctx.drawImage(img,
        Math.round(cx - (b.x + b.w / 2) - camX),
        bottom - (b.y + b.h) - camY);
    }
  },

  // ----- character walk sheets -----
  // Draws the actor anchored at the feet of tile-pixel (px,py), at NATIVE
  // resolution (source 32px art = 16 logical px on the 2x canvas — crisp,
  // and the same body-to-tile proportions as the source game).
  // frame: 0..3 walk cycle (0 = idle). Returns false for drawn fallback.
  drawActor(ctx, kind, facing, frame, px, py) {
    if (!CHAR_FILES.includes(kind)) return false;
    const img = this._lazy(this.chars, kind, `assets/characters/${kind}.png`);
    if (!this._ready(img)) return false;
    const fw = img.width / 4, fh = img.height / 4;
    const row = CHAR_ROW[facing] !== undefined ? CHAR_ROW[facing] : 0;
    const dw = fw / 2, dh = fh / 2;
    // half-logical-pixel rounding = whole device pixels on the 2x canvas
    const dx = Math.round((px + 8 - dw / 2) * 2) / 2;
    const dy = Math.round((py + 17 - dh) * 2) / 2;
    ctx.drawImage(img, frame * fw, row * fh, fw, fh, dx, dy, dw, dh);
    return true;
  },

  // ----- background warm-up -----
  // Kicks off every download AFTER boot so nothing waits on page load but
  // everything is cached by the time it's needed. Accessors stay lazy-safe.
  preloadAll() {
    for (const species of Object.keys(DEX_NUM)) {
      this.frontFor(species); this.backFor(species);
      this.frontFor(species, true); this.backFor(species, true);
    }
    for (const id of Object.keys(ITEM_ICON_FILE)) this.itemIconImg(id);
    for (const s of ['outdoor', 'indoor', 'flowers']) this._sheet(s);
    for (const b of ['house', 'house_player', 'house_rival', 'lab', 'center', 'mart']) this._building(b);
    for (const kind of CHAR_FILES) this._lazy(this.chars, kind, `assets/characters/${kind}.png`);
    for (const m of ['hometown', 'bedroom', 'house', 'rexhouse', 'lab']) this.mapBackground(m);
    for (const a of ['grass', 'lab', 'gym']) {
      this.battleBackground(a);
      this.battleSky(a);
      this.battleEnemyPad(a);
      this.battlePlayerPad(a);
    }
    for (const tid of Object.keys(this.TRAINER_SPRITE_FILE)) this.trainerFront(tid);
    this.trainerBack();
    this.trainerBackThrow();
    for (const fn of this.BATTLE_FX_FILES) this.battleFx(fn);
  },
};
