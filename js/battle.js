// ===== Battle system (Gen-3 mechanics + Authentic GBA FireRed Animations) =====

const STAGE_NAMES = {
  atk: 'Attack', def: 'Defense', spa: 'Sp. Atk', spd: 'Sp. Def', spe: 'Speed',
  acc: 'accuracy', eva: 'evasiveness',
};

function stageMul(s) { return s >= 0 ? (2 + s) / 2 : 2 / (2 - s); }
function accStageMul(s) { return s >= 0 ? (3 + s) / 3 : 3 / (3 - s); }
function freshStages() { return { atk: 0, def: 0, spa: 0, spd: 0, spe: 0, acc: 0, eva: 0 }; }
function freshVolatiles() { return { flinch: false, confuse: 0, seeded: false }; }

// Gen 3 damage formula: ((2L/5+2) * Power * A/D) / 50 + 2, then
// crit x2, STAB x1.5, type effectiveness, random 85-100%.
function calcDamage(user, target, move, uStages, tStages, crit) {
  const phys = move.cat === 'phys';
  let aStage = uStages[phys ? 'atk' : 'spa'];
  let dStage = tStages[phys ? 'def' : 'spd'];
  if (crit) { aStage = Math.max(0, aStage); dStage = Math.min(0, dStage); }
  let A = Math.floor((phys ? user.atk : user.spa) * stageMul(aStage));
  let D = Math.max(1, Math.floor((phys ? target.def : target.spd) * stageMul(dStage)));
  if (phys && user.status === 'BRN') A = Math.max(1, Math.floor(A / 2));
  let dmg = Math.floor(Math.floor(Math.floor(2 * user.level / 5 + 2) * move.power * A / D) / 50) + 2;
  if (crit) dmg *= 2;
  if (creatureTypes(user).includes(move.type)) dmg = Math.floor(dmg * 1.5);
  dmg = Math.floor(dmg * typeEffectiveness(move.type, creatureTypes(target)));
  dmg = Math.floor(dmg * randInt(85, 100) / 100);
  return Math.max(1, dmg);
}

// Confusion self-hit: 40-power typeless physical against own Defense.
function confusionSelfDamage(c) {
  const dmg = Math.floor(Math.floor(Math.floor(2 * c.level / 5 + 2) * 40 * c.atk / Math.max(1, c.def)) / 50) + 2;
  return Math.max(1, Math.floor(dmg * randInt(85, 100) / 100));
}

// Effective speed for turn order (paralysis quarters it).
function battleSpeed(c, stages) {
  let s = Math.floor(c.spe * stageMul(stages.spe));
  if (c.status === 'PAR') s = Math.floor(s / 4);
  return s;
}

function pickMultiHits(multi) {
  if (multi === 2) return 2;
  // Gen 3 distribution: 2 (3/8), 3 (3/8), 4 (1/8), 5 (1/8)
  const r = randInt(1, 8);
  if (r <= 3) return 2;
  if (r <= 6) return 3;
  if (r === 7) return 4;
  return 5;
}

// Native GBA FireRed battle stage layout (240x112 arena on 240x160 logical screen)
const FR_STAGE = {
  arenaY: 6,
  arenaH: 112,
  pHomeX: 34,
  pHomeY: 48,   // 48 + 64 = 112 (sits flush on bottom of arena!)
  eHomeX: 140,
  eHomeY: 14,   // 14 + 64 = 78 (centered on enemy oval platform!)
};

// Shared 64x64 offscreen composite buffer for GBA stat-wave & energy-tint masking
let _fxBuf = null;
let _fxCtx = null;
function getFxBuffer() {
  if (!_fxBuf && typeof document !== 'undefined') {
    _fxBuf = document.createElement('canvas');
    _fxBuf.width = 64;
    _fxBuf.height = 64;
    _fxCtx = _fxBuf.getContext('2d');
    if (_fxCtx) _fxCtx.imageSmoothingEnabled = false;
  }
  return _fxCtx;
}

const Battle = {
  active: null,
  // opts: { wild: creature } | { trainer: trainerDef }  ->  'win'|'lose'|'ran'|'catch'
  async start(opts) {
    const session = new BattleSession(opts);
    Battle.active = session;
    const result = await session.run();
    Battle.active = null;
    return result;
  },
};

class BattleSession {
  constructor(opts) {
    this.wild = !!opts.wild;
    this.trainer = opts.trainer || null;
    this.party = Game.player.party;
    if (this.wild) {
      this.enemyParty = [opts.wild];
    } else {
      this.enemyParty = this.trainer.party.map(([sp, lv]) => makeCreature(sp, lv));
    }
    this.enemyIdx = 0;
    this.playerIdx = this.party.findIndex(c => c.hp > 0);
    this.pStages = freshStages();
    this.eStages = freshStages();
    this.vol = { player: freshVolatiles(), enemy: freshVolatiles() };
    this.runAttempts = 0;
    this.participants = new Set();
    this.evoQueue = [];
    this.over = false;
    this.result = null;

    // GBA FireRed scene & combatant visual state (64x64 native sprites)
    this.scene = {
      curtain: 0,   // 1 -> 0 horizontal split scanline curtain wipe
      ePadDx: 0,    // enemy oval platform slide offset
      pPadDx: 0,    // player grass/floor platform slide offset
      bgTint: null, // { color, alpha, t0, dur, pulse }
    };
    this.fx = {
      p: {
        x: FR_STAGE.pHomeX, y: FR_STAGE.pHomeY, dx: 0, dy: 0,
        scale: 1, squashX: 1, squashY: 1,
        vis: false, flash: 0, isTrainer: false, trFrame: 0,
        silhouette: 0, tintColor: null, tintAlpha: 0,
        statWave: null, wavy: 0, clipBottom: FR_STAGE.arenaH,
        ghosts: [],
      },
      e: {
        x: FR_STAGE.eHomeX, y: FR_STAGE.eHomeY, dx: 0, dy: 0,
        scale: 1, squashX: 1, squashY: 1,
        vis: false, flash: 0, isTrainer: false,
        silhouette: 0, tintColor: null, tintAlpha: 0,
        statWave: null, wavy: 0, clipBottom: null,
        ghosts: [],
      },
      orb: {
        x: 0, y: 0, vis: false, rot: 0, openState: 0,
        ballRow: 0, locked: false, shadowY: 74,
      },
    };
    this.mapId = (typeof Game !== 'undefined' && Game.player) ? Game.player.map : null;
    this.particles = [];
    this.animSprites = [];
    this.shake = 0;
  }

  get active() { return this.party[this.playerIdx]; }
  get enemy() { return this.enemyParty[this.enemyIdx]; }

  battlerName(side, c) {
    return (side === 'enemy' && this.wild ? 'Wild ' : '') + creatureName(c);
  }

  sideFx(side) { return side === 'player' ? this.fx.p : this.fx.e; }

  // Center coordinates of a combatant in the 240x112 arena space
  combatantCenter(side) {
    const f = this.sideFx(side);
    return {
      cx: f.x + f.dx + 32,
      cy: f.y + f.dy + (side === 'player' ? 38 : 36),
      footY: side === 'player' ? 108 : 74,
    };
  }

  // ---------- Sprite-sheet FX spawners ----------
  addAnimSprite(cfg) {
    this.animSprites.push(Object.assign({
      sheet: 'fx_hit',
      cw: 48, ch: 48, row: 0,
      frames: [0],
      t0: performance.now(),
      dur: 300,
      delay: 0,
      x0: 0, y0: 0, x1: null, y1: null,
      arc: 0,
      s0: 1, s1: null,
      rot0: 0, rot1: null,
      alpha0: 1, alpha1: null,
      flipX: false,
      orbit: null,
    }, cfg));
  }

  flashArena(color, alpha = 0.38, dur = 360) {
    this.scene.bgTint = { color, alpha, t0: performance.now(), dur };
  }

  // Fallback + accent procedural particles (guarantees AGENTS.md Rule #2 fallback)
  impactFx(moveType, targetSide, typeMult) {
    const { cx, cy } = this.combatantCenter(targetSide);
    this.spawnEffect(moveType, cx, cy);
    this.shake = Math.max(this.shake, typeMult > 1 ? 14 : 9);
  }

  spawnEffect(type, cx, cy) {
    const P = this.particles;
    const add = (n, make) => { for (let i = 0; i < n; i++) P.push(make(i)); };
    const r = (a, b) => a + Math.random() * (b - a);
    switch (type) {
      case 'FIRE':
        add(12, () => ({ kind: 'rect', x: cx + r(-12, 12), y: cy + r(-4, 10), vx: r(-0.6, 0.6), vy: r(-1.9, -0.7), g: -0.02,
          life: r(16, 26), age: 0, size: r(2, 3.5), color: ['#f85820', '#f8a830', '#fff070'][randInt(0, 2)] }));
        break;
      case 'WATER': case 'ICE':
        add(12, () => ({ kind: 'rect', x: cx + r(-10, 10), y: cy + r(-10, 2), vx: r(-1.4, 1.4), vy: r(-2.1, -0.6), g: 0.16,
          life: r(18, 28), age: 0, size: r(2, 3), color: ['#4890f8', '#88d0ff', '#ffffff'][randInt(0, 2)] }));
        break;
      case 'ELECTRIC':
        add(10, () => ({ kind: 'streak', x: cx + r(-14, 14), y: cy + r(-14, 14), vx: r(-2.4, 2.4), vy: r(-2.4, 2.4), g: 0,
          life: r(8, 14), age: 0, size: r(4, 8), color: ['#f8d830', '#ffffb0'][randInt(0, 1)] }));
        break;
      case 'GRASS': case 'BUG':
        add(10, () => ({ kind: 'rect', x: cx + r(-14, 14), y: cy + r(-12, 4), vx: r(-0.8, 0.8), vy: r(0.2, 1.0), g: 0.01,
          life: r(18, 30), age: 0, size: r(2, 3), color: ['#58c048', '#90e870', '#d8f868'][randInt(0, 2)] }));
        break;
      case 'POISON':
        add(10, () => ({ kind: 'ring', x: cx + r(-11, 11), y: cy + r(-4, 10), vx: r(-0.3, 0.3), vy: r(-1.1, -0.4), g: 0,
          life: r(18, 28), age: 0, size: r(2, 4.5), color: ['#a840b8', '#e088f0'][randInt(0, 1)] }));
        break;
      case 'PSYCHIC':
        add(4, (i) => ({ kind: 'ring', x: cx, y: cy, vx: 0, vy: 0, g: 0, grow: 1.25,
          life: 16 + i * 4, age: 0, size: 3, color: ['#f85898', '#78d8ff'][i % 2] }));
        break;
      case 'GROUND': case 'ROCK': case 'STEEL':
        add(10, () => ({ kind: 'rect', x: cx + r(-12, 12), y: cy + r(-6, 6), vx: r(-1.5, 1.5), vy: r(-2.4, -0.9), g: 0.2,
          life: r(16, 26), age: 0, size: r(2, 3.5), color: ['#c8a850', '#989088', '#e8e0d0'][randInt(0, 2)] }));
        break;
      case 'CAUGHT_STARS':
        for (let i = 0; i < 3; i++) {
          const vx = (i - 1) * 18;
          this.addAnimSprite({
            sheet: 'pokeballs', cw: 24, ch: 24, row: 2, frames: [0],
            x0: cx, y0: cy, x1: cx + vx, y1: cy + 12, arc: 18,
            s0: 1.0, s1: 0.85, alpha0: 1, alpha1: 0.1, dur: 520,
          });
        }
        add(8, (i) => ({ kind: 'rect', x: cx, y: cy, vx: Math.cos(i * Math.PI / 4) * 1.5, vy: Math.sin(i * Math.PI / 4) * 1.5 - 0.8, g: 0.06,
          life: 24, age: 0, size: 2.5, color: ['#f8d830', '#fff8b8'][i % 2] }));
        break;
      case 'BALL_OPEN_BURST':
        // Authentic FireRed Poke Ball white/magenta light beams + sparkles
        for (let i = 0; i < 6; i++) {
          const ang = (i / 6) * Math.PI * 2;
          this.addAnimSprite({
            sheet: 'pokeballs', cw: 24, ch: 24, row: 2, frames: [1, 2],
            x0: cx, y0: cy,
            x1: cx + Math.cos(ang) * 26, y1: cy + Math.sin(ang) * 22,
            s0: 0.8, s1: 0.4, alpha0: 1, alpha1: 0, dur: 260,
          });
        }
        add(6, (i) => ({ kind: 'ring', x: cx, y: cy, vx: 0, vy: 0, g: 0, grow: 1.9,
          life: 9 + i * 2, age: 0, size: 2, color: i % 2 ? '#ff98f0' : '#ffffff' }));
        break;
      case 'BALL_BREAK':
        this.spawnEffect('BALL_OPEN_BURST', cx, cy);
        break;
      default:
        add(6, () => ({ kind: 'rect', x: cx + r(-10, 10), y: cy + r(-10, 10), vx: r(-1.2, 1.2), vy: r(-1.2, 1.2), g: 0,
          life: r(10, 16), age: 0, size: 2, color: '#fff8e0' }));
    }
  }

  // ---------- Authentic FireRed Entry Hop, Send-Out & Recall Choreography ----------
  async playEntryHop(side) {
    const f = this.sideFx(side);
    // Two brisk GBA hops + squash-and-stretch
    await this.tween(f, 'dy', -7, 75);
    f.squashX = 1.06; f.squashY = 0.94;
    await this.tween(f, 'dy', 0, 75);
    f.squashX = 0.96; f.squashY = 1.05;
    await this.tween(f, 'dy', -4, 65);
    f.squashX = 1; f.squashY = 1;
    await this.tween(f, 'dy', 0, 65);
  }

  async materializeFromBall(side, ballX, ballY, ballRow = 0) {
    const f = this.sideFx(side);
    const orb = this.fx.orb;
    orb.ballRow = ballRow;
    orb.x = ballX;
    orb.y = ballY;
    orb.rot = 0;
    orb.locked = false;
    orb.openState = 1;
    orb.vis = true;
    await wait(55);
    orb.openState = 2; // wide-open!
    this.spawnEffect('BALL_OPEN_BURST', ballX, ballY - 4);
    await wait(65);
    orb.vis = false;
    orb.openState = 0;

    // Pokemon expands from glowing magenta/white energy just like GBA FireRed
    f.isTrainer = false;
    f.vis = true;
    f.dx = 0; f.dy = 0;
    f.scale = 0.12;
    f.tintColor = '#ff88e8';
    f.tintAlpha = 0.92;
    await Promise.all([
      this.tween(f, 'scale', 1.0, 230),
      this.tween(f, 'tintAlpha', 0, 260),
    ]);
    f.tintColor = null;
    await this.playEntryHop(side);
  }

  async recallToBall(side) {
    const f = this.sideFx(side);
    if (!f.vis) return;
    const { cx, footY } = this.combatantCenter(side);
    Sfx.thrown();
    f.tintColor = '#ff70e0';
    f.tintAlpha = 0.15;
    this.spawnEffect('BALL_OPEN_BURST', cx, footY - 14);
    await Promise.all([
      this.tween(f, 'tintAlpha', 0.95, 120),
      this.tween(f, 'scale', 0.08, 220),
    ]);
    f.vis = false;
    f.scale = 1;
    f.tintAlpha = 0;
    f.tintColor = null;
  }

  // 5-frame Red Back-Sprite Throw while sliding left off-screen!
  async playerThrowAndSendOut(firstTurn = false) {
    const p = this.fx.p;
    const hasThrowStrip = !!(GameAssets.trainerBackThrow() || GameAssets.trainerBack());
    if (firstTurn && hasThrowStrip) {
      await UI.say(`Go! ${creatureName(this.active)}!`, { noWait: true, keep: true });
      p.isTrainer = true;
      p.vis = true;
      p.trFrame = 0;
      await wait(90);

      // Step through frames 1 -> 2 -> 3 -> 4 while Red slides left off-screen
      const slidePromise = this.tween(p, 'x', -72, 430);
      p.trFrame = 1; await wait(85);
      p.trFrame = 2; await wait(85);
      p.trFrame = 3; // Arm whipped forward — release Poke Ball!
      Sfx.thrown();
      const orb = this.fx.orb;
      orb.ballRow = 0;
      orb.openState = 0;
      orb.locked = false;
      orb.rot = 0;
      orb.shadowY = 104;
      orb.vis = true;
      const ballArc = this.arcTween(orb, Math.max(12, p.x + 44), 68, FR_STAGE.pHomeX + 32, 96, 260, 28);
      await wait(85);
      p.trFrame = 4;
      await Promise.all([slidePromise, ballArc]);
      p.vis = false;
      p.x = FR_STAGE.pHomeX;
      p.trFrame = 0;
      await this.materializeFromBall('player', FR_STAGE.pHomeX + 32, 96, 0);
      UI.setBox('player', this.active, { slide: true });
      await wait(160);
      UI.hideDialog();
      return;
    }

    // Mid-battle switch send-out: arc Poke Ball from bottom-left into player crest
    await UI.say(`Go! ${creatureName(this.active)}!`, { noWait: true, keep: true });
    Sfx.thrown();
    const orb = this.fx.orb;
    orb.ballRow = 0;
    orb.openState = 0;
    orb.locked = false;
    orb.rot = 0;
    orb.shadowY = 104;
    orb.vis = true;
    await this.arcTween(orb, 8, 92, FR_STAGE.pHomeX + 32, 96, 250, 30);
    p.x = FR_STAGE.pHomeX;
    p.y = FR_STAGE.pHomeY;
    await this.materializeFromBall('player', FR_STAGE.pHomeX + 32, 96, 0);
    UI.setBox('player', this.active, { slide: true });
    await wait(150);
    UI.hideDialog();
  }

  async enemyTrainerSendOut(firstTurn = false) {
    const e = this.fx.e;
    await UI.say(`${this.trainer.name} sent out ${creatureName(this.enemy)}!`, { noWait: true, keep: true });
    const slidePromises = [];
    if (firstTurn && e.isTrainer && e.vis) {
      slidePromises.push(this.tween(e, 'x', 262, 300));
    }
    Sfx.thrown();
    const orb = this.fx.orb;
    orb.ballRow = 0;
    orb.openState = 0;
    orb.locked = false;
    orb.rot = 0;
    orb.shadowY = 74;
    orb.vis = true;
    const startX = firstTurn ? 196 : 238;
    slidePromises.push(this.arcTween(orb, startX, 34, FR_STAGE.eHomeX + 32, 62, 280, 26));
    await Promise.all(slidePromises);
    e.x = FR_STAGE.eHomeX;
    e.y = FR_STAGE.eHomeY;
    e.clipBottom = null;
    await this.materializeFromBall('enemy', FR_STAGE.eHomeX + 32, 62, 0);
    UI.setBox('enemy', this.enemy, { slide: true });
    UI.setEnemyBalls(this.enemyParty);
    await wait(180);
    UI.hideDialog();
  }

  // ---------- Authentic FireRed Move Choreography (All 35 Moves!) ----------
  async playMoveAnim(moveId, move, side, targetSide, hitIdx = 0) {
    const atk = this.sideFx(side);
    const def = this.sideFx(targetSide);
    const dir = side === 'player' ? 1 : -1;
    const flip = side === 'enemy';
    const A = this.combatantCenter(side);
    const B = this.combatantCenter(targetSide);
    const dxAB = B.cx - A.cx;
    const dyAB = B.cy - A.cy;
    const angleAB = Math.atan2(dyAB, dxAB);

    const addHitStar = (cx = B.cx, cy = B.cy, dur = 240, sc = 1.0) => {
      this.addAnimSprite({
        sheet: 'fx_hit', cw: 48, ch: 48, frames: [0, 1, 2, 3],
        x0: cx, y0: cy, s0: sc, s1: sc * 1.08, dur,
      });
    };

    switch (moveId) {
      case 'quickattack': {
        // Ultra-fast darting with GBA translucent speed afterimages!
        atk.ghosts = [{ dx: -10 * dir, dy: 0, alpha: 0.45 }, { dx: -20 * dir, dy: 0, alpha: 0.25 }];
        await this.tween(atk, 'dx', -10 * dir, 55);
        await this.tween(atk, 'dx', 26 * dir, 65);
        addHitStar(B.cx, B.cy, 210, 1.1);
        this.addAnimSprite({
          sheet: 'fx_status', cw: 32, ch: 32, frames: [5],
          x0: B.cx - 10, y0: B.cy - 8, s0: 0.9, s1: 1.25, alpha0: 1, alpha1: 0, dur: 190,
        });
        await this.tween(atk, 'dx', 0, 110);
        atk.ghosts = [];
        break;
      }

      case 'scratch':
      case 'furyattack': {
        const off = (hitIdx % 2 === 0 ? -1 : 1) * 6;
        await this.tween(atk, 'dx', 12 * dir, 70);
        this.tween(atk, 'dx', 0, 120);
        this.addAnimSprite({
          sheet: 'fx_slash', cw: 48, ch: 48, frames: [0, 1, 2, 3, 4],
          x0: B.cx + off, y0: B.cy - 4, flipX: flip, s0: 1.05, dur: 240,
        });
        addHitStar(B.cx - off, B.cy + 4, 210, 0.85);
        await wait(160);
        break;
      }

      case 'slash': {
        this.flashArena('#0c101c', 0.45, 280);
        await this.tween(atk, 'dx', 15 * dir, 75);
        this.tween(atk, 'dx', 0, 130);
        this.addAnimSprite({
          sheet: 'fx_claw', cw: 48, ch: 48, frames: [0, 1, 2, 3, 4],
          x0: B.cx, y0: B.cy - 2, flipX: flip, s0: 1.25, dur: 280,
        });
        this.addAnimSprite({
          sheet: 'fx_slash', cw: 48, ch: 48, frames: [1, 2, 3, 4],
          x0: B.cx + 4, y0: B.cy + 2, flipX: !flip, s0: 1.1, dur: 250, delay: 45,
        });
        await wait(220);
        break;
      }

      case 'metalclaw': {
        this.flashArena('#c8e4ff', 0.30, 260);
        atk.tintColor = '#d8f0ff'; atk.tintAlpha = 0.7;
        await this.tween(atk, 'dx', 14 * dir, 80);
        this.tween(atk, 'tintAlpha', 0, 140);
        this.tween(atk, 'dx', 0, 130);
        this.addAnimSprite({
          sheet: 'fx_claw_metal', cw: 48, ch: 48, frames: [0, 1, 2, 3, 4],
          x0: B.cx, y0: B.cy, flipX: flip, s0: 1.25, dur: 280,
        });
        for (let i = 0; i < 3; i++) {
          this.addAnimSprite({
            sheet: 'fx_status', cw: 32, ch: 32, frames: [5],
            x0: B.cx + (i - 1) * 14, y0: B.cy + (i % 2 ? -10 : 8),
            s0: 0.7, s1: 1.1, alpha0: 1, alpha1: 0, dur: 220, delay: i * 45,
          });
        }
        await wait(230);
        atk.tintColor = null;
        break;
      }

      case 'bite':
      case 'hyperfang':
      case 'superfang': {
        if (moveId === 'bite') this.flashArena('#100820', 0.50, 360);
        const sc = moveId === 'superfang' ? 1.25 : 1.05;
        await this.tween(atk, 'dx', 10 * dir, 70);
        this.tween(atk, 'dx', 0, 130);
        // Upper jaw drops down + lower jaw snaps up!
        this.addAnimSprite({
          sheet: 'fx_fangs', cw: 64, ch: 24, frames: [0],
          x0: B.cx, y0: B.cy - 26, x1: B.cx, y1: B.cy - 6,
          s0: sc, s1: sc, dur: 210,
        });
        this.addAnimSprite({
          sheet: 'fx_fangs', cw: 64, ch: 24, frames: [1],
          x0: B.cx, y0: B.cy + 26, x1: B.cx, y1: B.cy + 7,
          s0: sc, s1: sc, dur: 210,
        });
        await wait(165);
        this.shake = Math.max(this.shake, 12);
        addHitStar(B.cx, B.cy, 220, sc);
        await wait(110);
        break;
      }

      case 'pursuit': {
        this.flashArena('#080814', 0.58, 340);
        atk.ghosts = [{ dx: -12 * dir, dy: 0, alpha: 0.5 }];
        await this.tween(atk, 'dx', 20 * dir, 80);
        this.addAnimSprite({
          sheet: 'fx_slash', cw: 48, ch: 48, frames: [1, 2, 3, 4],
          x0: B.cx, y0: B.cy, flipX: flip, s0: 1.15, dur: 230,
        });
        addHitStar(B.cx, B.cy, 200, 1.0);
        await this.tween(atk, 'dx', 0, 130);
        atk.ghosts = [];
        break;
      }

      case 'rapidspin': {
        this.addAnimSprite({
          sheet: 'fx_tornado', cw: 48, ch: 64, frames: [0, 1, 0, 1, 0, 1],
          x0: A.cx, y0: A.cy, s0: 0.85, s1: 1.0, alpha0: 0.9, alpha1: 0.2, dur: 260,
        });
        await wait(120);
        await this.tween(atk, 'dx', 18 * dir, 85);
        addHitStar(B.cx, B.cy, 220, 1.0);
        await this.tween(atk, 'dx', 0, 120);
        break;
      }

      case 'ember': {
        await this.tween(atk, 'dx', 6 * dir, 60);
        this.tween(atk, 'dx', 0, 100);
        for (let i = 0; i < 3; i++) {
          this.addAnimSprite({
            sheet: 'fx_fire', cw: 48, ch: 48, frames: [0, 1, 2, 3],
            x0: A.cx + 12 * dir, y0: A.cy - 6,
            x1: B.cx + (i - 1) * 10, y1: B.cy + 4, arc: 10,
            s0: 0.55, s1: 0.95, dur: 260, delay: i * 65,
          });
        }
        await wait(240);
        this.addAnimSprite({
          sheet: 'fx_fire', cw: 48, ch: 48, frames: [1, 2, 3, 4, 5],
          x0: B.cx, y0: B.cy + 4, s0: 1.05, s1: 1.15, dur: 300,
        });
        def.tintColor = '#ff6820'; def.tintAlpha = 0.55;
        this.tween(def, 'tintAlpha', 0, 260);
        await wait(200);
        def.tintColor = null;
        break;
      }

      case 'flamethrower': {
        this.flashArena('#ff4810', 0.36, 620);
        await this.tween(atk, 'dx', 8 * dir, 65);
        this.tween(atk, 'dx', 0, 140);
        for (let i = 0; i < 6; i++) {
          this.addAnimSprite({
            sheet: 'fx_fireball', cw: 64, ch: 64, frames: [0],
            x0: A.cx + 14 * dir, y0: A.cy - 6,
            x1: B.cx + (i % 3 - 1) * 8, y1: B.cy + (i % 2 ? -4 : 4),
            flipX: !flip, s0: 0.55, s1: 1.05, dur: 250, delay: i * 52,
          });
        }
        await wait(230);
        for (let k = -1; k <= 1; k++) {
          this.addAnimSprite({
            sheet: 'fx_fire', cw: 48, ch: 48, frames: [1, 2, 3, 4, 5, 3, 4],
            x0: B.cx + k * 14, y0: B.cy + 6 - Math.abs(k) * 4,
            s0: 1.0, s1: 1.25, dur: 380, delay: (k + 1) * 35,
          });
        }
        def.tintColor = '#ff5018'; def.tintAlpha = 0.72;
        this.shake = Math.max(this.shake, 14);
        await this.tween(def, 'tintAlpha', 0, 340);
        def.tintColor = null;
        break;
      }

      case 'bubble': {
        for (let i = 0; i < 6; i++) {
          const yOff = (i % 3 - 1) * 9;
          this.addAnimSprite({
            sheet: 'fx_bubble', cw: 32, ch: 32, frames: [0, 1, 0, 2],
            x0: A.cx + 12 * dir, y0: A.cy - 4,
            x1: B.cx + (i % 2 ? -8 : 8), y1: B.cy + yOff, arc: (i % 2 ? 8 : -6),
            s0: 0.55, s1: 1.0, dur: 340, delay: i * 50,
          });
        }
        await wait(380);
        this.addAnimSprite({
          sheet: 'fx_bubble', cw: 32, ch: 32, frames: [2, 3],
          x0: B.cx, y0: B.cy, s0: 1.0, s1: 1.3, alpha0: 1, alpha1: 0, dur: 180,
        });
        await wait(100);
        break;
      }

      case 'watergun': {
        this.flashArena('#3898ff', 0.24, 420);
        await this.tween(atk, 'dx', 8 * dir, 60);
        this.tween(atk, 'dx', 0, 120);
        for (let i = 0; i < 7; i++) {
          this.addAnimSprite({
            sheet: 'fx_water', cw: 48, ch: 48, frames: [0, 1],
            x0: A.cx + 14 * dir, y0: A.cy - 4,
            x1: B.cx, y1: B.cy,
            s0: 0.75, s1: 1.1, dur: 200, delay: i * 36,
          });
        }
        await wait(210);
        this.addAnimSprite({
          sheet: 'fx_water', cw: 48, ch: 48, frames: [2, 3, 4, 5],
          x0: B.cx, y0: B.cy + 4, s0: 1.1, s1: 1.3, dur: 300,
        });
        def.tintColor = '#58b8ff'; def.tintAlpha = 0.55;
        await this.tween(def, 'tintAlpha', 0, 240);
        def.tintColor = null;
        break;
      }

      case 'vinewhip': {
        await this.tween(atk, 'dx', 10 * dir, 65);
        this.tween(atk, 'dx', 0, 120);
        this.addAnimSprite({
          sheet: 'fx_vine', cw: 48, ch: 48, frames: [0, 1, 2, 3],
          x0: A.cx + 22 * dir, y0: A.cy - 6,
          x1: (A.cx + B.cx) / 2, y1: (A.cy + B.cy) / 2,
          flipX: flip, s0: 0.9, s1: 1.15, dur: 170,
        });
        await wait(110);
        this.addAnimSprite({
          sheet: 'fx_vine', cw: 48, ch: 48, frames: [4, 5, 3],
          x0: B.cx - 6 * dir, y0: B.cy, flipX: flip, s0: 1.2, dur: 230,
        });
        addHitStar(B.cx, B.cy, 210, 0.95);
        await wait(160);
        break;
      }

      case 'razorleaf': {
        // Leaves rise & spin around user, then slice cleanly through target!
        for (let i = 0; i < 5; i++) {
          const ly = A.cy - 16 + (i - 2) * 7;
          this.addAnimSprite({
            sheet: 'fx_leaf', cw: 24, ch: 24, frames: [0, 1, 2, 3, 4, 5, 6, 7],
            x0: A.cx + 8 * dir, y0: ly,
            x1: B.cx + 12 * dir, y1: B.cy + (i - 2) * 6,
            s0: 1.0, s1: 1.15, dur: 320, delay: i * 48,
          });
        }
        await wait(280);
        this.addAnimSprite({
          sheet: 'fx_slash', cw: 48, ch: 48, frames: [1, 2, 3, 4],
          x0: B.cx, y0: B.cy, flipX: flip, s0: 0.95, dur: 200,
        });
        await wait(140);
        break;
      }

      case 'leechseed': {
        for (let i = -1; i <= 1; i++) {
          this.addAnimSprite({
            sheet: 'fx_seed', cw: 24, ch: 24, frames: [0],
            x0: A.cx + 10 * dir, y0: A.cy - 6,
            x1: B.cx + i * 12, y1: B.footY - 4, arc: 28,
            s0: 0.9, s1: 1.0, dur: 340, delay: (i + 1) * 55,
          });
          this.addAnimSprite({
            sheet: 'fx_seed', cw: 24, ch: 24, frames: [0, 1, 2, 2],
            x0: B.cx + i * 12, y0: B.footY - 4,
            s0: 0.9, s1: 1.15, dur: 360, delay: 340 + (i + 1) * 55,
          });
        }
        await wait(680);
        break;
      }

      case 'sleeppowder':
      case 'stunspore':
      case 'poisonpowder': {
        const pal = moveId === 'sleeppowder'
          ? ['#88f8b0', '#c8ffd8']
          : moveId === 'stunspore'
            ? ['#f8e030', '#fff8a0']
            : ['#c858e8', '#f0a0ff'];
        this.flashArena(pal[0], 0.20, 460);
        for (let i = 0; i < 22; i++) {
          this.particles.push({
            kind: 'rect',
            x: B.cx + (Math.random() * 44 - 22),
            y: B.cy - 28 - Math.random() * 16,
            vx: Math.sin(i) * 0.35,
            vy: 0.65 + Math.random() * 0.55,
            g: 0,
            life: 32 + Math.random() * 14,
            age: 0,
            size: 2.5 + (i % 2),
            color: pal[i % 2],
          });
        }
        def.tintColor = pal[0]; def.tintAlpha = 0.5;
        await wait(320);
        await this.tween(def, 'tintAlpha', 0, 200);
        def.tintColor = null;
        break;
      }

      case 'poisonsting':
      case 'twineedle':
      case 'pinmissile': {
        await this.tween(atk, 'dx', 10 * dir, 60);
        this.tween(atk, 'dx', 0, 110);
        this.addAnimSprite({
          sheet: 'fx_needle', cw: 36, ch: 36, frames: [0, 1],
          x0: A.cx + 14 * dir, y0: A.cy - 4,
          x1: B.cx, y1: B.cy,
          rot0: angleAB, rot1: angleAB,
          s0: 1.0, s1: 1.1, dur: 175,
        });
        await wait(165);
        if (moveId === 'poisonsting' || moveId === 'twineedle') {
          this.addAnimSprite({
            sheet: 'fx_poison', cw: 36, ch: 36, frames: [3, 4],
            x0: B.cx, y0: B.cy + 4, s0: 1.0, s1: 1.2, dur: 210,
          });
        }
        addHitStar(B.cx, B.cy, 190, 0.85);
        await wait(90);
        break;
      }

      case 'stringshot': {
        // Silk thread shoots across -> sticky Spider Web expands over foe!
        for (let i = 0; i < 5; i++) {
          this.particles.push({
            kind: 'streak',
            x: A.cx + (B.cx - A.cx) * (i / 5),
            y: A.cy + (B.cy - A.cy) * (i / 5),
            vx: dxAB * 0.05, vy: dyAB * 0.05, g: 0,
            life: 16, age: 0, size: 4, color: '#ffffff',
          });
        }
        await wait(120);
        this.addAnimSprite({
          sheet: 'fx_web', cw: 64, ch: 64, frames: [0],
          x0: B.cx, y0: B.cy + 2,
          s0: 0.25, s1: 1.05, alpha0: 1, alpha1: 0.15, dur: 460,
        });
        await wait(420);
        break;
      }

      case 'gust':
      case 'twister': {
        if (moveId === 'twister') this.flashArena('#5848d8', 0.28, 480);
        const count = moveId === 'twister' ? 2 : 1;
        for (let k = 0; k < count; k++) {
          this.addAnimSprite({
            sheet: 'fx_tornado', cw: 48, ch: 64, frames: [0, 1, 0, 1, 0, 1, 0, 1],
            x0: B.cx - 20 + k * 14, y0: B.cy + 2,
            x1: B.cx + 18 - k * 14, y1: B.cy - 2,
            s0: 0.85, s1: 1.15, dur: 440, delay: k * 80,
          });
        }
        this.addAnimSprite({
          sheet: 'fx_wind', cw: 48, ch: 48, frames: [0, 1],
          x0: B.cx, y0: B.cy - 8, s0: 0.9, s1: 1.25, alpha0: 1, alpha1: 0, dur: 320, delay: 90,
        });
        await wait(400);
        break;
      }

      case 'wingattack': {
        await this.tween(atk, 'dy', -10, 70);
        await this.tween(atk, 'dx', 16 * dir, 70);
        this.tween(atk, 'dy', 0, 120);
        this.tween(atk, 'dx', 0, 120);
        this.addAnimSprite({
          sheet: 'fx_wind', cw: 48, ch: 48, frames: [0, 1],
          x0: B.cx - 8, y0: B.cy - 4, flipX: flip, s0: 1.2, dur: 220,
        });
        this.addAnimSprite({
          sheet: 'fx_wind', cw: 48, ch: 48, frames: [0, 1],
          x0: B.cx + 8, y0: B.cy + 4, flipX: !flip, s0: 1.2, dur: 220, delay: 40,
        });
        for (let i = 0; i < 3; i++) {
          this.addAnimSprite({
            sheet: 'fx_wind', cw: 48, ch: 48, frames: [2, 3],
            x0: B.cx + (i - 1) * 14, y0: B.cy - 10,
            x1: B.cx + (i - 1) * 20, y1: B.cy + 16,
            s0: 0.85, s1: 0.95, alpha0: 1, alpha1: 0, dur: 320, delay: i * 40,
          });
        }
        addHitStar(B.cx, B.cy, 210, 1.05);
        await wait(210);
        break;
      }

      case 'featherdance': {
        for (let i = 0; i < 6; i++) {
          const ox = (i - 2.5) * 11;
          this.addAnimSprite({
            sheet: 'fx_wind', cw: 48, ch: 48, frames: [2, 3, 2, 3],
            x0: B.cx + ox, y0: B.cy - 30,
            x1: B.cx + ox + (i % 2 ? 8 : -8), y1: B.cy + 20,
            s0: 0.95, s1: 0.95, alpha0: 1, alpha1: 0.15, dur: 520, delay: i * 55,
          });
        }
        await wait(540);
        break;
      }

      case 'confusion':
      case 'psybeam': {
        // Authentic GBA FireRed psychedelic sky flash + concentric rings + H-blank sine warp!
        this.flashArena(moveId === 'psybeam' ? '#ff48c8' : '#7848e8', 0.42, 540);
        this.addAnimSprite({
          sheet: 'fx_psychic', cw: 48, ch: 48, frames: [0],
          x0: A.cx, y0: A.cy - 6, s0: 0.6, s1: 1.15, alpha0: 1, alpha1: 0, dur: 260,
        });
        if (moveId === 'psybeam') {
          for (let i = 0; i < 6; i++) {
            this.addAnimSprite({
              sheet: 'fx_psychic', cw: 48, ch: 48, frames: [1, 2, 3],
              x0: A.cx + 12 * dir, y0: A.cy - 4,
              x1: B.cx, y1: B.cy,
              s0: 0.55, s1: 1.05, dur: 240, delay: i * 42,
            });
          }
          await wait(210);
        } else {
          for (let i = 0; i < 3; i++) {
            this.addAnimSprite({
              sheet: 'fx_psychic', cw: 48, ch: 48, frames: [1, 2, 3, 4],
              x0: B.cx, y0: B.cy, s0: 0.5, s1: 1.35, alpha0: 0.95, alpha1: 0.1, dur: 320, delay: i * 70,
            });
          }
        }
        def.wavy = 1.0;
        def.tintColor = '#e868ff'; def.tintAlpha = 0.5;
        await Promise.all([
          this.tween(def, 'wavy', 0, 360),
          this.tween(def, 'tintAlpha', 0, 360),
        ]);
        def.tintColor = null;
        break;
      }

      case 'supersonic':
      case 'growl': {
        for (let i = 0; i < 4; i++) {
          this.addAnimSprite({
            sheet: moveId === 'growl' ? 'fx_status' : 'fx_psychic',
            cw: moveId === 'growl' ? 32 : 48,
            ch: moveId === 'growl' ? 32 : 48,
            frames: moveId === 'growl' ? [2, 3] : [1, 2, 3],
            x0: A.cx + 12 * dir, y0: A.cy - 6,
            x1: B.cx, y1: B.cy + (i % 2 ? -8 : 6),
            s0: 0.65, s1: 1.15, alpha0: 1, alpha1: 0.15, dur: 340, delay: i * 70,
          });
        }
        await wait(440);
        break;
      }

      case 'tailwhip': {
        for (let i = 0; i < 2; i++) {
          await this.tween(atk, 'dx', -7, 65);
          await this.tween(atk, 'dx', 7, 65);
        }
        await this.tween(atk, 'dx', 0, 55);
        break;
      }

      case 'smokescreen':
      case 'sandattack': {
        for (let i = 0; i < 6; i++) {
          this.addAnimSprite({
            sheet: 'fx_status', cw: 32, ch: 32, frames: [4],
            x0: moveId === 'sandattack' ? A.cx + 12 * dir : B.cx + (i % 3 - 1) * 14,
            y0: moveId === 'sandattack' ? A.footY - 6 : B.cy + (i % 2 ? 8 : -6),
            x1: B.cx + (i % 3 - 1) * 14,
            y1: B.cy + (i % 2 ? -8 : 8),
            s0: 0.65, s1: 1.25, alpha0: 0.95, alpha1: 0.1, dur: 340, delay: i * 45,
          });
        }
        if (moveId === 'smokescreen') {
          def.tintColor = '#282830'; def.tintAlpha = 0.65;
        } else {
          def.tintColor = '#d8b068'; def.tintAlpha = 0.55;
        }
        await wait(340);
        await this.tween(def, 'tintAlpha', 0, 200);
        def.tintColor = null;
        break;
      }

      case 'sweetscent': {
        this.flashArena('#ff98c8', 0.25, 480);
        for (let i = 0; i < 6; i++) {
          this.addAnimSprite({
            sheet: 'fx_status', cw: 32, ch: 32, frames: [6],
            x0: A.cx + 8 * dir, y0: A.cy - 10,
            x1: B.cx + (i - 2.5) * 12, y1: B.cy + (i % 2 ? -10 : 8), arc: 14,
            s0: 0.75, s1: 1.1, alpha0: 1, alpha1: 0.15, dur: 440, delay: i * 50,
          });
        }
        await wait(480);
        break;
      }

      case 'scaryface': {
        this.flashArena('#05050c', 0.72, 520);
        this.addAnimSprite({
          sheet: 'fx_scaryface', cw: 112, ch: 76, frames: [0],
          x0: B.cx, y0: B.cy - 2,
          s0: 0.55, s1: 1.0, alpha0: 1, alpha1: 0.1, dur: 460,
        });
        this.shake = Math.max(this.shake, 10);
        await wait(440);
        break;
      }

      case 'agility': {
        atk.ghosts = [
          { dx: -16, dy: 0, alpha: 0.5 },
          { dx: 16, dy: 0, alpha: 0.5 },
        ];
        for (let i = 0; i < 2; i++) {
          await this.tween(atk, 'dx', 18, 55);
          await this.tween(atk, 'dx', -18, 55);
        }
        await this.tween(atk, 'dx', 0, 50);
        atk.ghosts = [];
        break;
      }

      case 'harden':
      case 'withdraw': {
        atk.tintColor = moveId === 'withdraw' ? '#88d8ff' : '#ffffff';
        atk.tintAlpha = 0.85;
        for (let i = 0; i < 3; i++) {
          this.addAnimSprite({
            sheet: 'fx_status', cw: 32, ch: 32, frames: [5],
            x0: A.cx + (i - 1) * 14, y0: A.cy - 8 + (i % 2) * 12,
            s0: 0.7, s1: 1.1, alpha0: 1, alpha1: 0, dur: 240, delay: i * 55,
          });
        }
        await this.tween(atk, 'tintAlpha', 0, 320);
        atk.tintColor = null;
        break;
      }

      case 'growth': {
        await this.tween(atk, 'scale', 1.12, 160);
        await this.tween(atk, 'scale', 1.0, 160);
        break;
      }

      default: {
        // Tackle / Struggle & default physical strike
        await this.tween(atk, 'dx', -6 * dir, 55);
        await this.tween(atk, 'dx', 18 * dir, 75);
        addHitStar(B.cx, B.cy, 230, 1.05);
        await this.tween(atk, 'dx', 0, 130);
        break;
      }
    }
  }

  // ---------- Authentic FireRed Scrolling Stat-Stage Wave & Status FX ----------
  async playStatWave(side, stat, stages) {
    const f = this.sideFx(side);
    if (stages > 0) Sfx.statUp(); else Sfx.statDown();
    f.statWave = {
      stat: stat || 'mix',
      dir: stages > 0 ? -1 : 1, // -1 scrolls UP on stat rise, +1 scrolls DOWN on stat fall!
      t0: performance.now(),
      dur: 620,
    };
    await wait(520);
  }

  async playStatusAnim(side, kind) {
    const f = this.sideFx(side);
    const C = this.combatantCenter(side);
    if (kind === 'SLP') {
      for (let i = 0; i < 3; i++) {
        this.addAnimSprite({
          sheet: 'fx_status', cw: 32, ch: 32, frames: [8],
          x0: C.cx + 6 * (side === 'player' ? 1 : -1), y0: C.cy - 14,
          x1: C.cx + (18 + i * 5) * (side === 'player' ? 1 : -1), y1: C.cy - 38,
          s0: 0.65, s1: 1.15, alpha0: 1, alpha1: 0.1, dur: 540, delay: i * 160,
        });
      }
      await wait(560);
    } else if (kind === 'CONFUSE') {
      // Orbiting yellow FireRed ducks around head!
      for (let i = 0; i < 3; i++) {
        this.addAnimSprite({
          sheet: 'fx_status', cw: 32, ch: 32, frames: [0, 1, 0, 1],
          orbit: { cx: C.cx, cy: C.cy - 20, rx: 18, ry: 5, phase0: (i * Math.PI * 2) / 3, turns: 1.2 },
          s0: 0.95, s1: 0.95, dur: 580,
        });
      }
      await wait(520);
    } else if (kind === 'PSN') {
      f.tintColor = '#a840d0'; f.tintAlpha = 0.65;
      for (let i = 0; i < 3; i++) {
        this.addAnimSprite({
          sheet: 'fx_poison', cw: 36, ch: 36, frames: [0, 1, 2, 3],
          x0: C.cx + (i - 1) * 14, y0: C.cy + 10,
          x1: C.cx + (i - 1) * 14, y1: C.cy - 10,
          s0: 0.95, s1: 1.05, alpha0: 1, alpha1: 0.15, dur: 420, delay: i * 70,
        });
      }
      await this.tween(f, 'tintAlpha', 0, 440);
      f.tintColor = null;
    } else if (kind === 'BRN') {
      f.tintColor = '#ff5820'; f.tintAlpha = 0.68;
      for (let i = -1; i <= 1; i++) {
        this.addAnimSprite({
          sheet: 'fx_fire', cw: 48, ch: 48, frames: [0, 1, 2, 3, 4],
          x0: C.cx + i * 12, y0: C.cy + 6, s0: 0.85, s1: 1.0, dur: 380, delay: (i + 1) * 50,
        });
      }
      await this.tween(f, 'tintAlpha', 0, 400);
      f.tintColor = null;
    } else if (kind === 'PAR') {
      f.tintColor = '#ffe840'; f.tintAlpha = 0.7;
      for (let i = 0; i < 4; i++) {
        this.addAnimSprite({
          sheet: 'fx_status', cw: 32, ch: 32, frames: [5],
          x0: C.cx + (i % 2 ? -12 : 12), y0: C.cy + (i < 2 ? -10 : 8),
          s0: 0.85, s1: 1.15, dur: 220, delay: i * 60,
        });
      }
      for (let j = 0; j < 3; j++) {
        f.dx = (j % 2 ? -3 : 3);
        await wait(55);
      }
      f.dx = 0;
      await this.tween(f, 'tintAlpha', 0, 180);
      f.tintColor = null;
    } else if (kind === 'FRZ') {
      f.tintColor = '#80e8ff'; f.tintAlpha = 0.72;
      this.spawnEffect('ICE', C.cx, C.cy);
      await this.tween(f, 'tintAlpha', 0, 380);
      f.tintColor = null;
    }
  }

  // ---------- Main Battle Flow ----------
  async run() {
    Game.mode = 'battle';
    Sfx.encounter();
    await UI.flashScreen(2);

    // Prepare opposing horizontal platform slide-in + GBA scanline curtain wipe!
    const trFront = this.trainer ? GameAssets.trainerFront(this.trainer.flag) : null;
    const hasPlayerBack = !!(GameAssets.trainerBackThrow() || GameAssets.trainerBack());

    this.scene.curtain = 1;
    this.scene.ePadDx = -220;
    this.scene.pPadDx = 220;

    this.fx.e.vis = true;
    this.fx.e.isTrainer = !!trFront;
    this.fx.e.silhouette = this.wild ? 1 : 0;
    this.fx.e.x = FR_STAGE.eHomeX - 220;
    this.fx.e.y = FR_STAGE.eHomeY;
    this.fx.e.scale = 1;

    this.fx.p.vis = hasPlayerBack;
    this.fx.p.isTrainer = hasPlayerBack;
    this.fx.p.trFrame = 0;
    this.fx.p.x = FR_STAGE.pHomeX + 220;
    this.fx.p.y = FR_STAGE.pHomeY;
    this.fx.p.scale = 1;

    await Promise.all([
      this.tween(this.scene, 'curtain', 0, 420),
      this.tween(this.scene, 'ePadDx', 0, 560),
      this.tween(this.scene, 'pPadDx', 0, 560),
      this.tween(this.fx.e, 'x', FR_STAGE.eHomeX, 560),
      this.tween(this.fx.p, 'x', FR_STAGE.pHomeX, 560),
    ]);

    if (this.wild) {
      UI.setEnemyBalls(null);
      await this.tween(this.fx.e, 'silhouette', 0, 200);
      await this.playEntryHop('enemy');
      UI.setBox('enemy', this.enemy, { slide: true });
      await UI.say(`A wild ${creatureName(this.enemy)} appeared!`);
      if (this.enemy.shiny) {
        Sfx.statUp();
        const E = this.combatantCenter('enemy');
        this.spawnEffect('CAUGHT_STARS', E.cx, E.cy - 6);
        await UI.say("Whoa... its colors are different. It's SHINY!");
      }
    } else {
      UI.setEnemyBalls(this.enemyParty);
      await UI.say(`${this.trainer.name} wants to battle!`);
      await this.enemyTrainerSendOut(true);
    }

    // Player 5-frame Red throw & Pokemon materialization
    await this.playerThrowAndSendOut(true);
    UI.setBattleAiPill(true, this.focusConceptId());
    this.participants.add(this.active.uid);

    while (!this.over) {
      await this.turn();
    }

    // post-battle evolutions (from battle EXP)
    await this.handleEvolutions();

    UI.hideBoxes();
    UI.hideDialog();
    Game.mode = 'overworld';
    Input.clearPressed();
    return this.result;
  }

  async turn() {
    const action = await this.playerAction();

    if (action.kind === 'run') {
      if (await this.tryRun()) return;
      await this.enemyTurnOnly();
      return;
    }
    if (action.kind === 'switch') {
      await this.doSwitch(action.idx);
      await this.enemyTurnOnly();
      return;
    }
    if (action.kind === 'item') {
      await this.useBagItem(action.id, action.target);
      await this.enemyTurnOnly();
      return;
    }
    if (action.kind === 'orb') {
      const caught = await this.throwOrb(action.id);
      if (caught) return;
      await this.enemyTurnOnly();
      return;
    }

    // FIGHT: resolve both moves in speed/priority order.
    const moveId = action.struggle ? 'struggle' : this.active.moves[action.moveIdx].id;
    const pMove = MOVES[moveId];
    const pSlot = action.struggle ? null : this.active.moves[action.moveIdx];
    const eSlot = this.enemyMoveChoice();
    const eMoveId = eSlot ? eSlot.id : 'struggle';
    const eMove = MOVES[eMoveId];

    const pPrio = pMove.prio || 0, ePrio = eMove.prio || 0;
    let playerFirst;
    if (pPrio !== ePrio) playerFirst = pPrio > ePrio;
    else {
      const ps = battleSpeed(this.active, this.pStages);
      const es = battleSpeed(this.enemy, this.eStages);
      playerFirst = ps === es ? Math.random() < 0.5 : ps > es;
    }

    const order = playerFirst
      ? [['player', moveId, pMove, pSlot], ['enemy', eMoveId, eMove, eSlot]]
      : [['enemy', eMoveId, eMove, eSlot], ['player', moveId, pMove, pSlot]];

    for (const [side, mId, move, slot] of order) {
      if (this.over) return;
      const user = side === 'player' ? this.active : this.enemy;
      if (user.hp <= 0) continue;
      await this.useMove(side, mId, move, slot);
      if (await this.resolveFaints()) return;
    }

    await this.endOfTurn();
  }

  async enemyTurnOnly() {
    if (this.over || this.enemy.hp <= 0) return;
    const slot = this.enemyMoveChoice();
    const mId = slot ? slot.id : 'struggle';
    await this.useMove('enemy', mId, MOVES[mId], slot);
    if (await this.resolveFaints()) return;
    await this.endOfTurn();
  }

  // ---------- player action selection ----------
  async playerAction() {
    while (true) {
      UI.setPrompt(`What will ${creatureName(this.active)} do?`);
      const a = await UI.choose(['FIGHT', 'BAG', 'SWITCH', 'RUN'], {
        cols: 2, canCancel: false,
        style: { right: '10px', bottom: '10px', width: '290px', height: '108px', padding: '20px 18px' },
      });

      if (a === 0) { // FIGHT
        if (this.active.moves.every(m => m.pp <= 0)) {
          await UI.say(`${creatureName(this.active)} has no moves left!`);
          return { kind: 'fight', struggle: true };
        }
        const mi = await UI.moveMenu(this.active);
        if (mi === -1) continue;
        return { kind: 'fight', moveIdx: mi };
      }
      if (a === 1) { // BAG
        const id = await UI.bagScreen({ inBattle: true });
        if (!id) continue;
        if (ITEMS[id].kind === 'orb') {
          if (!this.wild) {
            await UI.say("You can't capture another trainer's partner!");
            continue;
          }
          return { kind: 'orb', id };
        }
        const t = await UI.partyScreen({ title: 'Use on which team member?' });
        if (t === -1) continue;
        if (!this.itemWouldWork(id, this.party[t])) {
          await UI.say("It won't have any effect.");
          continue;
        }
        return { kind: 'item', id, target: t };
      }
      if (a === 2) { // SWITCH
        const idx = await UI.partyScreen({ title: 'Switch to which team member?' });
        if (idx === -1) continue;
        if (idx === this.playerIdx) { await UI.say(`${creatureName(this.active)} is already in battle!`); continue; }
        if (this.party[idx].hp <= 0) { await UI.say(`${creatureName(this.party[idx])} has no energy left to battle!`); continue; }
        return { kind: 'switch', idx };
      }
      return { kind: 'run' };
    }
  }

  itemWouldWork(id, c) {
    const it = ITEMS[id];
    if (c.hp <= 0) return false;
    if (it.kind === 'heal') return c.hp < c.maxHp;
    if (it.kind === 'cure') return c.status && (it.cures.includes('ALL') || it.cures.includes(c.status));
    return false;
  }

  async useBagItem(id, targetIdx) {
    const it = ITEMS[id];
    const c = this.party[targetIdx];
    Game.player.bag[id]--;
    if (it.kind === 'heal') {
      const from = c.hp;
      c.hp = Math.min(c.maxHp, c.hp + it.amount);
      Sfx.heal();
      if (targetIdx === this.playerIdx) {
        const P = this.combatantCenter('player');
        for (let i = 0; i < 4; i++) {
          this.addAnimSprite({
            sheet: 'pokeballs', cw: 24, ch: 24, row: 2, frames: [2],
            x0: P.cx + (i - 1.5) * 12, y0: P.cy + 12,
            x1: P.cx + (i - 1.5) * 12, y1: P.cy - 18,
            s0: 0.9, s1: 1.15, alpha0: 1, alpha1: 0, dur: 420, delay: i * 65,
          });
        }
      }
      await UI.say(`${creatureName(c)} recovered ${c.hp - from} HP!`, { noWait: true, keep: true });
      if (targetIdx === this.playerIdx) await UI.animateHp('player', from, c.hp, c.maxHp);
      await wait(300);
      UI.hideDialog();
    } else {
      const txt = STATUS_INFO[c.status] ? STATUS_INFO[c.status].text : 'unwell';
      c.status = null; c.sleepTurns = 0;
      Sfx.heal();
      if (targetIdx === this.playerIdx) UI.setStatusChip('player', null);
      await UI.say(`${creatureName(c)} is no longer ${txt}!`);
    }
  }

  // ---------- switching ----------
  async doSwitch(idx, recallMsg = true) {
    if (recallMsg && this.active && this.active.hp > 0) {
      await UI.say(`${creatureName(this.active)}, come back!`, { noWait: true, keep: true });
      await this.recallToBall('player');
      UI.hideDialog();
    } else {
      this.fx.p.vis = false;
    }
    this.playerIdx = idx;
    this.pStages = freshStages();
    this.vol.player = freshVolatiles();
    await this.playerThrowAndSendOut(false);
    if (this.enemy.hp > 0) this.participants.add(this.active.uid);
  }

  // ---------- running ----------
  async tryRun() {
    if (!this.wild) {
      await UI.say("No! There's no running from a trainer battle!");
      return false;
    }
    this.runAttempts++;
    const ps = battleSpeed(this.active, this.pStages);
    const es = Math.max(1, battleSpeed(this.enemy, this.eStages));
    const f = Math.floor(ps * 128 / es) + 30 * this.runAttempts;
    if (f > 255 || randInt(0, 255) < f) {
      Sfx.run();
      await UI.say('Got away safely!');
      this.over = true; this.result = 'ran';
      return true;
    }
    await UI.say("Can't escape!");
    return false;
  }

  // ---------- enemy AI ----------
  enemyMoveChoice() {
    const usable = this.enemy.moves.filter(m => m.pp > 0);
    if (!usable.length) return null;
    const damaging = usable.filter(m => MOVES[m.id].power > 0);
    const superEff = damaging.filter(m =>
      typeEffectiveness(MOVES[m.id].type, creatureTypes(this.active)) > 1);
    if (superEff.length && Math.random() < 0.6) return superEff[randInt(0, superEff.length - 1)];
    return usable[randInt(0, usable.length - 1)];
  }

  focusConceptId() {
    if (this.trainer && this.trainer.flag && typeof AITutor !== 'undefined') {
      const m = AITutor.TRAINER_CONCEPT_MAP[this.trainer.flag];
      if (m !== undefined) return m;
    }
    return null;
  }

  async runKnowledgeChallenge() {
    if (typeof AITutor === 'undefined' || !UI.askEncounterModal) return null;
    const enc = await AITutor.nextEncounter({
      wild: this.wild,
      trainer: this.trainer,
      enemyIdx: this.enemyIdx,
      enemyPartyLen: this.enemyParty.length,
    });
    return UI.askEncounterModal(enc);
  }

  // ---------- move execution ----------
  async useMove(side, moveId, move, slot) {
    const user = side === 'player' ? this.active : this.enemy;
    const target = side === 'player' ? this.enemy : this.active;
    const targetSide = side === 'player' ? 'enemy' : 'player';
    const uStages = side === 'player' ? this.pStages : this.eStages;
    const tStages = side === 'player' ? this.eStages : this.pStages;
    const uVol = this.vol[side];
    const name = this.battlerName(side, user);

    // Pre-move status / volatile checks with GBA FireRed animations!
    if (user.status === 'SLP') {
      user.sleepTurns--;
      if (user.sleepTurns > 0) {
        Sfx.statusFx();
        await UI.say(`${name} is fast asleep.`, { noWait: true, keep: true });
        await this.playStatusAnim(side, 'SLP');
        UI.hideDialog();
        return;
      }
      user.status = null;
      UI.setStatusChip(side, null);
      await UI.say(`${name} woke up!`);
    }
    if (user.status === 'FRZ') {
      if (Math.random() < 0.2) {
        user.status = null;
        UI.setStatusChip(side, null);
        await UI.say(`${name} thawed out!`);
      } else {
        Sfx.statusFx();
        await UI.say(`${name} is frozen solid!`, { noWait: true, keep: true });
        await this.playStatusAnim(side, 'FRZ');
        UI.hideDialog();
        return;
      }
    }
    if (uVol.flinch) {
      uVol.flinch = false;
      Sfx.statusFx();
      await UI.say(`${name} flinched and couldn't move!`);
      return;
    }
    if (uVol.confuse > 0) {
      uVol.confuse--;
      if (uVol.confuse === 0) {
        await UI.say(`${name} snapped out of its confusion!`);
      } else {
        Sfx.statusFx();
        await UI.say(`${name} is confused!`, { noWait: true, keep: true });
        await this.playStatusAnim(side, 'CONFUSE');
        UI.hideDialog();
        if (Math.random() < 0.5) {
          const dmg = confusionSelfDamage(user);
          const from = user.hp;
          user.hp = Math.max(0, user.hp - dmg);
          Sfx.hit();
          const fx = this.sideFx(side);
          const C = this.combatantCenter(side);
          fx.flash = 14;
          this.addAnimSprite({ sheet: 'fx_hit', cw: 48, ch: 48, frames: [0, 1, 2, 3], x0: C.cx, y0: C.cy, dur: 220 });
          await UI.say('It hurt itself in its confusion!', { noWait: true, keep: true });
          await UI.animateHp(side, from, user.hp, user.maxHp);
          UI.hideDialog();
          return;
        }
      }
    }
    if (user.status === 'PAR' && Math.random() < 0.25) {
      Sfx.statusFx();
      await UI.say(`${name} is fully paralyzed!`, { noWait: true, keep: true });
      await this.playStatusAnim(side, 'PAR');
      UI.hideDialog();
      return;
    }

    if (slot) slot.pp = Math.max(0, slot.pp - 1);

    // KnowledgeQuest 3-Model AI Combat Challenge on player moves!
    let aiPowerMul = 1.0;
    let aiGuaranteeHit = false;
    let aiNeuralStrike = false;
    let aiBadgeEarned = false;
    if (side === 'player' && typeof AITutor !== 'undefined') {
      UI.hideDialog();
      const aiTurn = await this.runKnowledgeChallenge();
      if (aiTurn && aiTurn.result) {
        const r = aiTurn.result;
        UI.setBattleAiPill(true, r.target_concept_id);
        aiBadgeEarned = !!r.badge_just_earned;
        const verdict = r.grader ? r.grader.verdict : 'correct';
        if (verdict === 'incorrect') {
          Sfx.weakHit();
          await UI.say(`${name}'s focus slipped — ${move.name} fizzled out!`);
          return;
        }
        aiGuaranteeHit = true;
        if (verdict === 'partial') {
          aiPowerMul = Math.max(0.35, r.grader.rpg_damage_multiplier || 0.55);
        } else {
          aiNeuralStrike = !!r.used_neural_grader;
          aiPowerMul = aiNeuralStrike ? 1.35 : 1.15;
        }
      }
    }

    await UI.say(`${name} used ${move.name}!`, { noWait: true, keep: true });
    await wait(240);

    // Accuracy check (with accuracy/evasion stages; skipped when AI challenge succeeded)
    if (move.acc !== true && !aiGuaranteeHit) {
      const stage = Math.max(-6, Math.min(6, uStages.acc - tStages.eva));
      const chance = move.acc * accStageMul(stage);
      if (randInt(1, 100) > chance) {
        await wait(160);
        await UI.say(`${name}'s attack missed!`);
        return;
      }
    }

    if (move.cat === 'status') {
      await this.playMoveAnim(moveId, move, side, targetSide, 0);
      if (move.special === 'leechseed') {
        await this.applyLeechSeed(targetSide, target);
      } else {
        await this.applyEffect(side, move.effect, user, target, uStages, tStages);
      }
      if (aiBadgeEarned && this.active.hp > 0) await this.gainExp(this.active, 35);
      UI.hideDialog();
      return;
    }

    // Damaging move
    const typeMult = typeEffectiveness(move.type, creatureTypes(target));
    if (typeMult === 0) {
      await wait(140);
      await UI.say(`It doesn't affect ${creatureName(target)}...`);
      return;
    }

    const defFx = this.sideFx(targetSide);
    const B = this.combatantCenter(targetSide);

    // Super Fang: fixed damage, no crit/STAB/type math beyond immunity.
    if (move.special === 'superfang') {
      const dmg = Math.max(1, Math.round(Math.floor(target.hp / 2) * aiPowerMul));
      await this.playMoveAnim(moveId, move, side, targetSide, 0);
      defFx.flash = 18;
      this.impactFx(move.type, targetSide, 1);
      Sfx.hit();
      const from = target.hp;
      target.hp = Math.max(0, target.hp - dmg);
      await UI.animateHp(targetSide, from, target.hp, target.maxHp);
      if (aiBadgeEarned && this.active.hp > 0) await this.gainExp(this.active, 35);
      UI.hideDialog();
      return;
    }

    const hits = move.multi ? pickMultiHits(move.multi) : 1;
    let landed = 0;
    let lastCrit = false;
    for (let h = 0; h < hits && target.hp > 0; h++) {
      const crit = Math.random() < (move.hiCrit ? 1 / 8 : 1 / 16);
      const rawDmg = calcDamage(user, target, move, uStages, tStages, crit);
      const dmg = Math.max(1, Math.round(rawDmg * aiPowerMul));
      lastCrit = crit;
      await this.playMoveAnim(moveId, move, side, targetSide, h);
      defFx.flash = 18;
      this.impactFx(move.type, targetSide, typeMult);
      if (aiNeuralStrike) this.spawnEffect('CAUGHT_STARS', B.cx, B.cy - 6);
      if (typeMult > 1) Sfx.superHit(); else if (typeMult < 1) Sfx.weakHit(); else Sfx.hit();
      const from = target.hp;
      target.hp = Math.max(0, target.hp - dmg);
      await UI.animateHp(targetSide, from, target.hp, target.maxHp);
      landed++;
      if (move.effect && move.multi && target.hp > 0 && Math.random() < move.effect.chance) {
        await this.applyEffect(side, move.effect, user, target, uStages, tStages);
      }
    }

    if (aiNeuralStrike) await UI.say('Neural Synthesis Strike! Boosted power!');
    else if (aiPowerMul < 1.0) await UI.say('Partial Synthesis — glancing blow!');
    if (lastCrit) await UI.say('A critical hit!');
    if (move.multi) await UI.say(`Hit ${landed} time(s)!`);
    if (typeMult > 1) await UI.say("It's super effective!");
    else if (typeMult < 1) await UI.say("It's not very effective...");

    // Fire moves thaw a frozen target.
    if (move.type === 'FIRE' && target.status === 'FRZ' && target.hp > 0) {
      target.status = null;
      UI.setStatusChip(targetSide, null);
      await UI.say(`${creatureName(target)} thawed out!`);
    }

    // Recoil (Struggle)
    if (move.recoil && landed > 0) {
      const rec = Math.max(1, Math.floor(user.maxHp / 8));
      const uFrom = user.hp;
      user.hp = Math.max(0, user.hp - rec);
      await UI.say(`${name} is hit with recoil!`, { noWait: true, keep: true });
      await UI.animateHp(side, uFrom, user.hp, user.maxHp);
      UI.hideDialog();
    }

    // Secondary effect (single-hit moves)
    if (move.effect && !move.multi && target.hp > 0 && Math.random() < move.effect.chance) {
      await this.applyEffect(side, move.effect, user, target, uStages, tStages);
    }
    if (aiBadgeEarned && this.active.hp > 0) await this.gainExp(this.active, 35);
    UI.hideDialog();
  }

  async applyEffect(side, eff, user, target, uStages, tStages) {
    if (!eff) return;
    const targetSide = side === 'player' ? 'enemy' : 'player';
    if (eff.status) {
      await this.inflictStatus(targetSide, target, eff.status);
      return;
    }
    if (eff.volatile === 'flinch') {
      this.vol[targetSide].flinch = true;
      return;
    }
    if (eff.volatile === 'confuse') {
      if (this.vol[targetSide].confuse > 0 || target.hp <= 0) {
        await UI.say('But it failed!');
        return;
      }
      this.vol[targetSide].confuse = randInt(2, 5);
      Sfx.statusFx();
      await this.playStatusAnim(targetSide, 'CONFUSE');
      await UI.say(`${this.battlerName(targetSide, target)} became confused!`);
      return;
    }
    // stat stage change -> authentic FireRed scrolling stat wave inside sprite!
    const self = eff.target === 'self';
    const statSide = self ? side : targetSide;
    const who = self ? user : target;
    const stages = self ? uStages : tStages;
    const stat = eff.stat;
    const cur = stages[stat];
    const next = Math.max(-6, Math.min(6, cur + eff.stages));
    const name = creatureName(who);
    if (next === cur) {
      await UI.say(`${name}'s ${STAGE_NAMES[stat]} won't go any ${eff.stages > 0 ? 'higher' : 'lower'}!`);
      return;
    }
    stages[stat] = next;
    await this.playStatWave(statSide, stat, eff.stages);
    const adv = Math.abs(eff.stages) >= 2 ? 'sharply ' : '';
    await UI.say(`${name}'s ${STAGE_NAMES[stat]} ${adv}${eff.stages > 0 ? 'rose' : 'fell'}!`);
  }

  async applyLeechSeed(targetSide, target) {
    if (creatureTypes(target).includes('GRASS')) {
      await UI.say(`It doesn't affect ${creatureName(target)}...`);
      return;
    }
    if (this.vol[targetSide].seeded) {
      await UI.say('But it failed!');
      return;
    }
    this.vol[targetSide].seeded = true;
    Sfx.statusFx();
    await UI.say(`${this.battlerName(targetSide, target)} was seeded!`);
  }

  async inflictStatus(targetSide, target, status) {
    if (target.status || target.hp <= 0) {
      await UI.say('But it failed!');
      return;
    }
    const types = creatureTypes(target);
    if (status === 'BRN' && types.includes('FIRE')) { await UI.say(`It doesn't affect ${creatureName(target)}...`); return; }
    if (status === 'PSN' && (types.includes('POISON') || types.includes('STEEL'))) { await UI.say(`It doesn't affect ${creatureName(target)}...`); return; }
    target.status = status;
    if (status === 'SLP') target.sleepTurns = randInt(2, 4);
    Sfx.statusFx();
    await this.playStatusAnim(targetSide, status);
    UI.setStatusChip(targetSide, status);
    await UI.say(`${this.battlerName(targetSide, target)} was ${STATUS_INFO[status].text}!`);
  }

  // ---------- end of turn ----------
  async endOfTurn() {
    if (this.over) return;
    for (const side of ['player', 'enemy']) {
      const c = side === 'player' ? this.active : this.enemy;
      if (c.hp <= 0) continue;
      if (c.status === 'PSN' || c.status === 'BRN') {
        const dmg = Math.max(1, Math.floor(c.maxHp / 8));
        const from = c.hp;
        c.hp = Math.max(0, c.hp - dmg);
        Sfx.statusFx();
        const verb = c.status === 'PSN' ? 'hurt by poison' : 'hurt by its burn';
        await UI.say(`${creatureName(c)} is ${verb}!`, { noWait: true, keep: true });
        await this.playStatusAnim(side, c.status);
        await UI.animateHp(side, from, c.hp, c.maxHp);
        UI.hideDialog();
        if (await this.resolveFaints()) return;
      }
      // Leech Seed drain with arcing green energy orbs!
      if (this.vol[side].seeded && c.hp > 0) {
        const otherSide = side === 'player' ? 'enemy' : 'player';
        const other = side === 'player' ? this.enemy : this.active;
        const dmg = Math.max(1, Math.floor(c.maxHp / 8));
        const from = c.hp;
        c.hp = Math.max(0, c.hp - dmg);
        Sfx.statusFx();
        const S = this.combatantCenter(side);
        const D = this.combatantCenter(otherSide);
        for (let i = 0; i < 4; i++) {
          this.addAnimSprite({
            sheet: 'pokeballs', cw: 24, ch: 24, row: 2, frames: [2],
            x0: S.cx, y0: S.cy, x1: D.cx, y1: D.cy, arc: 22,
            s0: 0.9, s1: 1.0, dur: 360, delay: i * 65,
          });
        }
        await UI.say(`${creatureName(c)}'s health is sapped by Leech Seed!`, { noWait: true, keep: true });
        await wait(280);
        await UI.animateHp(side, from, c.hp, c.maxHp);
        if (other && other.hp > 0) {
          const oFrom = other.hp;
          other.hp = Math.min(other.maxHp, other.hp + dmg);
          Sfx.heal();
          await UI.animateHp(otherSide, oFrom, other.hp, other.maxHp);
        }
        UI.hideDialog();
        if (await this.resolveFaints()) return;
      }
    }
    this.vol.player.flinch = false;
    this.vol.enemy.flinch = false;
  }

  // Returns true if the battle ended (or flow should stop this turn).
  async resolveFaints() {
    // Enemy faint
    if (this.enemy.hp <= 0 && !this.over) {
      Sfx.faint();
      await this.faintAnim('enemy');
      if (!this.wild) UI.setEnemyBalls(this.enemyParty);
      await UI.say(`${this.battlerName('enemy', this.enemy)} fainted!`);
      await this.awardExp();
      const next = this.enemyParty.findIndex(c => c.hp > 0);
      if (!this.wild && next !== -1) {
        this.enemyIdx = next;
        this.eStages = freshStages();
        this.vol.enemy = freshVolatiles();
        this.participants = new Set(this.active.hp > 0 ? [this.active.uid] : []);
        await this.enemyTrainerSendOut(false);
      } else {
        if (!this.wild) {
          // Slide defeated Trainer back onto the platform just like GBA FireRed!
          const trFront = GameAssets.trainerFront(this.trainer.flag);
          if (trFront) {
            const eb = document.getElementById('enemy-box');
            if (eb) eb.classList.add('hidden');
            this.fx.e.isTrainer = true;
            this.fx.e.vis = true;
            this.fx.e.dy = 0;
            this.fx.e.clipBottom = null;
            this.fx.e.x = 256;
            this.tween(this.fx.e, 'x', FR_STAGE.eHomeX, 340);
          }
          await UI.say(`You defeated ${this.trainer.name}!`);
          Game.player.money += this.trainer.prize;
          await UI.say(`You got $${this.trainer.prize} for winning!`);
        }
        this.over = true; this.result = 'win';
        return true;
      }
    }
    // Player faint
    if (this.active.hp <= 0 && !this.over) {
      this.active.status = null; this.active.sleepTurns = 0;
      this.vol.player = freshVolatiles();
      Sfx.faint();
      await this.faintAnim('player');
      UI.setStatusChip('player', null);
      await UI.say(`${creatureName(this.active)} fainted!`);
      const anyLeft = this.party.some(c => c.hp > 0);
      if (!anyLeft) {
        this.over = true; this.result = 'lose';
        return true;
      }
      let idx;
      do {
        idx = await UI.partyScreen({ title: 'Choose the next team member.', forced: true });
      } while (this.party[idx].hp <= 0);
      await this.doSwitch(idx, false);
      return true;
    }
    return this.over;
  }

  // Authentic GBA FireRed ground-clipped downward sink!
  async faintAnim(side) {
    const fx = this.sideFx(side);
    fx.clipBottom = side === 'enemy' ? (FR_STAGE.eHomeY + 64) : FR_STAGE.arenaH;
    await this.tween(fx, 'dy', 64, 230);
    fx.vis = false;
    fx.dy = 0;
    if (side === 'enemy') fx.clipBottom = null;
  }

  // ---------- EXP / levels ----------
  async awardExp() {
    if (this.enemyIsCaught) return;
    const sp = SPECIES[this.enemy.species];
    let base = Math.floor(sp.expYield * this.enemy.level / 7);
    if (!this.wild) base = Math.floor(base * 1.5);
    base = Math.max(1, base);
    for (const c of this.party) {
      if (c.hp <= 0 || c.level >= 100) continue;
      const amount = this.participants.has(c.uid) ? base : Math.max(1, Math.floor(base / 2));
      await this.gainExp(c, amount);
    }
  }

  async gainExp(c, amount) {
    if (c.level >= 100) return;
    Sfx.expTick();
    await UI.say(`${creatureName(c)} gained ${amount} EXP. Points!`);
    const isActive = c === this.active;
    while (c.level < 100) {
      const next = expForLevel(c.level + 1);
      if (c.exp + amount < next) {
        c.exp += amount;
        amount = 0;
        if (isActive) await UI.animateExp(0, expProgress(c));
        break;
      }
      amount -= (next - c.exp);
      c.exp = next;
      if (isActive) await UI.animateExp(expProgress(c), 1);
      await this.levelUp(c);
    }
  }

  async levelUp(c) {
    c.level++;
    const oldMax = c.maxHp;
    calcStats(c);
    c.hp = Math.min(c.maxHp, c.hp + (c.maxHp - oldMax));
    Sfx.levelUp();
    if (c === this.active) UI.setBox('player', this.active);
    await UI.say(`${creatureName(c)} grew to Lv${c.level}!`);
    const sp = SPECIES[c.species];
    for (const entry of sp.learnset.filter(e => e.lv === c.level)) {
      await this.learnMove(c, entry.move);
    }
    if (sp.evolve && c.level >= sp.evolve.at && !this.evoQueue.includes(c)) {
      this.evoQueue.push(c);
    }
  }

  async learnMove(c, moveId) {
    if (c.moves.some(m => m.id === moveId)) return;
    const mv = MOVES[moveId];
    if (c.moves.length < 4) {
      c.moves.push({ id: moveId, pp: mv.pp, maxPp: mv.pp });
      Sfx.statUp();
      await UI.say(`${creatureName(c)} learned ${mv.name}!`);
      return;
    }
    await UI.say(`${creatureName(c)} wants to learn ${mv.name}. But it already knows 4 moves.`);
    const yes = await UI.yesNo(`Forget a move to make room for ${mv.name}?`);
    if (!yes) {
      await UI.say(`${creatureName(c)} did not learn ${mv.name}.`);
      return;
    }
    UI.setPrompt('Forget which move?');
    const pick = await UI.choose(c.moves.map(m => MOVES[m.id].name).concat('CANCEL'),
      { style: { right: '14px', bottom: '126px' } });
    UI.hideDialog();
    if (pick === -1 || pick === 4) {
      await UI.say(`${creatureName(c)} did not learn ${mv.name}.`);
      return;
    }
    const old = MOVES[c.moves[pick].id].name;
    c.moves[pick] = { id: moveId, pp: mv.pp, maxPp: mv.pp };
    await UI.say(`1... 2... and... poof! ${creatureName(c)} forgot ${old} and learned ${mv.name}!`);
  }

  async handleEvolutions() {
    for (const c of this.evoQueue) {
      const sp = SPECIES[c.species];
      if (!sp.evolve || c.level < sp.evolve.at) continue;
      const oldName = creatureName(c);
      await UI.say(`What? ${oldName} is evolving!`);
      Sfx.evolve();
      await UI.flashScreen(4);
      const oldMax = c.maxHp;
      c.species = sp.evolve.to;
      calcStats(c);
      c.hp = Math.min(c.maxHp, c.hp + (c.maxHp - oldMax));
      if (c === this.active && this.fx.p.vis) UI.setBox('player', this.active);
      await UI.say(`Congratulations! Your ${oldName} evolved into ${creatureName(c)}!`);
      for (const entry of SPECIES[c.species].learnset.filter(e => e.lv === c.level)) {
        await this.learnMove(c, entry.move);
      }
    }
    this.evoQueue = [];
  }

  // ---------- Authentic 3-Frame FireRed Capture Choreography ----------
  async throwOrb(id) {
    Game.player.bag[id]--;
    await UI.say(`You threw a ${ITEMS[id].name}!`, { noWait: true, keep: true });
    Sfx.thrown();
    UI.hideBoxes();

    const orb = this.fx.orb;
    orb.ballRow = id === 'greatball' ? 1 : 0;
    orb.openState = 0;
    orb.locked = false;
    orb.rot = 0;
    orb.shadowY = 74;
    orb.vis = true;

    const E = this.combatantCenter('enemy');
    // High parabolic throw from player side to above enemy
    await this.arcTween(orb, 58, 96, E.cx, 32, 430, 38);

    // Ball pops half-open -> wide-open and pulls glowing magenta Pokemon inside!
    orb.openState = 1;
    await wait(55);
    orb.openState = 2;
    Sfx.statusFx();
    this.spawnEffect('BALL_OPEN_BURST', orb.x, orb.y + 4);
    this.fx.e.tintColor = '#ff78e8';
    this.fx.e.tintAlpha = 0.25;
    await Promise.all([
      this.tween(this.fx.e, 'tintAlpha', 0.95, 140),
      this.tween(this.fx.e, 'scale', 0.08, 300),
    ]);
    this.fx.e.vis = false;
    this.fx.e.scale = 1;
    this.fx.e.tintAlpha = 0;
    this.fx.e.tintColor = null;

    // Snap closed & drop onto the enemy grass oval with 2 GBA bounces!
    orb.openState = 1; await wait(45);
    orb.openState = 0;
    await this.tween(orb, 'y', 70, 210);
    await this.tween(orb, 'y', 58, 95);
    await this.tween(orb, 'y', 70, 95);
    await this.tween(orb, 'y', 65, 65);
    await this.tween(orb, 'y', 70, 65);

    // Gen 3 catch formula
    const en = this.enemy;
    const rate = SPECIES[en.species].catchRate;
    const ball = ITEMS[id].rate;
    const statusMul = (en.status === 'SLP' || en.status === 'FRZ') ? 2 : en.status ? 1.5 : 1;
    let a = Math.floor((3 * en.maxHp - 2 * en.hp) * rate * ball / (3 * en.maxHp)) * statusMul;
    a = Math.max(1, Math.floor(a));
    let shakes = 0;
    if (a >= 255) shakes = 4;
    else {
      const b = Math.floor(1048560 / Math.sqrt(Math.sqrt(Math.floor(16711680 / a))));
      for (let i = 0; i < 4; i++) {
        if (randInt(0, 65535) < b) shakes++;
        else break;
      }
    }

    for (let i = 0; i < Math.min(shakes, 3); i++) {
      await wait(460);
      Sfx.shake();
      await this.tween(orb, 'rot', -1.5, 105);
      await this.tween(orb, 'rot', 1.5, 165);
      await this.tween(orb, 'rot', 0, 105);
    }
    await wait(450);

    if (shakes === 4) {
      Sfx.catchClick();
      orb.locked = true;
      this.spawnEffect('CAUGHT_STARS', orb.x, orb.y - 6);
      await wait(560);
      Sfx.caught();
      UI.hideDialog();
      await UI.say(`Gotcha! ${creatureName(en)} was caught!`);
      orb.vis = false;
      orb.locked = false;
      this.enemyIsCaught = true;
      if (Game.player.party.length < 6) {
        Game.player.party.push(en);
        await UI.say(`${creatureName(en)} joined your team!`);
      } else {
        Game.player.vault.push(en);
        await UI.say(`Your team is full. ${creatureName(en)} was sent to the storage vault!`);
      }
      this.over = true;
      this.result = 'catch';
      return true;
    }

    // Breakout: ball snaps wide-open with white/pink burst as wild Pokemon rematerializes!
    Sfx.breakout();
    orb.openState = 2;
    this.spawnEffect('BALL_BREAK', orb.x, orb.y);
    this.shake = Math.max(this.shake, 10);
    await wait(70);
    orb.vis = false;
    orb.openState = 0;
    this.fx.e.vis = true;
    this.fx.e.scale = 0.2;
    this.fx.e.tintColor = '#ff88e8';
    this.fx.e.tintAlpha = 0.85;
    await Promise.all([
      this.tween(this.fx.e, 'scale', 1, 200),
      this.tween(this.fx.e, 'tintAlpha', 0, 220),
    ]);
    this.fx.e.tintColor = null;
    UI.setBox('enemy', this.enemy, { slide: true });
    UI.setBox('player', this.active, { slide: true });
    UI.setBattleAiPill(true, this.focusConceptId());
    UI.hideDialog();
    const msgs = [
      'Oh no! It broke free!',
      'Aww! It appeared to be caught!',
      'Aargh! Almost had it!',
      'Shoot! It was so close, too!',
    ];
    await UI.say(msgs[shakes]);
    return false;
  }

  // ---------- Tween helpers ----------
  tween(obj, prop, to, ms) {
    const from = obj[prop];
    return new Promise(res => {
      const t0 = performance.now();
      const step = (t) => {
        const k = Math.min(1, (t - t0) / ms);
        // Smooth ease-out for crisp GBA decelerations
        const e = 1 - Math.pow(1 - k, 1.65);
        obj[prop] = from + (to - from) * e;
        if (k < 1) requestAnimationFrame(step); else { obj[prop] = to; res(); }
      };
      requestAnimationFrame(step);
    });
  }

  arcTween(orb, x0, y0, x1, y1, ms, arcH = 42) {
    return new Promise(res => {
      const t0 = performance.now();
      const step = (t) => {
        const k = Math.min(1, (t - t0) / ms);
        orb.x = x0 + (x1 - x0) * k;
        orb.y = y0 + (y1 - y0) * k - arcH * Math.sin(Math.PI * k);
        if (k < 1) requestAnimationFrame(step); else res();
      };
      requestAnimationFrame(step);
    });
  }

  // ---------- Sprite-sheet & Combatant Rendering ----------
  drawCombatant(ctx, f, img, srcRect) {
    const sc = (f.scale === undefined ? 1 : f.scale);
    const dw = Math.round(64 * sc * (f.squashX || 1) * 2) / 2;
    const dh = Math.round(64 * sc * (f.squashY || 1) * 2) / 2;
    const baseX = Math.round((f.x + f.dx + (64 - dw) / 2) * 2) / 2;
    const baseY = Math.round((f.y + f.dy + (64 - dh)) * 2) / 2;

    const sx = srcRect ? srcRect.x : 0;
    const sy = srcRect ? srcRect.y : 0;
    const sw = srcRect ? srcRect.w : img.width;
    const sh = srcRect ? srcRect.h : img.height;

    ctx.save();
    if (f.clipBottom !== null && f.clipBottom !== undefined) {
      ctx.beginPath();
      ctx.rect(-80, -40, 420, f.clipBottom + 40);
      ctx.clip();
    }

    // Draw Quick Attack / Agility translucent afterimages
    if (f.ghosts && f.ghosts.length) {
      for (const g of f.ghosts) {
        ctx.save();
        ctx.globalAlpha = g.alpha || 0.4;
        ctx.drawImage(img, sx, sy, sw, sh, baseX + g.dx, baseY + (g.dy || 0), dw, dh);
        ctx.restore();
      }
    }

    // Check if we need pixel-masked composite effects (silhouette, energy tint, statWave, wavy)
    const now = performance.now();
    let activeWave = f.statWave;
    if (activeWave && now - activeWave.t0 >= activeWave.dur) {
      f.statWave = null;
      activeWave = null;
    }
    const needMask = (f.silhouette > 0.01) || (f.tintAlpha > 0.01 && f.tintColor) || !!activeWave;
    const fxCtx = needMask ? getFxBuffer() : null;

    let drawSrc = img;
    let dsx = sx, dsy = sy, dsw = sw, dsh = sh;

    if (fxCtx) {
      fxCtx.clearRect(0, 0, 64, 64);
      fxCtx.globalCompositeOperation = 'source-over';
      fxCtx.globalAlpha = 1;
      fxCtx.drawImage(img, sx, sy, sw, sh, 0, 0, 64, 64);

      // 1. Authentic GBA FireRed scrolling stat-stage chevron wave masked inside sprite!
      if (activeWave) {
        const waveImg = GameAssets.battleFx('stat_waves');
        if (waveImg) {
          const p = (now - activeWave.t0) / activeWave.dur;
          const col = (GameAssets.STAT_WAVE_INDEX[activeWave.stat] !== undefined)
            ? GameAssets.STAT_WAVE_INDEX[activeWave.stat]
            : 7;
          const scroll = ((now - activeWave.t0) * 0.14 * activeWave.dir) % 64;
          const wy = ((scroll % 64) + 64) % 64;
          const fade = p < 0.2 ? p / 0.2 : p > 0.8 ? (1 - p) / 0.2 : 1;
          fxCtx.globalCompositeOperation = 'source-atop';
          fxCtx.globalAlpha = 0.68 * Math.max(0, fade);
          fxCtx.drawImage(waveImg, col * 64, wy, 64, 64, 0, 0, 64, 64);
        }
      }

      // 2. Energy / Status color tint (Poke Ball pink beam, PSN purple, BRN orange, etc.)
      if (f.tintAlpha > 0.01 && f.tintColor) {
        fxCtx.globalCompositeOperation = 'source-atop';
        fxCtx.globalAlpha = Math.min(1, f.tintAlpha);
        fxCtx.fillStyle = f.tintColor;
        fxCtx.fillRect(0, 0, 64, 64);
      }

      // 3. Wild encounter intro dark GBA silhouette
      if (f.silhouette > 0.01) {
        fxCtx.globalCompositeOperation = 'source-atop';
        fxCtx.globalAlpha = Math.min(1, f.silhouette);
        fxCtx.fillStyle = '#18202c';
        fxCtx.fillRect(0, 0, 64, 64);
      }

      fxCtx.globalCompositeOperation = 'source-over';
      fxCtx.globalAlpha = 1;
      drawSrc = _fxBuf;
      dsx = 0; dsy = 0; dsw = 64; dsh = 64;
    }

    // GBA H-blank horizontal sine-wave distortion (Confusion / Psybeam)
    if (f.wavy > 0.02) {
      const slices = 16;
      const shSlice = dsh / slices;
      const dhSlice = dh / slices;
      for (let i = 0; i < slices; i++) {
        const wob = Math.sin(now * 0.028 + i * 0.65) * 4.5 * f.wavy;
        ctx.drawImage(
          drawSrc,
          dsx, dsy + i * shSlice, dsw, shSlice,
          baseX + wob, baseY + i * dhSlice, dw, dhSlice
        );
      }
    } else {
      ctx.drawImage(drawSrc, dsx, dsy, dsw, dsh, baseX, baseY, dw, dh);
    }

    ctx.restore();
  }

  drawAnimSprites(ctx) {
    const now = performance.now();
    const list = this.animSprites;
    for (let i = list.length - 1; i >= 0; i--) {
      const s = list[i];
      const elapsed = now - s.t0 - (s.delay || 0);
      if (elapsed < 0) continue;
      if (elapsed >= s.dur) { list.splice(i, 1); continue; }
      const k = elapsed / s.dur;

      const img = GameAssets.battleFx(s.sheet);
      if (!img) continue; // graceful fallback (particles still render)

      const fi = s.frames[Math.min(s.frames.length - 1, Math.floor(k * s.frames.length))];
      let x, y;
      if (s.orbit) {
        const o = s.orbit;
        const ang = o.phase0 + k * Math.PI * 2 * (o.turns || 1);
        x = o.cx + Math.cos(ang) * o.rx;
        y = o.cy + Math.sin(ang) * o.ry;
      } else {
        const x1 = s.x1 === null ? s.x0 : s.x1;
        const y1 = s.y1 === null ? s.y0 : s.y1;
        x = s.x0 + (x1 - s.x0) * k;
        y = s.y0 + (y1 - s.y0) * k - (s.arc ? s.arc * Math.sin(Math.PI * k) : 0);
      }

      const scale = s.s1 === null ? s.s0 : (s.s0 + (s.s1 - s.s0) * k);
      const rot = s.rot1 === null ? s.rot0 : (s.rot0 + (s.rot1 - s.rot0) * k);
      const alpha = s.alpha1 === null ? s.alpha0 : (s.alpha0 + (s.alpha1 - s.alpha0) * k);

      ctx.save();
      ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
      ctx.translate(Math.round(x * 2) / 2, Math.round(y * 2) / 2);
      if (rot) ctx.rotate(rot);
      if (s.flipX) ctx.scale(-1, 1);
      const dw = s.cw * scale;
      const dh = s.ch * scale;
      ctx.drawImage(
        img,
        fi * s.cw, (s.row || 0) * s.ch, s.cw, s.ch,
        -dw / 2, -dh / 2, dw, dh
      );
      ctx.restore();
    }
  }

  drawParticles(ctx) {
    const P = this.particles;
    for (let i = P.length - 1; i >= 0; i--) {
      const p = P[i];
      p.age++;
      if (p.age >= p.life) { P.splice(i, 1); continue; }
      p.x += p.vx; p.y += p.vy; p.vy += p.g || 0;
      const fade = 1 - p.age / p.life;
      ctx.globalAlpha = Math.max(0.15, fade);
      ctx.fillStyle = p.color;
      ctx.strokeStyle = p.color;
      if (p.kind === 'rect') {
        ctx.fillRect(p.x, p.y, p.size, p.size);
      } else if (p.kind === 'ring') {
        const rad = p.grow ? p.size + p.age * p.grow : p.size;
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(p.x, p.y, rad, 0, Math.PI * 2); ctx.stroke();
      } else if (p.kind === 'streak') {
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x + p.vx * p.size, p.y + p.vy * p.size); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  drawOrb(ctx, orb) {
    const pbSheet = GameAssets.battleFx('pokeballs');
    // Ground shadow on platform
    ctx.fillStyle = 'rgba(20, 36, 20, 0.28)';
    ctx.beginPath();
    ctx.ellipse(orb.x, orb.shadowY || 74, 7, 2.5, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.save();
    ctx.translate(Math.round((orb.x + orb.rot * 2.5) * 2) / 2, Math.round(orb.y * 2) / 2);
    ctx.rotate(orb.rot * 0.42);
    if (pbSheet) {
      const col = Math.min(2, Math.max(0, orb.openState || 0));
      const row = orb.ballRow || 0;
      ctx.drawImage(pbSheet, col * 24, row * 24, 24, 24, -12, -12, 24, 24);
      if (orb.locked) {
        ctx.globalCompositeOperation = 'source-atop';
        ctx.fillStyle = 'rgba(24, 24, 40, 0.35)';
        ctx.fillRect(-12, -12, 24, 24);
      }
    } else {
      // Fallback code-drawn Poke Ball (AGENTS.md Rule #2)
      ctx.fillStyle = '#303038';
      ctx.beginPath(); ctx.arc(0, 0, 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = orb.ballRow === 1 ? '#3878d8' : '#e04838';
      ctx.beginPath(); ctx.arc(0, -0.5, 4, Math.PI, 0); ctx.fill();
      ctx.fillStyle = '#f0f0f0';
      ctx.beginPath(); ctx.arc(0, 0.8, 4, 0, Math.PI); ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.fillRect(-1, -1, 2, 2);
    }
    ctx.restore();
  }

  draw(ctx) {
    ctx.save();
    if (this.shake > 0) {
      this.shake--;
      const m = Math.min(3.2, this.shake * 0.42);
      ctx.translate((Math.random() * 2 - 1) * m, (Math.random() * 2 - 1) * m);
    }

    const W = Game.viewW;
    const AY = FR_STAGE.arenaY;
    const AH = FR_STAGE.arenaH;
    const sceneX = Math.round((W - 240)) / 2;

    const sky = GameAssets.battleSky(this.mapId);
    const ePad = GameAssets.battleEnemyPad(this.mapId);
    const pPad = GameAssets.battlePlayerPad(this.mapId);
    const fullBg = GameAssets.battleBackground(this.mapId);

    // 1. Arena Sky / Scanline Backdrop across entire viewport width
    if (sky) {
      ctx.drawImage(sky, 0, 0, 8, 1, -4, -4, W + 8, AY + 4);
      ctx.drawImage(sky, 0, 0, 8, AH, -4, AY, W + 8, AH);
    } else if (fullBg) {
      ctx.drawImage(fullBg, 0, 0, 8, 1, -4, -4, W + 8, AY + 4);
      ctx.drawImage(fullBg, 0, 0, 8, AH, -4, AY, W + 8, AH);
      ctx.drawImage(fullBg, sceneX, AY, 240, AH);
    } else {
      ctx.fillStyle = '#d8f4d8';
      ctx.fillRect(-4, -4, W + 8, AY + AH + 4);
    }

    // Atmospheric move sky flash / dim (Flamethrower, Psychic, Slash, Scary Face...)
    if (this.scene.bgTint) {
      const bt = this.scene.bgTint;
      const p = (performance.now() - bt.t0) / bt.dur;
      if (p >= 1) {
        this.scene.bgTint = null;
      } else {
        const fade = Math.sin(Math.PI * p);
        ctx.save();
        ctx.globalAlpha = bt.alpha * fade;
        ctx.fillStyle = bt.color;
        ctx.fillRect(-4, -4, W + 8, AY + AH + 4);
        ctx.restore();
      }
    }

    // 2. Centered 240x112 GBA FireRed Battle Stage
    ctx.save();
    ctx.beginPath();
    ctx.rect(-4, 0, W + 8, AY + AH);
    ctx.clip();
    ctx.translate(sceneX, AY);

    // Sliding Enemy & Player Platforms (or fallback ellipses)
    const exOff = Math.round(this.scene.ePadDx * 2) / 2;
    const pxOff = Math.round(this.scene.pPadDx * 2) / 2;
    if (sky && ePad && pPad) {
      ctx.drawImage(ePad, 104 + exOff, 32);
      ctx.drawImage(pPad, 0 + pxOff, 84);
    } else if (!fullBg) {
      ctx.fillStyle = '#98e080';
      ctx.beginPath(); ctx.ellipse(172 + exOff, 68, 56, 18, 0, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.ellipse(64 + pxOff, 108, 58, 14, 0, 0, Math.PI * 2); ctx.fill();
    }

    const e = this.fx.e, p = this.fx.p, orb = this.fx.orb;
    if (e.flash > 0) e.flash--;
    if (p.flash > 0) p.flash--;

    // Ground contact shadow under enemy Pokemon
    if (e.vis && !e.isTrainer) {
      ctx.fillStyle = 'rgba(20, 44, 20, 0.22)';
      ctx.beginPath();
      ctx.ellipse(e.x + e.dx + 32, 74, 22 * (e.scale || 1), 5.5 * (e.scale || 1), 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw Enemy (Trainer or Pokemon)
    if (e.vis && (e.flash <= 0 || Math.floor(e.flash / 3) % 2 === 0)) {
      if (e.isTrainer) {
        const trImg = GameAssets.trainerFront(this.trainer && this.trainer.flag);
        if (trImg) this.drawCombatant(ctx, e, trImg, null);
      } else {
        const img = GameAssets.frontFor(this.enemy.species, this.enemy.shiny)
          || Sprites.creature(this.enemy.species);
        this.drawCombatant(ctx, e, img, null);
      }
    }

    // Draw Player (5-frame Red Back-Throw or Pokemon Back-Sprite)
    if (p.vis && (p.flash <= 0 || Math.floor(p.flash / 3) % 2 === 0)) {
      if (p.isTrainer) {
        const throwStrip = GameAssets.trainerBackThrow();
        const trBack = GameAssets.trainerBack();
        if (throwStrip) {
          const fi = Math.min(4, Math.max(0, p.trFrame || 0));
          this.drawCombatant(ctx, p, throwStrip, { x: fi * 64, y: 0, w: 64, h: 64 });
        } else if (trBack) {
          this.drawCombatant(ctx, p, trBack, null);
        }
      } else {
        const img = GameAssets.backFor(this.active.species, this.active.shiny)
          || Sprites.creature(this.active.species);
        this.drawCombatant(ctx, p, img, null);
      }
    }

    // Draw Poke Ball & Sprite-Sheet Move/Status Animations & Particles
    if (orb.vis) this.drawOrb(ctx, orb);
    this.drawAnimSprites(ctx);
    this.drawParticles(ctx);

    ctx.restore(); // end 240x112 arena clip

    // 3. Authentic GBA FireRed Bottom Battle Message Bar Frame (y = 118..160)
    const barY = AY + AH;
    const barH = 160 - barY;
    ctx.fillStyle = '#283038';
    ctx.fillRect(-4, barY, W + 8, barH + 4);
    ctx.fillStyle = '#c8a848';
    ctx.fillRect(2, barY + 2, W - 4, barH - 4);
    ctx.fillStyle = '#e8d078';
    ctx.fillRect(3, barY + 3, W - 6, 1);
    ctx.fillStyle = '#204858';
    ctx.fillRect(5, barY + 4, W - 10, barH - 8);

    // 4. GBA Horizontal Split Scanline Curtain Wipe during battle intro!
    if (this.scene.curtain > 0.005) {
      const c = this.scene.curtain;
      ctx.fillStyle = '#08080c';
      const bandH = 4;
      const span = (W + 16) * c;
      for (let y = 0, row = 0; y < 160; y += bandH, row++) {
        if (row % 2 === 0) {
          ctx.fillRect(-8, y, span, bandH);
        } else {
          ctx.fillRect(W + 8 - span, y, span, bandH);
        }
      }
    }

    ctx.restore();
  }
}
