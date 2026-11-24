// ===== WILDVALE — input + UI layer (dialog, menus, screens, battle HUD) =====

const wait = ms => new Promise(r => setTimeout(r, ms));
const randInt = (lo, hi) => lo + Math.floor(Math.random() * (hi - lo + 1)); // inclusive

// ---------------- Input ----------------
const Input = (() => {
  const held = {};
  const pressed = {};
  const listeners = [];

  function mapKey(key) {
    switch (key) {
      case 'ArrowUp': case 'w': case 'W': return 'up';
      case 'ArrowDown': case 's': case 'S': return 'down';
      case 'ArrowLeft': case 'a': case 'A': return 'left';
      case 'ArrowRight': case 'd': case 'D': return 'right';
      case 'z': case 'Z': case ' ': return 'a';
      case 'x': case 'X': case 'Escape': case 'Backspace': return 'b';
      case 'Enter': return 'start';
      case 'm': case 'M': return 'mute';
      default: return null;
    }
  }

  // Shared entry point for keyboard AND touch controls — both produce the
  // exact same press/release semantics (held state + one-shot dispatch).
  function press(name) {
    held[name] = true;
    Sfx.unlock();
    if (name === 'mute') { UI.toast(Sfx.toggle() ? 'SOUND: ON' : 'SOUND: OFF'); return; }
    // A waiting async flow consumes the press first.
    for (let i = 0; i < listeners.length; i++) {
      const li = listeners[i];
      if (!li.keys || li.keys.includes(name)) {
        listeners.splice(i, 1);
        li.resolve(name);
        return;
      }
    }
    pressed[name] = true;
  }
  function release(name) {
    held[name] = false;
  }

  window.addEventListener('keydown', (e) => {
    if (e.target && (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') && e.key !== 'Escape') return;
    const name = mapKey(e.key);
    if (!name) return;
    e.preventDefault();
    if (e.repeat) { held[name] = true; return; }
    press(name);
  });
  window.addEventListener('keyup', (e) => {
    const name = mapKey(e.key);
    if (name) release(name);
  });

  return {
    held,
    dirHeld() {
      for (const d of ['up', 'down', 'left', 'right']) if (held[d]) return d;
      return null;
    },
    consume(name) {
      if (pressed[name]) { pressed[name] = false; return true; }
      return false;
    },
    clearPressed() { for (const k of Object.keys(pressed)) pressed[k] = false; },
    waitButton(...keys) {
      return new Promise(resolve => listeners.push({ keys: keys.length ? keys : null, resolve }));
    },
    // Register a one-shot listener; returns an unsubscribe function.
    listen(keys, cb) {
      const li = { keys, resolve: cb };
      listeners.push(li);
      return () => { const i = listeners.indexOf(li); if (i >= 0) listeners.splice(i, 1); };
    },
    // Touch-control entry points (same semantics as keydown/keyup).
    press, release,
  };
})();

// ---------------- UI ----------------
const UI = (() => {
  const $ = id => document.getElementById(id);
  let dlg, dlgText, dlgArrow, dlgPortraitWrap, dlgPortrait, dlgName, menuLayer, fadeEl;

  function init() {
    dlg = $('dialog'); dlgText = $('dialog-text'); dlgArrow = $('dialog-arrow');
    dlgPortraitWrap = $('dialog-portrait-wrap');
    dlgPortrait = $('dialog-portrait');
    dlgName = $('dialog-name');
    menuLayer = $('menu-layer'); fadeEl = $('fade');
  }

  // Speaker profile map for portraits and badges (Pokemon Void visual reference)
  const SPEAKER_PROFILES = {
    'MAPLE': { name: 'Prof. Ada', portrait: 'assets/portraits/prof.png' },
    'PROF': { name: 'Prof. Ada', portrait: 'assets/portraits/prof.png' },
    'ADA': { name: 'Prof. Ada', portrait: 'assets/portraits/prof.png' },
    'MOM': { name: 'Mom', portrait: 'assets/portraits/mom.png' },
    'REX': { name: 'Rex', portrait: 'assets/portraits/rival.png' },
    'NURSE': { name: 'Nurse Joy', portrait: 'assets/portraits/nurse.png' },
    'CLERK': { name: 'Shop Clerk', portrait: 'assets/portraits/seren.png' },
    'LIN': { name: 'Technician Lin', portrait: 'assets/portraits/nia.png' },
    'RIVAL_MOM': { name: "Rex's Mother", portrait: 'assets/portraits/rival_mom.png' },
    'MOTHER': { name: "Rex's Mother", portrait: 'assets/portraits/rival_mom.png' },
    'DAISY': { name: "Rex's Mother", portrait: 'assets/portraits/rival_mom.png' },
  };

  // Creature thumbnail: real front sprite if loaded, else the drawn fallback.
  function creatureThumb(species, px, shiny) {
    const img = GameAssets.frontFor(species, shiny);
    if (img) {
      const el = document.createElement('img');
      el.src = img.src;
      el.style.cssText = `width:${px}px;height:${px}px;image-rendering:pixelated;object-fit:contain`;
      return el;
    }
    const cv = document.createElement('canvas');
    cv.width = 16; cv.height = 16;
    cv.getContext('2d').drawImage(Sprites.creature(species), 0, 0);
    cv.style.cssText = `width:${px}px;height:${px}px;image-rendering:pixelated`;
    return cv;
  }

  // Item icon HTML (empty string when no icon is available).
  function itemIconHtml(itemId) {
    const url = GameAssets.itemIconUrl(itemId);
    return url ? `<img src="${url}" class="item-icon" onerror="this.remove()" alt="">` : '';
  }

  // ----- dialog / typewriter -----
  async function say(rawText, opts = {}) {
    dlg.classList.remove('hidden');
    dlgArrow.classList.add('hidden');
    
    let displayText = rawText;
    let speakerName = opts.name || null;
    let speakerPortrait = opts.portrait || null;
    
    // Automatically detect Speaker tag (e.g. "MAPLE: Welcome to Netoria!")
    const prefixMatch = rawText.match(/^([A-Za-z0-9_]+):\s*([\s\S]*)$/);
    if (prefixMatch) {
      const tag = prefixMatch[1].toUpperCase();
      if (SPEAKER_PROFILES[tag]) {
        speakerName = speakerName || SPEAKER_PROFILES[tag].name;
        speakerPortrait = speakerPortrait || SPEAKER_PROFILES[tag].portrait;
        displayText = prefixMatch[2];
      }
    }
    
    if (speakerPortrait && dlgPortrait && dlgPortraitWrap) {
      dlgPortrait.src = speakerPortrait;
      dlgPortraitWrap.classList.remove('hidden');
    } else if (dlgPortraitWrap) {
      dlgPortraitWrap.classList.add('hidden');
    }
    
    if (speakerName && dlgName) {
      dlgName.textContent = speakerName;
      dlgName.classList.remove('hidden');
    } else if (dlgName) {
      dlgName.classList.add('hidden');
    }

    dlgText.textContent = '';
    let i = 0, skipped = false;
    const unsub = Input.listen(['a', 'b', 'start'], () => { skipped = true; });
    while (i < displayText.length) {
      if (skipped) { dlgText.textContent = displayText; break; }
      i += 2;
      dlgText.textContent = displayText.slice(0, i);
      if (i % 6 === 0) Sfx.blip();
      await wait(24);
    }
    unsub();
    if (opts.noWait) { if (!opts.keep) hideDialog(); return; }
    dlgArrow.classList.remove('hidden');
    await Input.waitButton('a', 'b', 'start');
    Sfx.select();
    dlgArrow.classList.add('hidden');
    if (!opts.keep) hideDialog();
  }

  async function sayLines(lines, opts = {}) {
    for (let i = 0; i < lines.length; i++) {
      await say(lines[i], { keep: opts.keep || i < lines.length - 1 });
    }
  }

  function setPrompt(text) {
    dlg.classList.remove('hidden');
    dlgArrow.classList.add('hidden');
    if (dlgPortraitWrap) dlgPortraitWrap.classList.add('hidden');
    if (dlgName) dlgName.classList.add('hidden');
    dlgText.textContent = text;
  }

  function hideDialog() {
    dlg.classList.add('hidden');
    dlgText.textContent = '';
    dlgArrow.classList.add('hidden');
    if (dlgPortraitWrap) dlgPortraitWrap.classList.add('hidden');
    if (dlgName) dlgName.classList.add('hidden');
  }

  // ----- generic menu -----
  // options: array of strings (or {label, dim}). opts: {style, cols, canCancel, startIdx}
  async function choose(options, opts = {}) {
    const box = document.createElement('div');
    box.className = 'menu gba-box' + (opts.cols === 2 ? ' grid-menu' : '');
    Object.assign(box.style, opts.style || { right: '14px', bottom: '126px' });
    const items = options.map(o => {
      const d = document.createElement('div');
      d.className = 'mi' + ((o && o.dim) ? ' dim' : '');
      d.textContent = typeof o === 'string' ? o : o.label;
      box.appendChild(d);
      return d;
    });
    menuLayer.appendChild(box);
    let idx = opts.startIdx || 0;
    const cols = opts.cols || 1;
    const paint = () => items.forEach((d, i) => d.classList.toggle('sel', i === idx));
    paint();
    try {
      while (true) {
        const k = await Input.waitButton('up', 'down', 'left', 'right', 'a', 'b', 'start');
        if (k === 'a' || k === 'start') { Sfx.select(); return idx; }
        if (k === 'b') {
          if (opts.canCancel === false) continue;
          Sfx.cancel(); return -1;
        }
        const n = options.length;
        let next = idx;
        if (cols === 1) {
          if (k === 'up') next = (idx - 1 + n) % n;
          if (k === 'down') next = (idx + 1) % n;
        } else {
          if (k === 'left' && idx % cols > 0) next = idx - 1;
          if (k === 'right' && idx % cols < cols - 1 && idx + 1 < n) next = idx + 1;
          if (k === 'up' && idx - cols >= 0) next = idx - cols;
          if (k === 'down' && idx + cols < n) next = idx + cols;
        }
        if (next !== idx) { idx = next; Sfx.cursor(); paint(); }
      }
    } finally {
      box.remove();
    }
  }

  async function yesNo(prompt) {
    if (prompt) await say(prompt, { noWait: true, keep: true });
    const r = await choose(['YES', 'NO'], { style: { right: '14px', bottom: '126px' } });
    hideDialog();
    return r === 0;
  }

  // ----- toast -----
  let toastTimer = null;
  function toast(text) {
    let t = document.getElementById('ui-toast');
    if (!t) {
      t = document.createElement('div');
      t.id = 'ui-toast';
      t.className = 'gba-box';
      Object.assign(t.style, {
        position: 'absolute', left: '12px', top: '12px', padding: '10px 14px',
        fontSize: '11px', zIndex: 95,
      });
      menuLayer.appendChild(t);
    }
    t.textContent = text;
    t.style.display = 'block';
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.style.display = 'none'; }, 1200);
  }

  // ----- fades / flashes -----
  function fadeOut(ms = 300) {
    fadeEl.classList.remove('flash');
    fadeEl.style.transition = `opacity ${ms}ms linear`;
    fadeEl.classList.add('on');
    return wait(ms + 30);
  }
  function fadeIn(ms = 300) {
    fadeEl.classList.remove('flash');
    fadeEl.style.transition = `opacity ${ms}ms linear`;
    fadeEl.classList.remove('on');
    return wait(ms + 30);
  }
  async function flashScreen(times = 3) {
    fadeEl.classList.add('flash');
    for (let i = 0; i < times; i++) {
      fadeEl.classList.add('on'); await wait(70);
      fadeEl.classList.remove('on'); await wait(70);
    }
    fadeEl.classList.remove('flash');
  }

  // ----- battle HUD -----
  function hpClass(frac) { return frac > 0.5 ? '' : frac > 0.2 ? 'mid' : 'low'; }

  function boxEls(side) {
    const root = $(side === 'player' ? 'player-box' : 'enemy-box');
    return {
      root,
      name: root.querySelector('.ib-name'),
      lv: root.querySelector('.ib-lv'),
      st: root.querySelector('.ib-status'),
      fill: root.querySelector('.hp-fill'),
      hpnum: root.querySelector('.ib-hpnum'),
      exp: root.querySelector('.exp-fill'),
    };
  }

  function setBox(side, c, opts = {}) {
    const e = boxEls(side);
    const wasHidden = e.root.classList.contains('hidden');
    e.root.classList.remove('hidden');
    if (opts.slide || wasHidden) {
      e.root.classList.remove('slide-enter');
      void e.root.offsetWidth;
      e.root.classList.add('slide-enter');
    }
    e.name.textContent = creatureName(c);
    e.lv.textContent = 'Lv' + c.level;
    setStatusChip(side, c.status);
    const frac = Math.max(0, c.hp / c.maxHp);
    e.fill.style.width = (frac * 100) + '%';
    e.fill.className = 'hp-fill ' + hpClass(frac);
    if (e.hpnum) e.hpnum.textContent = `${Math.max(0, c.hp)}/ ${c.maxHp}`;
    if (e.exp && side === 'player') e.exp.style.width = (expProgress(c) * 100) + '%';
  }

  function setStatusChip(side, status) {
    const e = boxEls(side);
    const rect = (status && GameAssets.STATUS_ICON_RECT && GameAssets.battleFx('type_status_icons'))
      ? GameAssets.STATUS_ICON_RECT[status]
      : null;
    if (rect) {
      e.st.className = `ib-status on px-icon st-${status}`;
      e.st.style.backgroundPosition = `${-rect.x * 2}px ${-rect.y * 2}px`;
      e.st.textContent = status;
    } else {
      e.st.className = 'ib-status' + (status ? ` on st-${status}` : '');
      e.st.style.backgroundPosition = '';
      e.st.textContent = status || '';
    }
  }

  // Trainer battles: show the opponent's team as ball icons, darkening each
  // one as it faints. Pass null (or a wild battle) to hide the row.
  function setEnemyBalls(party) {
    const row = document.querySelector('#enemy-box .ib-balls');
    if (!row) return;
    row.innerHTML = '';
    if (!party) return;
    const url = GameAssets.itemIconUrl('pokeball');
    for (const c of party) {
      const img = document.createElement('img');
      img.src = url;
      img.alt = '';
      if (c.hp <= 0) img.classList.add('down');
      row.appendChild(img);
    }
  }

  function hideBoxes() {
    $('player-box').classList.add('hidden');
    $('enemy-box').classList.add('hidden');
    setBattleAiPill(false);
  }

  function escHtml(str) {
    return String(str == null ? '' : str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function setBattleAiPill(visible, focusConceptId = null) {
    let pill = document.getElementById('ai-battle-pill');
    if (!visible || typeof AITutor === 'undefined') {
      if (pill) pill.classList.add('hidden');
      return;
    }
    if (!pill) {
      pill = document.createElement('div');
      pill.id = 'ai-battle-pill';
      menuLayer.parentElement.appendChild(pill);
    }
    const st = AITutor.getStatus();
    const cid = (focusConceptId !== null && focusConceptId !== undefined)
      ? focusConceptId
      : (st.frontier_concept_id || 0);
    const cInfo = (st.concepts && st.concepts[cid]) || { short_title: `C${cid}`, mastery: 0.4 };
    const mPct = Math.round((cInfo.mastery || 0) * 100);
    const live = AITutor.isOnline();
    pill.innerHTML = `<span class="ai-dot ${live ? '' : 'offline'}"></span>` +
      `<span>${live ? '3-MODEL AI' : 'LOCAL AI'} &middot; C${cid} ${escHtml(cInfo.short_title || '')} ${mPct}% &middot; &#9733; ${st.badges_earned_count || 0}/5</span>`;
    pill.classList.remove('hidden');
  }

  // Tween HP from -> to over ~600ms.
  function animateHp(side, from, to, max) {
    const e = boxEls(side);
    return new Promise(res => {
      const dur = Math.min(900, 250 + Math.abs(from - to) * 14);
      const t0 = performance.now();
      function step(t) {
        const k = Math.min(1, (t - t0) / dur);
        const hp = Math.round(from + (to - from) * k);
        const frac = Math.max(0, hp / max);
        e.fill.style.width = (frac * 100) + '%';
        e.fill.className = 'hp-fill ' + hpClass(frac);
        if (e.hpnum) e.hpnum.textContent = `${Math.max(0, hp)}/ ${max}`;
        if (k < 1) requestAnimationFrame(step); else res();
      }
      requestAnimationFrame(step);
    });
  }

  function animateExp(fromFrac, toFrac) {
    const e = boxEls('player');
    return new Promise(res => {
      const dur = 500;
      const t0 = performance.now();
      function step(t) {
        const k = Math.min(1, (t - t0) / dur);
        e.exp.style.width = ((fromFrac + (toFrac - fromFrac) * k) * 100) + '%';
        if (k < 1) requestAnimationFrame(step); else res();
      }
      requestAnimationFrame(step);
    });
  }

  // ----- battle move menu -----
  async function moveMenu(c) {
    const box = document.createElement('div');
    box.className = 'menu gba-box grid-menu';
    Object.assign(box.style, {
      left: '10px', bottom: '10px', width: '486px', height: '108px',
      padding: '18px 20px',
    });
    const panel = document.createElement('div');
    panel.className = 'gba-box';
    panel.id = 'move-panel';
    menuLayer.appendChild(box);
    menuLayer.appendChild(panel);
    const items = c.moves.map(m => {
      const d = document.createElement('div');
      d.className = 'mi';
      d.textContent = MOVES[m.id].name;
      box.appendChild(d);
      return d;
    });
    let idx = 0;
    const paint = () => {
      items.forEach((d, i) => d.classList.toggle('sel', i === idx));
      const m = c.moves[idx];
      const tName = MOVES[m.id].type;
      const tr = (GameAssets.TYPE_ICON_RECT && GameAssets.battleFx('type_status_icons'))
        ? GameAssets.TYPE_ICON_RECT[tName]
        : null;
      const typeHtml = tr
        ? `TYPE/ <span class="fr-type-badge" style="background-position:${-tr.x * 2}px ${-tr.y * 2}px" title="${tName}"></span>`
        : `TYPE/${tName}`;
      panel.innerHTML = `PP&nbsp;&nbsp;${m.pp}/${m.maxPp}<br>${typeHtml}`;
    };
    paint();
    try {
      while (true) {
        const k = await Input.waitButton('up', 'down', 'left', 'right', 'a', 'b', 'start');
        if (k === 'a' || k === 'start') {
          if (c.moves[idx].pp <= 0) { Sfx.cancel(); setPrompt("There's no PP left for this move!"); await wait(900); continue; }
          Sfx.select(); return idx;
        }
        if (k === 'b') { Sfx.cancel(); return -1; }
        const n = items.length;
        let next = idx;
        if (k === 'left' && idx % 2 === 1) next = idx - 1;
        if (k === 'right' && idx % 2 === 0 && idx + 1 < n) next = idx + 1;
        if (k === 'up' && idx >= 2) next = idx - 2;
        if (k === 'down' && idx + 2 < n) next = idx + 2;
        if (next !== idx) { idx = next; Sfx.cursor(); paint(); }
      }
    } finally {
      box.remove(); panel.remove();
    }
  }

  // ----- party screen -----
  // mode: 'view' | 'switch' | 'target'  forced: can't cancel
  async function partyScreen(opts = {}) {
    const party = Game.player.party;
    const scr = document.createElement('div');
    scr.className = 'full-screen';
    scr.innerHTML = `<div class="fs-title">${opts.title || 'PARTY'}</div>
      <div class="fs-foot">${opts.forced ? 'Z: choose' : 'Z: choose &nbsp; X: back'}</div>`;
    const list = document.createElement('div');
    scr.appendChild(list);
    menuLayer.appendChild(scr);

    if (!party.length) {
      list.innerHTML = '<div style="font-size:12px;color:#9ab;padding:20px;line-height:1.8">Your party is empty.<br>Visit Prof. MAPLE\'s lab for your first partner!</div>';
      await Input.waitButton('a', 'b', 'start');
      Sfx.cancel();
      scr.remove();
      return -1;
    }

    function render(sel) {
      list.innerHTML = '';
      party.forEach((c, i) => {
        const slot = document.createElement('div');
        slot.className = 'party-slot' + (i === sel ? ' sel' : '') + (c.hp <= 0 ? ' fainted' : '');
        const cv = creatureThumb(c.species, 48, c.shiny);
        const frac = Math.max(0, c.hp / c.maxHp);
        const cls = frac > 0.5 ? '' : frac > 0.2 ? 'mid' : 'low';
        const stTag = c.hp <= 0 ? 'FNT' : c.status;
        slot.appendChild(cv);
        const info = document.createElement('div');
        info.className = 'ps-info';
        info.innerHTML = `<div class="ps-name">${creatureName(c)} <span style="color:#9ab">Lv${c.level}</span>
            ${stTag ? `<span class="ps-st st-${stTag}">${stTag}</span>` : ''}</div>
          <div class="ps-hpline"><div class="ps-hptrack"><div class="ps-hpfill ${cls}" style="width:${frac * 100}%"></div></div>
          <span>${Math.max(0, c.hp)}/${c.maxHp}</span></div>`;
        slot.appendChild(info);
        list.appendChild(slot);
      });
    }

    let idx = 0;
    render(idx);
    try {
      while (true) {
        const k = await Input.waitButton('up', 'down', 'a', 'b', 'start');
        if (k === 'up') { idx = (idx - 1 + party.length) % party.length; Sfx.cursor(); render(idx); }
        else if (k === 'down') { idx = (idx + 1) % party.length; Sfx.cursor(); render(idx); }
        else if (k === 'b') {
          if (opts.forced) { Sfx.cancel(); continue; }
          Sfx.cancel(); return -1;
        }
        else if (k === 'a' || k === 'start') { Sfx.select(); return idx; }
      }
    } finally {
      scr.remove();
    }
  }

  // ----- bag screen -----
  // Returns itemId or null. opts: {inBattle}
  async function bagScreen(opts = {}) {
    const entries = Object.entries(Game.player.bag).filter(([, n]) => n > 0);
    const scr = document.createElement('div');
    scr.className = 'full-screen';
    scr.innerHTML = `<div class="fs-title">BAG</div>
      <div class="fs-foot">Z: use &nbsp; X: back</div>`;
    const list = document.createElement('div');
    const desc = document.createElement('div');
    desc.className = 'bag-desc';
    scr.appendChild(list); scr.appendChild(desc);
    menuLayer.appendChild(scr);

    if (!entries.length) {
      list.innerHTML = '<div style="font-size:12px;color:#9ab;padding:20px">The bag is empty.</div>';
      await Input.waitButton('a', 'b', 'start');
      Sfx.cancel();
      scr.remove();
      return null;
    }

    let idx = 0;
    const render = () => {
      list.innerHTML = '';
      entries.forEach(([id, n], i) => {
        const row = document.createElement('div');
        row.className = 'bag-row' + (i === idx ? ' sel' : '');
        row.innerHTML = `<span>${itemIconHtml(id)}${ITEMS[id].name}</span><span>x${String(n).padStart(2, ' ')}</span>`;
        list.appendChild(row);
      });
      desc.textContent = ITEMS[entries[idx][0]].desc;
    };
    render();
    try {
      while (true) {
        const k = await Input.waitButton('up', 'down', 'a', 'b', 'start');
        if (k === 'up') { idx = (idx - 1 + entries.length) % entries.length; Sfx.cursor(); render(); }
        else if (k === 'down') { idx = (idx + 1) % entries.length; Sfx.cursor(); render(); }
        else if (k === 'b') { Sfx.cancel(); return null; }
        else if (k === 'a' || k === 'start') { Sfx.select(); return entries[idx][0]; }
      }
    } finally {
      scr.remove();
    }
  }

  // ----- shop -----
  async function shopScreen() {
    const scr = document.createElement('div');
    scr.className = 'full-screen';
    scr.innerHTML = `<div class="fs-title">POKE MART</div>
      <div class="money-chip gba-box">$<span id="shop-money"></span></div>
      <div class="fs-foot">Z: buy &nbsp; LEFT/RIGHT: qty &nbsp; X: leave</div>`;
    const list = document.createElement('div');
    const desc = document.createElement('div');
    desc.className = 'bag-desc';
    scr.appendChild(list); scr.appendChild(desc);
    menuLayer.appendChild(scr);

    let idx = 0, qty = 1;
    const render = () => {
      document.getElementById('shop-money').textContent = Game.player.money;
      list.innerHTML = '';
      SHOP_STOCK.forEach((id, i) => {
        const row = document.createElement('div');
        row.className = 'bag-row' + (i === idx ? ' sel' : '');
        const q = i === idx ? ` x${qty} = $${ITEMS[id].price * qty}` : '';
        row.innerHTML = `<span>${itemIconHtml(id)}${ITEMS[id].name}</span><span>$${ITEMS[id].price}${q}</span>`;
        list.appendChild(row);
      });
      desc.textContent = ITEMS[SHOP_STOCK[idx]].desc;
    };
    render();
    try {
      while (true) {
        const k = await Input.waitButton('up', 'down', 'left', 'right', 'a', 'b', 'start');
        if (k === 'up') { idx = (idx - 1 + SHOP_STOCK.length) % SHOP_STOCK.length; qty = 1; Sfx.cursor(); render(); }
        else if (k === 'down') { idx = (idx + 1) % SHOP_STOCK.length; qty = 1; Sfx.cursor(); render(); }
        else if (k === 'left') { qty = Math.max(1, qty - 1); Sfx.cursor(); render(); }
        else if (k === 'right') { qty = Math.min(99, qty + 1); Sfx.cursor(); render(); }
        else if (k === 'b') { Sfx.cancel(); return; }
        else if (k === 'a' || k === 'start') {
          const id = SHOP_STOCK[idx];
          const cost = ITEMS[id].price * qty;
          if (Game.player.money < cost) { Sfx.cancel(); toast('Not enough money!'); continue; }
          Game.player.money -= cost;
          Game.player.bag[id] = (Game.player.bag[id] || 0) + qty;
          Sfx.buy(); toast(`Bought ${qty} ${ITEMS[id].name}!`);
          qty = 1; render();
        }
      }
    } finally {
      scr.remove();
    }
  }

  // ----- starter choice -----
  async function starterChoice() {
    const ids = ['bulbasaur', 'charmander', 'squirtle'];
    const scr = document.createElement('div');
    scr.className = 'full-screen';
    scr.innerHTML = `<div class="fs-title" style="text-align:center;margin-top:20px">CHOOSE YOUR PARTNER!</div>
      <div id="starter-row"></div>
      <div class="fs-foot">LEFT/RIGHT: browse &nbsp; Z: choose &nbsp; X/ESC: back</div>`;
    menuLayer.appendChild(scr);
    const row = scr.querySelector('#starter-row');
    const cards = ids.map(id => {
      const sp = SPECIES[id];
      const card = document.createElement('div');
      card.className = 'starter-card';
      card.appendChild(creatureThumb(id, 96));
      const t = sp.types[0];
      card.insertAdjacentHTML('beforeend',
        `<div class="sc-name">${sp.name}</div>
         <div class="sc-type" style="background:${TYPES[t].color}">${t}</div>
         <div class="sc-blurb">${sp.blurb}</div>`);
      row.appendChild(card);
      return card;
    });
    let idx = 1;
    const paint = () => cards.forEach((c, i) => c.classList.toggle('sel', i === idx));
    paint();
    try {
      while (true) {
        const k = await Input.waitButton('left', 'right', 'a', 'b', 'start');
        if (k === 'left') { idx = (idx + 2) % 3; Sfx.cursor(); paint(); }
        else if (k === 'right') { idx = (idx + 1) % 3; Sfx.cursor(); paint(); }
        else if (k === 'b') { Sfx.cancel(); return null; }
        else { Sfx.select(); return ids[idx]; }
      }
    } finally {
      scr.remove();
    }
  }

  // ----- save / import modals -----
  function saveModal(code) {
    return new Promise((resolve) => {
      const scr = document.createElement('div');
      scr.className = 'full-screen save-modal';
      scr.innerHTML = `<div class="fs-title">SAVE CODE</div>
        <div style="font-size:10px;line-height:1.8;color:#b8c4cc;margin-bottom:10px">
          Copy this code somewhere safe. Use IMPORT CODE on the title screen to restore.<br>
          (Also saved to this browser.)</div>
        <textarea readonly></textarea>
        <div class="modal-btns">
          <button id="sm-copy">COPY</button>
          <button id="sm-close">CLOSE</button>
        </div>
        <div class="fs-foot">X: close</div>`;
      const ta = scr.querySelector('textarea');
      ta.value = code;
      menuLayer.appendChild(scr);
      ta.focus(); ta.select();
      const unsub = Input.listen(['b'], () => done());
      function done() {
        unsub();
        Sfx.cancel();
        scr.remove();
        resolve();
      }
      scr.querySelector('#sm-copy').onclick = () => {
        ta.focus(); ta.select();
        let ok = false;
        try { ok = document.execCommand('copy'); } catch (e) { /* fall through */ }
        if (!ok && navigator.clipboard) navigator.clipboard.writeText(code).then(() => {}, () => {});
        Sfx.select();
        toast('Copied to clipboard!');
      };
      scr.querySelector('#sm-close').onclick = done;
    });
  }

  function importModal() {
    return new Promise((resolve) => {
      const scr = document.createElement('div');
      scr.className = 'full-screen save-modal';
      scr.innerHTML = `<div class="fs-title">IMPORT SAVE CODE</div>
        <div style="font-size:10px;line-height:1.8;color:#b8c4cc;margin-bottom:10px">
          Paste your save code below, then press LOAD.</div>
        <textarea></textarea>
        <div class="modal-btns">
          <button id="im-load">LOAD</button>
          <button id="im-cancel">CANCEL</button>
        </div>
        <div class="fs-foot">ESC: cancel</div>`;
      const ta = scr.querySelector('textarea');
      menuLayer.appendChild(scr);
      ta.focus();
      const done = (val) => {
        unsub();
        scr.remove();
        resolve(val);
      };
      const unsub = Input.listen(['b'], () => { Sfx.cancel(); done(null); });
      scr.querySelector('#im-load').onclick = () => {
        const code = ta.value.trim();
        if (!code) { Sfx.cancel(); toast('Paste a save code first!'); ta.focus(); return; }
        Sfx.select();
        done(code);
      };
      scr.querySelector('#im-cancel').onclick = () => { Sfx.cancel(); done(null); };
    });
  }

  // ----- AI Encounter Modal (Double-DQN + DeBERTa-v3-Small + Final-DKT-GRU) -----
  function askEncounterModal(enc) {
    return new Promise((resolve) => {
      const hasChoices = Array.isArray(enc.choices) && enc.choices.length > 0;
      let openTextMode = !hasChoices;
      let selIdx = 0;
      let retries = 1;
      let hintRevealed = !!enc.podcast_hint;
      let phase = 'question'; // 'question' | 'grading' | 'feedback'
      let turnResult = null;
      let unsub = null;
      const tStart = performance.now();

      const backdrop = document.createElement('div');
      backdrop.className = 'ai-modal-backdrop';
      const card = document.createElement('div');
      card.className = 'ai-modal-card';
      backdrop.appendChild(card);
      menuLayer.appendChild(backdrop);

      function armInput() {
        if (unsub) unsub();
        unsub = Input.listen(null, (k) => {
          unsub = null;
          onGameKey(k);
        });
      }

      function cleanup() {
        if (unsub) { unsub(); unsub = null; }
        window.removeEventListener('keydown', onRawKey, true);
        if (typeof AITutor !== 'undefined') AITutor.stopPodcast();
        backdrop.remove();
        Input.clearPressed();
      }

      function finish() {
        const out = { encounter: enc, result: turnResult };
        cleanup();
        resolve(out);
      }

      async function submitAnswer(rawAns) {
        if (phase !== 'question') return;
        const cleaned = String(rawAns || '').trim();
        if (openTextMode && cleaned.length < 2) {
          retries++;
          Sfx.cancel();
          toast('Type a short explanation first!');
          const ta = card.querySelector('.ai-open-textarea');
          if (ta) ta.focus();
          armInput();
          return;
        }
        phase = 'grading';
        if (typeof AITutor !== 'undefined') AITutor.stopPodcast();
        const elapsedSec = Math.max(0.4, (performance.now() - tStart) / 1000);
        render();
        Sfx.select();
        turnResult = await AITutor.submitEncounter(enc, cleaned, elapsedSec, retries, hintRevealed);
        if (turnResult.badge_just_earned) Sfx.badge();
        else if (turnResult.grader.verdict === 'correct') Sfx.statUp();
        else if (turnResult.grader.verdict === 'partial') Sfx.select();
        else Sfx.statDown();
        phase = 'feedback';
        render();
        armInput();
      }

      function triggerPodcastPlay() {
        hintRevealed = true;
        const txt = enc.podcast_hint || enc.available_podcast_hint;
        const aud = enc.podcast_audio || enc.available_podcast_audio;
        Sfx.select();
        render();
        if (typeof AITutor !== 'undefined') {
          AITutor.playPodcast(aud, txt);
        }
      }

      function onGameKey(k) {
        if (phase === 'grading') { armInput(); return; }
        if (phase === 'feedback') {
          if (k === 'a' || k === 'b' || k === 'start') {
            Sfx.select();
            finish();
            return;
          }
          armInput();
          return;
        }
        // Question phase
        if (openTextMode) {
          if (k === 'b' && hasChoices) {
            Sfx.cancel();
            openTextMode = false;
            render();
          }
          armInput();
          return;
        }
        if (k === 'up') {
          selIdx = (selIdx - 1 + enc.choices.length) % enc.choices.length;
          Sfx.cursor();
          render();
          armInput();
        } else if (k === 'down') {
          selIdx = (selIdx + 1) % enc.choices.length;
          Sfx.cursor();
          render();
          armInput();
        } else if (k === 'a' || k === 'start') {
          submitAnswer(String(selIdx + 1));
        } else {
          armInput();
        }
      }

      function onRawKey(e) {
        if (phase === 'feedback') {
          if (e.key === 'Enter') {
            e.preventDefault();
            e.stopPropagation();
            Sfx.select();
            finish();
          }
          return;
        }
        if (phase !== 'question') return;
        const inTa = e.target && e.target.tagName === 'TEXTAREA';
        if (inTa) {
          if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            e.stopPropagation();
            submitAnswer(e.target.value);
          }
          return;
        }
        if (!openTextMode && hasChoices && ['1', '2', '3', '4'].includes(e.key)) {
          const idx = parseInt(e.key, 10) - 1;
          if (idx < enc.choices.length) {
            e.preventDefault();
            e.stopPropagation();
            selIdx = idx;
            submitAnswer(String(idx + 1));
          }
        } else if (!openTextMode && (e.key === 't' || e.key === 'T')) {
          e.preventDefault();
          e.stopPropagation();
          openTextMode = true;
          Sfx.select();
          render();
        } else if ((e.key === 'h' || e.key === 'H') && (enc.podcast_hint || enc.available_podcast_hint)) {
          e.preventDefault();
          e.stopPropagation();
          triggerPodcastPlay();
        }
      }

      function render() {
        const st = typeof AITutor !== 'undefined' ? AITutor.getStatus() : { affect: {} };
        const live = typeof AITutor !== 'undefined' && AITutor.isOnline();
        const isBoss = enc.rl_action === 4 || openTextMode;
        const mPct = Math.round((enc.target_mastery !== undefined ? enc.target_mastery : 0.4) * 100);
        const topHtml = `
          <div class="ai-top-bar">
            <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
              <span class="ai-dot ${live ? '' : 'offline'}"></span>
              <span class="ai-badge-rl ${isBoss ? 'boss' : ''}">RL [${enc.rl_action}] ${escHtml(enc.rl_action_name)}</span>
            </div>
            <div class="ai-badge-concept">C${enc.target_concept_id}: ${escHtml(enc.target_concept_short || enc.target_concept_title)} &middot; ${mPct}%</div>
          </div>
        `;

        if (phase === 'grading') {
          card.innerHTML = `${topHtml}
            <div class="ai-body-area" style="align-items:center;justify-content:center;text-align:center">
              <div style="font-size:11px;color:#88ccff;margin-bottom:10px">NEURAL EVALUATION IN PROGRESS...</div>
              <div style="font-family:var(--dialog-font);font-size:19px;color:#d0e2f5">Running DeBERTa-v3 Cross-Encoder + Final-DKT-GRU Telemetry...</div>
            </div>`;
          return;
        }

        if (phase === 'feedback' && turnResult) {
          const v = turnResult.grader.verdict;
          const semPct = Math.round((turnResult.grader.semantic_score || 0) * 100);
          const cid = turnResult.target_concept_id;
          const mBefore = Math.round((turnResult.mastery_before[cid] || 0) * 1000) / 10;
          const mAfter = Math.round((turnResult.mastery_after[cid] || 0) * 1000) / 10;
          const aff = turnResult.affect || { concentrating: 0.7, frustrated: 0.25, bored: 0.15 };
          const lat = turnResult.latencies_ms || { total_ai_turn_ms: 0 };

          let vTitle = '';
          if (v === 'correct') {
            vTitle = turnResult.used_neural_grader
              ? `&#10003; CORRECT (${semPct}% Semantic Match) &mdash; NEURAL CRITICAL STRIKE (1.35x DMG)!`
              : `&#10003; CORRECT &mdash; FOCUSED STRIKE (1.15x DMG)!`;
          } else if (v === 'partial') {
            vTitle = `&#9680; PARTIAL UNDERSTANDING (${semPct}% Semantic Match) &mdash; GLANCING HIT!`;
          } else {
            const rightOpt = (hasChoices && enc.correct_idx != null)
              ? ` (Correct Move: [${enc.correct_idx + 1}])`
              : '';
            vTitle = `&#10007; INCORRECT${rightOpt} &mdash; ATTACK FIZZLED!`;
          }

          const badgeBanner = turnResult.badge_just_earned
            ? `<div class="ai-badge-unlock">&#9733; CONCEPT SYNTHESIS BADGE UNLOCKED: C${cid} ${escHtml(turnResult.target_concept_title)}! (+Bonus EXP)</div>`
            : '';
          const reactLine = turnResult.persona_reaction
            || (enc.persona_banter ? (v === 'correct' ? enc.persona_banter.on_correct : (v === 'partial' ? enc.persona_banter.on_partial : enc.persona_banter.on_incorrect)) : '');
          const reactHtml = reactLine
            ? `<div class="ai-persona-react">${escHtml(reactLine)}</div>`
            : '';

          card.innerHTML = `${topHtml}
            <div class="ai-body-area ai-scroll">
              <div>
                <div class="ai-verdict-banner ${v}">${vTitle}</div>
                ${badgeBanner}
                ${reactHtml}
                <div class="ai-ref-box"><strong>Prof. Ada's Insight:</strong> ${escHtml(turnResult.reference_answer)}</div>
                <div class="ai-telem-grid">
                  <div>GRADER: ${escHtml(turnResult.grader.graded_by)}</div>
                  <div>DKT MASTERY C${cid}: ${mBefore}% &rarr; <strong>${mAfter}%</strong></div>
                  <div>AFFECT: Focus ${Math.round(aff.concentrating * 100)}% &middot; Frust ${Math.round(aff.frustrated * 100)}% &middot; Bored ${Math.round(aff.bored * 100)}%</div>
                  <div>LATENCY: ${lat.total_ai_turn_ms} ms (Resp: ${turnResult.response_sec}s)</div>
                </div>
              </div>
              <div class="ai-footer-bar">
                <div class="ai-foot-hint">Press Z / SPACE / ENTER or click CONTINUE</div>
                <button class="ai-btn-primary" id="ai-cont-btn">EXECUTE MOVE &rarr;</button>
              </div>
            </div>`;
          const contBtn = card.querySelector('#ai-cont-btn');
          if (contBtn) contBtn.onclick = () => { Sfx.select(); finish(); };
          return;
        }

        // Question Phase
        const banter = enc.persona_banter || null;
        const lm = enc.llm_meta || null;
        const llmTag = (lm && (lm.reasoner_source === 'live_llm' || lm.stylizer_source === 'live_llm' || lm.stylizer_source === 'live_qlora'))
          ? `<span class="ai-llm-chip">${escHtml(lm.reasoner_model || 'DeepSeek-R1')} + ${escHtml(lm.stylizer_model || 'Llama-3.1')}${lm.cache_hit ? ' &middot; INSTANT' : ''}</span>`
          : '';
        const personaHtml = (banter && banter.intro_taunt)
          ? `<div class="ai-persona-bar">
               <div class="ai-persona-text">${escHtml(banter.intro_taunt)}</div>
               ${llmTag}
             </div>`
          : '';
        const hintStr = enc.podcast_hint || (hintRevealed ? enc.available_podcast_hint : null);
        const hintIntroStr = (hintStr && banter && banter.hint_intro) ? `${banter.hint_intro} ` : '';
        const podcastHtml = hintStr
          ? `<div class="ai-podcast-bar">
               <div class="ai-podcast-text">&#127911; ${escHtml(hintIntroStr + hintStr)}</div>
               <button class="ai-btn-mini" id="ai-play-pod">&#9654; AUDIO</button>
             </div>`
          : '';

        let responseHtml = '';
        if (!openTextMode && hasChoices) {
          const btns = enc.choices.map((ch, i) => `
            <button type="button" class="ai-choice-btn ${i === selIdx ? 'sel' : ''}" data-cidx="${i}">
              <span class="ai-ckey">${i + 1}</span>
              <span>${escHtml(ch)}</span>
            </button>
          `).join('');
          responseHtml = `<div class="ai-choices-list">${btns}</div>`;
        } else {
          responseHtml = `
            <div class="ai-open-wrap">
              <div class="ai-open-label">&#9876; DEBERTA-V3-SMALL SEMANTIC GRADING &mdash; EXPLAIN IN YOUR OWN WORDS:</div>
              <textarea class="ai-open-textarea" placeholder="Type your explanation here... (Ctrl+Enter or click SUBMIT)"></textarea>
            </div>
          `;
        }

        const hintBtnHtml = (!hintStr && enc.available_podcast_hint)
          ? `<button type="button" class="ai-btn-mini" id="ai-req-hint">&#127911; HINT [H]</button>`
          : '';
        const modeBtnHtml = (!openTextMode && hasChoices)
          ? `<button type="button" class="ai-btn-mini" id="ai-to-open">&#9997; OWN WORDS [T]</button>`
          : (openTextMode && hasChoices
            ? `<button type="button" class="ai-btn-mini" id="ai-to-mc">&larr; 4 CHOICES [ESC]</button>`
            : '');
        const submitBtnHtml = openTextMode
          ? `<button type="button" class="ai-btn-primary" id="ai-sub-open">SUBMIT TO DeBERTa &rarr;</button>`
          : '';

        const footHint = openTextMode
          ? 'Type answer &middot; Ctrl+Enter or click SUBMIT'
          : 'UP/DOWN + Z (or keys 1-4) &middot; [T] Own Words &middot; [H] Podcast';

        card.innerHTML = `${topHtml}
          ${personaHtml}
          ${podcastHtml}
          <div class="ai-question-box">${escHtml(enc.question)}</div>
          <div class="ai-body-area ai-scroll">
            ${responseHtml}
            <div class="ai-footer-bar">
              <div class="ai-foot-hint">${footHint}</div>
              <div class="ai-action-row">
                ${hintBtnHtml}
                ${modeBtnHtml}
                ${submitBtnHtml}
              </div>
            </div>
          </div>`;

        const playBtn = card.querySelector('#ai-play-pod');
        if (playBtn) playBtn.onclick = () => triggerPodcastPlay();
        const reqHintBtn = card.querySelector('#ai-req-hint');
        if (reqHintBtn) reqHintBtn.onclick = () => triggerPodcastPlay();
        const toOpenBtn = card.querySelector('#ai-to-open');
        if (toOpenBtn) {
          toOpenBtn.onclick = () => { openTextMode = true; Sfx.select(); render(); };
        }
        const toMcBtn = card.querySelector('#ai-to-mc');
        if (toMcBtn) {
          toMcBtn.onclick = () => { openTextMode = false; Sfx.cancel(); render(); armInput(); };
        }
        card.querySelectorAll('.ai-choice-btn').forEach(btn => {
          btn.onclick = () => {
            const idx = parseInt(btn.dataset.cidx, 10);
            selIdx = idx;
            submitAnswer(String(idx + 1));
          };
        });
        const ta = card.querySelector('.ai-open-textarea');
        if (ta) {
          setTimeout(() => ta.focus(), 20);
          const subBtn = card.querySelector('#ai-sub-open');
          if (subBtn) subBtn.onclick = () => submitAnswer(ta.value);
        }
      }

      window.addEventListener('keydown', onRawKey, true);
      render();
      armInput();
      // Auto-play spoken 15s podcast clip when Double-DQN selects Action [0]!
      if (enc.rl_action === 0 && (enc.podcast_audio || enc.podcast_hint) && typeof AITutor !== 'undefined') {
        AITutor.playPodcast(enc.podcast_audio, enc.podcast_hint);
      }
    });
  }

  // ----- KnowledgeQuest 3-Tab AI Field Journal & Study Hub -----
  async function questScreen(initialTab = 0) {
    if (typeof AITutor !== 'undefined') await AITutor.refreshStatus();
    return new Promise((resolve) => {
      let tab = Number(initialTab) || 0; // 0: Mastery & Affect, 1: Study Notes, 2: AI Lab & New Course
      let selConcept = 0;
      let unsub = null;

      const scr = document.createElement('div');
      scr.className = 'full-screen quest-modal';
      menuLayer.appendChild(scr);
      Sfx.select();

      function armInput() {
        if (unsub) unsub();
        unsub = Input.listen(['up', 'down', 'left', 'right', 'a', 'b', 'start'], (k) => {
          unsub = null;
          onKey(k);
        });
      }

      function closeJournal() {
        if (unsub) { unsub(); unsub = null; }
        if (typeof AITutor !== 'undefined') AITutor.stopPodcast();
        Sfx.cancel();
        scr.remove();
        Input.clearPressed();
        resolve();
      }

      function onKey(k) {
        if (k === 'b' || k === 'start') {
          closeJournal();
          return;
        }
        if (k === 'left') {
          tab = (tab + 2) % 3;
          Sfx.cursor();
          render();
        } else if (k === 'right') {
          tab = (tab + 1) % 3;
          Sfx.cursor();
          render();
        } else if (k === 'up') {
          selConcept = (selConcept + 4) % 5;
          Sfx.cursor();
          render();
        } else if (k === 'down') {
          selConcept = (selConcept + 1) % 5;
          Sfx.cursor();
          render();
        } else if (k === 'a' && tab === 1) {
          const st = AITutor.getStatus();
          const c = st.concepts[selConcept];
          if (c) {
            Sfx.select();
            AITutor.playPodcast(c.podcast_audio, c.podcast_hint);
            toast(`Playing C${selConcept} Podcast Clip...`);
          }
        }
        armInput();
      }

      function render() {
        const st = typeof AITutor !== 'undefined' ? AITutor.getStatus() : { concepts: [], affect: {} };
        const live = typeof AITutor !== 'undefined' && AITutor.isOnline();
        const aff = st.affect || { concentrating: 0.68, frustrated: 0.30, bored: 0.18 };

        let bodyHtml = '';
        if (tab === 0) {
          const rows = (st.concepts || []).map((c, idx) => {
            const pct = Math.round((c.mastery || 0) * 100);
            const preTxt = c.prereqs && c.prereqs.length ? `Prereq: C${c.prereqs.join(', C')}` : 'Foundation';
            const badgeHtml = c.badge
              ? '<span style="color:#ffd848">&#9733; BADGE EARNED</span>'
              : `<span style="color:#8aa2bc">${preTxt}</span>`;
            return `
              <div class="kq-concept-row ${idx === selConcept ? 'sel' : ''}" data-cidx="${idx}">
                <div class="kq-crow-top">
                  <span>C${c.concept_id}: ${escHtml(c.title)}</span>
                  <span>${badgeHtml} &nbsp; <strong>${pct}%</strong></span>
                </div>
                <div class="kq-bar-track"><div class="kq-bar-fill" style="width:${pct}%"></div></div>
              </div>
            `;
          }).join('');

          const lt = st.last_turn && st.last_turn.latencies_ms
            ? `Last AI Turn: RL ${st.last_turn.latencies_ms.rl_gamemaster_ms}ms | DeBERTa ${st.last_turn.latencies_ms.deberta_grader_ms}ms | DKT ${st.last_turn.latencies_ms.final_dkt_gru_ms}ms`
            : 'Fight wild creatures or trainers to record live 3-model telemetry!';

          bodyHtml = `
            <div class="kq-scroll ai-scroll">
              ${rows}
              <div class="kq-affect-strip">
                <div class="kq-affect-card">FOCUS: <strong style="color:#68e090">${Math.round(aff.concentrating * 100)}%</strong><br><span style="color:#7890a8">Behavioral Only</span></div>
                <div class="kq-affect-card">FRUSTRATED: <strong style="color:#ff9488">${Math.round(aff.frustrated * 100)}%</strong><br><span style="color:#7890a8">Zero Webcam</span></div>
                <div class="kq-affect-card">BORED: <strong style="color:#f8d058">${Math.round(aff.bored * 100)}%</strong><br><span style="color:#7890a8">Badges: ${st.badges_earned_count || 0}/5</span></div>
              </div>
              <div style="margin-top:7px;font-size:7.5px;color:#8aa4c0">${escHtml(lt)}</div>
            </div>
          `;
        } else if (tab === 1) {
          const c = (st.concepts && st.concepts[selConcept]) || { concept_id: 0, title: '', summary_bullets: [], podcast_hint: '' };
          const pills = (st.concepts || []).map((ci, i) => `
            <button type="button" class="ai-btn-mini" data-pickc="${i}" style="${i === selConcept ? 'border-color:#f8d048;color:#f8d048' : ''}">C${i}: ${escHtml(ci.short_title || ('C' + i))}</button>
          `).join('');
          const bullets = (c.summary_bullets || []).map(b => `<div class="kq-note-bullet">&bull; ${escHtml(b)}</div>`).join('');
          bodyHtml = `
            <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px">${pills}</div>
            <div class="kq-scroll ai-scroll" style="background:#101722;border:1px solid #2e4058;border-radius:6px;padding:10px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:8px">
                <div style="font-size:9px;color:#f8d048">C${c.concept_id}: ${escHtml(c.title)}</div>
                <button type="button" class="ai-btn-mini" id="kq-play-note-pod">&#9654; PLAY 15s PODCAST [Z]</button>
              </div>
              ${bullets}
              <div class="ai-podcast-bar" style="margin-top:8px">
                <div class="ai-podcast-text">&#127911; ${escHtml(c.podcast_hint || '')}</div>
              </div>
            </div>
          `;
        } else {
          const m = st.models || { dkt: {}, rl: {}, grader: {}, dino: {}, vlm: {} };
          const availCourses = (st.available_courses || []).map(ac => `
            <button type="button" class="ai-btn-mini" data-course="${escHtml(ac.course_id)}" data-prof="average" style="${st.course_id === ac.course_id ? 'border-color:#58f898;color:#58f898' : ''}">${escHtml(ac.title.slice(0, 28))}</button>
          `).join('');
          bodyHtml = `
            <div class="kq-scroll ai-scroll" style="background:#101722;border:1px solid #2e4058;border-radius:6px;padding:10px">
              <div style="font-size:8.5px;color:#68c0f8;margin-bottom:6px">LOADED PYTORCH &amp; KAGGLE PIPELINE (5-MODEL MULTIMODAL ENGINE)</div>
              <div style="font-family:var(--dialog-font);font-size:15.5px;line-height:1.28;color:#dce8f6;margin-bottom:8px">
                &bull; <strong>Model #1 (DKT):</strong> ${escHtml((m.dkt && m.dkt.name) || 'Final-DKT-GRU')} (AUC: 0.7996) &nbsp;
                &bull; <strong>Model #2 (RL GM):</strong> ${escHtml((m.rl && m.rl.name) || 'Double-DQN')}<br>
                &bull; <strong>Model #3 (Grader):</strong> ${escHtml((m.grader && m.grader.name) || 'DeBERTa-v3-Small')} &nbsp;
                &bull; <strong>Model #4 (Spatial):</strong> ${escHtml((m.dino && m.dino.name) || 'LLMDet-Large / Grounding-DINO')}<br>
                &bull; <strong>Model #5 (VLM):</strong> ${escHtml((m.vlm && m.vlm.name) || 'Qwen/Qwen3-VL-8B-Instruct')} (Dual-Vision Fused IoU: 0.993)
              </div>
              <div style="margin-bottom:8px">
                <button type="button" class="ai-btn-primary" id="kq-open-dino-btn">&#128269; LAUNCH VLM + GROUNDING DINO LAB MONITOR (10 PUZZLES + VQA)</button>
              </div>
              <div style="font-size:8.5px;color:#f8d048;margin-bottom:5px">ACTIVE COURSES (CLICK TO SWITCH)</div>
              <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:9px">
                ${availCourses || '<button type="button" class="ai-btn-mini" data-course="vision_yolo_ocr" data-prof="average">YOLO &amp; OCR</button>'}
              </div>
              <div style="background:#142236;border:1px solid #4888c8;border-radius:6px;padding:8px">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:5px">
                  <span style="font-size:8.5px;color:#58f898">+ UPLOAD PDF / PPTX / WORD (.DOCX) &mdash; VLM AUTO-EXTRACTS PHOTOS &amp; CONCEPTS</span>
                  <div style="display:flex;gap:4px">
                    <button type="button" class="ai-btn-mini" id="kq-preset-os">Preset: OS &amp; Memory</button>
                    <button type="button" class="ai-btn-mini" id="kq-preset-net">Preset: Networks &amp; TCP</button>
                  </div>
                </div>
                <div style="background:#0b1320;border:1px dashed #4c88c8;border-radius:5px;padding:6px 8px;margin-bottom:5px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px">
                  <label style="font-family:var(--dialog-font);font-size:15.5px;color:#78f0a8;cursor:pointer">
                    &#128196; <strong>1. Select PDF, PPTX, Word (.docx), or Image:</strong>
                    <input type="file" id="kq-new-img" accept=".pdf,.pptx,.ppt,.docx,.doc,image/*" style="font-size:11.5px;color:#dce8f6;margin-left:6px" />
                  </label>
                </div>
                <input type="text" id="kq-new-title" placeholder="2. Optional Course Title (auto-filled from PDF/PPTX/DOCX filename if left empty)" style="width:100%;box-sizing:border-box;background:#0b1320;color:#fff;border:1px solid #385880;border-radius:4px;padding:5px 8px;font-family:var(--dialog-font);font-size:15px;margin-bottom:5px" />
                <textarea id="kq-new-notes" rows="2" placeholder="3. Optional Notes (Leave empty when uploading PDF/PPTX/Word — Qwen3-VL-8B will inspect the document & extract photos/concepts automatically!)" style="width:100%;box-sizing:border-box;background:#0b1320;color:#dce8f6;border:1px solid #385880;border-radius:4px;padding:5px 8px;font-family:var(--dialog-font);font-size:14.5px;margin-bottom:5px"></textarea>
                <div style="display:flex;align-items:center;justify-content:flex-end;gap:6px">
                  <button type="button" class="ai-btn-primary" id="kq-ingest-btn" style="width:100%;padding:7px 12px">&#9889; INGEST DOCUMENT WITH ${escHtml(((m.vlm && m.vlm.short_name) || 'QWEN3-VL-8B').toUpperCase())} + ${escHtml(((m.dino && m.dino.name) || 'LLMDET-LARGE').toUpperCase())} &amp; LAUNCH LAB</button>
                </div>
              </div>
            </div>
          `;
        }

        scr.innerHTML = `
          <div class="kq-journal-win">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
              <div>
                <div class="quest-title-badge">ADAPTIVE E-LEARNING AI FIELD JOURNAL</div>
                <div class="quest-subtitle">${escHtml(st.course_title || 'Deep Learning Vision: CNNs, YOLO & OCR')}</div>
              </div>
              <div style="display:flex;align-items:center;gap:8px">
                <span class="ai-dot ${live ? '' : 'offline'}"></span>
                <span style="font-size:8px;color:${live ? '#78f098' : '#f8b848'}">${live ? '5-MODEL SERVER ONLINE' : 'OFFLINE FALLBACK'}</span>
                <button type="button" class="ai-btn-mini" id="kq-close-btn">&#10005; CLOSE</button>
              </div>
            </div>
            <div class="kq-tabs">
              <button type="button" class="kq-tab ${tab === 0 ? 'active' : ''}" data-tab="0">[1] AI MASTERY &amp; AFFECT</button>
              <button type="button" class="kq-tab ${tab === 1 ? 'active' : ''}" data-tab="1">[2] STUDY NOTES &amp; AUDIO</button>
              <button type="button" class="kq-tab ${tab === 2 ? 'active' : ''}" data-tab="2">[3] VLM, DINO LAB &amp; NEW COURSE</button>
            </div>
            ${bodyHtml}
            <div style="margin-top:7px;font-size:7.5px;color:#8aa0be;text-align:right">
              LEFT/RIGHT: Switch Tab &middot; UP/DOWN: Select Concept &middot; X / ESC: Close
            </div>
          </div>
        `;

        const closeBtn = scr.querySelector('#kq-close-btn');
        if (closeBtn) closeBtn.onclick = closeJournal;
        const dinoBtn = scr.querySelector('#kq-open-dino-btn');
        if (dinoBtn) {
          dinoBtn.onclick = async () => {
            if (unsub) { unsub(); unsub = null; }
            scr.remove();
            await dinoDiagramScreen(selConcept);
            resolve();
          };
        }
        const presetOsBtn = scr.querySelector('#kq-preset-os');
        if (presetOsBtn) {
          presetOsBtn.onclick = () => {
            const tEl = scr.querySelector('#kq-new-title');
            const nEl = scr.querySelector('#kq-new-notes');
            if (tEl) tEl.value = 'Operating Systems: Virtual Memory & Paging';
            if (nEl) nEl.value = [
              '1. Virtual Address Spaces & Page Tables: Translating virtual page numbers (VPN) to physical frame numbers (PFN)',
              '2. Translation Lookaside Buffer (TLB): Hardware associative cache accelerating address translation in 1 cycle',
              '3. Page Faults & Demand Paging: OS trap handler loading missing pages from NVMe swap into RAM frames',
              '4. Page Replacement Algorithms (LRU & Clock): Evicting least recently used frames to prevent thrashing',
              '5. End-to-End MMU + TLB + OS Virtual Memory Pipeline: Full hardware-software address translation synthesis',
            ].join('\n');
            Sfx.cursor();
          };
        }
        const presetNetBtn = scr.querySelector('#kq-preset-net');
        if (presetNetBtn) {
          presetNetBtn.onclick = () => {
            const tEl = scr.querySelector('#kq-new-title');
            const nEl = scr.querySelector('#kq-new-notes');
            if (tEl) tEl.value = 'Computer Networks: TCP/IP, Routing & Congestion';
            if (nEl) nEl.value = [
              '1. OSI & TCP/IP Encapsulation: Wrapping application payloads into transport segments, IP packets, and link frames',
              '2. TCP 3-Way Handshake & Reliable Sliding Window: SYN, SYN-ACK, ACK sequencing and cumulative acknowledgments',
              '3. IPv4 Subnetting & Longest Prefix Match Routing: Forwarding packets across router forwarding tables',
              '4. TCP Congestion Control (AIMD & Fast Recovery): Additive increase, multiplicative decrease and congestion window sizing',
              '5. End-to-End Full-Stack Network Packet Journey: DNS resolution, TLS handshake, TCP transport, and BGP/OSPF routing',
            ].join('\n');
            Sfx.cursor();
          };
        }
        const fileInputEl = scr.querySelector('#kq-new-img');
        if (fileInputEl) {
          fileInputEl.onchange = () => {
            const tEl = scr.querySelector('#kq-new-title');
            if (fileInputEl.files && fileInputEl.files[0] && tEl && !tEl.value.trim()) {
              const rawName = fileInputEl.files[0].name.replace(/\.[^.]+$/, '');
              tEl.value = rawName.replace(/[_-]+/g, ' ').trim();
            }
          };
        }
        const ingestBtn = scr.querySelector('#kq-ingest-btn');
        if (ingestBtn) {
          ingestBtn.onclick = async () => {
            const tEl = scr.querySelector('#kq-new-title');
            const nEl = scr.querySelector('#kq-new-notes');
            const fEl = scr.querySelector('#kq-new-img');
            const hasFile = !!(fEl && fEl.files && fEl.files[0]);
            const fileName = hasFile ? fEl.files[0].name : null;
            const titleVal = (tEl && tEl.value.trim()) || (hasFile ? fileName.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ') : 'Operating Systems: Virtual Memory & Paging');
            const notesVal = (nEl && nEl.value.trim()) || (hasFile ? '' : [
              '1. Virtual Address Spaces & Page Tables: Translating virtual page numbers to physical frames',
              '2. Translation Lookaside Buffer (TLB): Hardware cache accelerating address lookup',
              '3. Page Faults & Demand Paging: OS trap handler loading pages from disk swap',
              '4. Page Replacement (LRU & Clock): Evicting optimal frames to stop thrashing',
              '5. End-to-End MMU + TLB + OS Pipeline: Hardware-software virtual memory synthesis',
            ].join('\n'));
            ingestBtn.disabled = true;
            ingestBtn.textContent = hasFile
              ? `VLM INSPECTING '${fileName.slice(0, 26)}' & EXTRACTING PHOTOS...`
              : 'INGESTING WITH QWEN3-VL-8B + GROUNDING DINO...';
            Sfx.select();

            let fileB64 = null;
            if (hasFile) {
              fileB64 = await new Promise((resB64) => {
                const reader = new FileReader();
                reader.onload = () => resB64(reader.result);
                reader.onerror = () => resB64(null);
                reader.readAsDataURL(fEl.files[0]);
              });
            }
            try {
              const res = await AITutor.ingestCustomCourse(
                {
                  courseTitle: titleVal,
                  rawNotes: notesVal,
                  fileName,
                  fileBase64: fileB64,
                  imageBase64: fileB64,
                  profile: 'average',
                },
                (pct, stepMsg) => {
                  ingestBtn.textContent = `${pct}% — ${stepMsg}`;
                }
              );
              const nPhotos = (res && res.num_photos_extracted) || 0;
              toast(nPhotos > 0
                ? `VLM extracted ${nPhotos} photos & 5 concepts from "${fileName}"!`
                : `Ingested "${titleVal}"! Launching Lab Monitor...`);
              if (unsub) { unsub(); unsub = null; }
              scr.remove();
              await dinoDiagramScreen(0);
              resolve();
            } catch (err) {
              ingestBtn.disabled = false;
              ingestBtn.textContent = 'RETRY INGEST DOCUMENT';
              toast('Error: Ensure the game server is running!');
            }
          };
        }
        scr.querySelectorAll('.kq-tab').forEach(b => {
          b.onclick = () => { tab = parseInt(b.dataset.tab, 10); Sfx.cursor(); render(); };
        });
        scr.querySelectorAll('[data-cidx]').forEach(r => {
          r.onclick = () => { selConcept = parseInt(r.dataset.cidx, 10); tab = 1; Sfx.select(); render(); };
        });
        scr.querySelectorAll('[data-pickc]').forEach(b => {
          b.onclick = () => { selConcept = parseInt(b.dataset.pickc, 10); Sfx.cursor(); render(); };
        });
        const playNoteBtn = scr.querySelector('#kq-play-note-pod');
        if (playNoteBtn) {
          playNoteBtn.onclick = () => {
            const c = st.concepts[selConcept];
            if (c) {
              Sfx.select();
              AITutor.playPodcast(c.podcast_audio, c.podcast_hint);
            }
          };
        }
        scr.querySelectorAll('[data-course]').forEach(b => {
          b.onclick = async () => {
            Sfx.select();
            await AITutor.resetSession(b.dataset.course, b.dataset.prof);
            toast(`Switched to ${b.dataset.course} (${b.dataset.prof})!`);
            tab = 0;
            render();
          };
        });
      }

      render();
      armInput();
    });
  }

  // ----- Model #4: Grounding DINO Interactive Lab Diagram Calibration Terminal -----
  async function dinoDiagramScreen(initialConceptId = null) {
    const puzzles = (typeof AITutor !== 'undefined' && AITutor.getVisionPuzzles)
      ? await AITutor.getVisionPuzzles()
      : [];
    if (!puzzles.length) return;

    return new Promise((resolve) => {
      let pIdx = 0;
      if (initialConceptId !== null && initialConceptId !== undefined) {
        const found = puzzles.findIndex(p => Number(p.concept_id) === Number(initialConceptId));
        if (found >= 0) pIdx = found;
      }

      let evalResult = null;
      let busy = false;
      let dragStart = null;
      let dragCurr = null;
      let tStart = performance.now();
      let unsub = null;
      const imgCache = {};

      const scr = document.createElement('div');
      scr.className = 'full-screen quest-modal';
      menuLayer.appendChild(scr);
      Sfx.select();

      function armInput() {
        if (unsub) unsub();
        unsub = Input.listen(['left', 'right', 'a', 'b', 'start'], (k) => {
          unsub = null;
          if (k === 'b' || k === 'start') {
            closeScreen();
            return;
          }
          if (k === 'left') {
            pIdx = (pIdx - 1 + puzzles.length) % puzzles.length;
            evalResult = null;
            tStart = performance.now();
            Sfx.cursor();
            render();
          } else if (k === 'right' || (k === 'a' && evalResult)) {
            pIdx = (pIdx + 1) % puzzles.length;
            evalResult = null;
            tStart = performance.now();
            Sfx.cursor();
            render();
          }
          armInput();
        });
      }

      function closeScreen() {
        if (unsub) { unsub(); unsub = null; }
        Sfx.cancel();
        scr.remove();
        Input.clearPressed();
        resolve();
      }

      function drawDiagramCanvas(cvs, p) {
        if (!cvs) return;
        const ctx = cvs.getContext('2d');
        const W = 720, H = 420;
        ctx.fillStyle = '#141c2c';
        ctx.fillRect(0, 0, W, H);

        const src = p.diagram_file;
        let im = imgCache[src];
        if (!im) {
          im = new Image();
          imgCache[src] = im;
          im.onload = () => drawDiagramCanvas(cvs, p);
          im.src = src;
        }
        if (im && im.complete && im.naturalWidth > 0) {
          ctx.drawImage(im, 0, 0, W, H);
        } else {
          // Instant fallback schematic while PNG loads
          ctx.fillStyle = '#1c2a42';
          ctx.fillRect(20, 60, W - 40, H - 80);
          ctx.strokeStyle = '#4890d8';
          ctx.lineWidth = 2;
          ctx.strokeRect(20, 60, W - 40, H - 80);
          const tb = p.target_box || [100, 100, 300, 300];
          ctx.fillStyle = '#284868';
          ctx.fillRect(tb[0], tb[1], tb[2] - tb[0], tb[3] - tb[1]);
          ctx.strokeStyle = '#f8c848';
          ctx.strokeRect(tb[0], tb[1], tb[2] - tb[0], tb[3] - tb[1]);
        }

        // Draw live drag selection box
        if (dragStart && dragCurr) {
          const x1 = Math.min(dragStart.x, dragCurr.x);
          const y1 = Math.min(dragStart.y, dragCurr.y);
          const w = Math.abs(dragCurr.x - dragStart.x);
          const h = Math.abs(dragCurr.y - dragStart.y);
          ctx.save();
          ctx.fillStyle = 'rgba(255, 216, 72, 0.22)';
          ctx.fillRect(x1, y1, w, h);
          ctx.strokeStyle = '#ffd848';
          ctx.lineWidth = 3;
          ctx.setLineDash([6, 4]);
          ctx.strokeRect(x1, y1, w, h);
          ctx.restore();
        }

        // Draw Grounding DINO / VLM evaluation result overlay
        if (evalResult) {
          const gb = evalResult.grounded_box || p.target_box;
          if (gb && gb.length === 4) {
            ctx.save();
            ctx.fillStyle = 'rgba(56, 248, 144, 0.20)';
            ctx.fillRect(gb[0], gb[1], gb[2] - gb[0], gb[3] - gb[1]);
            ctx.strokeStyle = '#38f890';
            ctx.lineWidth = 4;
            ctx.strokeRect(gb[0], gb[1], gb[2] - gb[0], gb[3] - gb[1]);
            const tagText = `VLM+DINO: ${(evalResult.target_label || '').slice(0, 24)} (${Math.round((evalResult.dino_confidence || 0.48) * 100)}%)`;
            ctx.font = 'bold 12px monospace';
            const tagW = Math.min(W - 8, Math.max(180, ctx.measureText(tagText).width + 14));
            const tagX = Math.max(2, Math.min(W - tagW - 2, gb[0]));
            const tagY = gb[1] >= 26 ? gb[1] - 24 : Math.min(H - 24, gb[3] + 4);
            ctx.fillStyle = 'rgba(14, 30, 24, 0.92)';
            ctx.fillRect(tagX, tagY, tagW, 20);
            ctx.fillStyle = '#58ffaa';
            ctx.fillText(tagText, tagX + 6, tagY + 14);
            ctx.restore();
          }

          if (evalResult.user_box && evalResult.user_box.length === 4) {
            const ub = evalResult.user_box;
            ctx.save();
            ctx.strokeStyle = '#ffd848';
            ctx.lineWidth = 3;
            ctx.setLineDash([5, 4]);
            ctx.strokeRect(ub[0], ub[1], ub[2] - ub[0], ub[3] - ub[1]);
            ctx.restore();
          } else if (evalResult.click_point && evalResult.click_point.length === 2) {
            const [cx, cy] = evalResult.click_point;
            ctx.save();
            ctx.strokeStyle = evalResult.verdict === 'correct' ? '#ffd848' : '#ff6868';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(cx, cy, 12, 0, Math.PI * 2);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(cx - 18, cy); ctx.lineTo(cx + 18, cy);
            ctx.moveTo(cx, cy - 18); ctx.lineTo(cx, cy + 18);
            ctx.stroke();
            ctx.restore();
          }
        }
      }

      function toCanvasCoords(cvs, clientX, clientY) {
        const rect = cvs.getBoundingClientRect();
        const scaleX = 720 / Math.max(1, rect.width);
        const scaleY = 420 / Math.max(1, rect.height);
        return {
          x: Math.max(0, Math.min(720, (clientX - rect.left) * scaleX)),
          y: Math.max(0, Math.min(420, (clientY - rect.top) * scaleY)),
        };
      }

      async function handleSubmission(clickPt, boxArr) {
        if (busy) return;
        busy = true;
        const p = puzzles[pIdx];
        const elapsed = Math.max(0.5, (performance.now() - tStart) / 1000);
        Sfx.select();
        evalResult = await AITutor.submitVisionPuzzle(
          p.puzzle_id,
          clickPt ? clickPt.x : null,
          clickPt ? clickPt.y : null,
          boxArr,
          elapsed
        );
        busy = false;
        if (evalResult.verdict === 'correct') {
          Sfx.statUp();
          if (typeof Game !== 'undefined' && Game.player && Game.player.party && Game.player.party[0]) {
            Game.player.party[0].exp = (Game.player.party[0].exp || 0) + (evalResult.xp_gained || 45);
          }
        } else if (evalResult.verdict === 'partial') {
          Sfx.select();
        } else {
          Sfx.statDown();
        }
        render();
      }

      function render() {
        const p = puzzles[pIdx];
        const live = typeof AITutor !== 'undefined' && AITutor.isOnline();
        const pills = puzzles.map((item, idx) => {
          const tag = `C${item.concept_id}-${idx % 2 === 0 ? 'A' : 'B'}`;
          return `<button type="button" class="ai-btn-mini" data-pidx="${idx}" style="padding:3px 7px;${idx === pIdx ? 'border-color:#58f898;color:#58f898' : ''}">${tag}</button>`;
        }).join('');

        let statusFooter = '';
        if (evalResult) {
          const cid = evalResult.concept_id;
          const mBefore = Math.round(((evalResult.mastery_before && evalResult.mastery_before[cid]) || 0.4) * 1000) / 10;
          const mAfter = Math.round(((evalResult.mastery_after && evalResult.mastery_after[cid]) || 0.5) * 1000) / 10;
          const vCol = evalResult.verdict === 'correct' ? '#58f898' : (evalResult.verdict === 'partial' ? '#f8d048' : '#ff8080');
          statusFooter = `
            <div style="flex-shrink:0;max-height:64px;overflow-y:auto;background:#0d1624;border:1px solid ${vCol};border-radius:5px;padding:4px 8px;margin-top:4px">
              <div style="font-size:7.5px;color:${vCol};margin-bottom:2px">
                ${escHtml(evalResult.graded_by)} &middot; IoU: ${evalResult.iou} &middot; DKT C${cid}: ${mBefore}% &rarr; <strong>${mAfter}%</strong> (+${evalResult.xp_gained || 0} EXP)
              </div>
              <div style="font-family:var(--dialog-font);font-size:14px;color:#dce8f6;line-height:1.18">
                ${escHtml(evalResult.feedback)}
              </div>
            </div>
          `;
        } else {
          statusFooter = `
            <div style="flex-shrink:0;font-family:var(--dialog-font);font-size:14.5px;color:#9cc4e8;margin-top:4px;line-height:1.15">
              &#128070; <strong>Click</strong> OR <strong>click-and-drag a box</strong> around the target component on the figure above! (100% Privacy-First Vision AI)
            </div>
          `;
        }

        const vqaQ = p.vlm_vqa_question || `Explain how [${p.target_label}] functions inside ${p.concept_title}.`;
        const vqaRef = p.vlm_vqa_reference || p.explanation || '';
        const stInfo = (typeof AITutor !== 'undefined' && AITutor.getStatus) ? AITutor.getStatus() : {};
        const modInfo = stInfo.models || {};
        const vlmShort = (modInfo.vlm && modInfo.vlm.short_name) || 'Qwen3-VL-8B';
        const dinoShort = ((modInfo.dino && modInfo.dino.name) || 'LLMDet-Large').replace(/\s*\(Offline Fallback\)/i, '');

        scr.innerHTML = `
          <div class="kq-journal-win" style="padding:8px 12px;overflow:hidden">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:nowrap;flex-shrink:0;margin-bottom:4px">
              <div class="quest-title-badge" style="color:#58f898;font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">LAB MONITOR &mdash; ${escHtml(vlmShort)} + ${escHtml(dinoShort.toUpperCase())}</div>
              <div style="display:flex;align-items:center;gap:5px;flex-shrink:0">
                <span class="ai-dot ${live ? '' : 'offline'}"></span>
                <span style="font-size:7px;color:${live ? '#78f098' : '#f8b848'};white-space:nowrap">${live ? 'CUDA LIVE' : 'CACHED'}</span>
                <button type="button" class="ai-btn-mini" id="dino-new-course-btn" style="padding:4px 7px;border-color:#58f898;color:#58f898">+ COURSE</button>
                <button type="button" class="ai-btn-mini" id="dino-next-btn" style="padding:4px 7px">NEXT &rarr;</button>
                <button type="button" class="ai-btn-mini" id="dino-close-btn" style="padding:4px 7px">&#10005;</button>
              </div>
            </div>
            <div style="display:flex;gap:4px;flex-wrap:nowrap;overflow-x:auto;flex-shrink:0;margin-bottom:4px">${pills}</div>
            <div style="flex-shrink:0;background:#142238;border:1px solid #3c6ea8;border-radius:5px;padding:4px 8px;margin-bottom:4px;font-family:var(--dialog-font);font-size:14.5px;line-height:1.18;color:#f8e078">
              ${escHtml(p.prompt)}
            </div>
            <div style="flex:1 1 0;min-height:140px;position:relative;display:flex;align-items:center;justify-content:center;background:#0a101a;border:2px solid #2e4668;border-radius:6px;overflow:hidden;padding:2px">
              <canvas id="dino-diagram-canvas" width="720" height="420" style="height:100%;width:auto;max-width:100%;max-height:100%;aspect-ratio:720/420;display:block;cursor:crosshair;touch-action:none"></canvas>
            </div>
            ${statusFooter}
            <div style="flex-shrink:0;background:#101b2c;border:1px solid #3a5c88;border-radius:5px;padding:5px 8px;margin-top:4px">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;margin-bottom:3px">
                <span style="font-size:7.5px;color:#68c8ff">VLM VISUAL QUESTION (${escHtml(vlmShort)} &rarr; DeBERTa-v3 Grader):</span>
                <button type="button" class="ai-btn-mini" id="dino-vqa-fill" style="padding:3px 7px">Auto-Fill</button>
              </div>
              <div style="font-family:var(--dialog-font);font-size:14px;line-height:1.15;color:#e4eef8;margin-bottom:3px">${escHtml(vqaQ)}</div>
              <div style="display:flex;gap:6px">
                <input type="text" id="dino-vqa-input" placeholder="Type 1-sentence answer about this diagram..." style="flex:1;background:#09111c;color:#fff;border:1px solid #35567e;border-radius:4px;padding:3px 7px;font-family:var(--dialog-font);font-size:14px" />
                <button type="button" class="ai-btn-primary" id="dino-vqa-submit" style="width:auto;padding:3px 9px;font-size:8px">GRADE VQA</button>
              </div>
              <div id="dino-vqa-out" style="font-family:var(--dialog-font);font-size:13.5px;color:#88e8a8;margin-top:2px"></div>
            </div>
          </div>
        `;

        const closeBtn = scr.querySelector('#dino-close-btn');
        if (closeBtn) closeBtn.onclick = closeScreen;
        const newCourseBtn = scr.querySelector('#dino-new-course-btn');
        if (newCourseBtn) {
          newCourseBtn.onclick = async () => {
            if (unsub) { unsub(); unsub = null; }
            scr.remove();
            await questScreen(2);
            resolve();
          };
        }
        const vqaFillBtn = scr.querySelector('#dino-vqa-fill');
        if (vqaFillBtn) {
          vqaFillBtn.onclick = () => {
            const inp = scr.querySelector('#dino-vqa-input');
            if (inp) inp.value = vqaRef;
            Sfx.cursor();
          };
        }
        const vqaSubBtn = scr.querySelector('#dino-vqa-submit');
        if (vqaSubBtn) {
          vqaSubBtn.onclick = async () => {
            const inp = scr.querySelector('#dino-vqa-input');
            const outEl = scr.querySelector('#dino-vqa-out');
            const ans = (inp && inp.value.trim()) || vqaRef;
            if (!ans) return;
            vqaSubBtn.disabled = true;
            vqaSubBtn.textContent = 'GRADING...';
            Sfx.select();
            try {
              const res = await AITutor.submitVisionVQA(p.concept_id, ans, 9.5);
              const pct = Math.round((res.score || 0) * 100);
              const col = res.verdict === 'correct' ? '#58f898' : (res.verdict === 'partial' ? '#f8d048' : '#ff8080');
              if (outEl) {
                outEl.style.color = col;
                outEl.innerHTML = `<strong>DeBERTa-v3 Verdict: ${escHtml((res.verdict || '').toUpperCase())} (${pct}%)</strong> in ${res.grader_latency_ms || 12}ms &mdash; DKT C${p.concept_id} Mastery Updated!`;
              }
              if (res.verdict === 'correct') Sfx.statUp();
            } catch (e) {
              if (outEl) outEl.textContent = 'Start server.py to grade live with DeBERTa-v3!';
            }
            vqaSubBtn.disabled = false;
            vqaSubBtn.textContent = 'GRADE VQA';
          };
        }
        const nextBtn = scr.querySelector('#dino-next-btn');
        if (nextBtn) {
          nextBtn.onclick = () => {
            pIdx = (pIdx + 1) % puzzles.length;
            evalResult = null;
            tStart = performance.now();
            Sfx.cursor();
            render();
          };
        }
        scr.querySelectorAll('[data-pidx]').forEach(b => {
          b.onclick = () => {
            pIdx = parseInt(b.dataset.pidx, 10);
            evalResult = null;
            tStart = performance.now();
            Sfx.cursor();
            render();
          };
        });

        const cvs = scr.querySelector('#dino-diagram-canvas');
        drawDiagramCanvas(cvs, p);

        if (cvs) {
          cvs.onpointerdown = (e) => {
            if (busy) return;
            cvs.setPointerCapture(e.pointerId);
            dragStart = toCanvasCoords(cvs, e.clientX, e.clientY);
            dragCurr = { ...dragStart };
            drawDiagramCanvas(cvs, p);
          };
          cvs.onpointermove = (e) => {
            if (!dragStart) return;
            dragCurr = toCanvasCoords(cvs, e.clientX, e.clientY);
            drawDiagramCanvas(cvs, p);
          };
          cvs.onpointerup = async (e) => {
            if (!dragStart) return;
            const endPt = toCanvasCoords(cvs, e.clientX, e.clientY);
            const dx = Math.abs(endPt.x - dragStart.x);
            const dy = Math.abs(endPt.y - dragStart.y);
            const startCopy = { ...dragStart };
            dragStart = null;
            dragCurr = null;
            if (dx >= 14 && dy >= 14) {
              await handleSubmission(null, [startCopy.x, startCopy.y, endPt.x, endPt.y]);
            } else {
              await handleSubmission(endPt, null);
            }
          };
        }
      }

      render();
      armInput();
    });
  }

  // ----- Title Screen Course Upload Gate (NEW GAME unlocks only after course upload completes) -----
  function titleCourseMenu({ hasSave = false } = {}) {
    return new Promise((resolve) => {
      let uploadedCourse = (typeof AITutor !== 'undefined' && AITutor.getUploadedCourse)
        ? AITutor.getUploadedCourse()
        : null;
      let uploading = false;
      let progressPct = uploadedCourse ? 100 : 0;
      let progressStep = uploadedCourse
        ? `Ready! "${uploadedCourse.course_title}" (${uploadedCourse.num_photos_extracted || 0} photos extracted)`
        : 'Insert a PDF, PPTX, Word (.docx), or slide image to unlock NEW GAME.';
      let errorMsg = '';

      const wrap = document.createElement('div');
      wrap.style.cssText = 'position:absolute;inset:0;pointer-events:auto;z-index:20;';
      menuLayer.appendChild(wrap);

      function done(result) {
        wrap.remove();
        Input.clearPressed();
        resolve(result);
      }

      function render() {
        const unlocked = Boolean(uploadedCourse && !uploading);
        const stMeta = (typeof AITutor !== 'undefined' && AITutor.getStatus) ? AITutor.getStatus() : {};
        const vlmTag = ((stMeta.models && stMeta.models.vlm && stMeta.models.vlm.short_name) || 'QWEN3-VL-8B').toUpperCase();
        const dinoTag = ((stMeta.models && stMeta.models.dino && stMeta.models.dino.name) || 'LLMDET-LARGE').toUpperCase();
        wrap.innerHTML = `
          <div style="position:absolute;left:16px;bottom:42px;width:210px;background:rgba(12,20,34,0.95);border:3px solid #f0e8d0;border-radius:8px;padding:10px;box-shadow:0 6px 18px rgba(0,0,0,0.6)">
            <div style="font-size:8px;color:#f8d048;margin-bottom:8px;text-align:center">MAIN MENU</div>
            ${hasSave ? '<button type="button" id="ts-continue" class="ai-btn-primary" style="width:100%;margin-bottom:7px;background:#2a4870;border-color:#78b8f8">CONTINUE</button>' : ''}
            <button type="button" id="ts-newgame" class="ai-btn-primary" ${unlocked ? '' : 'disabled'} style="width:100%;margin-bottom:7px;${unlocked ? 'background:#1d6b42;border-color:#58f898;color:#fff;cursor:pointer;box-shadow:0 0 12px rgba(88,248,152,0.45)' : 'background:#222c3c;border-color:#445268;color:#7888a0;opacity:0.58;cursor:not-allowed'}">
              ${unlocked ? '&#9654; NEW GAME' : '&#128274; NEW GAME'}
            </button>
            <button type="button" id="ts-import" class="ai-btn-mini" style="width:100%;padding:6px 8px">IMPORT CODE</button>
            <div style="font-family:var(--dialog-font);font-size:13.5px;color:${unlocked ? '#78f8a8' : '#b8c8dc'};margin-top:6px;line-height:1.2;text-align:center">
              ${unlocked ? 'Course verified! Click NEW GAME to begin.' : 'Upload a course on the right to unlock NEW GAME.'}
            </div>
          </div>

          <div style="position:absolute;right:16px;bottom:42px;width:448px;background:rgba(12,20,34,0.95);border:3px solid ${unlocked ? '#58f898' : '#58a8f8'};border-radius:8px;padding:10px 12px;box-shadow:0 6px 18px rgba(0,0,0,0.6)">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px">
              <span style="font-size:8px;color:${unlocked ? '#58f898' : '#68c8ff'};white-space:nowrap">&#128218; INSERT COURSE</span>
              <span style="font-size:7px;color:#f8d048;white-space:nowrap">${escHtml(vlmTag)} + ${escHtml(dinoTag)}</span>
            </div>
            <div style="background:#09111c;border:1px dashed #4a82bc;border-radius:5px;padding:6px 8px;margin-bottom:6px">
              <input type="file" id="ts-course-file" accept=".pdf,.pptx,.ppt,.docx,.doc,image/*" ${uploading ? 'disabled' : ''} style="width:100%;font-size:11.5px;color:#dce8f6;cursor:pointer" />
            </div>
            <div style="display:flex;gap:6px;margin-bottom:6px">
              <input type="text" id="ts-course-title" ${uploading ? 'disabled' : ''} value="${escHtml((uploadedCourse && uploadedCourse.course_title) || '')}" placeholder="Course Title (auto-filled from file name)" style="flex:1;background:#09111c;color:#fff;border:1px solid #35547a;border-radius:4px;padding:5px 8px;font-family:var(--dialog-font);font-size:15px" />
              <button type="button" id="ts-upload-btn" class="ai-btn-primary" ${uploading ? 'disabled' : ''} style="width:auto;padding:5px 12px;white-space:nowrap">
                ${uploading ? 'ANALYZING...' : (unlocked ? 'RE-UPLOAD' : '&#9889; UPLOAD')}
              </button>
            </div>
            <div style="background:#0a1220;border:1px solid #2b4464;border-radius:4px;height:12px;overflow:hidden;margin-bottom:5px">
              <div id="ts-prog-bar" style="height:100%;width:${progressPct}%;background:${unlocked ? '#38e888' : '#48a0f8'};transition:width 0.35s ease"></div>
            </div>
            <div id="ts-prog-text" style="font-family:var(--dialog-font);font-size:14.5px;color:${errorMsg ? '#ff8888' : (unlocked ? '#78f8a8' : '#b8d4f0')};line-height:1.25">
              ${escHtml(errorMsg || progressStep)}
            </div>
          </div>
        `;

        const contBtn = wrap.querySelector('#ts-continue');
        if (contBtn) {
          contBtn.onclick = () => { Sfx.select(); done({ action: 'CONTINUE' }); };
        }
        const impBtn = wrap.querySelector('#ts-import');
        if (impBtn) {
          impBtn.onclick = () => { Sfx.select(); done({ action: 'IMPORT CODE' }); };
        }
        const newBtn = wrap.querySelector('#ts-newgame');
        if (newBtn) {
          newBtn.onclick = () => {
            if (!unlocked) {
              Sfx.statDown();
              toast('Insert a course and wait for upload to complete first!');
              return;
            }
            Sfx.select();
            done({ action: 'NEW GAME', course: uploadedCourse });
          };
        }

        const fileEl = wrap.querySelector('#ts-course-file');
        const titleEl = wrap.querySelector('#ts-course-title');
        if (fileEl) {
          fileEl.onchange = () => {
            if (fileEl.files && fileEl.files[0] && titleEl && !titleEl.value.trim()) {
              titleEl.value = fileEl.files[0].name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim();
            }
          };
        }

        const upBtn = wrap.querySelector('#ts-upload-btn');
        if (upBtn) {
          upBtn.onclick = async () => {
            if (uploading) return;
            const hasFile = Boolean(fileEl && fileEl.files && fileEl.files[0]);
            if (!hasFile) {
              errorMsg = 'Please select a .pdf, .pptx, .docx, or image file first!';
              Sfx.statDown();
              render();
              return;
            }
            const fileObj = fileEl.files[0];
            const fileName = fileObj.name;
            const titleVal = (titleEl && titleEl.value.trim()) || fileName.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim();

            uploading = true;
            uploadedCourse = null;
            errorMsg = '';
            progressPct = 4;
            progressStep = `Reading '${fileName}' and uploading to Kaggle GPU server...`;
            Sfx.select();
            render();

            const fileB64 = await new Promise((resB64) => {
              const reader = new FileReader();
              reader.onload = () => resB64(reader.result);
              reader.onerror = () => resB64(null);
              reader.readAsDataURL(fileObj);
            });

            if (!fileB64) {
              uploading = false;
              errorMsg = 'Failed to read selected file.';
              progressPct = 0;
              render();
              return;
            }

            try {
              const res = await AITutor.ingestCustomCourse(
                {
                  courseTitle: titleVal,
                  rawNotes: '',
                  fileName,
                  fileBase64: fileB64,
                  profile: 'average',
                },
                (pct, stepMsg) => {
                  progressPct = Math.max(progressPct, Number(pct) || 10);
                  progressStep = stepMsg || 'Analyzing with Qwen3-VL-8B + LLMDet-Large...';
                  const bar = wrap.querySelector('#ts-prog-bar');
                  const txt = wrap.querySelector('#ts-prog-text');
                  if (bar) bar.style.width = `${progressPct}%`;
                  if (txt) txt.textContent = `${progressPct}% — ${progressStep}`;
                }
              );
              uploading = false;
              progressPct = 100;
              uploadedCourse = {
                course_id: res.course_id,
                course_title: res.course_title || titleVal,
                num_photos_extracted: res.num_photos_extracted || 0,
                source_file: res.source_file || fileName,
              };
              progressStep = `Course ready! "${uploadedCourse.course_title}" (${uploadedCourse.num_photos_extracted} extracted photos & 10 VLM/DINO puzzles).`;
              Sfx.statUp();
              render();
            } catch (err) {
              uploading = false;
              progressPct = 0;
              errorMsg = `Upload error: ${err && err.message ? err.message : 'Check server connection.'}`;
              Sfx.statDown();
              render();
            }
          };
        }
      }

      render();
    });
  }

  return {
    init, say, sayLines, setPrompt, hideDialog, choose, yesNo, toast,
    fadeOut, fadeIn, flashScreen,
    setBox, setStatusChip, setEnemyBalls, hideBoxes, setBattleAiPill, animateHp, animateExp,
    moveMenu, partyScreen, bagScreen, shopScreen, starterChoice,
    saveModal, importModal, askEncounterModal, questScreen, dinoDiagramScreen, titleCourseMenu,
  };
})();
