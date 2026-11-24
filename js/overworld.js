// ===== Overworld engine =====
const DELTA = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
const OPPOSITE = { up: 'down', down: 'up', left: 'right', right: 'left' };

// The rival always picks the starter that counters the player's choice.
const RIVAL_COUNTER = { bulbasaur: 'charmander', charmander: 'squirtle', squirtle: 'bulbasaur' };

const Overworld = (() => {
  const TILE = 16, VIEW_W = 240, VIEW_H = 160;
  const STEP_TIME = 0.18;   // seconds per tile
  const FACE_DELAY = 0.07;  // tap-to-turn grace period

  let moving = null;        // { fromX, fromY, t }
  let faceTimer = 0;
  let bumpCooldown = 0;
  let blockedDir = null;    // direction of a gate that just refused us; re-arm on release

  function map() { return MAPS[Game.player.map]; }
  function mapW() { return map().rows[0].length; }
  function mapH() { return map().rows.length; }
  function tileAt(x, y) {
    const m = map();
    if (y < 0 || y >= m.rows.length || x < 0 || x >= m.rows[y].length) return null;
    return m.rows[y][x];
  }
  function visibleNpcs() {
    return (map().npcs || []).filter(n => !(n.hiddenIf && Game.player.flags[n.hiddenIf]));
  }
  function npcAt(x, y) {
    return visibleNpcs().find(n => n.x === x && n.y === y) || null;
  }
  function isSolid(x, y) {
    const ch = tileAt(x, y);
    if (ch === null) return true;
    if (SOLID_TILES.has(ch)) return true;
    // tiles under a building image are solid (doors stay enterable)
    if (ch !== 'D' && GameAssets.decalSolid(Game.player.map, x, y)) return true;
    if (npcAt(x, y)) return true;
    return false;
  }

  function runAsync(fn) {
    if (Game.busy) return;
    Game.busy = true;
    Promise.resolve(fn()).catch(e => console.error(e)).finally(() => {
      Game.busy = false;
      Input.clearPressed();
    });
  }

  // ---------------- update ----------------
  function update(dt) {
    if (Game.mode !== 'overworld' || Game.busy) return;
    const p = Game.player;
    if (bumpCooldown > 0) bumpCooldown -= dt;

    if (moving) {
      moving.t += dt / STEP_TIME;
      if (moving.t >= 1) {
        moving = null;
        onStepComplete();
      }
      return;
    }

    if (Input.consume('start')) { runAsync(startMenu); return; }
    if (Input.consume('a')) { runAsync(interact); return; }
    Input.consume('b');

    const dir = Input.dirHeld();
    if (!dir) { faceTimer = 0; blockedDir = null; return; }
    if (blockedDir && dir !== blockedDir) blockedDir = null;

    if (p.facing !== dir) {
      p.facing = dir;
      faceTimer = FACE_DELAY;
      return;
    }
    if (faceTimer > 0) { faceTimer -= dt; return; }

    const [dx, dy] = DELTA[dir];
    const tx = p.x + dx, ty = p.y + dy;

    // map edge → link to neighbor map
    if (tileAt(tx, ty) === null) {
      if (dir === blockedDir) return; // gate already warned; wait for key release
      const link = (map().links || {})[dir];
      if (link) runAsync(() => traverseLink(dir, link));
      return;
    }
    if (isSolid(tx, ty)) {
      if (bumpCooldown <= 0) { Sfx.bump(); bumpCooldown = 0.4; }
      return;
    }
    moving = { fromX: p.x, fromY: p.y, t: 0 };
    p.x = tx; p.y = ty;
  }

  function onStepComplete() {
    const p = Game.player;
    const ch = tileAt(p.x, p.y);

    // warp tiles (doors / exit mats / stairs)
    const warp = (map().warps || {})[`${p.x},${p.y}`];
    if (warp) { runAsync(() => doWarp(warp)); return; }

    // wild encounters in tall grass
    const enc = map().encounters;
    if (ch === 't' && enc && Math.random() < enc.rate) {
      runAsync(() => startWildEncounter(enc));
      return;
    }

    // trainer line-of-sight
    for (const npc of visibleNpcs()) {
      if (!npc.trainer || npc.trainer.range <= 0) continue;
      if (Game.player.flags[npc.trainer.flag]) continue;
      if (seesPlayer(npc)) {
        runAsync(() => trainerApproach(npc));
        return;
      }
    }
  }

  // Trainers watch all four directions — sneaking behind them doesn't work.
  // When they spot the player they spin to face that direction first.
  function seesPlayer(npc) {
    const p = Game.player;
    for (const dir of ['up', 'down', 'left', 'right']) {
      const [dx, dy] = DELTA[dir];
      for (let i = 1; i <= npc.trainer.range; i++) {
        const x = npc.x + dx * i, y = npc.y + dy * i;
        if (p.x === x && p.y === y) { npc.facing = dir; return true; }
        const ch = tileAt(x, y);
        if (ch === null || SOLID_TILES.has(ch) || npcAt(x, y)) break;
      }
    }
    return false;
  }

  // ---------------- warps & links ----------------
  async function doWarp(warp) {
    Sfx.door();
    await UI.fadeOut(280);
    Game.player.map = warp.map;
    Game.player.x = warp.x;
    Game.player.y = warp.y;
    Game.player.facing = warp.facing || 'down';
    // Reaching a healing center makes it the blackout respawn point.
    if ((MAPS[warp.map].npcs || []).some(n => n.special === 'nurse')) {
      Game.player.respawn = { map: warp.map, x: warp.x, y: warp.y };
    }
    await UI.fadeIn(280);
  }

  async function traverseLink(dir, targetId) {
    const p = Game.player;
    if (targetId === 'route1' && p.map === 'hometown' && !p.flags.starter) {
      await UI.say("Whoa there! Wild computational creatures roam the tall grass out that way.");
      await UI.say('Meet Professor ADA down at the Research Lab first. Her lab is the large brick building at the south end of town.');
      blockedDir = dir; // don't re-trigger until the key is released
      return;
    }
    const target = MAPS[targetId];
    await UI.fadeOut(220);
    p.map = targetId;
    if (dir === 'up') p.y = target.rows.length - 1;
    if (dir === 'down') p.y = 0;
    UI.toast(target.name);
    await UI.fadeIn(220);
  }

  // ---------------- interaction ----------------
  async function interact() {
    const p = Game.player;
    const [dx, dy] = DELTA[p.facing];
    let tx = p.x + dx, ty = p.y + dy;

    let npc = npcAt(tx, ty);
    // talking across a counter
    if (!npc && tileAt(tx, ty) === 'k') npc = npcAt(tx + dx, ty + dy);
    if (npc) { await talkTo(npc); return; }

    const ch = tileAt(tx, ty);
    const sign = (map().signs || {})[`${tx},${ty}`];
    if (sign) { await UI.say(sign); return; }
    if (ch === '1' || ch === '2' || ch === '3') { await pedestal(tx, ty); return; }
    if (ch === 'H') {
      await UI.say("ADA'S LAB CONSOLE: Booting Grounding DINO Open-Vocabulary Vision Calibration Terminal...");
      await UI.dinoDiagramScreen();
      return;
    }
    if (ch === 'k' && Game.player.map === 'lab') {
      const cid = Math.max(0, Math.min(4, (tx - 1) % 5));
      const st = (typeof AITutor !== 'undefined') ? AITutor.getStatus() : null;
      const c = st && st.concepts ? st.concepts[cid] : null;
      if (c && typeof AITutor !== 'undefined') {
        AITutor.playPodcast(c.podcast_audio, c.podcast_hint);
        await UI.say(`RESEARCH AUDIO LOG [C${cid} - ${c.short_title || c.title}]:\n${c.podcast_hint}`);
        AITutor.stopPodcast();
        return;
      }
    }
    if (ch === 'B') { await UI.say('The bed is neatly made. No time for napping — adventure calls!'); return; }
    if (ch === 'b') { await UI.say('Field guides, maps, and research notebooks on learning algorithms.'); return; }
  }

  function getStoryGreeting(npcKey, fallbackText) {
    const story = (typeof AITutor !== 'undefined' && AITutor.getCourseStory) ? AITutor.getCourseStory() : null;
    if (story && story.npc_greetings && story.npc_greetings[npcKey]) {
      return story.npc_greetings[npcKey];
    }
    return fallbackText;
  }

  async function askNpcChoice(prompt, options) {
    if (prompt) await UI.say(prompt, { noWait: true, keep: true });
    const idx = await UI.choose(options, { style: { right: '14px', bottom: '126px' } });
    UI.hideDialog();
    return idx;
  }

  async function talkTo(npc) {
    // face the player
    npc.facing = OPPOSITE[Game.player.facing];
    if (npc.special === 'nurse') return nurseFlow();
    if (npc.special === 'shop') return shopFlow();
    if (npc.special === 'prof') return profFlow();
    if (npc.special === 'parent') return parentFlow();
    if (npc.trainer) return trainerFlow(npc);
    const greet = getStoryGreeting('villager', (npc.dialog && npc.dialog[0]) || 'TOWNIE: Lovely day for studying!');
    await UI.say(greet);
    if (UI.npcChatModal) {
      const pick = await askNpcChoice('Chat with this townie about your course?', ['ASK ABOUT COURSE', 'BYE']);
      if (pick === 0) {
        await UI.npcChatModal({ npcKey: 'villager', speaker: 'TOWNIE', greeting: greet });
      }
    }
  }

  async function parentFlow() {
    const f = Game.player.flags;
    const momGreet = getStoryGreeting('mom', 'MOM: Up already? Today is the day your KnowledgeQuest research begins!');
    if (!f.sent_off) {
      const story = (typeof AITutor !== 'undefined' && AITutor.getCourseStory) ? AITutor.getCourseStory() : null;
      const lines = [momGreet];
      if (story && Array.isArray(story.story_intro) && story.story_intro.length) {
        lines.push(story.story_intro[0]);
      } else {
        lines.push('MOM: Professor ADA sent word from the laboratory. She needs your help restoring the Knowledge Network.');
      }
      lines.push('MOM: I packed a POTION in your bag — and remember, you can ask anyone in town questions about your course!');
      await UI.sayLines(lines);
      Game.player.bag.potion = (Game.player.bag.potion || 0) + 1;
      UI.toast('Got a Potion!');
      f.sent_off = true;
      return;
    }
    await UI.say(momGreet);
    const pick = await askNpcChoice('MOM: What do you need, sweetheart?', ['REST & HEAL', 'ASK ABOUT COURSE', 'BYE']);
    if (pick === 1 && UI.npcChatModal) {
      await UI.npcChatModal({ npcKey: 'mom', speaker: 'MOM', greeting: momGreet });
      return;
    }
    if (pick !== 0) { await UI.say('MOM: Take care out there. Dinner is whenever you come home.'); return; }
    await UI.fadeOut(300);
    Sfx.heal();
    for (const c of Game.player.party) {
      c.hp = c.maxHp;
      c.status = null;
      c.sleepTurns = 0;
      for (const m of c.moves) m.pp = m.maxPp;
    }
    Game.player.respawn = { map: Game.player.map, x: Game.player.x, y: Game.player.y };
    await wait(500);
    await UI.fadeIn(300);
    await UI.say('MOM: There. Good as new, the whole lot of you.');
  }

  async function nurseFlow() {
    const greet = getStoryGreeting('nurse', 'NURSE: Welcome to the POKEMON CENTER.');
    await UI.say(greet);
    const pick = await askNpcChoice('NURSE: How can I help today?', ['HEAL TEAM', 'ASK ABOUT COURSE', 'BYE']);
    if (pick === 1 && UI.npcChatModal) {
      await UI.npcChatModal({ npcKey: 'nurse', speaker: 'NURSE', greeting: greet });
      return;
    }
    if (pick !== 0) { await UI.say('NURSE: Of course. We are here whenever you need us.'); return; }
    await UI.fadeOut(250);
    Sfx.heal();
    for (const c of Game.player.party) {
      c.hp = c.maxHp;
      c.status = null;
      c.sleepTurns = 0;
      for (const m of c.moves) m.pp = m.maxPp;
    }
    Game.player.respawn = { map: Game.player.map, x: Game.player.x, y: Game.player.y };
    await wait(500);
    await UI.fadeIn(250);
    await UI.sayLines([
      'NURSE: All done! Your team is in perfect shape.',
      'NURSE: We hope to see you again... though preferably not too soon!',
    ]);
  }

  async function shopFlow() {
    const greet = getStoryGreeting('clerk', 'CLERK: Welcome to the POKE MART, trainer! Potions for bruises, study tips on the house!');
    await UI.say(greet);
    const pick = await askNpcChoice('CLERK: What catches your eye?', ['BROWSE SHOP', 'ASK ABOUT COURSE', 'BYE']);
    if (pick === 1 && UI.npcChatModal) {
      await UI.npcChatModal({ npcKey: 'clerk', speaker: 'CLERK', greeting: greet });
      return;
    }
    if (pick !== 0) { await UI.say('CLERK: Safe roads out there. Mind the tall grass!'); return; }
    await UI.shopScreen();
    await UI.say('CLERK: Safe roads out there. Mind the tall grass!');
  }

  async function profFlow() {
    const f = Game.player.flags;
    const story = (typeof AITutor !== 'undefined' && AITutor.getCourseStory) ? AITutor.getCourseStory() : null;
    if (!f.starter) {
      if (story && Array.isArray(story.story_intro) && story.story_intro.length) {
        await UI.sayLines([
          ...story.story_intro,
          'ADA: Choose a research companion from those three consoles: BULBASAUR, CHARMANDER, or SQUIRTLE!',
        ]);
      } else {
        await UI.sayLines([
          "ADA: Ah, welcome! I am Professor ADA of the Willowbrook Research Lab.",
          'ADA: Talk to anyone in town to ask questions about your course, and battle trainers to level up your DKT Mastery!',
          'ADA: Choose a companion from those three consoles: BULBASAUR, CHARMANDER, or SQUIRTLE.',
        ]);
      }
      if (UI.npcChatModal) {
        const askNow = await askNpcChoice('ADA: Ready to start, or want to ask about the course?', ['PICK STARTER', 'ASK ADA']);
        if (askNow === 1) {
          await UI.npcChatModal({
            npcKey: 'prof_ada',
            speaker: 'PROF. ADA',
            greeting: getStoryGreeting('prof_ada', 'ADA: Ask me anything about this course!'),
          });
        }
      }
    } else {
      const partnerName = creatureName(Game.player.party[0]);
      let adaLines = null;
      if (typeof AITutor !== 'undefined' && AITutor.isOnline() && AITutor.getNpcDialogue) {
        const dlg = await AITutor.getNpcDialogue('prof_ada', partnerName);
        if (dlg && Array.isArray(dlg.lines) && dlg.lines.length) {
          adaLines = dlg.lines;
        }
      }
      if (adaLines) {
        await UI.sayLines(adaLines);
      } else {
        await UI.say(getStoryGreeting('prof_ada', `ADA: How is ${partnerName}? Looking sharp! Ask me anything about this course!`));
      }
      if (UI.npcChatModal) {
        const pick = await askNpcChoice('ADA: How can I help with your research?', ['ASK ADA', 'LAB MONITOR', 'BYE']);
        if (pick === 0) {
          await UI.npcChatModal({
            npcKey: 'prof_ada',
            speaker: 'PROF. ADA',
            greeting: (adaLines && adaLines[0]) || getStoryGreeting('prof_ada', 'ADA: Fire away with any course question!'),
          });
        } else if (pick === 1 && UI.dinoDiagramScreen) {
          await UI.dinoDiagramScreen();
        }
      }
    }
  }

  async function pedestal(tx, ty) {
    if (Game.player.flags.starter) {
      const pFlags = Game.player.flags;
      const leftCol = pFlags.starter_id === 'bulbasaur' ? 10 : (pFlags.starter_id === 'charmander' ? 8 : 9);
      if (tx === leftCol) {
        await UI.say("ADA: That research companion is reserved for the next prospective researcher.");
      } else {
        await UI.say('The console is empty.');
      }
      return;
    }
    const id = await UI.starterChoice();
    if (!id) return; // backed out — the desks stay available
    const ok = await UI.yesNo(`Initialize research partner ${SPECIES[id].name}?`);
    if (!ok) return;
    const starter = makeCreature(id, 5);
    Game.player.party.push(starter);
    Game.player.flags.starter = true;
    Game.player.flags.starter_id = id;
    Sfx.caught();
    await UI.say(`You partnered with ${SPECIES[id].name}!`);
    await UI.sayLines([
      'ADA: A magnificent choice! Your learning philosophies align well.',
      'ADA: Take these 5 POKE BALLS for field study. You can record and capture wild creatures once they are weakened in battle.',
    ]);
    Game.player.bag.pokeball = (Game.player.bag.pokeball || 0) + 5;
    UI.toast('Got 5 Poke Balls!');

    // The rival grabs the counter-pick and challenges immediately with course-aware LLM humor!
    const rivalId = RIVAL_COUNTER[id];
    const rexGreet = getStoryGreeting(
      'tr_rex',
      `REX: Done analyzing? Then I'll choose ${SPECIES[rivalId].name}! Let's run a diagnostic battle right here!`
    );
    await UI.sayLines([
      rexGreet,
      `REX: I picked ${SPECIES[rivalId].name} as your counter-pick. Let's test your course knowledge right now!`,
    ]);
    const tr = {
      flag: 'tr_rex', name: 'Rival REX',
      party: [[rivalId, 5]], prize: 175,
    };
    const result = await Battle.start({ trainer: tr });
    Game.player.flags.rex_gone = true;
    if (result === 'win') {
      Game.player.flags.tr_rex = true;
      await UI.sayLines([
        'REX: ...Tsk! An unexpected variance in my calculations. You got lucky with your tactical tempo!',
        'REX: Enjoy your beginner\'s advantage. Next time I\'ll have optimized my team\'s parameters. Count on it!',
      ]);
      await UI.say('REX dashed out of the lab.');
      await UI.sayLines([
        'ADA: Remarkable composure in your first trial! Rex\'s competitive drive will keep you sharp.',
        'ADA: Talk to anyone in town to ask questions about your course, or head north up FERNWAY TRAIL toward STONEGATE GYM!',
      ]);
    } else if (result === 'lose') {
      await blackout();
      await UI.sayLines([
        'You hear REX analyzing his victory data outside the lab.',
        'No matter. Review your battle data, ask Prof. ADA questions, and formulate a new strategy.',
      ]);
    }
  }

  // ---------------- trainers ----------------
  async function trainerFlow(npc) {
    const tr = npc.trainer;
    const trGreet = getStoryGreeting(tr.flag, (tr.intro && tr.intro[0]) || `${tr.name}: Ready to test your knowledge?`);
    if (Game.player.flags[tr.flag]) {
      await UI.say(trGreet);
      if (UI.npcChatModal) {
        const pick = await askNpcChoice(`${tr.name}: Want to discuss the course?`, ['ASK ABOUT COURSE', 'BYE']);
        if (pick === 0) {
          await UI.npcChatModal({ npcKey: tr.flag, speaker: tr.name.toUpperCase(), greeting: trGreet });
        }
      }
      return;
    }
    await UI.say(trGreet);
    if (UI.npcChatModal) {
      const pick = await askNpcChoice(`${tr.name}: What's your move?`, ['BATTLE!', 'ASK ABOUT COURSE', 'NOT YET']);
      if (pick === 1) {
        await UI.npcChatModal({ npcKey: tr.flag, speaker: tr.name.toUpperCase(), greeting: trGreet });
        const ready = await UI.yesNo(`${tr.name}: Got the info you needed? Ready to battle now?`);
        if (!ready) return;
      } else if (pick === 2 || pick < 0) {
        return;
      }
    }
    const result = await Battle.start({ trainer: tr });
    if (result === 'win') {
      Game.player.flags[tr.flag] = true;
      await UI.say(`${tr.name}: ${tr.loseText}`);
      if (tr.badge && !Game.player.badges.includes('GRANITE BADGE')) {
        Game.player.badges.push('GRANITE BADGE');
        Sfx.badge();
        await UI.say('You received the GRANITE BADGE from MASON!');
        await UI.say('The GRANITE BADGE proves you can move what refuses to be moved.');
      }
      await UI.sayLines(tr.after);
    } else if (result === 'lose') {
      await blackout();
    }
  }

  async function trainerApproach(npc) {
    npc.alert = 1;
    Sfx.encounter();
    await wait(700);
    npc.alert = 0;
    // walk along the line of sight until adjacent to the player
    const p = Game.player;
    while (Math.abs(npc.x - p.x) + Math.abs(npc.y - p.y) > 1) {
      await stepNpc(npc, npc.facing);
    }
    p.facing = OPPOSITE[npc.facing];
    await trainerFlow(npc);
  }

  async function stepNpc(npc, dir) {
    const [dx, dy] = DELTA[dir];
    npc.ox = 0; npc.oy = 0;
    for (let k = 1; k <= 8; k++) {
      npc.ox = dx * 2 * k; npc.oy = dy * 2 * k;
      await wait(20);
    }
    npc.x += dx; npc.y += dy;
    npc.ox = 0; npc.oy = 0;
  }

  // ---------------- battles from the overworld ----------------
  async function startWildEncounter(enc) {
    const total = enc.table.reduce((s, [, w]) => s + w, 0);
    let roll = randInt(1, total);
    let speciesId = enc.table[0][0];
    for (const [id, w] of enc.table) {
      roll -= w;
      if (roll <= 0) { speciesId = id; break; }
    }
    const level = randInt(enc.levels[0], enc.levels[1]);
    const result = await Battle.start({ wild: makeCreature(speciesId, level) });
    if (result === 'lose') await blackout();
  }

  async function blackout() {
    await UI.say('You have no team members left that can fight!');
    Game.player.money = Math.floor(Game.player.money / 2);
    await UI.say('You blacked out!');
    await UI.fadeOut(600);
    for (const c of Game.player.party) {
      c.hp = c.maxHp; c.status = null; c.sleepTurns = 0;
      for (const m of c.moves) m.pp = m.maxPp;
    }
    const r = Game.player.respawn;
    Game.player.map = r.map; Game.player.x = r.x; Game.player.y = r.y;
    Game.player.facing = 'down';
    await wait(400);
    await UI.fadeIn(400);
    await UI.say('You hurried somewhere safe, shielding your exhausted team...');
  }

  // ---------------- start menu ----------------
  async function startMenu() {
    Sfx.select();
    while (true) {
      const pick = await UI.choose(['JOURNAL', 'DINO LAB', 'NEW COURSE', 'PARTY', 'BAG', 'BADGES', 'SAVE', 'EXIT'], {
        style: { right: '12px', top: '12px' },
      });
      if (pick === -1 || pick === 7) return;
      if (pick === 0) await UI.questScreen(0);
      if (pick === 1) await UI.dinoDiagramScreen();
      if (pick === 2) await UI.questScreen(2);
      if (pick === 3) await partyManage();
      if (pick === 4) await overworldBag();
      if (pick === 5) {
        const b = Game.player.badges;
        await UI.say(b.length ? `BADGES: ${b.join(', ')}` : 'You have no badges yet. The STONEGATE gym awaits!');
      }
      if (pick === 6) {
        const s = await UI.choose(['SAVE TO BROWSER', 'GET SAVE CODE', 'CANCEL'], {
          style: { right: '12px', top: '12px' },
        });
        if (s === 0) {
          UI.toast(SaveSys.toLocal() ? 'Game saved!' : 'Save failed!');
          Sfx.heal();
        } else if (s === 1) {
          SaveSys.toLocal();
          await UI.saveModal(SaveSys.exportCode());
        }
      }
    }
  }

  async function partyManage() {
    while (true) {
      const i = await UI.partyScreen({ title: 'PARTY' });
      if (i === -1) return;
      if (Game.player.party.length < 2) { UI.toast('Nothing to swap with yet!'); continue; }
      const j = await UI.partyScreen({ title: 'Swap with which team member?' });
      if (j === -1 || j === i) continue;
      const party = Game.player.party;
      [party[i], party[j]] = [party[j], party[i]];
      Sfx.select();
    }
  }

  async function overworldBag() {
    while (true) {
      const id = await UI.bagScreen({});
      if (!id) return;
      if (ITEMS[id].kind === 'orb') { await UI.say('Best saved for battle!'); continue; }
      const t = await UI.partyScreen({ title: 'Use on which team member?' });
      if (t === -1) continue;
      const c = Game.player.party[t];
      const it = ITEMS[id];
      if (it.kind === 'heal') {
        if (c.hp <= 0 || c.hp >= c.maxHp) { await UI.say("It won't have any effect."); continue; }
        const from = c.hp;
        c.hp = Math.min(c.maxHp, c.hp + it.amount);
        Game.player.bag[id]--;
        Sfx.heal();
        await UI.say(`${creatureName(c)} recovered ${c.hp - from} HP!`);
      } else {
        const works = c.hp > 0 && c.status && (it.cures.includes('ALL') || it.cures.includes(c.status));
        if (!works) { await UI.say("It won't have any effect."); continue; }
        const txt = STATUS_INFO[c.status].text;
        c.status = null; c.sleepTurns = 0;
        Game.player.bag[id]--;
        Sfx.heal();
        await UI.say(`${creatureName(c)} is no longer ${txt}!`);
      }
    }
  }

  // ---------------- drawing ----------------
  function playerRenderPos() {
    const p = Game.player;
    let px = p.x * TILE, py = p.y * TILE;
    if (moving) {
      px = (moving.fromX + (p.x - moving.fromX) * moving.t) * TILE;
      py = (moving.fromY + (p.y - moving.fromY) * moving.t) * TILE;
    }
    return [px, py];
  }

  function draw(ctx) {
    const m = map();
    const w = mapW(), h = mapH();
    const VW = Game.viewW; // logical viewport width adapts to the screen
    const [prx, pry] = playerRenderPos();

    let camX, camY;
    camX = w * TILE <= VW ? -(VW - w * TILE) / 2
      : Math.max(0, Math.min(w * TILE - VW, prx - (VW - TILE) / 2));
    camY = h * TILE <= VIEW_H ? -(VIEW_H - h * TILE) / 2
      : Math.max(0, Math.min(h * TILE - VIEW_H, pry - (VIEW_H - TILE) / 2));
    // snap the camera to whole device pixels (the canvas runs at 2x) —
    // fractional scroll offsets cause visible seams between tile rows
    camX = Math.round(camX * 2) / 2;
    camY = Math.round(camY * 2) / 2;

    ctx.fillStyle = '#101018';
    ctx.fillRect(0, 0, VW, VIEW_H);

    const mapId = Game.player.map;
    const indoor = !m.outdoor;
    const mapBg = GameAssets.mapBackground(mapId);
    if (mapBg) {
      ctx.drawImage(mapBg, Math.round(-camX), Math.round(-camY));
      if (mapId === 'lab') {
        const ballImg = GameAssets.labBallImg();
        const pFlags = Game.player.flags;
        const leftCol = pFlags.starter_id === 'bulbasaur' ? 10 : (pFlags.starter_id === 'charmander' ? 8 : 9);
        for (let col = 8; col <= 10; col++) {
          if (pFlags.starter && col !== leftCol) continue;
          const bx = col * TILE - camX;
          const by = 4 * TILE + 2 - camY;
          if (ballImg) {
            ctx.drawImage(ballImg, Math.round(bx), Math.round(by));
          } else {
            ctx.fillStyle = '#d03028'; ctx.fillRect(Math.round(bx) + 4, Math.round(by) + 4, 8, 4);
            ctx.fillStyle = '#f8f8f8'; ctx.fillRect(Math.round(bx) + 4, Math.round(by) + 8, 8, 4);
            ctx.fillStyle = '#202020'; ctx.fillRect(Math.round(bx) + 3, Math.round(by) + 7, 10, 2);
            ctx.fillRect(Math.round(bx) + 6, Math.round(by) + 6, 4, 4);
            ctx.fillStyle = '#ffffff'; ctx.fillRect(Math.round(bx) + 7, Math.round(by) + 7, 2, 2);
          }
        }
      }
    } else {
      const x0 = Math.floor(Math.max(0, camX) / TILE) - 1;
      const y0 = Math.floor(Math.max(0, camY) / TILE) - 1;
      const xSpan = Math.ceil(VW / TILE) + 3;
      for (let y = Math.max(0, y0); y < Math.min(h, y0 + 14); y++) {
        for (let x = Math.max(0, x0); x < Math.min(w, x0 + xSpan); x++) {
          // building footprints render as grass; the decal image goes on top
          const ch = GameAssets.tileCovered(mapId, x, y) ? '.' : m.rows[y][x];
          const dx = x * TILE - camX, dy = y * TILE - camY;
          if (!GameAssets.drawTile(ctx, ch, dx, dy, indoor)) {
            ctx.drawImage(Sprites.tile(ch), dx, dy);
          }
        }
      }
      GameAssets.drawDecals(ctx, mapId, camX, camY);
    }

    // entities sorted by y so lower ones draw in front
    const ents = [];
    for (const npc of visibleNpcs()) {
      ents.push({
        y: npc.y * TILE + (npc.oy || 0),
        draw: () => {
          const nx = npc.x * TILE + (npc.ox || 0) - camX;
          const ny = npc.y * TILE + (npc.oy || 0) - camY;
          const stepping = (npc.ox || npc.oy);
          const frame = stepping ? (Math.floor(Math.abs(npc.ox + npc.oy) / 4) % 4) : 0;
          if (!GameAssets.drawActor(ctx, npc.kind, npc.facing, frame, nx, ny)) {
            ctx.drawImage(Sprites.human(npc.kind, npc.facing), nx, ny - 1);
          }
          if (npc.alert) {
            ctx.fillStyle = '#fff';
            ctx.fillRect(nx + 4, ny - 17, 8, 10);
            ctx.fillStyle = '#d03028';
            ctx.fillRect(nx + 7, ny - 15, 2, 4);
            ctx.fillRect(nx + 7, ny - 10, 2, 2);
          }
        },
      });
    }
    const bob = moving && moving.t > 0.2 && moving.t < 0.7 ? -1 : 0;
    const pFrame = moving ? (Math.floor(moving.t * 4) % 4) : 0;
    ents.push({
      y: pry,
      draw: () => {
        if (!GameAssets.drawActor(ctx, 'player', Game.player.facing, pFrame, prx - camX, pry - camY)) {
          ctx.drawImage(Sprites.human('player', Game.player.facing), prx - camX, pry - camY - 1 + bob);
        }
        // tall grass partly covers the player
        if (tileAt(Game.player.x, Game.player.y) === 't' && !moving) {
          const dx = Game.player.x * TILE - camX, dy = Game.player.y * TILE - camY;
          ctx.save();
          ctx.beginPath();
          ctx.rect(dx, dy + 9, 16, 7);
          ctx.clip();
          if (!GameAssets.drawTile(ctx, 't', dx, dy, indoor)) {
            ctx.drawImage(Sprites.tile('t'), 0, 9, 16, 7, dx, dy + 9, 16, 7);
          }
          ctx.restore();
        }
      },
    });
    ents.sort((a, b) => a.y - b.y);
    for (const e of ents) e.draw();
  }

  return { update, draw };
})();
