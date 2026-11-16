// ===== Map data =====
// Tile legend:
//   . grass   : path    t tall grass (encounters)   T tree    W water
//   F flowers f fence   r roof   w wall   D door(warp)   s sign
//   _ floor   k counter m exit mat(warp)  b bookshelf  O boulder
//   B bed     h stairs(warp)
//   R center roof   M mart roof   c center emblem   g mart emblem
//   H healing machine   G goods shelf   n bench
//   1 2 3 starter desks   x void
// Solid: T W f r w s k b O B R M c g H G n 1 2 3 x  (D, m, h are walkable warps)
const SOLID_TILES = new Set(['T', 'W', 'f', 'r', 'w', 's', 'k', 'b', 'O', 'B',
  'R', 'M', 'c', 'g', 'H', 'G', 'n', '1', '2', '3', 'x']);

// All names and dialogue below are original.
const MAPS = {
  // ------------------------------------------------ Player's bedroom (upstairs)
  bedroom: {
    name: 'YOUR ROOM',
    rows: [
      'wwwwwwwwwwww',
      'wkk_bb_wwwww',
      'w______wwh_w',
      'w______www_w',
      'wBB___kk___w',
      'wBB________w',
      'w__________w',
      'w__________w',
      'wwwwwwwwwwww',
    ],
    warps: { '9,2': { map: 'house', x: 10, y: 3, facing: 'down' } },
    signs: {},
    npcs: [],
  },

  // ------------------------------------------------ Player's house (downstairs)
  house: {
    name: 'YOUR HOUSE',
    rows: [
      'wwwwwwwwwwwww',
      'wkkkkk____hww',
      'w_________hww',
      'w___________w',
      'w____bbb____w',
      'w____bbb____w',
      'w___________w',
      'wh_________hw',
      'wwwwmwwwwwwww',
      'xxxxxxxxxxxxx',
    ],
    warps: {
      '10,2': { map: 'bedroom',  x: 10, y: 2, facing: 'down' },
      '4,8':  { map: 'hometown', x: 6, y: 8, facing: 'down' },
    },
    signs: {},
    npcs: [
      { id: 'mom', x: 8, y: 5, kind: 'parent', facing: 'left', special: 'parent' },
    ],
  },

  // ------------------------------------------------ WILLOWBROOK (start town)
  // Full authentic 24x20 Pallet Town geometry with multi-directional solid collisions.
  hometown: {
    name: 'WILLOWBROOK',
    outdoor: true,
    rows: [
      'TTTTTTTTTTTT..TTTTTTTTTT',
      'TTTTTTTTTTTT..TTTTTTTTTT',
      'T......................T',
      'T......................T',
      'T....rrrrr....rrrrr....T',
      'T....rrrrr....rrrrr....T',
      'T....wwwww....wwwww....T',
      'T...swDwww...swDwww....T',
      'T......................T',
      'T............rrrrrrr...T',
      'T............rrrrrrr...T',
      'T....fffff...rrrrrrr...T',
      'T....FFFF....wwwwwww...T',
      'T....FFFF....wwwDwww...T',
      'T....s.............s...T',
      'T......................T',
      'T......WWWW..fffffff...T',
      'T......WWWW............T',
      'T......WWWW............T',
      'TTTTTTTTTTTTTTTTTTTTTTTT',
    ],
    links: { up: 'route1' },
    warps: {
      '6,7':   { map: 'house',    x: 4, y: 7, facing: 'up' },
      '15,7':  { map: 'rexhouse', x: 4, y: 7, facing: 'up' },
      '16,13': { map: 'lab',      x: 6, y: 11, facing: 'up' },
    },
    signs: {
      '4,7':   "YOUR HOUSE\nThe warm porch light is on.",
      '13,7':  "REX'S HOUSE\nA note on the door: 'At the lab. Do not disturb.'",
      '5,14':  "WILLOWBROOK BOTANICAL GARDEN\n\"Nurture knowledge from small seeds.\"",
      '19,14': "ADA RESEARCH LAB\n\"Curiosity powers discovery.\"",
    },
    npcs: [
      { id: 'hv1', x: 18, y: 15, kind: 'villager', facing: 'down',
        dialog: [
          'Going into the tall grass without your own partner? Bad idea, kid.',
          'Wild computational creatures roam the tall grass!',
          'If you want a partner of your own, Professor ADA is the person to see. Her Research Lab is the large brick building south of here.',
        ] },
      { id: 'hv2', x: 10, y: 8, kind: 'villager2', facing: 'right',
        dialog: [
          'Every system has underlying rules! In battle, elemental and logical types govern the outcome.',
          'WATER douses FIRE, FIRE burns GRASS, GRASS drinks WATER.',
          'Understand the principles and weaknesses, and you will solve any challenge on your KnowledgeQuest!',
        ] },
    ],
  },

  // ------------------------------------------------ Rex's house
  rexhouse: {
    name: "REX'S HOUSE",
    rows: [
      'wwwwwwwwwwwww',
      'wbb_k_bb_bb_w',
      'w___________w',
      'w___________w',
      'w____bbb____w',
      'w____bbb____w',
      'w___________w',
      'wh_________hw',
      'wwwwmwwwwwwww',
      'xxxxxxxxxxxxx',
    ],
    warps: { '4,8': { map: 'hometown', x: 15, y: 8, facing: 'down' } },
    signs: {},
    npcs: [
      { id: 'rival_mom', x: 8, y: 5, kind: 'rival_mom', facing: 'left',
        dialog: [
          "MOTHER: Oh, good morning! Are you heading to Prof. ADA's laboratory?",
          "MOTHER: REX bounded out the door before finishing his breakfast, shouting about research hypotheses and being the top cadet.",
          "MOTHER: He is always in such a rush to prove himself! Please keep an eye on him for me, dear.",
          "MOTHER: If you ever need a warm cup of tea or a moment to rest, my door is always open.",
        ] },
    ],
  },

  // ------------------------------------------------ Maple's lab
  lab: {
    name: "ADA'S LAB",
    rows: [
      'wwwwwwwwwwwww',
      'wkkkkkk_bbbbw',
      'w___________w',
      'wHH_________w',
      'wHH_____123_w',
      'w___________w',
      'w___________w',
      'wbbbb___bbbbw',
      'wbbbb___bbbbw',
      'w___________w',
      'w___________w',
      'wh_________hw',
      'wwwwwwmwwwwww',
      'xxxxxxxxxxxxx',
    ],
    warps: { '6,12': { map: 'hometown', x: 16, y: 14, facing: 'down' } },
    signs: {},
    npcs: [
      { id: 'prof', x: 6, y: 3, kind: 'prof', facing: 'down', special: 'prof' },
      { id: 'rex', x: 5, y: 4, kind: 'rival', facing: 'right', hiddenIf: 'rex_gone',
        dialog: [
          "REX: Oh, it's you. Took you long enough to roll out of bed.",
          "REX: I'm only waiting because Prof. ADA insisted we both be here. Hurry up and pick so I can analyze your choice!",
        ] },
    ],
  },

  // ------------------------------------------------ FERNWAY TRAIL (route)
  route1: {
    name: 'FERNWAY TRAIL',
    outdoor: true,
    rows: [
      'TTTTTTTTT::TTTTTTTTT',
      'T........::........T',
      'T..ttt...::...ttt..T',
      'T..ttt...::...ttt..T',
      'T.ttttt..::..ttttt.T',
      'T........::........T',
      'T.tttt...::...tttt.T',
      'T.tttt...::...tttt.T',
      'T.tttt...::...tttt.T',
      'T........::........T',
      'T..s.....::........T',
      'T........::........T',
      'T........::..ttttt.T',
      'T........::..ttttt.T',
      'T........::..ttttt.T',
      'T........::........T',
      'T.WW.....::.....WW.T',
      'T.WW.....::.....WW.T',
      'T........::........T',
      'Tttt.....::.....tttT',
      'Tttt.....::.....tttT',
      'T........::........T',
      'T........::........T',
      'TTTTTTTTT::TTTTTTTTT',
    ],
    links: { up: 'stonegate', down: 'hometown' },
    warps: {},
    signs: {
      '3,10': 'FERNWAY TRAIL\nWild creatures lurk in the tall grass!',
    },
    encounters: {
      rate: 0.14,
      levels: [2, 5],
      table: [
        ['pidgey', 30], ['rattata', 30], ['caterpie', 20], ['weedle', 20],
      ],
    },
    npcs: [
      { id: 'cal', x: 8, y: 8, kind: 'scout', facing: 'right',
        trainer: {
          flag: 'tr_cal', name: 'Scout CAL', range: 3,
          party: [['rattata', 5]], prize: 100,
          intro: ['Halt! My bounding boxes just locked onto you with 0.94 IoU!',
                  'Let us see if your Non-Maximum Suppression is sharp enough for battle!'],
          loseText: 'Suppressed like a duplicate box!',
          after: ['Clean localization! Keep your IoU tight and your false alarms low.',
                  'Picnicker MIRA up ahead trains single-stage YOLO detectors. Watch her speed!'],
        } },
      { id: 'mira', x: 11, y: 13, kind: 'picnic', facing: 'left',
        trainer: {
          flag: 'tr_mira', name: 'Picnicker MIRA', range: 3,
          party: [['caterpie', 4], ['weedle', 4]], prize: 80,
          intro: ['Two-stage detectors are way too slow to spot tiny bugs in tall grass!',
                  'You Only Look Once out here — show me your YOLO mastery!'],
          loseText: 'Spotted at stride 8 in a single pass!',
          after: ['Anchor-free multi-scale grids spot even the tiniest CATERPIE!',
                  'Up in STONEGATE GYM, ROCCO and Leader MASON test OCR and full Vision Systems!'],
        } },
    ],
  },

  // ------------------------------------------------ STONEGATE (gym town)
  stonegate: {
    name: 'STONEGATE',
    outdoor: true,
    rows: [
      'TTTTTTTTTTTTTTTTTTTT',
      'T..................T',
      'T..................T',
      'T..................T',
      'T..................T',
      'T....rrrrrrr.......T',
      'T....rrrrrrr.......T',
      'T....wwwDwww..s....T',
      'T.......:..........T',
      'T.......:..........T',
      'T.......:..........T',
      'T.RRRRR.:....MMMMM.T',
      'T.RRRRR.:....MMMMM.T',
      'T.wcDww.:....wDgww.T',
      'T.s.:...:.....:..s.T',
      'T...::::::::::::...T',
      'T........:.........T',
      'T..s.....:.........T',
      'T........:.........T',
      'TTTTTTTTT::TTTTTTTTT',
    ],
    links: { down: 'route1' },
    warps: {
      '4,13':  { map: 'healstone', x: 6, y: 5, facing: 'up' },
      '14,13': { map: 'shop1',     x: 5, y: 4, facing: 'up' },
      '8,7':   { map: 'gym',       x: 6, y: 9, facing: 'up' },
    },
    signs: {
      '14,7': 'STONEGATE GYM\nLeader: MASON, the Wall That Walks.',
      '2,14': 'POKEMON CENTER\nHeal your team — free of charge.',
      '17,14': 'POKE MART\nEverything a trainer needs.',
      '3,17': 'STONEGATE\n"Built on rock, run on grit."',
    },
    npcs: [
      { id: 'sv1', x: 13, y: 16, kind: 'villager2', facing: 'down',
        dialog: [
          'Challenging MASON? Listen close, because everyone learns this the hard way.',
          'His team shrugs off weak hits. Chip damage will not get you anywhere.',
          'Bring something that hits HARD, or wear him down with poison and powders.',
        ] },
      { id: 'sv2', x: 16, y: 16, kind: 'villager', facing: 'left',
        dialog: [
          'The POKEMON CENTER fixed my PIDGEY up in seconds. Marvelous machine they have.',
          'Rest your team before the gym. Walking in tired is walking out crying.',
        ] },
    ],
  },

  // ------------------------------------------------ Interiors (Stonegate)
  healstone: {
    name: 'POKEMON CENTER',
    rows: [
      'wwwwwwwwwwww',
      'w__H_______w',
      'w__kkkkkk__w',
      'w__________w',
      'w_n______n_w',
      'w__________w',
      'w_____m____w',
      'wwwwwwwwwwww',
    ],
    warps: { '6,6': { map: 'stonegate', x: 4, y: 14, facing: 'down' } },
    signs: {},
    npcs: [
      { id: 'nurse1', x: 5, y: 1, kind: 'nurse', facing: 'down', special: 'nurse' },
    ],
  },

  shop1: {
    name: 'POKE MART',
    rows: [
      'wwwwwwwwww',
      'w___G_GG_w',
      'w_kkkk___w',
      'w________w',
      'w______G_w',
      'w____m___w',
      'wwwwwwwwww',
    ],
    warps: { '5,5': { map: 'stonegate', x: 14, y: 14, facing: 'down' } },
    signs: {},
    npcs: [
      { id: 'clerk1', x: 3, y: 1, kind: 'clerk', facing: 'down', special: 'shop' },
    ],
  },

  gym: {
    name: 'STONEGATE GYM',
    rows: [
      'wwwwwwwwwwww',
      'w__________w',
      'w__________w',
      'w__O____O__w',
      'w__________w',
      'w____O_____w',
      'w__________w',
      'w__________w',
      'w__O____O__w',
      'w__________w',
      'w_____m____w',
      'wwwwwwwwwwww',
    ],
    warps: { '6,10': { map: 'stonegate', x: 8, y: 8, facing: 'down' } },
    signs: {},
    npcs: [
      { id: 'rocco', x: 3, y: 6, kind: 'hiker', facing: 'right',
        trainer: {
          flag: 'tr_rocco', name: 'Trainee ROCCO', range: 3,
          party: [['metapod', 9], ['kakuna', 9]], prize: 180,
          intro: ['Can you read ancient stone inscriptions tilted at 35 degrees?',
                  'DBNet polygons, CRNN columns, and CTC blank tokens — prove your OCR skills!'],
          loseText: 'Decoded with zero Character Error Rate!',
          after: ['You mastered CTC alignment without a single character box!',
                  'Leader MASON awaits ahead. His final Pokemon demands an Open-Text Vision Pipeline synthesis!'],
        } },
      { id: 'mason', x: 5, y: 1, kind: 'leader', facing: 'down', special: 'leader',
        trainer: {
          flag: 'tr_mason', name: 'Leader MASON', range: 0,
          party: [['pidgeotto', 11], ['raticate', 13]], prize: 1300, badge: true,
          intro: ['Welcome to STONEGATE GYM. Separate models crumble under real-world edge cases.',
                  'Combine YOLO detection, perspective rectification, and OCR into an unbreakable pipeline — battle!'],
          loseText: 'Flawless architecture... the Granite Vision trial is yours.',
          after: ['Take the GRANITE BADGE. Your Computer Vision mastery stood firm under live neural grading.',
                  'Keep checking your Field Journal until all five Concept Synthesis Badges shine!'],
        } },
    ],
  },
};
