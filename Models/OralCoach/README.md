# Oral coach: live signals + live recommendations

Reads your webcam and microphone while you practise, and shows live scores in a camera window: your mirrored
webcam image with a box around your face (green = looking at the camera, yellow = looking away), what you just said
as subtitles, and a score panel on the right. Every snapshot is saved as JSON in a session log. On top of that sits the **coach**: it spots patterns over time (stressed for
20 s, confident then a drop, voice tense while the face looks calm...), retrieves advice from a small library of
cited tip cards, lets an LLM word one short tip with a "why", shows it on the camera image, then checks whether it
helped (IT WORKED, or a stronger tip with a different strategy). See [The coach](#the-coach-recommendations) below.
The window has a chunky retro handheld-RPG look with game sound effects. See [Retro look and sounds](#retro-look-and-sounds).

## Setup (once)

1. Python 3.11 to 3.13, ideally in a virtual environment.
2. Install PyTorch **with CUDA** from https://pytorch.org/get-started/locally/ (pick your OS, pip, CUDA 12.x).
3. `pip install -r requirements.txt`
4. `python setup_models.py` downloads all models (~2 GB) so the first run starts fast
   (add `--whisper-model base` if you run with the base speech-to-text model).

## Run

| Command | What it does |
| --- | --- |
| `python run.py` | Webcam + mic, camera window with scores and tips |
| `python run.py --whisper-model base` | Use a faster, less accurate speech-to-text model |
| `python run.py --speak` | Read each tip out loud from the start (short version), so you can keep looking at the camera |

In the camera window: `q` or Esc quits, `s` turns the sound on or off, `v` turns spoken tips on or off. Each session is saved to `logs/session-<time>.jsonl`.

**Calibration:** for the first 30 seconds of speech the coach learns *your* normal pitch, volume, pace and
expression. After that, 0.5 means "like your normal" and the bars move when you drift from it.
So start each session by talking normally for half a minute.

## Retro look and sounds

The camera window is drawn like an old handheld RPG around your normal (mirrored) webcam image: chunky
brackets around your face, a status screen with HP-style bars, and the tip in a dialog box that types itself
out. Everything is drawn in code (`coach/pixelfont.py` is a small hand-made pixel font that also does French
accents), and every sound is synthesised from square and triangle waves in `coach/sfx.py`, so nothing is downloaded.

| When | Sound | On screen |
| --- | --- | --- |
| Window opens / closes | start-up jingle / power-down | |
| Calibration done | rising power-up | "CALIBRATED!" |
| A tip appears | chime (level 1), double ping (level 2), soft calm chord (level 3 pause), heartbeat (freeze) | dialog box slides in |
| Tip text appears | little text blips, like RPG dialog | letters typed one by one |
| A bar stays red for 2 s | soft double beep (at most every 8 s) | the bar and its name blink |
| That bar is back to green | small up blip | |
| "Did it help?" says yes | success jingle + coin | "IT WORKED!" drops in, green flash, +XP |
| ... says not yet | gentle two-note down | "NOT YET..." shakes |
| Steady streak (praise) | sparkle | "NICE!", gold flash, +XP |
| Enough XP | level-up fanfare | "LEVEL UP!" |
| Key press | menu tick | |

XP and levels only live in the window (tip worked +50, praise +30, calibration +20, +5 per calm 10 s). They are
not logged, so the JSON events stay exactly as before and the game can use its own scoring.
`coach/juice.py` is the one place that maps events to sounds and pop-ups, if you want to change them.

Sound plays through `sounddevice` (already installed for the microphone), with Windows' built-in `winsound` as
a fallback. **Use headphones for real sessions**: from speakers, the microphone would hear the coach's own
sounds and count them as your voice.

## The coach (recommendations)

```
1 s snapshots ──► pattern detector ──► is a tip allowed now? ──► retrieve cards ──► best match ──► LLM words it
 (channels,       (last 30-90 s,         (cooldown, nothing        for this pattern     (a new one     tip + why
  "why" parts)    face vs voice)          being checked)           (BM25 + filter)      if escalating) (+ source)
                                                                                                   │
   IT WORKED (rest) / not yet (escalate next time) ◄── "did it help?" (badness 6-20 s after vs 8 s before) ◄──┘
```

**Patterns** (`coach/recommend/patterns.py`, thresholds at the top). Only after calibration.

| Pattern | Fires when | Uses |
| --- | --- | --- |
| You froze | silent ≥ 4 s right after speaking | voice |
| Stressed for a while | stress ≥ 0.66 for 15 s | face + voice |
| Speeding up while nervous | pace ≥ 0.65 and stress ≥ 0.6 for 10 s | face + voice |
| **Voice tense, face calm** | voice side of stress ≥ 0.68 while the face side ≤ 0.58, for 10 s | face vs voice |
| **Face tense, voice fine** | the opposite, for 12 s | face vs voice |
| Confidence dropped | confidence fell by ≥ 0.15 compared with 20-45 s ago; says which part dropped most | face + voice |
| Not looking at the audience | eye contact ≤ 35 % for 12 s (adds "looking down, notes?") | face |
| Lots of filler words | ≥ 6 fillers/min for 8 s | words |
| Talking fast / Flat, monotone voice / Low energy | pace ≥ 0.75 for 15 s / voice variety ≤ 0.25 for 20 s / energy ≤ 0.32 for 20 s | voice |

The two face-vs-voice patterns are the multimodal part: one model alone cannot see them, only the comparison can.

**More patterns** (`coach/recommend/more_patterns.py`, thresholds at the top). Face and voice *together*, and slow
changes over the session. "Your start" = the average of your first 60 s of talking after calibration.

| Pattern | Fires when | Uses |
| --- | --- | --- |
| Reading from your notes | looking down ≥ 40 % and eye contact ≤ 40 % *and* a flat voice (≤ 0.40), for 10 s | face + voice |
| Stressed and getting quieter | stress ≥ 0.62 *and* volume ≤ 0.35 (quieter than your normal), for 10 s | face + voice |
| Flat voice, blank face | voice variety ≤ 0.30 *and* positivity ≤ 0.38 (no smile), not stressed, for 15 s | face + voice |
| Energy fading since your start | energy over the last 25 s is ≥ 0.15 below your start | voice, over time |
| Confident start, then a slow slide | confidence over the last 30 s is ≥ 0.12 below your start (sudden drops stay "Confidence dropped") | face + voice, over time |
| Speeding up little by little | pace higher in each of the last three 15 s blocks, +0.12 in total, now ≥ 0.6 | voice, over time |
| Stress creeping up | stress higher in each of the last three 15 s blocks, +0.10 in total, now 0.55-0.66 | face + voice, over time |

The four "over time" patterns are early or slow versions of an older one (low energy, confidence dropped, talking
fast, stressed for a while). They share its cards and its escalation, like the fast lane does. They
cost almost nothing: a few averages over the snapshots the detector already keeps.

**Good news too** (`coach/recommend/progress.py`). Two `progress` events, shown as a green pop-up with a sound and XP:
*recovered* (a tip worked and you are still better 30 s later: "Still better 30 s after slow breathing: stress 0.81
-> 0.55") and *improving* (a signal got better in each of the last three 20 s blocks with no tip: "Getting better:
eye contact 41% -> 58% -> 77% over 60 s").

**Spoken tips** (`coach/speaker.py`, off by default). With `--speak`, or after pressing `v`, each tip is read out
loud in a few words, so you don't have to look at the screen and lose eye contact. It uses the offline Windows
voices (`pip install pyttsx3`; without it, Windows PowerShell's voice), in its own thread, after the tip sound.
While it talks the mic analysis ignores the audio, so the coach's voice never counts as yours (no fake pace,
fillers, stress or "you froze"), and the sound effects get quieter. Use headphones so it doesn't reach the mic.

**When it speaks.** At most one tip every 25 s, never while the last tip is still being checked (a freeze may
interrupt). The most important active problem wins.

**Escalation.** If the same problem is still there after a tip that did not help (within 90 s), the next tip goes
one level up and prefers a *different* strategy: level 1 *hint* ("Slow down a little"), level 2 *action* ("Say
your next sentence at half speed"), level 3 *pause* ("Stop. Breathe, then restart slower"). A freeze starts at
level 2. If a level-3 pause did not help either, the coach leaves that problem alone for 90 s instead of nagging.

**Did it help?** Every pattern has a "badness" number (stress, 1 - eye contact, fillers...). The coach compares
its average in the 8 s before the tip with seconds 6-20 after it. Enough drop = it worked. You see the result on
screen ("That helped: stress 0.81 -> 0.56"). If it worked, the coach leaves that problem alone for 45 s; if not,
the next tip on it is escalated (see above).

**Fresh numbers after a tip.** Filler words per minute and pace are counted over the last 30 s. Right after a
filler or pace tip, that count starts again from the tip (`ChannelBuilder.restart`, looking back at least 6 s), so
the check and the next decision only see what you did *after* reading the tip. Without this, the fillers you said
just before the tip kept the number high for 30 s, and the coach said "not yet" and escalated even though you had
stopped. The panel's FILLERS and PACE numbers restart too. Stress, eye contact and the other scores already use
windows of 3-5 s, so they need no restart. While a spoken tip plays (`--speak`) the mic is off; that time is left
out of fillers per minute and pace, so it doesn't look like "no fillers" or "slower".

**Timing rules:**
- An urgent moment (freeze, panic...) does not wait for a normal tip that the LLM is still writing: that tip is
  dropped and the urgent one is shown at once.
- If the problem stops while the LLM is writing, the tip is dropped instead of showing up late.
- Silent for 8 s or more (thinking, notes): only freeze help, no eye contact or face tips. Silent for 30 s or more
  counts as a break, not a freeze: no more freeze tips, no "steady" praise, progress or XP until you talk again.
- A tip with no verdict (e.g. your face was lost during the check) does not count as "did not help", so the next
  one is not escalated because of it.
- When the face is not found, eye contact is "unknown" (no tips), not "looking away".

**Choosing the strategy.** Same advice for everyone, nothing is saved between sessions: the coach takes the card
that best matches this moment (BM25 score). When escalating, the strategy that just failed is pushed down so a
different one is tried. The tip says why: "slow breathing: best match for this moment", or "switching to relaxing
shoulders and throat: slow breathing did not help".

**RAG library** (`knowledge/`). `cards/*.md` holds 31 tip cards. Each one has a strategy, the patterns it fits,
wording for the 3 levels, a one-line "why", a short source key and a passage. `sources.md` has the full references:
two open public speaking textbooks (*Stand up, Speak out*; *Exploring Public Speaking*) and studies on
breathing, reframing arousal, smiling, fillers, eye contact, pitch variety, gestures, vocal confidence, choking
under pressure and the recency effect (`cards/more.md` has the 13 newest cards). It does not use psychiatry material. Retrieval first keeps only cards
tagged for the pattern, then ranks them with BM25 against the pattern and its evidence. To use real textbook text,
paste chapters as `.txt` files into `knowledge/books/` (first line = citation). They are split into passages,
searched too, and given to the LLM as extra context.

**LLM** (`coach/recommend/writer.py`). It only *rewords* the chosen card for this moment, in at most 10 words plus
a "why" that quotes your evidence. It never picks the strategy and never makes up the source, which always
comes from the card. If it takes longer than 6 s or returns something unusable, the card's own wording is used.
The coach tries these in order, and a red line tells you which one was missing:
- **Ollama (free, offline, runs on your GPU):** install from https://ollama.com, then
  `ollama pull qwen2.5:3b` (about 2 GB, also good in French).
- **Claude API:** `pip install anthropic` and set `ANTHROPIC_API_KEY`.
- **None:** the card's own wording. Instant, always available.

**Events for the game.** Everything is a JSON line with a `type`, and every session log has them:

| type | key fields |
| --- | --- |
| `tick` | the 1-second snapshot, as before |
| `pattern_start` / `pattern_end` | `pattern`, `label`, `modality`, `evidence` / `duration_s` |
| `tip` | `id`, `pattern`, `level` (1-3), `level_name`, `text`, `why`, `strategy`, `card`, `source`, `source_full`, `choice` (why this strategy), `escalated_because`, `restart` (rolling counts restarted at this tip), `evidence`, `retrieved` (cards with scores), `backend` |
| `tip_result` | `tip_id`, `worked`, `metric`, `before`, `after`, `message` |
| `praise` | `text`, `reasons` (after 45 s with no problem) |
| `progress` | `kind` (`recovered` / `improving`), `text`, `why`, and `pattern` + `strategy` or `signal` |

A game can turn `tip_result.worked` and `praise` into points and streaks, and `level` into urgency.

## What each score is made of

The `why` field in every JSON snapshot lists these parts with their values, so a tip can say *why* it fired.

| Channel | Built from |
| --- | --- |
| stress | voice arousal up, voice valence down (audEERING), face arousal up, face valence down (HSEmotion), pitch above your normal |
| confidence | voice dominance (audEERING), louder than your normal, eye contact, pitch not raised |
| energy | voice arousal, pitch variety, loudness |
| positivity | face valence, smile (MediaPipe), voice valence |
| eye_contact | share of the last 5 s with head and eyes towards the camera (MediaPipe) |
| expressiveness | pitch variation vs your normal; low = monotone (Parselmouth) |
| pace / pace_wpm | words per minute over the last 30 s, fillers excluded (faster-whisper) |
| pace_now / pace_now_wpm | the same over only the last 10 s of words (ending at the newest word, since speech-to-text runs a little behind). The panic fast lane uses it, because a 30 s average hides a sudden speed-up |
| fillers_per_min | um, uh, euh, ben... (faster-whisper; "like", "du coup" counted separately as soft fillers) |
| silence_now_s | how long you have been silent right now (Silero VAD) |

If a model fails to load, the scores that need it simply leave that part out, and a red warning line says which one.

## Files

```
run.py              start here
setup_models.py     downloads models
coach/config.py     every threshold and window length
coach/face.py       webcam: MediaPipe + HSEmotion
coach/voice.py      mic: Silero VAD, Parselmouth, audEERING, faster-whisper
coach/channels.py   raw measurements -> channels, personal baseline
coach/overlay.py    camera window (OpenCV), retro style: feed, face brackets, status bars, tip dialog
coach/pixelfont.py  hand-made pixel font for the window
coach/sfx.py        retro sound effects, synthesised in code, and the little mixer that plays them
coach/juice.py      game feel: events -> sounds, XP, pop-ups, flashes (display only)
coach/speaker.py    spoken tips (--speak / v): offline text-to-speech, mutes the mic analysis while talking
coach/recommend/    the coach: patterns.py, urgent.py (fast lane), more_patterns.py (face+voice, over time),
                    progress.py (good news), knowledge.py (BM25 + strategy names),
                    writer.py (LLM), coach.py (when/what/did it help)
knowledge/          tip cards, sources, optional book chapters
```

## Known limits

- Built and checked in a cloud machine without a webcam, microphone or GPU. MediaPipe,
  Silero VAD, the pitch code and the audEERING model wiring all ran there. The real HSEmotion, audEERING and Whisper
  weights could not be downloaded there, so the first real run on your laptop is the real test.
- faster-whisper on GPU needs CUDA 12 and cuDNN 9 libraries. If it errors, try `--whisper-model base` or set
  `device = "cpu"` in `coach/config.py`.
- The coach's thresholds were tuned on made-up data. Expect to adjust a few (top of `patterns.py`) after your
  first real sessions.
- "Did it help?" can't fully separate the tip from things that would have calmed down anyway. A fair next step
  is to sometimes stay silent on purpose and use that as a baseline.
- The tip cards are short paraphrases written for this project. Check a reference in `knowledge/sources.md`
  before quoting it in your report.
- The sound effects were checked as data but could not be listened to on the cloud machine, which has no speakers.
- Lab emotion models see a nervous presenter mostly as "neutral". That's why stress and confidence mix several cues
  instead of trusting one label.
