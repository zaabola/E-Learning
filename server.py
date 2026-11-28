r"""
KnowledgeQuest Live 7-Model Game & AI Server (`server.py`)
==========================================================
Serves the vanilla HTML/CSS/JS GBA RPG (`index.html`, `js/`, `css/`, `assets/`,
`course_material/`) AND exposes real-time REST endpoints powered by all 3 custom
PyTorch models (100% local, zero cloud APIs, zero webcam):

  1. Model #1 (`Final-DKT-GRU`)    -> Cognitive 5-Concept Mastery Tracing
  2. Model #2 (`Double-DQN` GM)    -> RL Adaptive Action & Difficulty Selection
  3. Model #3 (`DeBERTa-v3-Small`) -> Semantic Open-Text Short-Answer Grading

Run:
  .\.venv\Scripts\python.exe server.py
Then open:
  http://127.0.0.1:8000
"""

import argparse
import functools
import json
import mimetypes
import os
import sys
import threading
import uuid
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(REPO_ROOT, 'Models')
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from engine.knowledgequest_engine import KnowledgeQuestAIEngine
try:
    from OralCoach.oral_coach_service import OralCoachService
except Exception as _oc_imp_err:
    print(f"  [OralCoach Warning] Bridge unavailable: {_oc_imp_err}", flush=True)
    OralCoachService = None

mimetypes.add_type('audio/wav', '.wav')
mimetypes.add_type('font/ttf', '.ttf')
mimetypes.add_type('application/javascript', '.js')


class GameServerState:
    def __init__(
        self,
        default_course: str = 'vision_yolo_ocr',
        default_profile: str = 'average',
        vlm_mode: str = 'hybrid',
        vlm_model_id: str = 'Qwen/Qwen3-VL-8B-Instruct',
        llm_mode: str = 'fallback',
        reasoner_model_id: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        stylizer_model_id: str = 'NousResearch/Meta-Llama-3.1-8B-Instruct',
    ):
        self.lock = threading.Lock()
        self.engine = KnowledgeQuestAIEngine(
            vlm_mode=vlm_mode,
            vlm_model_id=vlm_model_id,
            llm_mode=llm_mode,
            reasoner_model_id=reasoner_model_id,
            stylizer_model_id=stylizer_model_id,
        )
        self.session = self.engine.create_session(profile=default_profile, course_id=default_course)
        self.ingest_jobs = {}
        self.oral_coach = (
            OralCoachService(llm_orchestrator=self.engine.llm_orchestrator)
            if OralCoachService is not None
            else None
        )

    def reset_session(self, course_id: str = 'vision_yolo_ocr', profile: str = 'average') -> dict:
        with self.lock:
            self.session = self.engine.create_session(profile=profile, course_id=course_id)
            return self.session.get_status_snapshot()


class KnowledgeQuestHTTPHandler(SimpleHTTPRequestHandler):
    server_state: GameServerState = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=REPO_ROOT, **kwargs)

    def end_headers(self):
        # Allow both http://127.0.0.1:8000 AND file:// index.html to talk to /api/*
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        parsed = urlparse(self.path).path
        if parsed.startswith('/api/') or parsed.endswith(('.js', '.css', '.html', '.json')):
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

    def log_message(self, fmt, *args):
        # Keep static asset logs quiet; log API calls concisely
        if self.path.startswith('/api/'):
            pass

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _read_json_body(self) -> dict:
        length = int(self.headers.get('Content-Length', '0') or '0')
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route == '/api/status':
            st = self.server_state
            with st.lock:
                snap = st.session.get_status_snapshot()
            self._send_json(snap)
            return
        if route == '/api/vision/puzzles':
            st = self.server_state
            with st.lock:
                puzzles = st.session.get_diagram_puzzles()
            self._send_json({'puzzles': puzzles})
            return
        if route == '/api/course/story':
            st = self.server_state
            with st.lock:
                sess = st.session
                ctx = sess.prepare_course_story_ctx()
            story = sess.run_course_story_from_ctx(ctx, allow_blocking=True)
            with st.lock:
                story = sess.finalize_course_story(story)
            self._send_json(story)
            return
        if route == '/api/npc/dialogue':
            st = self.server_state
            qs = parse_qs(parsed.query)
            npc_key = (qs.get('npc') or ['prof_ada'])[0]
            starter_name = (qs.get('starter') or ['your partner'])[0]
            with st.lock:
                sess = st.session
                ctx = sess.prepare_npc_dialogue_ctx(npc_key=npc_key, starter_name=starter_name)
            dlg = sess.run_npc_dialogue_from_ctx(ctx)
            print(
                f"  [NPC DIALOGUE] {npc_key} | {dlg.get('model_name', 'Stylizer')} "
                f"| src={dlg.get('source')} | cache={dlg.get('cache_hit')}",
                flush=True,
            )
            self._send_json(dlg)
            return
        if route == '/api/course/ingest/status':
            st = self.server_state
            qs = parse_qs(parsed.query)
            job_id = (qs.get('job_id') or [''])[0]
            job = st.ingest_jobs.get(job_id)
            if not job:
                self._send_json({'error': f'Unknown ingest job_id: {job_id}'}, status=404)
                return
            self._send_json(job)
            return
        if route == '/api/oral_coach/status':
            st = self.server_state
            if st.oral_coach is None:
                self._send_json({'active': False, 'error': 'OralCoachService unavailable'}, status=503)
                return
            qs = parse_qs(parsed.query)
            try:
                since_seq = int((qs.get('since_seq') or ['0'])[0])
            except ValueError:
                since_seq = 0
            out = st.oral_coach.get_status(since_seq=since_seq)
            if isinstance(out.get('capabilities'), dict):
                is_loopback = self.client_address[0] in ('127.0.0.1', '::1')
                is_tunneled = bool(self.headers.get('CF-Connecting-IP') or self.headers.get('X-Forwarded-For'))
                out['capabilities']['is_local_client'] = bool(is_loopback and not is_tunneled)
            self._send_json(out)
            return
        super().do_GET()

    def do_POST(self):
        route = urlparse(self.path).path
        st = self.server_state
        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_json({'error': f'Invalid JSON body: {exc}'}, status=400)
            return

        if route == '/api/session/reset':
            course_id = body.get('course_id') or st.session.course_id or 'vision_yolo_ocr'
            profile = body.get('profile') or 'average'
            snap = st.reset_session(course_id=course_id, profile=profile)
            print(f"  [SESSION RESET] Course={snap['course_id']} | Profile={profile}")
            self._send_json(snap)
            return

        if route == '/api/session/sync':
            with st.lock:
                req_course = body.get('course_id')
                if req_course and req_course != st.session.course_id:
                    st.session = st.engine.create_session(
                        profile=st.session.profile, course_id=req_course
                    )
                st.session.sync_progress(
                    consolidated=body.get('consolidated'),
                    mastery=body.get('mastery'),
                )
                snap = st.session.get_status_snapshot()
            self._send_json(snap)
            return

        if route == '/api/encounter/next':
            focus_c = body.get('focus_concept_id', None)
            force_act = body.get('force_action', None)
            persona_key = str(body.get('persona_key') or 'wild')
            if focus_c is not None:
                focus_c = max(0, min(4, int(focus_c)))
            if force_act is not None:
                force_act = max(0, min(4, int(force_act)))
            with st.lock:
                sess = st.session
                prep = sess.prepare_encounter_spec(
                    focus_concept_id=focus_c,
                    force_action=force_act,
                    persona_key=persona_key,
                )
            llm_bundle = sess.generate_encounter_llm_bundle(prep)
            with st.lock:
                enc = sess.finalize_encounter(prep, llm_bundle)
                snap = sess.get_status_snapshot()
            lm = enc.get('llm_meta') or {}
            llm_tag = f" | LLM={lm.get('reasoner_model','Bank')}+{lm.get('stylizer_model','Tmpl')} (hit={lm.get('cache_hit')})"
            print(
                f"  [RL TURN #{enc['turn']:02d}] Double-DQN -> [{enc['rl_action']}] {enc['rl_action_name']} "
                f"| Target: C{enc['target_concept_id']} ({enc['target_concept_short']}) "
                f"| Persona={persona_key} | {enc['rl_latency_ms']:.2f}ms{llm_tag}",
                flush=True,
            )
            self._send_json({'encounter': enc, 'status': snap})
            return

        if route == '/api/encounter/submit':
            user_resp = str(body.get('user_response', ''))
            resp_sec = float(body.get('response_sec', 10.0))
            retries = max(1, int(body.get('retries', 1)))
            hint_override = bool(body.get('hint_used', False))
            with st.lock:
                enc = st.session.active_encounter or body.get('encounter')
                if not enc:
                    enc = st.session.plan_next_encounter()
                rec = st.session.submit_turn_response(
                    encounter=enc,
                    user_response=user_resp,
                    response_sec=resp_sec,
                    retries=retries,
                    hint_used_override=hint_override,
                )
                snap = st.session.get_status_snapshot()
            badge_str = " [* BADGE UNLOCKED!]" if rec['badge_just_earned'] else ""
            print(
                f"  [GRADED #{rec['turn']:02d}] {rec['grader']['graded_by']} -> {rec['grader']['verdict'].upper()} "
                f"({rec['grader']['semantic_score']*100:.1f}%) | "
                f"C{rec['target_concept_id']} Mastery: {rec['mastery_before'][rec['target_concept_id']]*100:.1f}% -> "
                f"{rec['mastery_after'][rec['target_concept_id']]*100:.1f}% | "
                f"AI={rec['latencies_ms']['total_ai_turn_ms']:.1f}ms{badge_str}"
            )
            self._send_json({'result': rec, 'status': snap})
            return

        if route == '/api/vision/evaluate':
            puzzle_id = str(body.get('puzzle_id', 'c0_p1'))
            click_x = body.get('click_x', None)
            click_y = body.get('click_y', None)
            user_box = body.get('user_box', None)
            resp_sec = float(body.get('response_sec', 8.0))
            with st.lock:
                rec = st.session.submit_diagram_interaction(
                    puzzle_id=puzzle_id,
                    click_x=click_x,
                    click_y=click_y,
                    user_box=user_box,
                    response_sec=resp_sec,
                )
                snap = st.session.get_status_snapshot()
            cid = rec['concept_id']
            print(
                f"  [VISION #{rec['turn']:02d}] {rec['graded_by']} -> {rec['verdict'].upper()} "
                f"(IoU={rec['iou']:.2f}, Conf={rec['dino_confidence']:.2f}) | "
                f"Target: [{rec['target_label']}] | "
                f"C{cid} Mastery: {rec['mastery_before'][cid]*100:.1f}% -> {rec['mastery_after'][cid]*100:.1f}%"
            )
            self._send_json({'result': rec, 'status': snap})
            return

        if route == '/api/vision/vqa':
            cid = int(body.get('concept_id', 0))
            student_ans = str(body.get('student_answer', ''))
            resp_sec = float(body.get('response_sec', 11.0))
            with st.lock:
                rec = st.session.submit_vlm_vqa_answer(
                    concept_id=cid,
                    student_answer=student_ans,
                    response_sec=resp_sec,
                )
                snap = st.session.get_status_snapshot()
            print(
                f"  [VLM->DeBERTa VQA C{cid}] Verdict={rec['verdict'].upper()} "
                f"(Score={rec['score']*100:.1f}%) | "
                f"C{cid} Mastery: {rec['mastery_before'][cid]*100:.1f}% -> {rec['mastery_after'][cid]*100:.1f}% "
                f"({rec['grader_latency_ms']:.1f}ms)"
            )
            self._send_json({'result': rec, 'status': snap})
            return

        if route == '/api/npc/chat':
            npc_key = str(body.get('npc_key') or 'prof_ada')
            student_q = str(body.get('question') or '').strip()
            if not student_q:
                self._send_json({'error': 'Question cannot be empty'}, status=400)
                return
            with st.lock:
                sess = st.session
                ctx = sess.prepare_npc_chat_ctx(npc_key=npc_key, student_question=student_q)
            res = sess.run_npc_chat_from_ctx(ctx)
            with st.lock:
                sess.record_npc_chat_turn(ctx['npc_key'], ctx['question'], res['reply'])
            print(
                f"  [NPC CHAT] {npc_key} ({res.get('model_name')}) | Q={student_q[:45]!r} "
                f"| R={res.get('reasoner_ms',0):.0f}ms + S={res.get('latency_ms',0):.0f}ms",
                flush=True,
            )
            self._send_json(res)
            return

        if route == '/api/course/ingest':
            course_title = str(body.get('course_title', '')).strip()
            raw_notes = str(body.get('raw_notes', '')).strip()
            image_base64 = body.get('image_base64', None)
            file_name = body.get('file_name', None)
            file_base64 = body.get('file_base64', None)
            remote_vlm_url = body.get('remote_vlm_url', None)
            profile = str(body.get('profile', 'average'))

            job_id = uuid.uuid4().hex[:12]
            st.ingest_jobs[job_id] = {
                'job_id': job_id,
                'job_status': 'running',
                'progress': 5,
                'step': f"Received '{file_name or course_title}'. Starting VLM + Document Extraction...",
            }

            def _run_ingest_worker():
                def _on_prog(pct, msg):
                    st.ingest_jobs[job_id]['progress'] = int(pct)
                    st.ingest_jobs[job_id]['step'] = str(msg)

                try:
                    with st.lock:
                        st.session, pkg = st.engine.ingest_new_course(
                            course_title=course_title,
                            raw_notes=raw_notes,
                            image_base64=image_base64,
                            file_name=file_name,
                            file_base64=file_base64,
                            remote_vlm_url=remote_vlm_url,
                            profile=profile,
                            progress_cb=_on_prog,
                        )
                        snap = st.session.get_status_snapshot()
                        puzzles = st.session.get_diagram_puzzles()

                    print(
                        f"  [NEW COURSE INGESTED] '{pkg['title']}' (`{pkg['course_id']}`) | "
                        f"Source='{pkg.get('source_file', 'notes')}' | "
                        f"{pkg.get('num_photos_extracted', 0)} Extracted Photos + {len(puzzles)} VLM/DINO Puzzles!",
                        flush=True,
                    )
                    st.ingest_jobs[job_id] = {
                        'job_id': job_id,
                        'job_status': 'done',
                        'progress': 100,
                        'step': (
                            f"Ready! Extracted {pkg.get('num_photos_extracted', 0)} photos & "
                            f"{len(puzzles)} VLM/Grounding-DINO puzzles from '{pkg['title']}'."
                        ),
                        'status': snap,
                        'puzzles': puzzles,
                        'course_id': pkg['course_id'],
                        'course_title': pkg['title'],
                        'num_photos_extracted': pkg.get('num_photos_extracted', 0),
                        'source_file': pkg.get('source_file', ''),
                    }
                except Exception as exc:
                    print(f"  [INGEST ERROR] job={job_id}: {exc}", flush=True)
                    st.ingest_jobs[job_id] = {
                        'job_id': job_id,
                        'job_status': 'error',
                        'progress': 0,
                        'step': f'Ingestion failed: {exc}',
                        'error': str(exc),
                    }

            threading.Thread(target=_run_ingest_worker, daemon=True).start()
            self._send_json(st.ingest_jobs[job_id], status=202)
            return

        if route.startswith('/api/oral_coach/'):
            if st.oral_coach is None:
                self._send_json({'error': 'OralCoachService unavailable'}, status=503)
                return
            if route == '/api/oral_coach/start':
                out = st.oral_coach.start_session(
                    whisper_model=body.get('whisper_model') or None,
                    language=body.get('language') or None,
                    course_title=str(body.get('course_title') or ''),
                    tip_writer_mode=str(body.get('tip_writer_mode') or 'auto'),
                )
                print(
                    f"  [ORAL COACH] Session started: {out.get('session_id')} | "
                    f"Course={out.get('course_title', '')!r}",
                    flush=True,
                )
                self._send_json(out)
                return
            if route == '/api/oral_coach/ingest':
                try:
                    since_seq = int(body.get('since_seq') or 0)
                except (TypeError, ValueError):
                    since_seq = 0
                try:
                    sample_rate = int(body.get('sample_rate') or 16000)
                except (TypeError, ValueError):
                    sample_rate = 16000
                out = st.oral_coach.ingest(
                    frame_b64=body.get('frame_b64') or None,
                    pcm16_b64=body.get('pcm16_b64') or None,
                    sample_rate=sample_rate,
                    transcript=body.get('transcript') or None,
                    since_seq=since_seq,
                )
                self._send_json(out)
                return
            if route == '/api/oral_coach/mute':
                try:
                    seconds = float(body.get('seconds', 3.0))
                except (TypeError, ValueError):
                    seconds = 3.0
                self._send_json(st.oral_coach.mute(seconds))
                return
            if route == '/api/oral_coach/stop':
                out = st.oral_coach.stop_session()
                print(f"  [ORAL COACH] Session stopped: {out.get('summary')}", flush=True)
                self._send_json(out)
                return
            if route == '/api/oral_coach/launch_desktop':
                is_loopback = self.client_address[0] in ('127.0.0.1', '::1')
                is_tunneled = bool(self.headers.get('CF-Connecting-IP') or self.headers.get('X-Forwarded-For'))
                if not is_loopback or is_tunneled:
                    self._send_json(
                        {'launched': False, 'error': 'Desktop window is only available on localhost.'},
                        status=403,
                    )
                    return
                out = st.oral_coach.launch_desktop(
                    whisper_model=body.get('whisper_model') or None,
                    speak=bool(body.get('speak', False)),
                )
                self._send_json(out)
                return

        self._send_json({'error': f'Unknown API route: {route}'}, status=404)


def main():
    parser = argparse.ArgumentParser(description='KnowledgeQuest 5-Model AI + RPG Web Server')
    parser.add_argument('--host', default='127.0.0.1', help='Bind host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000, help='Port (default: 8000)')
    parser.add_argument('--course', default='vision_yolo_ocr', help='Initial course ID')
    parser.add_argument('--profile', default='average', choices=['struggling', 'average', 'advanced'])
    parser.add_argument('--live-vlm', action='store_true', help='Load live 8B/7B VLM on GPU (for Kaggle 2x T4)')
    parser.add_argument('--vlm-model', default='Qwen/Qwen3-VL-8B-Instruct', help='Hugging Face VLM model ID')
    parser.add_argument('--live-llm', action='store_true', help='Load live Stage 4 Reasoner (7B) + Stylizer (8B) LLMs on GPU')
    parser.add_argument('--reasoner-model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', help='Model #6 Pedagogical Reasoner ID')
    parser.add_argument('--stylizer-model', default='NousResearch/Meta-Llama-3.1-8B-Instruct', help='Model #7 Persona Stylizer ID')
    parser.add_argument('--open', action='store_true', help='Open browser automatically')
    args = parser.parse_args()

    print("=" * 88)
    print("  KNOWLEDGEQUEST 7-MODEL AI + BROWSER RPG SERVER")
    print("=" * 88)
    print("  Loading Final-DKT-GRU + Double-DQN + DeBERTa-v3 + Grounding-DINO + Qwen-VL + Stage 4 LLMs...")
    state = GameServerState(
        default_course=args.course,
        default_profile=args.profile,
        vlm_mode='live' if args.live_vlm else 'hybrid',
        vlm_model_id=args.vlm_model,
        llm_mode='live' if args.live_llm else 'fallback',
        reasoner_model_id=args.reasoner_model,
        stylizer_model_id=args.stylizer_model,
    )
    KnowledgeQuestHTTPHandler.server_state = state

    eng = state.engine
    llm_r = eng.llm_info['reasoner']
    llm_s = eng.llm_info['stylizer']
    print(f"  [Model #1] {eng.dkt_info['name']:<20} | AUC={eng.dkt_info['auc']:.4f}")
    print(f"  [Model #2] {eng.rl_info['name']:<20} | Reward={eng.rl_info['reward']:.2f}")
    print(f"  [Model #3] {eng.grader_info['name']:<20} | 3-Way Acc={eng.grader_info['acc_3way']*100:.2f}% ({eng.grader_info['device']})")
    print(f"  [Model #4] {eng.dino_info['name']:<20} | Mean Box IoU={eng.dino_info['mean_box_iou']:.3f} | {eng.dino_info['puzzles']} Lab Diagrams ({eng.dino_info['device']})")
    print(f"  [Model #5] {eng.vlm_info['name']:<20} | Mode={eng.vlm_info['mode'].upper()} | {eng.vlm_info['vqa_challenges']} VQA Rubrics")
    print(f"  [Model #6] {llm_r['name']:<20} | Live={llm_r['live']}")
    print(f"  [Model #7] {llm_s['name']:<20} | Live={llm_s['live']} | QLoRA={llm_s['qlora_active']}")
    print(f"  Active Course : {state.session.course_title} (`{state.session.course_id}`)")
    print(f"  Boot Time     : {eng.boot_time_sec:.2f}s (100% Local, Zero Webcam)")
    print("-" * 88)

    url = f"http://{args.host}:{args.port}"
    httpd = ThreadingHTTPServer((args.host, args.port), KnowledgeQuestHTTPHandler)
    print(f"  RPG Live URL  : {url}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 88)

    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down KnowledgeQuest server. Goodbye!")
        httpd.server_close()


if __name__ == '__main__':
    main()
