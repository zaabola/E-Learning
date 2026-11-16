"""
KnowledgeQuest Live 3-Model Game & AI Server (`server.py`)
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
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(REPO_ROOT, 'Models')
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from engine.knowledgequest_engine import KnowledgeQuestAIEngine

mimetypes.add_type('audio/wav', '.wav')
mimetypes.add_type('font/ttf', '.ttf')
mimetypes.add_type('application/javascript', '.js')


class GameServerState:
    def __init__(self, default_course: str = 'vision_yolo_ocr', default_profile: str = 'average'):
        self.lock = threading.Lock()
        self.engine = KnowledgeQuestAIEngine()
        self.session = self.engine.create_session(profile=default_profile, course_id=default_course)

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
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
        route = urlparse(self.path).path
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
            if focus_c is not None:
                focus_c = max(0, min(4, int(focus_c)))
            if force_act is not None:
                force_act = max(0, min(4, int(force_act)))
            with st.lock:
                enc = st.session.plan_next_encounter(
                    focus_concept_id=focus_c,
                    force_action=force_act,
                )
                snap = st.session.get_status_snapshot()
            print(
                f"  [RL TURN #{enc['turn']:02d}] Double-DQN -> [{enc['rl_action']}] {enc['rl_action_name']} "
                f"| Target: C{enc['target_concept_id']} ({enc['target_concept_short']}) "
                f"| {enc['rl_latency_ms']:.2f}ms"
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

        self._send_json({'error': f'Unknown API route: {route}'}, status=404)


def main():
    parser = argparse.ArgumentParser(description='KnowledgeQuest 4-Model AI + RPG Web Server')
    parser.add_argument('--host', default='127.0.0.1', help='Bind host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000, help='Port (default: 8000)')
    parser.add_argument('--course', default='vision_yolo_ocr', help='Initial course ID')
    parser.add_argument('--profile', default='average', choices=['struggling', 'average', 'advanced'])
    parser.add_argument('--open', action='store_true', help='Open browser automatically')
    args = parser.parse_args()

    print("=" * 88)
    print("  KNOWLEDGEQUEST 4-MODEL AI + BROWSER RPG SERVER")
    print("=" * 88)
    print("  Loading Final-DKT-GRU + Double-DQN + DeBERTa-v3-Small + Grounding-DINO-Tiny...")
    state = GameServerState(default_course=args.course, default_profile=args.profile)
    KnowledgeQuestHTTPHandler.server_state = state

    eng = state.engine
    print(f"  [Model #1] {eng.dkt_info['name']:<20} | AUC={eng.dkt_info['auc']:.4f}")
    print(f"  [Model #2] {eng.rl_info['name']:<20} | Reward={eng.rl_info['reward']:.2f}")
    print(f"  [Model #3] {eng.grader_info['name']:<20} | 3-Way Acc={eng.grader_info['acc_3way']*100:.2f}% ({eng.grader_info['device']})")
    print(f"  [Model #4] {eng.dino_info['name']:<20} | Mean Box IoU={eng.dino_info['mean_box_iou']:.3f} | {eng.dino_info['puzzles']} Lab Diagrams ({eng.dino_info['device']})")
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
