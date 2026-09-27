"""
emotion_trigger_server.py

Lightweight HTTP server (stdlib only — no FastAPI/uvicorn dependency) that
runs inside the voice_engine.py process. Exposes POST /emotion-trigger for
the hardware server to call whenever it detects a sustained emotion.

Started automatically by voice_engine.run() — no separate process, no extra
dependencies to install.

COOLDOWN BEHAVIOR
─────────────────
After a trigger is successfully processed (on_trigger() runs), the server
closes its "listening window" for a random interval between
COOLDOWN_MIN_S and COOLDOWN_MAX_S seconds. Any /emotion-trigger POSTs that
arrive during that window are accepted (200 OK) but NOT processed — they're
reported back as {"status": "cooling_down"} so the sender knows why nothing
happened. Once the window elapses, the very next POST is processed normally
and a new random cooldown window starts.

This is a *server-side* gate, independent of and in addition to any
rate-limiting the sender (emotion_sender.py) already does.
"""

import json
import logging
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("voice.emotion_trigger_server")

# Random cooldown window bounds (seconds) after each processed trigger.
COOLDOWN_MIN_S = 30.0
COOLDOWN_MAX_S = 45.0


def _make_handler(on_trigger):
    # Shared mutable state across requests (the handler class is
    # instantiated fresh per-request by BaseHTTPRequestHandler, so this
    # state has to live outside it).
    state = {
        "cooldown_until": 0.0,  # monotonic timestamp; 0 = not cooling down
    }
    state_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep the terminal clean; use logger for real diagnostics

        def _reply(self, status_code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            try:
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # The sender (camera-feed pipeline or hardware mic side)
                # already closed its connection — e.g. it hit its own
                # request timeout — before we finished writing the
                # response. The trigger itself already ran (on_trigger
                # was already called by the time _reply is reached), so
                # nothing is lost; there's just no one left to hand the
                # ack to. Safe to drop silently rather than let it bubble
                # up as an unhandled exception in socketserver's request
                # thread.
                logger.info("client disconnected before response was sent — dropping ack")

        def do_POST(self):
            if self.path != "/emotion-trigger":
                self.send_response(404)
                self.end_headers()
                return

            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                payload = json.loads(raw or b"{}")
                emotion = payload.get("emotion")
            except Exception as e:
                logger.warning("bad payload on /emotion-trigger: %s", e)
                self.send_response(400)
                self.end_headers()
                return

            if not emotion:
                self.send_response(400)
                self.end_headers()
                return

            now = time.monotonic()

            # --- Cooldown gate: check first, before touching on_trigger ---
            with state_lock:
                if now < state["cooldown_until"]:
                    remaining = round(state["cooldown_until"] - now, 1)
                    logger.info(
                        "emotion '%s' ignored — cooling down for %.1fs more",
                        emotion, remaining,
                    )
                    self._reply(200, {
                        "status": "cooling_down",
                        "retry_after_s": remaining,
                    })
                    return

            # --- Outside cooldown: process this one, then re-arm cooldown ---
            try:
                spoken = on_trigger(emotion)
            except Exception as e:
                logger.error("on_trigger callback failed: %s", e)
                spoken = False

            with state_lock:
                window = random.uniform(COOLDOWN_MIN_S, COOLDOWN_MAX_S)
                state["cooldown_until"] = time.monotonic() + window
            logger.info(
                "emotion '%s' processed (spoken=%s) — next listen window opens in %.1fs",
                emotion, spoken, window,
            )

            self._reply(200, {"status": "spoken" if spoken else "ignored"})

    return Handler


def start_emotion_trigger_server(on_trigger, host: str = "0.0.0.0", port: int = 5000) -> ThreadingHTTPServer:
    """
    on_trigger: callable(emotion: str) -> bool
        Called synchronously for each incoming trigger that lands outside
        the cooldown window. Return True if Veena spoke, False if the
        trigger was ignored (e.g. mid-conversation).

    Runs the HTTP server in a daemon thread and returns immediately — the
    voice engine's mic loop is completely unaffected. Returns the server
    object so callers can call .shutdown() on exit if desired.

    After each processed trigger, incoming POSTs are accepted but skipped
    for a random COOLDOWN_MIN_S–COOLDOWN_MAX_S second window before the
    server "re-opens" for the next emotion change.
    """
    handler_cls = _make_handler(on_trigger)
    server = ThreadingHTTPServer((host, port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"📡  Emotion trigger server listening on {host}:{port}/emotion-trigger")
    return server