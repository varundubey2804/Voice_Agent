"""
app.py — Veena AI  (v4 — Rock-Solid 2-Way Conversation)
═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURE
────────────
Previous versions used simple bool flags and a flat record-loop, which
caused race conditions and false barge-in because the mic never stopped
recording while Veena was speaking.

v4 uses an explicit STATE MACHINE with four states:

    IDLE  →  USER_SPEAKING  →  AI_THINKING  →  AI_SPEAKING  →  IDLE
             (mic on)          (mic off)        (mic off)

Transitions are guarded by a threading.Lock so only one thread can
change state at a time.  The audio producer checks state BEFORE queueing
a chunk; if state is not USER_SPEAKING it drains and discards the mic
buffer (hardware mute equivalent).

SPEECH-TO-TEXT (STT)
────────────────────
  • faster-whisper 'small' model, int8, CPU
  • Silence detection (RMS-based, per-frame voting) skips empty chunks
  • Emotion detection from prosody (speed + amplitude)
  • Minimum transcript length guard (≥ 3 chars) suppresses Whisper
    hallucinations on noise ("Hmm", ".", etc.)

TEXT-TO-SPEECH (TTS)
────────────────────
  • edge-tts sentence-streaming pipeline (see voice_service.py)
  • Dedicated background event loop — no asyncio.run() from executor
  • Uses pygame.mixer.Sound (not Music channel) — no single-channel conflict
  • 500 ms post-TTS grace + queue flush before re-enabling mic

FALSE BARGE-IN FIX
──────────────────
  The state machine is the primary fix.  Additionally:
  • Audio producer reads and discards mic frames while state ≠ IDLE
    (drains PyAudio's internal ring buffer so no stale audio accumulates)
  • After TTS ends, 500 ms silence + full queue flush before IDLE

CLEAN SHUTDOWN
──────────────
  Ctrl-C sets _shutdown Event → producer + consumer threads exit their
  loops → vs.shutdown() stops TTS and shuts down its event loop → no
  'cannot schedule new futures after shutdown' crash.
"""

from __future__ import annotations

import asyncio
import enum
import http.server
import io
import json
import os
import queue
import socketserver
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudio
import soundfile as sf
import websockets
from faster_whisper import WhisperModel
from websockets.server import serve

import finance_tools
import voice_service as vs
from agentic_rag import build_agent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

WHISPER_MODEL_SIZE   = "small"
CHUNK_SECONDS        = 7         # audio chunk length fed to Whisper
SILENCE_THRESHOLD    = 800        # RMS below this → silent frame
SILENCE_RATIO        = 0.85       # ≥85% silent frames → discard chunk
POST_TTS_GRACE_SEC   = 0.6        # seconds to wait after TTS before re-enabling mic
MIN_TRANSCRIPT_CHARS = 3          # shorter = Whisper noise artefact → discard
HTTP_PORT            = 8080
WS_PORT              = 8765
EXECUTOR             = ThreadPoolExecutor(max_workers=6)


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION STATE MACHINE
# ══════════════════════════════════════════════════════════════════════════════

class State(enum.Enum):
    IDLE         = "idle"          # waiting for user to speak
    AI_THINKING  = "ai_thinking"   # LLM generating response  (mic OFF)
    AI_SPEAKING  = "ai_speaking"   # TTS playing              (mic OFF)

class ConversationFSM:
    """
    Thread-safe finite state machine for the conversation turn.

    Only IDLE → AI_THINKING → AI_SPEAKING → IDLE is valid.
    Any attempt to transition from a wrong state is a no-op and logged.
    """
    def __init__(self) -> None:
        self._state = State.IDLE
        self._lock  = threading.Lock()

    @property
    def state(self) -> State:
        return self._state

    def mic_active(self) -> bool:
        """True only when it is safe to accept microphone input."""
        return self._state == State.IDLE

    def transition(self, new_state: State) -> bool:
        """Attempt state transition. Returns True on success."""
        _VALID: dict[State, set[State]] = {
            State.IDLE:        {State.AI_THINKING},
            State.AI_THINKING: {State.AI_SPEAKING, State.IDLE},   # IDLE on error
            State.AI_SPEAKING: {State.IDLE},
        }
        with self._lock:
            if new_state in _VALID.get(self._state, set()):
                prev = self._state
                self._state = new_state
                print(f"🔄 State: {prev.value} → {new_state.value}")
                return True
            print(f"⚠️  Invalid transition {self._state.value} → {new_state.value} (ignored)")
            return False

fsm = ConversationFSM()


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════

agent             = build_agent()
connected_clients: set = set()
main_loop: asyncio.AbstractEventLoop | None = None
audio_queue: queue.Queue = queue.Queue(maxsize=8)
_shutdown         = threading.Event()   # set on Ctrl-C


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO / STT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _rms(data: np.ndarray) -> float:
    if data.ndim > 1:
        data = data[:, 0]
    return float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))


def _is_silence(data: np.ndarray) -> bool:
    if data.ndim > 1:
        data = data[:, 0]
    fsize  = 1024
    n      = len(data) // fsize
    silent = sum(1 for i in range(n) if _rms(data[i*fsize:(i+1)*fsize]) < SILENCE_THRESHOLD)
    return (silent / n) >= SILENCE_RATIO if n > 0 else True


def _record_chunk(stream: pyaudio.Stream) -> io.BytesIO | None:
    """Record CHUNK_SECONDS of audio. Returns BytesIO WAV or None if silent."""
    n      = int(16000 / 1024 * CHUNK_SECONDS)
    frames = [stream.read(1024, exception_on_overflow=False) for _ in range(n)]
    arr    = np.frombuffer(b"".join(frames), dtype=np.int16)
    if _is_silence(arr):
        return None
    buf = io.BytesIO()
    sf.write(buf, arr, samplerate=16000, format="WAV")
    buf.seek(0)
    return buf


def _detect_emotion(wav: io.BytesIO, text: str) -> str:
    wav.seek(0)
    data, sr = sf.read(wav)
    wav.seek(0)
    if data.ndim > 1:
        data = data[:, 0]
    dur   = len(data) / sr
    wps   = len(text.split()) / dur if dur > 0 else 0
    amp   = float(np.max(np.abs(data))) if len(data) else 0.0
    if wps > 3 or amp > 0.8:
        return "stressed"
    if wps < 1.5 and amp < 0.3:
        return "calm"
    if any(w in text.lower() for w in ("confused", "how", "what", "?")):
        return "confused"
    return "neutral"


def _transcribe(model: WhisperModel, wav: io.BytesIO) -> tuple[str, str, str]:
    """Returns (text, language, emotion)."""
    segments, info = model.transcribe(wav, beam_size=3)
    text           = " ".join(s.text for s in segments).strip()
    emotion        = _detect_emotion(wav, text) if text else "neutral"
    return text, info.language, emotion


def _flush_queue() -> None:
    """Discard all buffered audio chunks (echo prevention after TTS)."""
    n = 0
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
            n += 1
        except queue.Empty:
            break
    if n:
        print(f"🧹 Flushed {n} stale audio chunk(s)")


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET
# ══════════════════════════════════════════════════════════════════════════════

async def _broadcast(msg: dict) -> None:
    if connected_clients:
        data = json.dumps(msg)
        await asyncio.gather(
            *[c.send(data) for c in connected_clients],
            return_exceptions=True,
        )


async def handle_websocket(websocket, path):
    connected_clients.add(websocket)
    print(f"👤 Client connected. Total: {len(connected_clients)}")
    try:
        await websocket.send(json.dumps({
            "type": "connection_status", "status": "connected",
            "message": "Connected to Veena AI",
        }))
        async for raw in websocket:
            try:
                await _handle_ws_message(json.loads(raw), websocket)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"👤 Client disconnected. Total: {len(connected_clients)}")


async def _handle_ws_message(data: dict, ws) -> None:
    t = data.get("type")
    if t == "get_dashboard_data":
        loop = asyncio.get_running_loop()
        dash = await loop.run_in_executor(EXECUTOR, finance_tools.get_dashboard_data)
        await ws.send(json.dumps({"type": "dashboard_data", "data": dash}))

    elif t == "text_input":
        text = data.get("text", "").strip()
        if text:
            await _run_turn(
                text,
                language=data.get("language", "en"),
                emotion=data.get("emotion", "neutral"),
            )

    elif t in ("start_listening", "stop_listening"):
        status = "listening_started" if t == "start_listening" else "listening_stopped"
        await ws.send(json.dumps({"type": status}))


# ══════════════════════════════════════════════════════════════════════════════
# CORE CONVERSATION TURN
# ══════════════════════════════════════════════════════════════════════════════

_FALLBACK = {
    "hi": "मुझे खेद है, मैं अभी आपकी बात ठीक से समझ नहीं पाई। क्या आप अपना सवाल दोबारा पूछ सकते हैं?",
    "en": "I'm sorry, I couldn't process that. Could you please rephrase?",
}


async def _run_turn(user_text: str, language: str = "en", emotion: str = "neutral") -> None:
    """
    Single conversation turn:  user_text → LLM → TTS

    State machine guards prevent overlapping turns.
    """
    # ── Guard: only start a turn when idle ────────────────────────────────────
    if not fsm.transition(State.AI_THINKING):
        print(f"⏭  Busy — dropped: {user_text[:50]}")
        return

    print(f"🗣  Customer ({language}/{emotion}): {user_text}")

    await _broadcast({
        "type": "user_message", "text": user_text,
        "emotion_detected": emotion,
        "timestamp": datetime.now().isoformat(),
    })

    # ── LLM call ──────────────────────────────────────────────────────────────
    try:
        loop         = asyncio.get_running_loop()
        tagged_input = f"[{emotion}] {user_text}" if emotion != "neutral" else user_text

        def _invoke() -> str:
            try:
                res = agent.invoke({"input": tagged_input, "language": language})
                out = res.get("output", "").strip()
                return out if out and "agent stopped" not in out.lower() \
                       else _FALLBACK.get(language, _FALLBACK["en"])
            except Exception as e:
                print(f"❌ Agent error: {e}")
                return _FALLBACK.get(language, _FALLBACK["en"])

        response = await loop.run_in_executor(EXECUTOR, _invoke)
        print(f"🤖 Veena: {response}")

    except Exception as e:
        print(f"❌ Turn error: {e}")
        fsm.transition(State.IDLE)
        await _broadcast({"type": "error", "message": "Error generating response"})
        return

    # ── Transition to speaking ─────────────────────────────────────────────────
    if not fsm.transition(State.AI_SPEAKING):
        # Should never happen (THINKING → SPEAKING is always valid)
        fsm.transition(State.IDLE)
        return

    await _broadcast({
        "type": "agent_response", "text": response,
        "timestamp": datetime.now().isoformat(),
    })
    await _broadcast({"type": "speaking_started"})

    # ── TTS (non-blocking — runs in executor, state machine stays correct) ─────
    asyncio.create_task(_play_and_finish(response, language))


async def _play_and_finish(text: str, language: str) -> None:
    """Play TTS, then transition back to IDLE and re-enable mic."""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(EXECUTOR, vs.play_tts, text, language)
    except Exception as e:
        print(f"❌ TTS task error: {e}")
    finally:
        # Grace period: let last audio decay in the room
        await asyncio.sleep(POST_TTS_GRACE_SEC)
        _flush_queue()              # discard any mic echo captured during TTS
        fsm.transition(State.IDLE)  # ← mic re-enabled HERE
        await _broadcast({"type": "speaking_finished"})
        print("🎤 Mic re-enabled — ready for user input")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO PRODUCER (mic recording)
# ══════════════════════════════════════════════════════════════════════════════

def _audio_producer() -> None:
    """
    Continuously records audio from the microphone.

    KEY DESIGN:
    While the FSM state is not IDLE the producer reads mic frames and
    DISCARDS them.  This drains PyAudio's internal ring buffer so stale
    audio never accumulates and cannot be transcribed as user speech.

    This is the hardware-level fix for false barge-in.
    """
    pa     = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16, channels=1, rate=16000,
        input=True, frames_per_buffer=1024,
    )
    print("🎙  Audio producer started")
    try:
        while not _shutdown.is_set():
            # ── MIC MUTE: drain when AI is active ─────────────────────────────
            if not fsm.mic_active():
                stream.read(1024, exception_on_overflow=False)
                continue

            # ── Record a chunk ─────────────────────────────────────────────────
            buf = _record_chunk(stream)
            if buf is None:
                continue

            if audio_queue.full():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
            audio_queue.put(buf)

    except Exception as e:
        if not _shutdown.is_set():
            print(f"❌ Producer error: {e}")
    finally:
        try:
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception:
            pass
        print("🎙  Audio producer stopped")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO CONSUMER (STT + dispatch)
# ══════════════════════════════════════════════════════════════════════════════

def _audio_consumer() -> None:
    """
    Pulls audio chunks from the queue, transcribes them with Whisper,
    and dispatches complete utterances to the async conversation pipeline.

    Gating (safety net — producer should already have muted the mic):
    ┌─────────────────────┬──────────────────────────────────────────────┐
    │  FSM state          │  Action                                      │
    ├─────────────────────┼──────────────────────────────────────────────┤
    │  IDLE               │  Transcribe → dispatch                       │
    │  AI_THINKING        │  Discard chunk silently                      │
    │  AI_SPEAKING        │  Discard chunk silently                      │
    └─────────────────────┴──────────────────────────────────────────────┘
    """
    model = _load_whisper()
    print("🎧  Audio consumer (STT) started")

    while not _shutdown.is_set():
        try:
            buf = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Safety net: skip if not idle
        if not fsm.mic_active():
            continue

        text, lang, emotion = _transcribe(model, buf)

        # Filter Whisper noise artefacts
        if not text or len(text.strip()) < MIN_TRANSCRIPT_CHARS:
            continue

        # Dispatch to async pipeline
        if main_loop and not main_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                _run_turn(text, lang, emotion),
                main_loop,
            )

    print("🎧  Audio consumer stopped")


def _load_whisper() -> WhisperModel:
    print(f"🔍 Loading Whisper '{WHISPER_MODEL_SIZE}' (CPU, int8) …")
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        num_workers=2,
        cpu_threads=4,
    )


def _audio_loop() -> None:
    prod = threading.Thread(target=_audio_producer, daemon=True, name="AudioProducer")
    cons = threading.Thread(target=_audio_consumer, daemon=True, name="AudioConsumer")
    prod.start()
    cons.start()
    prod.join()
    cons.join()


# ══════════════════════════════════════════════════════════════════════════════
# HTTP SERVER
# ══════════════════════════════════════════════════════════════════════════════

def _start_http() -> None:
    frontend = Path(__file__).parent / "frontend"
    if not (frontend / "index.html").exists():
        print(f"❌ index.html not found at {frontend}")
        return

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(frontend), **kw)
        def log_message(self, *_): pass
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

    try:
        with socketserver.TCPServer(("", HTTP_PORT), _Handler) as httpd:
            print(f"🌍 HTTP  →  http://localhost:{HTTP_PORT}")
            threading.Thread(
                target=lambda: (time.sleep(2),
                                webbrowser.open(f"http://localhost:{HTTP_PORT}/index.html")),
                daemon=True,
            ).start()
            httpd.serve_forever()
    except OSError as e:
        print(f"❌ HTTP error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET SERVER
# ══════════════════════════════════════════════════════════════════════════════

async def _start_ws() -> None:
    global main_loop
    main_loop = asyncio.get_running_loop()
    print(f"🌐 WebSocket  →  ws://localhost:{WS_PORT}")
    async with serve(
        handle_websocket, "localhost", WS_PORT,
        ping_interval=20, ping_timeout=10,
    ):
        await asyncio.Future()   # run forever


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("🎯  VEENA AI  —  Voice Financial Advisor  (v4)")
    print("=" * 70)

    threading.Thread(target=_start_http,  daemon=True, name="HTTP").start()
    time.sleep(0.8)

    threading.Thread(target=_audio_loop, daemon=True, name="AudioLoop").start()
    time.sleep(0.8)

    print()
    print(f"🚀  Opening browser …  http://localhost:{HTTP_PORT}/index.html")
    print()
    print("✅  All systems ready!")
    print(f"🌐  Web       :  http://localhost:{HTTP_PORT}/index.html")
    print(f"🔌  WebSocket :  ws://localhost:{WS_PORT}")
    print(f"🎙️   Voice     :  Active  (chunk={CHUNK_SECONDS}s, model={WHISPER_MODEL_SIZE})")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)

    try:
        asyncio.run(_start_ws())
    except KeyboardInterrupt:
        print("\n🛑  Shutting down Veena AI …")
        _shutdown.set()
        vs.shutdown()       # graceful TTS + event-loop teardown
        time.sleep(0.8)
        print("👋  Goodbye!")


if __name__ == "__main__":
    main()
