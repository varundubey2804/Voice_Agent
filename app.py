"""
app.py — Optimized Veena AI (Mental Wellness Companion)
Key improvements:
  - Reduced chunk length (4s) for faster response
  - Smarter silence detection with RMS energy
  - Audio queue for parallel record→transcribe pipeline
  - Whisper runs in a thread pool (non-blocking)
  - TTS streaming via edge_tts (first-chunk play)
  - WebSocket ping/pong keepalive
"""

import os
import io
import json
import asyncio
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from agentic_rag import build_agent
import voice_service as vs
import soundfile as sf
import websockets
from websockets.server import serve
import threading
import queue
from datetime import datetime
from pathlib import Path
import webbrowser
import http.server
import socketserver
import time
import wellness_tools
from concurrent.futures import ThreadPoolExecutor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Tunable constants ─────────────────────────────────────────────────────────
DEFAULT_MODEL_SIZE   = "small"          # ✅ Was "medium" — 3-4x faster, ~same quality
DEFAULT_CHUNK_LENGTH = 4                # ✅ Was 8s — cuts first-response latency in half
SILENCE_THRESHOLD    = 800             # RMS threshold (was simple peak; more accurate)
SILENCE_RATIO        = 0.85            # If 85%+ frames are silent, skip chunk
HTTP_PORT            = 8080
WS_PORT              = 8765
EXECUTOR             = ThreadPoolExecutor(max_workers=4)  # Shared thread pool

# ── State ─────────────────────────────────────────────────────────────────────
agent            = build_agent()
whisper_model    = None
audio_device     = None
mic_stream       = None
connected_clients= set()
is_speaking      = False
is_thinking      = False
current_ai_text  = ""
main_loop        = None
audio_queue      = queue.Queue(maxsize=8)   # ✅ Producer/consumer decoupling


# ── Audio helpers ─────────────────────────────────────────────────────────────

def rms_energy(data: np.ndarray) -> float:
    """More accurate than peak amplitude for silence detection."""
    if data.ndim > 1:
        data = data[:, 0]
    return float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))


def is_silence(data: np.ndarray) -> bool:
    """Returns True if chunk is mostly silent (RMS-based, per-frame voting)."""
    if data.ndim > 1:
        data = data[:, 0]
    frame_size = 1024
    silent_frames = 0
    total_frames  = len(data) // frame_size
    for i in range(total_frames):
        frame = data[i * frame_size:(i + 1) * frame_size]
        if rms_energy(frame) < SILENCE_THRESHOLD:
            silent_frames += 1
    return (silent_frames / total_frames) >= SILENCE_RATIO if total_frames > 0 else True


def is_echo(transcription: str, ai_text: str) -> bool:
    if not transcription or not ai_text:
        return False
    t_words = set(transcription.lower().replace('.','').replace(',','').split())
    a_words = set(ai_text.lower().replace('.','').replace(',','').split())
    if not t_words:
        return False
    overlap_pct = len(t_words & a_words) / len(t_words)
    return overlap_pct > 0.5


def record_chunk_in_memory(stream, length_sec=DEFAULT_CHUNK_LENGTH):
    """Record a chunk; return BytesIO WAV or None if silent."""
    frames     = [stream.read(1024) for _ in range(int(16000 / 1024 * length_sec))]
    raw_bytes  = b"".join(frames)
    audio_arr  = np.frombuffer(raw_bytes, dtype=np.int16)

    if is_silence(audio_arr):
        return None

    buf = io.BytesIO()
    sf.write(buf, audio_arr, samplerate=16000, format='WAV')
    buf.seek(0)
    return buf


def detect_emotion(wav_buffer: io.BytesIO, text: str) -> str:
    wav_buffer.seek(0)
    data, sr = sf.read(wav_buffer)
    wav_buffer.seek(0)
    if data.ndim > 1:
        data = data[:, 0]
    duration       = len(data) / sr
    words_per_sec  = len(text.split()) / duration if duration > 0 else 0
    amplitude      = np.max(np.abs(data))

    if words_per_sec > 3 or amplitude > 0.8:
        return "stressed"
    elif words_per_sec < 1.5 and amplitude < 0.3:
        return "calm"
    elif any(w in text.lower() for w in ("confused", "how", "what", "?")):
        return "confused"
    return "neutral"


def transcribe(model: WhisperModel, wav_buffer: io.BytesIO):
    segments, info = model.transcribe(wav_buffer, beam_size=3)   # ✅ Was 5; 3 is faster
    text    = " ".join(s.text for s in segments).strip()
    emotion = detect_emotion(wav_buffer, text) if text else "neutral"
    return text, info.language, emotion


def load_whisper() -> WhisperModel:
    print(f"🔍 Loading Whisper '{DEFAULT_MODEL_SIZE}' on CPU …")
    return WhisperModel(
        DEFAULT_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        num_workers=4,
        cpu_threads=4,           # ✅ Explicit thread count helps on multi-core
    )


# ── WebSocket broadcast ───────────────────────────────────────────────────────

async def broadcast_message(message: dict):
    if connected_clients:
        await asyncio.gather(
            *[c.send(json.dumps(message)) for c in connected_clients],
            return_exceptions=True,
        )


async def handle_websocket(websocket, path):
    connected_clients.add(websocket)
    print(f"👤 Client connected ({len(connected_clients)} total)")
    try:
        await websocket.send(json.dumps({
            "type": "connection_status",
            "status": "connected",
            "message": "Connected to Veena AI",
        }))
        async for raw in websocket:
            try:
                data = json.loads(raw)
                await handle_client_message(data, websocket)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"👤 Client disconnected ({len(connected_clients)} total)")


async def handle_client_message(data: dict, websocket):
    msg_type = data.get("type")

    if msg_type == "get_dashboard_data":
        loop = asyncio.get_running_loop()
        dashboard = await loop.run_in_executor(EXECUTOR, wellness_tools.get_dashboard_data)
        await websocket.send(json.dumps({"type": "dashboard_data", "data": dashboard}))

    elif msg_type == "text_input":
        text     = data.get("text", "")
        language = data.get("language", "en")
        emotion  = data.get("emotion", "neutral")
        if text:
            await process_user_input(text, language=language, emotion=emotion)

    elif msg_type in ("start_listening", "stop_listening"):
        status = "listening_started" if msg_type == "start_listening" else "listening_stopped"
        await websocket.send(json.dumps({"type": status}))


# ── Core conversation pipeline ────────────────────────────────────────────────

async def process_user_input(user_text: str, language="en", emotion="neutral"):
    global is_thinking, is_speaking, current_ai_text

    print(f"🗣  Customer [{language}/{emotion}]: {user_text}")
    is_thinking = True

    await broadcast_message({
        "type": "user_message",
        "text": user_text,
        "emotion_detected": emotion,
        "timestamp": datetime.now().isoformat(),
    })

    try:
        loop = asyncio.get_running_loop()

        _FALLBACK = {
            "hi": "मुझे खेद है, मैं अभी आपकी बात ठीक से समझ नहीं पाई। क्या आप दोबारा बता सकते हैं कि आप कैसा महसूस कर रहे हैं?",
            "en": "I'm sorry, I couldn't process that properly. Could you please tell me again how you're feeling?",
        }

        stress_state = wellness_tools.assess_stress(user_text)
        effective_emotion = emotion
        if stress_state["level"] in {"medium", "high"}:
            effective_emotion = "stressed"

        def invoke_agent():
            if wellness_tools.detect_crisis(user_text):
                return wellness_tools.get_crisis_resources()
            result = agent.invoke({
                "input": user_text,
                "language": language,
                "emotion": effective_emotion,
            })
            output = result.get("output", "").strip()
            # LangChain returns this literal string when max_iterations is hit
            if not output or "agent stopped" in output.lower():
                return _FALLBACK.get(language, _FALLBACK["en"])
            return output

        response = await loop.run_in_executor(EXECUTOR, invoke_agent)
        print(f"🤖 Veena: {response}")

        is_thinking = False
        is_speaking = True
        current_ai_text = response

        await broadcast_message({
            "type": "agent_response",
            "text": response,
            "timestamp": datetime.now().isoformat(),
        })
        await broadcast_message({"type": "speaking_started"})

        asyncio.create_task(play_tts_and_notify(response, language))

    except Exception as e:
        is_thinking = False
        print(f"❌ Agent error: {e}")
        await broadcast_message({"type": "error", "message": "Error generating response"})


async def play_tts_and_notify(text: str, language="en"):
    global is_speaking, current_ai_text
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(EXECUTOR, vs.play_text_to_speech_stream, text, language)
    except Exception as e:
        print(f"❌ TTS error: {e}")
    finally:
        is_speaking = False
        current_ai_text = ""
        await broadcast_message({"type": "speaking_finished"})


# ── Audio pipeline: producer + consumer threads ───────────────────────────────

def audio_producer():
    """Continuously records chunks and pushes non-silent ones to the queue."""
    global audio_device, mic_stream

    audio_device = pyaudio.PyAudio()
    mic_stream   = audio_device.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024,
    )
    print("🎙  Audio producer started …")
    try:
        while True:
            buf = record_chunk_in_memory(mic_stream, DEFAULT_CHUNK_LENGTH)
            if buf is None:
                continue
            # Drop oldest if queue full (prefer freshness)
            if audio_queue.full():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
            audio_queue.put(buf)
    except Exception as e:
        print(f"❌ Producer error: {e}")
    finally:
        if mic_stream:
            mic_stream.stop_stream()
            mic_stream.close()
        if audio_device:
            audio_device.terminate()


def audio_consumer():
    """Pulls chunks from queue, transcribes, and dispatches to main loop."""
    global is_speaking, is_thinking, current_ai_text, main_loop

    model = load_whisper()
    print("🎧  Audio consumer (transcriber) started …")

    import pygame

    while True:
        try:
            buf = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Skip while AI is thinking — avoids processing background noise
        if is_thinking:
            continue

        user_text, language, emotion = transcribe(model, buf)
        if not user_text:
            continue

        # Echo / barge-in logic
        if is_speaking:
            if is_echo(user_text, current_ai_text):
                print(f"🔇 Echo suppressed: {user_text[:50]}")
                continue
            else:
                print(f"🛑 Barge-in: {user_text}")
                if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                is_speaking = False

        if main_loop and not main_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                process_user_input(user_text, language, emotion),
                main_loop,
            )


def audio_recording_loop():
    """Start producer and consumer in separate threads."""
    producer_thread = threading.Thread(target=audio_producer, daemon=True, name="AudioProducer")
    consumer_thread = threading.Thread(target=audio_consumer, daemon=True, name="AudioConsumer")
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()


# ── HTTP server ───────────────────────────────────────────────────────────────

def start_http_server():
    app_dir      = Path(__file__).parent
    frontend_dir = app_dir / "frontend"

    if not (frontend_dir / "index.html").exists():
        print(f"❌ index.html not found at {frontend_dir}")
        return

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(frontend_dir), **kw)
        def log_message(self, *_): pass
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            super().end_headers()

    try:
        with socketserver.TCPServer(("", HTTP_PORT), Handler) as httpd:
            print(f"🌍 HTTP  → http://localhost:{HTTP_PORT}")
            threading.Thread(
                target=lambda: (time.sleep(2), webbrowser.open(f"http://localhost:{HTTP_PORT}/index.html")),
                daemon=True,
            ).start()
            httpd.serve_forever()
    except OSError as e:
        print(f"❌ HTTP error: {e}")


# ── WebSocket server ──────────────────────────────────────────────────────────

async def start_websocket_server():
    global main_loop
    main_loop = asyncio.get_running_loop()
    print(f"🌐 WebSocket → ws://localhost:{WS_PORT}")
    async with serve(
        handle_websocket,
        "localhost",
        WS_PORT,
        ping_interval=20,      # ✅ Keepalive pings every 20s
        ping_timeout=10,
    ):
        await asyncio.Future()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("🎯  VEENA AI — Mental Wellness Voice Companion")
    print("=" * 70)

    threading.Thread(target=start_http_server, daemon=True).start()
    time.sleep(1)

    threading.Thread(target=audio_recording_loop, daemon=True).start()
    time.sleep(1)

    print()
    print("✅  All systems ready!")
    print(f"🌐  Web     : http://localhost:{HTTP_PORT}/index.html")
    print(f"🔌  WS      : ws://localhost:{WS_PORT}")
    print(f"🎙️   Voice   : Active  (chunk={DEFAULT_CHUNK_LENGTH}s, model={DEFAULT_MODEL_SIZE})")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)

    try:
        asyncio.run(start_websocket_server())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Veena AI … Goodbye!")


if __name__ == "__main__":
    main()