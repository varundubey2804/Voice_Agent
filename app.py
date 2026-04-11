import os
import io
import json
import asyncio
import numpy as np
import pyaudio
import soundfile as sf
import websockets
from websockets.server import serve
import threading
import signal
from datetime import datetime
from pathlib import Path
import webbrowser
import http.server
import socketserver
import time
import pygame
from faster_whisper import WhisperModel
from dotenv import load_dotenv

# Internal imports
import finance_tools
import voice_service as vs
from agentic_rag import build_agent

# Initialize environment and professional monitoring
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
DEFAULT_MODEL_SIZE = "base"  # Optimized for industry-standard latency
DEFAULT_CHUNK_LENGTH = 4    # Snappier responsiveness
HTTP_PORT = 8080
WS_PORT = 8765

# State
agent = build_agent()
connected_clients = set()
is_speaking = False
is_thinking = False
current_ai_text = ""
main_loop = None
_shutdown_event = threading.Event()

# --- WebSocket Handlers (Corrected for websockets 13.0+) ---

async def handle_websocket(websocket): # Removed 'path' argument
    """Modern connection handler that prevents handshake failures."""
    connected_clients.add(websocket)
    print(f"👤 Client connected. Total clients: {len(connected_clients)}")
    
    try:
        await websocket.send(json.dumps({
            "type": "connection_status",
            "status": "connected",
            "message": "Connected to Veena AI"
        }))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_client_message(data, websocket)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print("👤 Client disconnected.")

async def handle_client_message(data, websocket):
    msg_type = data.get("type")
    
    if msg_type == "get_dashboard_data":
        # Industry-ready data bridge
        loop = asyncio.get_running_loop()
        dashboard_data = await loop.run_in_executor(None, finance_tools.get_dashboard_data)
        await websocket.send(json.dumps({"type": "dashboard_data", "data": dashboard_data}))
            
    elif msg_type == "text_input":
        text = data.get("text", "")
        language = data.get("language", "en")
        if text:
            await process_user_input(text, language=language)

# --- AI Logic & Audio Orchestration ---

async def process_user_input(user_text, language="en", emotion="neutral"):
    global is_thinking, is_speaking, current_ai_text
    is_thinking = True
    
    await broadcast_message({
        "type": "user_message", 
        "text": user_text, 
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        loop = asyncio.get_running_loop()
        # Invoke agent with industry-standard persona
        response = await loop.run_in_executor(
            None, 
            lambda: agent.invoke({"input": user_text, "language": language, "emotion": emotion})["output"].strip()
        )
        
        is_thinking = False
        is_speaking = True
        current_ai_text = response
        
        await broadcast_message({"type": "agent_response", "text": response})
        await broadcast_message({"type": "speaking_started"})
        
        # Audio Playback
        await loop.run_in_executor(None, lambda: vs.play_text_to_speech_stream(response, language))
        
    finally:
        is_speaking = False
        current_ai_text = ""
        await broadcast_message({"type": "speaking_finished"})

async def broadcast_message(message):
    if connected_clients:
        payload = json.dumps(message)
        await asyncio.gather(*[client.send(payload) for client in connected_clients], return_exceptions=True)

# --- Background Loops ---

def audio_recording_loop():
    global main_loop, is_speaking, is_thinking
    model = WhisperModel(DEFAULT_MODEL_SIZE, device="cpu", compute_type="int8")
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    print("🎙️ Audio recording active.")
    while not _shutdown_event.is_set():
        # Record chunk
        frames = [stream.read(1024, exception_on_overflow=False) for _ in range(int(16000 / 1024 * DEFAULT_CHUNK_LENGTH))]
        if is_thinking: continue
        
        audio_data = b"".join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Simple silence detection
        if np.max(np.abs(audio_array)) < 1500: continue
        
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_array, 16000, format='WAV')
        wav_io.seek(0)
        
        segments, info = model.transcribe(wav_io)
        text = " ".join(s.text for s in segments).strip()
        
        if text and main_loop:
            # Barge-in support: stop AI if user interrupts
            if is_speaking:
                pygame.mixer.music.stop()
            asyncio.run_coroutine_threadsafe(process_user_input(text, info.language), main_loop)

async def start_websocket_server():
    global main_loop
    main_loop = asyncio.get_running_loop()
    
    # Graceful shutdown handlers
    def _signal_handler(sig, frame):
        _shutdown_event.set()
        main_loop.stop()
    signal.signal(signal.SIGINT, _signal_handler)

    async with serve(handle_websocket, "localhost", WS_PORT):
        print(f"🚀 WebSocket online: ws://localhost:{WS_PORT}")
        await asyncio.Future() 

if __name__ == "__main__":
    # Start HTTP dashboard server
    threading.Thread(target=lambda: socketserver.TCPServer(("", HTTP_PORT), http.server.SimpleHTTPRequestHandler).serve_forever(), daemon=True).start()
    # Start Audio thread
    threading.Thread(target=audio_recording_loop, daemon=True).start()
    
    try:
        asyncio.run(start_websocket_server())
    except Exception:
        pass
