import os
import io
import json
import asyncio
import numpy as np
import pyaudio
from scipy.io.wavfile import read as wav_read
from faster_whisper import WhisperModel
from agentic_rag import build_agent
import voice_service as vs
import soundfile as sf
import websockets
from websockets.server import serve
import threading
import base64
from datetime import datetime
from langdetect import detect, LangDetectException

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_MODEL_SIZE = "base"
DEFAULT_CHUNK_LENGTH = 8
agent = build_agent()
whisper_model = None
audio = None
stream = None

# WebSocket clients
connected_clients = set()

def is_silence(data, threshold=3000):
    if data.ndim > 1:
        data = data[:, 0]
    return np.max(np.abs(data)) <= threshold

def record_chunk_in_memory(stream, length_sec=DEFAULT_CHUNK_LENGTH):
    """Record audio directly to BytesIO instead of saving to disk."""
    frames = [stream.read(1024) for _ in range(int(16000 / 1024 * length_sec))]
    audio_bytes = b"".join(frames)
    
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    if is_silence(audio_array):
        return None
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, samplerate=16000, format='WAV')
    buffer.seek(0)
    return buffer

def transcribe(model, wav_buffer):
    segments, info = model.transcribe(wav_buffer, beam_size=5)
    text = " ".join(segment.text for segment in segments)
    return text.strip(), info.language

def load_whisper():
    # Use multilingual model (remove .en suffix) to support Hindi
    size = DEFAULT_MODEL_SIZE
    try:
        print("🔍 Loading Whisper on CUDA ...")
        return WhisperModel(size, device="cuda", compute_type="float16", num_workers=4)
    except Exception as e:
        print(f"⚠ CUDA failed: {e} → CPU fallback")
        return WhisperModel(size, device="cpu", compute_type="int8", num_workers=2)

async def broadcast_message(message):
    """Send message to all connected clients"""
    if connected_clients:
        await asyncio.gather(
            *[client.send(json.dumps(message)) for client in connected_clients],
            return_exceptions=True
        )

async def handle_websocket(websocket, path):
    """Handle WebSocket connections"""
    connected_clients.add(websocket)
    print(f"👤 Client connected. Total clients: {len(connected_clients)}")
    
    # Send connection status
    await websocket.send(json.dumps({
        "type": "connection_status",
        "status": "connected",
        "message": "Connected to Veena AI"
    }))
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_client_message(data, websocket)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)
        print(f"👤 Client disconnected. Total clients: {len(connected_clients)}")

async def handle_client_message(data, websocket):
    """Handle messages from frontend"""
    message_type = data.get("type")
    
    if message_type == "start_listening":
        await websocket.send(json.dumps({
            "type": "listening_started",
            "message": "Started listening..."
        }))
        
    elif message_type == "stop_listening":
        await websocket.send(json.dumps({
            "type": "listening_stopped",
            "message": "Stopped listening"
        }))
        
    elif message_type == "audio_data":
        # Handle audio data from frontend (if implementing browser-based recording)
        audio_base64 = data.get("audio")
        if audio_base64:
            # Process audio data here
            pass
            
    elif message_type == "text_input":
        # Handle direct text input
        text = data.get("text", "")
        if text:
            # For text input, we can auto-detect language or default to en
            try:
                lang = detect(text)
            except:
                lang = 'en'
            await process_user_input(text, lang)

async def process_user_input(user_text, input_lang='en'):
    """Process user input and generate response"""
    print(f"🗣 Customer ({input_lang}): {user_text}")
    
    # Send user message to frontend
    await broadcast_message({
        "type": "user_message",
        "text": user_text,
        "timestamp": datetime.now().isoformat()
    })
    
    # Generate AI response
    try:
        # Append language instruction to input if needed, or rely on agent prompt
        response = agent.invoke({"input": user_text})["output"].strip()
        print(f"🤖 Veena: {response}")
        
        # Detect response language for TTS
        try:
            response_lang = detect(response)
        except LangDetectException:
            response_lang = input_lang # Fallback to input language

        print(f"🗣 TTS Language: {response_lang}")

        # Send agent response to frontend
        await broadcast_message({
            "type": "agent_response",
            "text": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send speaking status
        await broadcast_message({
            "type": "speaking_started"
        })
        
        # Play TTS (this will run in background)
        asyncio.create_task(play_tts_and_notify(response, response_lang))
        
    except Exception as e:
        print(f"Error generating response: {e}")
        await broadcast_message({
            "type": "error",
            "message": "Error generating response"
        })

async def play_tts_and_notify(text, language='en'):
    """Play TTS and notify when finished"""
    try:
        # Play TTS in a separate thread to avoid blocking
        def play_tts():
            vs.play_text_to_speech_stream(text, language=language)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, play_tts)
        
        # Notify frontend that speaking is finished
        await broadcast_message({
            "type": "speaking_finished"
        })
        
    except Exception as e:
        print(f"Error playing TTS: {e}")
        await broadcast_message({
            "type": "speaking_finished"
        })

def audio_recording_loop():
    """Background audio recording loop"""
    global whisper_model, audio, stream
    
    whisper_model = load_whisper()
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )
    
    print("🎙 Audio recording started...")
    
    try:
        while True:
            wav_buffer = record_chunk_in_memory(stream)
            if not wav_buffer:
                continue
            
            user_text, lang = transcribe(whisper_model, wav_buffer)
            if not user_text:
                continue
            
            # Send to WebSocket clients
            asyncio.run(process_user_input(user_text, lang))
            
    except KeyboardInterrupt:
        print("\n🛑 Audio recording stopped.")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        if audio:
            audio.terminate()
        print("🎤 Audio stream closed.")

async def start_websocket_server():
    """Start the WebSocket server"""
    print("🌐 Starting WebSocket server on ws://localhost:8765")
    async with serve(handle_websocket, "localhost", 8765):
        await asyncio.Future()  # Keep server running

def main():
    # Start audio recording in a separate thread
    audio_thread = threading.Thread(target=audio_recording_loop, daemon=True)
    audio_thread.start()
    
    # Start WebSocket server
    try:
        asyncio.run(start_websocket_server())
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")

if __name__ == "__main__":
    main()
