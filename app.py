import os
import io
import numpy as np
import pyaudio
from scipy.io.wavfile import read as wav_read
from faster_whisper import WhisperModel
from agentic_rag import build_agent
import voice_service as vs  # Updated to have play_text_to_speech_stream
import soundfile as sf  # For in-memory WAV

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_MODEL_SIZE   = "base"
DEFAULT_CHUNK_LENGTH = 8  # seconds

agent = build_agent()

def is_silence(data, threshold=3000):
    if data.ndim > 1:
        data = data[:, 0]
    return np.max(np.abs(data)) <= threshold

def record_chunk_in_memory(stream, length_sec=DEFAULT_CHUNK_LENGTH):
    """Record audio directly to BytesIO instead of saving to disk."""
    frames = [stream.read(1024) for _ in range(int(16000 / 1024 * length_sec))]
    audio_bytes = b"".join(frames)

    # Convert raw PCM bytes to numpy array
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

    if is_silence(audio_array):
        return None

    # Save to in-memory WAV buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, samplerate=16000, format='WAV')
    buffer.seek(0)
    return buffer

def transcribe(model, wav_buffer):
    segments, _info = model.transcribe(wav_buffer, beam_size=5)
    text = " ".join(segment.text for segment in segments)
    return text.strip()

def load_whisper():
    size = DEFAULT_MODEL_SIZE + ".en"
    try:
        print("🔍 Loading Whisper on CUDA …")
        return WhisperModel(size, device="cuda", compute_type="float16", num_workers=4)
    except Exception as e:
        print(f"⚠️ CUDA failed: {e} → CPU fallback")
        return WhisperModel(size, device="cpu", compute_type="int8", num_workers=2)

def main():
    model  = load_whisper()
    audio  = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000,
                        input=True, frames_per_buffer=1024)

    try:
        print("🎙️  Listening…  (Ctrl+C to quit)")
        while True:
            wav_buffer = record_chunk_in_memory(stream)
            if not wav_buffer:
                print("… (silence)")
                continue

            user_text = transcribe(model, wav_buffer)
            if not user_text:
                continue

            print(f"🗣️  Customer: {user_text}")
            
            response = agent.invoke({"input": user_text})["output"].strip()
            print(f"🤖 Veena: {response}")

            # Stream TTS directly
            vs.play_text_to_speech_stream(response)

    except KeyboardInterrupt:
        print("\n🛑  Stopped by user.")
    finally:
        stream.stop_stream(); stream.close(); audio.terminate()
        print("🎤  Audio stream closed.")

if __name__ == "__main__":
    main()
