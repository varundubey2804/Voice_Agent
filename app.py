

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from agentic_rag import build_agent
agent = build_agent()

import wave, pyaudio, numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
import voice_service as vs

DEFAULT_MODEL_SIZE   = "base"
DEFAULT_CHUNK_LENGTH = 8  # seconds

def is_silence(data, threshold=3000):
    if data.ndim > 1:
        data = data[:, 0]
    return np.max(np.abs(data)) <= threshold

def record_chunk(audio, stream, length_sec=DEFAULT_CHUNK_LENGTH):
    frames = [stream.read(1024) for _ in range(int(16000 / 1024 * length_sec))]
    tmp = "temp_chunk.wav"
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b"".join(frames))

    sr, data = wavfile.read(tmp)
    if is_silence(data):
        os.remove(tmp)
        return None
    return tmp
def transcribe(model, wav_path):
    segments, _info = model.transcribe(wav_path, beam_size=5)
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
            wav_path = record_chunk(audio, stream)
            if not wav_path:
                print("… (silence)")
                continue

            user_text = transcribe(model, wav_path)
            os.remove(wav_path)

            if not user_text:
                continue

            print(f"🗣️  Customer: {user_text}")
            
            response = agent.invoke({"input": user_text})["output"].strip()
            print(f"🤖 Veena: {response}")
            vs.play_text_to_speech(response)

    except KeyboardInterrupt:
        print("\n🛑  Stopped by user.")
    finally:
        stream.stop_stream(); stream.close(); audio.terminate()
        print("🎤  Audio stream closed.")

if __name__ == "__main__":
    main()
