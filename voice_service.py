import asyncio
import edge_tts
import pygame
import os

# Initialize mixer once at the start to improve latency
pygame.mixer.init()

async def generate_speech(text, output_file, language=None):
    voice = "hi-IN-SwaraNeural" if (language == "hi" or any(u'\u0900' <= c <= u'\u097f' for c in text)) else "en-IN-NeerjaNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

def play_text_to_speech_stream(text, language=None):
    output_file = "temp_audio.mp3"
    try:
        asyncio.run(generate_speech(text, output_file, language))
        
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # UNLOAD is critical to allow file deletion
        pygame.mixer.music.unload() 
        
        if os.path.exists(output_file):
            os.remove(output_file)
    except Exception as e:
        print(f"🔊 TTS Error: {e}")