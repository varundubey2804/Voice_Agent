# voice_service.py

import os
import time
import pygame
from gtts import gTTS

def play_text_to_speech(text, language='en', slow=False):
    """Convert text to speech and play it."""
    try:
        tts = gTTS(text=text, lang=language, slow=slow)
        temp_audio_file = "temp_audio.mp3"
        tts.save(temp_audio_file)

        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()
        pygame.mixer.quit()
        os.remove(temp_audio_file)

    except Exception as e:
        print(f"🔊 TTS Error: {e}")
