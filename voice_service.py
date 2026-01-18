# voice_service.py
import asyncio
import edge_tts
import pygame
import os

# Voice Configuration
VOICE_EN = "en-IN-NeerjaNeural"  # Indian English Female
VOICE_HI = "hi-IN-SwaraNeural"   # Hindi Female

async def generate_speech(text, output_file="temp_audio.mp3", language=None):
    """
    Generate TTS audio. 
    Uses provided language or automatically detects Hindi characters to switch voice.
    """
    
    selected_voice = VOICE_EN
    
    # If language is provided, use it
    if language:
        if language in ["hi", "hindi"]:
            selected_voice = VOICE_HI
            print(f"🗣️ Language provided: HINDI (Using {VOICE_HI})")
        else:
            selected_voice = VOICE_EN
            print(f"🗣️ Language provided: ENGLISH (Using {VOICE_EN})")
    else:
        # Check for Devanagari unicode range (Hindi characters)
        # If found, switch to the Hindi voice model
        if any(u'\u0900' <= c <= u'\u097f' for c in text):
            selected_voice = VOICE_HI
            print(f"🗣️ Language detected from text: HINDI (Using {VOICE_HI})")
        else:
            selected_voice = VOICE_EN
            print(f"🗣️ Language detected from text: ENGLISH (Using {VOICE_EN})")
        
    communicate = edge_tts.Communicate(text, selected_voice)
    await communicate.save(output_file)

def play_text_to_speech_stream(text, language=None):
    """Wrapper to run async generation and play audio synchronously."""
    output_file = "temp_audio.mp3"
    
    try:
        # Create a new event loop for this thread to run the async task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate_speech(text, output_file, language=language))
        loop.close()

        # Play audio using Pygame
        pygame.mixer.init()
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()
        pygame.mixer.quit()
        
        # Cleanup
        if os.path.exists(output_file):
            os.remove(output_file)

    except Exception as e:
        print(f"🔊 TTS Error: {e}")