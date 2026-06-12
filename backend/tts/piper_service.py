import os
import io
import wave
import asyncio
from concurrent.futures import ThreadPoolExecutor
from piper.voice import PiperVoice
from backend.core.config import settings
from backend.core.logger import logger

class PiperTTSService:
    def __init__(self, model_path: str = None):
        """
        Initializes Piper TTS.
        """
        self.model_path = model_path or settings.PIPER_VOICE_MODEL
        if not os.path.exists(self.model_path):
            logger.warning(f"Piper model not found at {self.model_path}. You need to download it.")
            self.voice = None
        else:
            try:
                self.voice = PiperVoice.load(self.model_path)
                logger.info("Piper TTS initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Piper TTS: {e}")
                self.voice = None

        self.executor = ThreadPoolExecutor(max_workers=2)

    def synthesize_sync(self, text: str) -> bytes:
        """
        Synchronously synthesizes text to audio bytes (WAV format).
        """
        if not self.voice:
            return b""

        try:
            # Piper writes to a wave file object
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wav_file:
                # Basic parameters (assumed from model metadata normally)
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2) # 16-bit
                wav_file.setframerate(settings.SAMPLE_RATE)

                self.voice.synthesize(text, wav_file)

            return buf.getvalue()
        except Exception as e:
            logger.error(f"Error during TTS synthesis: {e}")
            return b""

    async def synthesize(self, text: str) -> bytes:
        """
        Asynchronously synthesizes text to audio bytes.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.synthesize_sync, text)
