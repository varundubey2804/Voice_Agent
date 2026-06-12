import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from whisper_cpp_python import Whisper
from backend.core.config import settings
from backend.core.logger import logger

class WhisperSTT:
    def __init__(self, model_path: str = None):
        """
        Initializes the Whisper.cpp STT model.
        Args:
            model_path: Path to the quantized GGUF/bin model.
        """
        self.model_path = model_path or settings.WHISPER_MODEL_PATH
        if not os.path.exists(self.model_path):
            logger.warning(f"Whisper model not found at {self.model_path}. You will need to download it.")
            self.model = None
        else:
            try:
                self.model = Whisper(model_path=self.model_path, n_threads=4)
                logger.info("Whisper STT initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Whisper STT: {e}")
                self.model = None

        self.executor = ThreadPoolExecutor(max_workers=2)

    def transcribe_sync(self, audio_data: bytes) -> str:
        """
        Synchronously transcribes audio data.
        audio_data: Raw 16kHz, 16-bit mono PCM bytes or numpy array.
        """
        if not self.model:
            return ""

        try:
            # Whisper.cpp python binding expects numpy array of float32

            # Assuming incoming data is raw bytes of 16-bit PCM
            if isinstance(audio_data, bytes):
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_np = audio_data # already numpy

            result = self.model.transcribe(audio_np)

            text = ""
            for segment in result.get('segments', []):
                text += segment.get('text', '') + " "

            return text.strip()
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""

    async def transcribe(self, audio_data: bytes) -> str:
        """
        Asynchronously transcribes audio data.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.transcribe_sync, audio_data)
