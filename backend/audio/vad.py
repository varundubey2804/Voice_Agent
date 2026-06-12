import torch
from backend.core.logger import logger
from backend.core.config import settings

class VADService:
    def __init__(self, threshold: float = 0.5):
        """
        Initializes Silero VAD.
        """
        self.threshold = threshold
        try:
            # Silero VAD is lightweight enough to be loaded directly via torch hub
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True # Use ONNX for faster CPU inference
            )
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = self.utils

            logger.info("Silero VAD initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize VADService: {e}")
            raise

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Checks if a given audio chunk contains speech.
        Args:
            audio_chunk: 16-bit PCM audio data as numpy array.
            sample_rate: Expected to be 16000.
        """
        try:
            # Convert numpy array to torch tensor
            # Silero VAD expects float32 in range [-1, 1]
            if audio_chunk.dtype == np.int16:
                audio_float32 = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_float32 = audio_chunk

            tensor = torch.from_numpy(audio_float32)

            # Get speech probability
            speech_prob = self.model(tensor, sample_rate).item()
            return speech_prob >= self.threshold
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}")
            return False
