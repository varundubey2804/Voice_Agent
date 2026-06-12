import openwakeword
from openwakeword.model import Model
from backend.core.config import settings
from backend.core.logger import logger

class WakeWordDetector:
    def __init__(self, model_paths: list[str] = None):
        """
        Initializes the OpenWakeWord model.
        Args:
            model_paths: List of paths to custom wake word models (.onnx or .tflite).
                         If None, loads the default pre-trained models.
        """
        try:
            openwakeword.utils.download_models() # Ensures default models are present if no custom path
            self.model = Model(
                wakeword_models=model_paths or ["hey_mycroft"], # Default fallback if custom is absent
                inference_framework="onnx"
            )
            logger.info("WakeWordDetector initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize WakeWordDetector: {e}")
            raise

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Process an audio chunk and return True if a wake word is detected.
        Args:
            audio_chunk: 16-bit PCM audio data as numpy array (16kHz).
        """
        try:
            # openwakeword expects shape (N,)
            prediction = self.model.predict(audio_chunk)

            # prediction is a dictionary of {model_name: score}
            for model_name, score in prediction.items():
                if score > 0.5:  # threshold
                    logger.info(f"Wake word detected! Model: {model_name}, Score: {score}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error processing audio chunk in WakeWordDetector: {e}")
            return False
