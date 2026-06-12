import fasttext
import urllib.request
import os
from backend.core.config import settings
from backend.core.logger import logger

class LanguageDetector:
    def __init__(self):
        self.model_path = os.path.join(settings.MODELS_DIR, "lid.176.ftz")
        self._ensure_model_downloaded()
        try:
            self.model = fasttext.load_model(self.model_path)
            logger.info("FastText language detection initialized.")
        except Exception as e:
            logger.error(f"Failed to load FastText model: {e}")
            self.model = None

    def _ensure_model_downloaded(self):
        if not os.path.exists(self.model_path):
            logger.info(f"Downloading FastText language identification model to {self.model_path}...")
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                logger.info("Download complete.")
            except Exception as e:
                logger.error(f"Failed to download FastText model: {e}")

    def detect_language(self, text: str) -> tuple[str, float]:
        """
        Detects the language of the given text.
        Returns a tuple of (language_code, confidence).
        """
        if not self.model or not text.strip():
            return ("en", 0.0) # default fallback

        try:
            text = text.replace('\n', ' ')
            predictions = self.model.predict(text, k=1)
            # predictions looks like: (('__label__en',), array([0.99]))
            lang_label = predictions[0][0]
            confidence = float(predictions[1][0])

            # Extract language code, e.g., '__label__en' -> 'en'
            lang_code = lang_label.replace('__label__', '')
            return (lang_code, confidence)
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return ("en", 0.0)
