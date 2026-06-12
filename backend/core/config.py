from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Handheld AI Assistant"
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = DATA_DIR / "models"
    DB_DIR: Path = DATA_DIR / "db"
    DOCS_DIR: Path = DATA_DIR / "docs"

    # Audio settings
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 512

    # STT & Wake Word
    WAKE_WORD_MODEL: str = "hey_assistant"
    WHISPER_MODEL_PATH: str = str(MODELS_DIR / "ggml-tiny.en.bin")

    # LLM
    LLM_MODEL: str = "qwen-3-4b" # or gemma-3-4b, phi-4-mini
    LLM_CONTEXT_WINDOW: int = 4096

    # RAG
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    FAISS_INDEX_PATH: str = str(DB_DIR / "faiss_index.idx")

    # TTS
    PIPER_VOICE_MODEL: str = str(MODELS_DIR / "en_US-lessac-medium.onnx")

    # Security
    ENCRYPTION_KEY: str = "CHANGE_THIS_IN_PRODUCTION_AES_256_KEY_0000000"  # Should be 32 bytes in prod or env

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

# Ensure directories exist
for d in [settings.DATA_DIR, settings.MODELS_DIR, settings.DB_DIR, settings.DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
