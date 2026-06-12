import pytest
from backend.core.config import settings

def test_settings_loaded():
    assert settings.APP_NAME == "Handheld AI Assistant"
    assert settings.SAMPLE_RATE == 16000

def test_encryption_manager():
    from backend.security.encryption import encryption_manager
    original_text = "Highly confidential data."
    encrypted = encryption_manager.encrypt_data(original_text)
    assert encrypted != original_text
    decrypted = encryption_manager.decrypt_data(encrypted)
    assert decrypted == original_text
