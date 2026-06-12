import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from backend.core.config import settings
from backend.core.logger import logger

class EncryptionManager:
    def __init__(self):
        self._key = self._generate_key(settings.ENCRYPTION_KEY)
        self.fernet = Fernet(self._key)
        logger.info("EncryptionManager initialized.")

    def _generate_key(self, raw_key: str) -> bytes:
        # Generate a stable key from a string using a fixed salt for local storage
        # In a real secure app, salt should be securely stored or user-provided
        salt = b'local_ai_assistant_salt'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(raw_key.encode()))
        return key

    def encrypt_data(self, data: str) -> str:
        """Encrypts string data to a secure string representation."""
        try:
            encrypted_bytes = self.fernet.encrypt(data.encode('utf-8'))
            return encrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypts a secure string representation back to original string data."""
        try:
            decrypted_bytes = self.fernet.decrypt(encrypted_data.encode('utf-8'))
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def save_secure_file(self, filepath: str, data: str):
        """Encrypts and saves data to a file."""
        encrypted = self.encrypt_data(data)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(encrypted)

    def read_secure_file(self, filepath: str) -> str:
        """Reads and decrypts data from a file."""
        if not os.path.exists(filepath):
            return ""
        with open(filepath, 'r', encoding='utf-8') as f:
            encrypted_data = f.read()
        if not encrypted_data:
            return ""
        return self.decrypt_data(encrypted_data)

encryption_manager = EncryptionManager()
