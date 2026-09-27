"""
thread_store.py — Encrypted conversation-thread storage.

DESIGN
──────
• Every conversation ("chat") lives under a `thread_id` (uuid4 hex).
• The thread's full message history is stored as ONE encrypted blob per
  thread in a local SQLite DB. Encryption is Fernet (AES-128-CBC + HMAC),
  keyed by a single master key so the whole system can decrypt any thread
  it owns — no per-thread key juggling required by the caller.
• `thread_id` is the lookup KEY: you need it to fetch/append to a thread.
  Without it, the row can't even be found (it's the primary key), and even
  a full DB dump is useless without VOICE_MASTER_KEY on top of that.
• Master key resolution order:
    1. VOICE_MASTER_KEY env var (base64 urlsafe 32-byte Fernet key)
    2. ./.voice_master.key file (created automatically on first run)
  This means: set VOICE_MASTER_KEY in prod / secrets manager; in local
  dev it just works and persists across restarts via the key file.

USAGE
─────
    store = ThreadStore("threads.db")
    thread_id = store.create_thread(meta={"persona": "veena"})
    store.append_message(thread_id, "user", "I'm feeling anxious")
    store.append_message(thread_id, "assistant", "That sounds hard...")
    history = store.get_history(thread_id)   # decrypted list[dict]
    store.list_threads()                     # [{"thread_id":..., "created_at":...}, ...]
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

_KEY_FILE = Path(".voice_master.key")


def _resolve_master_key() -> bytes:
    env_key = os.environ.get("VOICE_MASTER_KEY")
    if env_key:
        return env_key.encode() if isinstance(env_key, str) else env_key

    if _KEY_FILE.exists():
        return _KEY_FILE.read_bytes().strip()

    key = Fernet.generate_key()
    _KEY_FILE.write_bytes(key)
    try:
        os.chmod(_KEY_FILE, 0o600)
    except OSError:
        pass
    print(
        "🔑  No VOICE_MASTER_KEY set — generated one and saved it to "
        f"'{_KEY_FILE}'. For production, set VOICE_MASTER_KEY explicitly "
        "and keep it out of source control / the DB's blast radius."
    )
    return key


class ThreadStore:
    def __init__(self, db_path: str = "threads.db", master_key: Optional[bytes] = None):
        self.db_path = db_path
        self._fernet = Fernet(master_key or _resolve_master_key())
        self._lock = threading.Lock()
        self._init_db()

    # ── schema ──────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id   TEXT PRIMARY KEY,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    meta_json   TEXT NOT NULL DEFAULT '{}',
                    payload     BLOB NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ── encryption helpers ──────────────────────────────────────────────
    def _encrypt(self, messages: list[dict]) -> bytes:
        raw = json.dumps(messages, ensure_ascii=False).encode("utf-8")
        return self._fernet.encrypt(raw)

    def _decrypt(self, blob: bytes) -> list[dict]:
        try:
            raw = self._fernet.decrypt(bytes(blob))
        except InvalidToken as exc:
            raise ValueError(
                "Could not decrypt thread — wrong VOICE_MASTER_KEY or corrupted data."
            ) from exc
        return json.loads(raw.decode("utf-8"))

    # ── public API ──────────────────────────────────────────────────────
    def create_thread(self, meta: Optional[dict] = None, thread_id: Optional[str] = None) -> str:
        """Create a new empty thread and return its thread_id (the access key)."""
        thread_id = thread_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO threads (thread_id, created_at, updated_at, meta_json, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (thread_id, now, now, json.dumps(meta or {}), self._encrypt([])),
            )
            conn.commit()
        return thread_id

    def thread_exists(self, thread_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return row is not None

    def append_message(self, thread_id: str, role: str, content: str, extra: Optional[dict] = None) -> None:
        """Append one message to a thread. Read-modify-write under a lock (thread-safe)."""
        entry = {
            "role": role,
            "content": content,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            entry.update(extra)

        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown thread_id: {thread_id}")
            messages = self._decrypt(row["payload"])
            messages.append(entry)
            conn.execute(
                "UPDATE threads SET payload = ?, updated_at = ? WHERE thread_id = ?",
                (self._encrypt(messages), datetime.now(timezone.utc).isoformat(), thread_id),
            )
            conn.commit()

    def get_history(self, thread_id: str) -> list[dict]:
        """Return the decrypted message list for a thread. Empty list if unknown."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        if row is None:
            return []
        return self._decrypt(row["payload"])

    def get_history_as_text(self, thread_id: str, max_turns: int = 20) -> str:
        """Convenience: last N turns rendered as 'Human: ...\\nAI: ...' for LangChain memory seeding."""
        history = self.get_history(thread_id)[-max_turns:]
        lines = []
        for m in history:
            speaker = "Human" if m["role"] == "user" else "AI"
            lines.append(f"{speaker}: {m['content']}")
        return "\n".join(lines)

    def list_threads(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT thread_id, created_at, updated_at, meta_json FROM threads "
                "ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {
                "thread_id": r["thread_id"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "meta": json.loads(r["meta_json"]),
            }
            for r in rows
        ]

    def delete_thread(self, thread_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
            conn.commit()
