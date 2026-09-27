"""
user_profile.py — Persistent, cross-thread "who is this user" profile.

WHY THIS IS SEPARATE FROM ThreadStore
──────────────────────────────────────
ThreadStore keys everything by thread_id, and a new process starts a
brand-new thread by default. That's correct for CONVERSATIONS ("what did
we just talk about") but wrong for IDENTITY ("what's my name", "I have
two kids", "I work night shifts") — that should survive across every
thread, forever, until the user explicitly resets it.

This module stores exactly ONE profile (this lamp = one user, for the
hackathon). If you ever need multi-user support, key by a device/voice-id
column instead of a single fixed row.

Storage: same SQLite file as ThreadStore, a new table, the same Fernet
encryption scheme (reuses thread_store's key resolution so there's only
one secret to manage, not two).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from thread_store import _resolve_master_key

_ROW_ID = 1  # single-row table — one user per lamp for now

# (field_key, spoken question). Keep this short — every question here is
# one extra voice round-trip before the user gets to actually talk to her.
ONBOARDING_QUESTIONS: list[tuple[str, str]] = [
    ("name", "Hi, I'm Veena. Before we start, what should I call you?"),
    ("context", "Good to meet you. What's going on in your life right now that you'd like some support with?"),
    ("support_style", "When you're stressed, do you usually want someone to just listen, or to help you work through it?"),
]


class UserProfileStore:
    def __init__(self, db_path: str = "threads.db", master_key: Optional[bytes] = None):
        self.db_path = db_path
        self._fernet = Fernet(master_key or _resolve_master_key())
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profile (
                    id          INTEGER PRIMARY KEY,
                    updated_at  TEXT NOT NULL,
                    payload     BLOB NOT NULL
                )
                """
            )
            conn.commit()

    def get(self) -> Optional[dict]:
        """Returns the decrypted profile dict, or None if onboarding hasn't run yet."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM user_profile WHERE id = ?", (_ROW_ID,)
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(self._fernet.decrypt(bytes(row["payload"])).decode("utf-8"))
        except InvalidToken:
            return None

    def save(self, profile: dict) -> None:
        raw = json.dumps(profile, ensure_ascii=False).encode("utf-8")
        blob = self._fernet.encrypt(raw)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO user_profile (id, updated_at, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at, payload = excluded.payload",
                (_ROW_ID, now, blob),
            )
            conn.commit()

    def reset(self) -> None:
        """Wipe the profile so onboarding runs again on next start."""
        with self._connect() as conn:
            conn.execute("DELETE FROM user_profile WHERE id = ?", (_ROW_ID,))
            conn.commit()


def profile_summary_text(profile: Optional[dict]) -> str:
    """
    Compact block to prepend to the persona's system prompt. Empty string
    if onboarding hasn't happened yet (so nothing weird gets injected).
    """
    if not profile:
        return ""
    lines = [f"- {key.replace('_', ' ')}: {val}" for key, val in profile.items() if val]
    if not lines:
        return ""
    return (
        "What you already know about this user — use it naturally when "
        "relevant, don't recite it back like a form:\n" + "\n".join(lines)
    )