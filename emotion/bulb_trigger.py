"""
emotion/bulb_trigger.py — Cooldown-guarded wrapper around ir_trigger,
called from inside the ONE emotion-trigger HTTP endpoint that already runs
inside voice_engine.py (see emotion_trigger_server.py).

WHY THIS ISN'T ITS OWN SERVER
──────────────────────────────
This started life as bulb_trigger_server.py — its own ThreadingHTTPServer
on 0.0.0.0:5000/emotion-trigger for the vision pipeline's
emotion/emotion_sender.py to call. But voice_engine.py already binds that
exact host/port/path for the hardware mic-trigger flow (start_emotion_
trigger_server in emotion_trigger_server.py). Two processes can't bind the
same port, so running both would fail with "address already in use".

Instead, this module plugs into that one existing endpoint:
VoiceEngine.handle_emotion_trigger() calls handle_bulb_trigger() to fire
the bulb UNCONDITIONALLY (mood lighting shouldn't wait on whether Veena
happens to be mid-conversation), then separately decides whether to speak
a check-in line (which DOES get dropped if she's busy — no queueing).

INDEPENDENT COOLDOWN (defense in depth) — unchanged from the original
────────────────────────────────────────────────────────────────────
ir_trigger.trigger() already skips exact duplicate actions, but has no
time-based cooldown. This adds one: even if the same action flaps quickly
(noisy vision-pipeline output, a retried request), the MacroDroid webhook
fires at most once per COOLDOWN_S.

The device ID itself is NOT duplicated here — it stays single-sourced in
config.py, resolved through ir_trigger.py like everywhere else.
"""

from __future__ import annotations

import os
import threading
import time

from . import ir_trigger

COOLDOWN_S = float(os.getenv("EMOTION_HTTP_COOLDOWN_S", "15.0"))

_last_trigger: str | None = None
_last_ts = 0.0
_lock = threading.Lock()


def handle_bulb_trigger(emotion_or_action: str, force: bool = False, blocking: bool = False) -> bool:
    """
    Fire the bulb/IR webhook for `emotion_or_action`, gated by a cooldown +
    change check independent of ir_trigger's own dedup. Returns True if it
    actually fired, False if it was swallowed by the cooldown/dedup guard.
    """
    global _last_trigger, _last_ts
    key = (emotion_or_action or "").lower().strip()
    action = ir_trigger.EMOTION_TO_TRIGGER.get(key, key)

    now = time.monotonic()
    with _lock:
        changed = action != _last_trigger
        cooled_down = (now - _last_ts) >= COOLDOWN_S
        fire = force or (changed and cooled_down)
        if fire:
            _last_trigger = action
            _last_ts = now

    if fire:
        # force=True here because we already did our own change+cooldown
        # check above — no need for ir_trigger's separate dedup on top.
        ir_trigger.trigger(action, force=True, blocking=blocking)
    return fire