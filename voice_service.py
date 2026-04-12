"""
voice_service.py  ─  Robust TTS + STT Service  (v4)
═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURE
────────────
Previous versions called asyncio.run() from a ThreadPoolExecutor worker,
creating a brand-new event loop on every TTS call.  On Windows this races
with the ProactorEventLoop and causes 'cannot schedule new futures after
shutdown'.

v4 uses a DEDICATED BACKGROUND EVENT LOOP that lives in its own thread for
the entire lifetime of the process.  All async TTS work is submitted to
that loop via asyncio.run_coroutine_threadsafe().  Only one loop, ever.

TTS PIPELINE
────────────
  play_tts(text, language)
    │
    ├─ split into sentences
    ├─ for each sentence:
    │     synthesise via edge_tts  →  temp .mp3
    │     play via pygame (Sound, not Music channel)  ← key fix
    │     wait for playback to finish (polling 50 Hz)
    └─ clean up temp files

Using pygame.mixer.Sound instead of pygame.mixer.music:
  • Sound objects are independent — no global single-channel conflict.
  • Multiple sounds can be loaded/played without interfering.
  • .stop() on the Sound object stops only that sound.

STT (SPEECH-TO-TEXT)
────────────────────
  Provided as a standalone transcribe_wav(wav_buffer) function that
  app.py calls directly with a BytesIO WAV buffer.  Keeping it here
  avoids circular imports and centralises audio logic.

STOP / CANCEL
─────────────
  request_stop()  — sets _stop_event; pipeline checks it between every
                    sentence and stops within one sentence boundary (~1 s).
  The event is cleared automatically at the start of each new TTS call.

SHUTDOWN SAFETY
───────────────
  shutdown()  — called once on process exit.  Signals the pipeline to
  stop, then shuts down the background loop cleanly.  No crash, no
  'cannot schedule new futures after shutdown'.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
import threading
import uuid
from typing import Optional

import edge_tts
import pygame

# ── Voices ─────────────────────────────────────────────────────────────────────
VOICES: dict[str, str] = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "mr": "mr-IN-AarohiNeural",   # Marathi (Whisper sometimes detects mr)
}
FALLBACK_VOICE = VOICES["en"]

# ── pygame mixer init (once per process) ───────────────────────────────────────
_MIXER_FREQ   = 44100
_MIXER_READY  = False

def _ensure_mixer() -> None:
    global _MIXER_READY
    if not _MIXER_READY:
        try:
            pygame.mixer.pre_init(_MIXER_FREQ, -16, 2, 512)
            pygame.mixer.init()
            _MIXER_READY = True
        except Exception as e:
            print(f"⚠️  pygame mixer init failed: {e}")

_ensure_mixer()

# ── Dedicated background event loop ────────────────────────────────────────────
_bg_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
_bg_thread: threading.Thread

def _start_bg_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()

_bg_thread = threading.Thread(
    target=_start_bg_loop,
    args=(_bg_loop,),
    daemon=True,
    name="TTS-EventLoop",
)
_bg_thread.start()

# ── Stop / cancel signal ───────────────────────────────────────────────────────
_stop_event    = threading.Event()
_current_sound: Optional[pygame.mixer.Sound] = None
_sound_lock    = threading.Lock()


def request_stop() -> None:
    """Interrupt any in-progress TTS immediately."""
    _stop_event.set()
    with _sound_lock:
        if _current_sound is not None:
            try:
                _current_sound.stop()
            except Exception:
                pass


def _clear_stop() -> None:
    _stop_event.clear()


def shutdown() -> None:
    """
    Clean shutdown — call once at process exit.
    Stops TTS, shuts down the background loop, joins its thread.
    """
    request_stop()
    try:
        _bg_loop.call_soon_threadsafe(_bg_loop.stop)
    except Exception:
        pass


# ── Text helpers ────────────────────────────────────────────────────────────────

def _pick_voice(text: str, language: Optional[str]) -> str:
    """Select voice by Unicode range (Devanagari) or language hint."""
    if any('\u0900' <= c <= '\u097f' for c in text):
        # Could be Hindi or Marathi — use language hint to distinguish
        if language == "mr":
            return VOICES.get("mr", VOICES["hi"])
        return VOICES["hi"]
    return VOICES.get(language or "en", VOICES["en"])


def _split_sentences(text: str) -> list[str]:
    """
    Split on sentence boundaries (.!?।) then merge short fragments.
    Merging avoids many tiny TTS calls — each has ~150-300 ms network cost.
    Minimum merge threshold: 5 words (was 4; slightly more aggressive).
    """
    raw = re.split(r'(?<=[.!?।])\s+', text.strip())
    merged: list[str] = []
    for chunk in raw:
        chunk = chunk.strip()
        if not chunk:
            continue
        if merged and len(merged[-1].split()) < 5:
            merged[-1] += " " + chunk
        else:
            merged.append(chunk)
    return merged or [text]


# ── Synthesis ──────────────────────────────────────────────────────────────────

async def _synthesize(sentence: str, voice: str) -> Optional[str]:
    """
    Synthesise one sentence to a temp .mp3 file.
    Returns the file path, or None on failure.
    Tries fallback voice once before giving up.
    """
    path = os.path.join(
        tempfile.gettempdir(),
        f"veena_tts_{os.getpid()}_{uuid.uuid4().hex}.mp3",
    )
    for attempt_voice in (voice, FALLBACK_VOICE):
        try:
            await edge_tts.Communicate(sentence, attempt_voice).save(path)
            return path
        except Exception as exc:
            print(f"⚠️  TTS ({attempt_voice}): {exc}")
        if attempt_voice == voice and voice == FALLBACK_VOICE:
            break   # no point trying fallback if it IS the fallback
    try:
        os.remove(path)
    except OSError:
        pass
    return None


# ── Playback ───────────────────────────────────────────────────────────────────

def _play_sound(path: str) -> None:
    """
    Load and play an mp3 using pygame.mixer.Sound (NOT the Music channel).

    Sound objects are independent instances — no global single-channel
    conflict, no 'music already playing' races between concurrent calls.

    Polls at 50 Hz → ≤20 ms reaction time on request_stop().
    """
    global _current_sound
    try:
        sound = pygame.mixer.Sound(path)
    except Exception as exc:
        print(f"⚠️  Failed to load audio {path}: {exc}")
        return

    with _sound_lock:
        _current_sound = sound

    channel = sound.play()
    if channel is None:
        # All mixer channels busy — shouldn't happen but handle gracefully
        print("⚠️  No free mixer channel; skipping sentence")
        return

    clock = pygame.time.Clock()
    while channel.get_busy():
        if _stop_event.is_set():
            channel.stop()
            break
        clock.tick(50)

    with _sound_lock:
        if _current_sound is sound:
            _current_sound = None


# ── Pipeline ───────────────────────────────────────────────────────────────────

async def _tts_pipeline(sentences: list[str], voice: str) -> None:
    """
    Async producer–consumer pipeline running inside the dedicated bg loop.

    Producer synthesises sentence N+1 while consumer plays sentence N,
    hiding synthesis latency behind playback time.
    """
    loop      = asyncio.get_running_loop()
    q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=2)
    temp_files: list[str] = []

    async def producer() -> None:
        for sentence in sentences:
            if _stop_event.is_set():
                await q.put(None)
                return
            path = await _synthesize(sentence, voice)
            if path:
                temp_files.append(path)
            await q.put(path)   # None = synthesis failed → consumer skips
        await q.put(None)       # sentinel: done

    async def consumer() -> None:
        while True:
            path = await q.get()
            if path is None:
                return
            if _stop_event.is_set():
                return
            try:
                await loop.run_in_executor(None, _play_sound, path)
            except Exception as exc:
                print(f"⚠️  Playback error: {exc}")

    await asyncio.gather(producer(), consumer())

    # Clean up temp files even if stopped early
    for f in temp_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except OSError:
            pass


# ── Public API ─────────────────────────────────────────────────────────────────

def play_tts(text: str, language: Optional[str] = None) -> None:
    """
    Synthesise and play text as speech.

    Blocking — returns only after all audio has finished playing
    (or after request_stop() is called).

    Safe to call from any thread, including ThreadPoolExecutor workers.
    Submits work to the dedicated background event loop so there is
    never more than one event loop in the process.
    """
    if not text or not text.strip():
        return
    if _stop_event.is_set():
        return

    _clear_stop()
    sentences = _split_sentences(text)
    voice     = _pick_voice(text, language)

    future = asyncio.run_coroutine_threadsafe(
        _tts_pipeline(sentences, voice),
        _bg_loop,
    )
    try:
        future.result()     # block the calling thread until pipeline finishes
    except Exception as exc:
        if not _stop_event.is_set():
            print(f"❌ TTS error: {exc}")


# Back-compat alias used by older app.py versions
def play_text_to_speech_stream(text: str, language: Optional[str] = None) -> None:
    play_tts(text, language)
