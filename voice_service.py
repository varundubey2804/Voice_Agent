"""
voice_service.py — Low-Latency Streaming TTS

KEY IMPROVEMENTS OVER PREVIOUS VERSION
───────────────────────────────────────
1. SENTENCE-LEVEL STREAMING  ← biggest latency win
   - Response text is split into sentences; each sentence is synthesised
     and played back-to-back instead of waiting for the full paragraph.
   - Time-to-first-audio drops from ~2-4 s → ~0.3-0.8 s.

2. ASYNC PIPELINE (generate-while-playing)
   - Sentence N+1 is generated while sentence N is still playing via an
     asyncio queue between the producer coroutine and the pygame playback
     loop.  Net gain: near-zero inter-sentence gap.

3. LIVE STREAMING FROM THE LLM  ← new in this version
   - speak_sentences_live_async() consumes an ASYNC ITERATOR of sentences
     (e.g. streaming_llm.stream_reply()) instead of a pre-split list.
   - This means synthesis of sentence 1 can start while the LLM is still
     generating sentence 4 — the LLM-generation and TTS-synthesis stages
     now overlap too, not just synthesis-and-playback.
   - play_text_to_speech_stream() (the old full-text entry point) still
     works exactly as before — it's now a thin wrapper that turns a
     complete string into a one-shot async iterator and feeds the same
     underlying pipeline. Use it for non-streamed content (tool outputs,
     CrisisResources text, etc.) where you already have the full string.

4. GRACEFUL CANCELLATION
   - A threading.Event (`_stop_event`) lets the caller interrupt TTS mid-
     stream (barge-in).  The audio consumer checks the flag between
     sentences, so playback stops cleanly within one sentence boundary.

5. ROBUST TEMP-FILE CLEANUP
   - Files are tracked in a list; cleanup runs even if pygame raises.
   - Names include PID so parallel processes never collide.

6. MIXER RE-INIT GUARD
   - Mixer is initialised once at module level; re-init is skipped if
     it is already running.

7. VOICE FALLBACK
   - If edge_tts raises for a voice, we fall back to the English voice
     rather than crashing the whole response.

8. CROSS-PLATFORM AUDIO DEVICE MESSAGING
   - Device-selection log messages no longer assume Windows; same code
     path works on Linux (ALSA/PulseAudio/PipeWire via SDL) and Windows.
   - See the Linux setup notes / setup_linux.sh for required system
     packages — pygame's mixer needs SDL2 + an MP3-capable SDL_mixer
     backend, which isn't always present out of the box on a minimal
     Arch install.
"""

import asyncio
import os
import re
import time
import uuid
import tempfile
import threading
import unicodedata
from typing import AsyncIterator, Optional

import edge_tts
import pygame
from dotenv import load_dotenv

load_dotenv()

AUDIO_OUTPUT_AVAILABLE = False

# ── Voices ────────────────────────────────────────────────────────────────────
# Maps ISO-639-1 language codes → edge-tts voice names.
# Add more entries here as needed; voice names from: edge-tts --list-voices
VOICES = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",       # Telugu
    "ta": "ta-IN-PallaviNeural",      # Tamil
    "kn": "kn-IN-SapnaNeural",        # Kannada
    "ml": "ml-IN-SobhanaNeural",      # Malayalam
    "bn": "bn-IN-TanishaaNeural",     # Bengali
    "mr": "mr-IN-AarohiNeural",       # Marathi
    "gu": "gu-IN-DhwaniNeural",       # Gujarati
    "pa": "pa-IN-Ojaswineural",       # Punjabi
    "ur": "ur-IN-GulNeural",          # Urdu
    "zh": "zh-CN-XiaoxiaoNeural",     # Chinese (Mandarin)
    "fr": "fr-FR-DeniseNeural",       # French
    "de": "de-DE-KatjaNeural",        # German
    "es": "es-ES-ElviraNeural",       # Spanish
    "ar": "ar-SA-ZariyahNeural",      # Arabic
}
FALLBACK_VOICE = VOICES["en"]
TTS_REQUEST_TIMEOUT_S = 20
TTS_PLAYBACK_TIMEOUT_S = 30
# Optional explicit device name (e.g. a PulseAudio/PipeWire sink name on
# Linux, or a Windows device name). Leave unset to use the system default.
TTS_AUDIO_DEVICE = os.environ.get("TTS_AUDIO_DEVICE") or None

# Unicode script ranges → language code (used for script-based auto-detection)
_SCRIPT_RANGES: list[tuple[str, str, str]] = [
    ("\u0C00", "\u0C7F", "te"),   # Telugu
    ("\u0B80", "\u0BFF", "ta"),   # Tamil
    ("\u0C80", "\u0CFF", "kn"),   # Kannada
    ("\u0D00", "\u0D7F", "ml"),   # Malayalam
    ("\u0980", "\u09FF", "bn"),   # Bengali
    ("\u0900", "\u097F", "hi"),   # Devanagari (Hindi / Marathi / Sanskrit)
    ("\u0A80", "\u0AFF", "gu"),   # Gujarati
    ("\u0A00", "\u0A7F", "pa"),   # Gurmukhi (Punjabi)
    ("\u0600", "\u06FF", "ur"),   # Arabic/Urdu
    ("\u4E00", "\u9FFF", "zh"),   # CJK (Chinese)
    ("\u0600", "\u06FF", "ar"),   # Arabic
]

# ── Mixer init (once per process) ────────────────────────────────────────────
def _init_mixer() -> bool:
    """
    Initialise pygame's audio mixer once per process. Works the same way on
    Linux (SDL routes to PulseAudio/PipeWire/ALSA) and Windows (SDL routes
    to WASAPI/DirectSound) — no platform branching needed here, the
    platform-specific part is which SDL backend is installed on the system
    (see setup_linux.sh).

    IMPORTANT: devicename is passed directly to init(), never via
    pre_init(). pygame.mixer.pre_init(devicename=...) sets that name as a
    PERSISTENT default for every future init() call in the process —
    calling mixer.quit() does NOT clear it. If TTS_AUDIO_DEVICE names a
    device that doesn't exist here (e.g. a name copied over from a
    different machine's .env file), a pre_init-based fallback would keep
    silently retrying that same broken device name and fail a second time
    too. Passing it straight to init() means a plain init() retry below is
    guaranteed to use real, unqualified defaults.
    """
    global AUDIO_OUTPUT_AVAILABLE
    if pygame.mixer.get_init():
        AUDIO_OUTPUT_AVAILABLE = True
        return True

    try:
        if TTS_AUDIO_DEVICE:
            pygame.mixer.init(devicename=TTS_AUDIO_DEVICE)
        else:
            pygame.mixer.init()
        selected_device = TTS_AUDIO_DEVICE or "system default output"
        print(f"TTS output: {selected_device}")
        AUDIO_OUTPUT_AVAILABLE = True
        return True
    except pygame.error as exc:
        if TTS_AUDIO_DEVICE:
            print(f"TTS device '{TTS_AUDIO_DEVICE}' unavailable: {exc}")
            print("Falling back to the system default output device")
            try:
                pygame.mixer.quit()
                pygame.mixer.init()  # no devicename kwarg — true system default
                print("TTS output: system default output")
                AUDIO_OUTPUT_AVAILABLE = True
                return True
            except pygame.error as exc2:
                print(f"System default output also failed: {exc2}")

        print("No audio device available for TTS; continuing without spoken output.")
        AUDIO_OUTPUT_AVAILABLE = False
        return False


AUDIO_OUTPUT_AVAILABLE = _init_mixer()

# ── Stop signal (set externally to cancel mid-stream TTS) ────────────────────
_stop_event = threading.Event()


def request_stop() -> None:
    """Call this from audio_consumer to implement barge-in."""
    _stop_event.set()


def _clear_stop() -> None:
    _stop_event.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick_voice(text: str, language: Optional[str]) -> str:
    """
    Choose the best TTS voice:
    1. Try to auto-detect the script from Unicode character ranges in the text.
    2. Fall back to the Whisper-detected language code hint.
    3. Final fallback → English.

    NOTE on live streaming: when called from the streaming path, `text` is
    only the FIRST sentence (we haven't generated the rest yet). That's
    fine in practice — a reply doesn't switch scripts mid-stream given the
    persona's strict "mirror the user's language" instruction — but if you
    ever see wrong-voice output on streamed replies, prefer passing an
    explicit `language` hint (from STT) rather than relying on script
    detection of a short first sentence.
    """
    script_counts: dict[str, int] = {}
    for ch in text:
        for lo, hi, lang in _SCRIPT_RANGES:
            if lo <= ch <= hi:
                script_counts[lang] = script_counts.get(lang, 0) + 1
                break
    if script_counts:
        dominant = max(script_counts, key=lambda k: script_counts[k])
        if dominant in VOICES:
            return VOICES[dominant]
    if language and language in VOICES:
        return VOICES[language]
    return VOICES["en"]


def _speech_text(text: str) -> str:
    """
    Remove visual formatting and symbols that should not be spoken aloud.

    NOTE: apostrophes (straight ' and curly ’) are deliberately kept. They
    fall under Unicode category "Po" (Punctuation, other), same as commas
    and periods, so the old exception list (".!?।,") stripped them like
    any other punctuation — turning "don't" into "don t" whenever it hit
    the TTS engine. Contractions are extremely common in spoken replies,
    so apostrophes are treated as content, not formatting.
    """
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("So") or category.startswith("Sk"):
            cleaned.append(" ")
        elif category.startswith("P") and char not in ".!?।,'’":
            cleaned.append(" ")
        else:
            cleaned.append(char)
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentence-sized chunks for streaming playback.
    Uses a simple regex; handles Hindi too (splits on । as well).
    Chunks under 4 chars are merged with the previous chunk to avoid
    generating TTS audio for stray punctuation.

    Used only by the full-text entry point (play_text_to_speech_stream).
    The live-streaming entry point gets pre-split sentences directly from
    streaming_llm.stream_reply(), which uses matching boundary logic.
    """
    text = _speech_text(text)
    if not text:
        return []
    raw = re.split(r'(?<=[.!?।])\s+', text)
    merged: list[str] = []
    for chunk in raw:
        chunk = chunk.strip()
        if not chunk:
            continue
        if merged and len(merged[-1]) < 4:
            merged[-1] += " " + chunk
        else:
            merged.append(chunk)
    return merged or [text]


async def _synthesize_to_file(sentence: str, voice: str) -> Optional[str]:
    """
    Synthesise one sentence → temp .mp3 file path.
    Returns None on failure (caller skips the sentence).
    """
    tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, f"tts_{os.getpid()}_{uuid.uuid4().hex}.mp3")
    try:
        communicate = edge_tts.Communicate(sentence, voice)
        await asyncio.wait_for(communicate.save(path), timeout=TTS_REQUEST_TIMEOUT_S)
        return path
    except Exception as exc:
        print(f"⚠️  TTS synthesis error ({voice}): {exc}")
        if voice != FALLBACK_VOICE:
            try:
                communicate = edge_tts.Communicate(sentence, FALLBACK_VOICE)
                await asyncio.wait_for(communicate.save(path), timeout=TTS_REQUEST_TIMEOUT_S)
                return path
            except Exception as exc2:
                print(f"⚠️  TTS fallback also failed: {exc2}")
        if os.path.exists(path):
            os.remove(path)
        return None


def _play_file(path: str) -> None:
    """
    Load and play an mp3 synchronously; poll at 20 Hz so we can honour
    _stop_event between poll ticks (≤50 ms reaction time).
    """
    if not AUDIO_OUTPUT_AVAILABLE:
        print("⚠️  Skipping TTS playback because no audio device is available.")
        return

    started_at = time.monotonic()
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            if _stop_event.is_set():
                break
            if time.monotonic() - started_at >= TTS_PLAYBACK_TIMEOUT_S:
                print("⚠️  TTS playback timed out; continuing to the next turn")
                break
            clock.tick(20)
    finally:
        pygame.mixer.music.stop()
        try:
            pygame.mixer.music.unload()
        except pygame.error:
            pass


async def _pipeline_stream(sentence_aiter: AsyncIterator[str], voice: str) -> None:
    """
    Shared async producer–consumer pipeline. This is now the ONE place
    that does synthesis+playback overlap — both the full-text entry point
    and the live-LLM-streaming entry point funnel through here.

    - Producer coroutine pulls the next sentence from `sentence_aiter`
      (which may itself still be waiting on LLM tokens or edge_tts) and
      pushes its synthesised audio file onto a queue.
    - Consumer coroutine plays files back-to-back in a thread-pool
      executor so the event loop stays free to keep producing.
    - A small asyncio.Queue(maxsize=2) decouples them: at most one
      sentence is buffered ahead of what's currently playing.
    """
    sentinel = object()
    queue: asyncio.Queue[object] = asyncio.Queue(maxsize=2)
    loop = asyncio.get_running_loop()
    temp_files: list[str] = []

    async def producer():
        async for sentence in sentence_aiter:
            if _stop_event.is_set():
                break
            # _speech_text is idempotent, so this is safe even for sentences
            # that already went through it upstream (the full-text path via
            # _split_sentences). The live-streaming path (speak_sentences_
            # live_async, fed by streaming_llm.stream_reply) never ran
            # _speech_text at all before this point — sentences came
            # straight from the LLM's token stream — so this is the only
            # place that guarantees BOTH entry points reach edge_tts with
            # the same symbol/formatting cleanup applied.
            spoken = _speech_text(sentence)
            if not spoken:
                continue
            path = await _synthesize_to_file(spoken, voice)
            if path:
                temp_files.append(path)
                await queue.put(path)
        await queue.put(sentinel)

    async def consumer():
        while True:
            path = await queue.get()
            if path is sentinel:
                break
            if _stop_event.is_set():
                break
            try:
                await loop.run_in_executor(None, _play_file, path)
            except Exception as exc:
                print(f"⚠️  Playback error: {exc}")

    await asyncio.gather(producer(), consumer())

    for f in temp_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except OSError:
            pass


async def _alist_iter(items: list[str]) -> AsyncIterator[str]:
    """Wrap a plain list as a one-shot async iterator (for the full-text path)."""
    for item in items:
        yield item


async def _achain_first(first: str, rest: AsyncIterator[str]) -> AsyncIterator[str]:
    """Re-attach a peeked-at first item to the front of an async iterator."""
    yield first
    async for item in rest:
        yield item


# ── Public entry point 1: FULL TEXT already known (unchanged behaviour) ──────

def play_text_to_speech_stream(text: str, language: Optional[str] = None) -> None:
    """
    Public entry point for content you already have in full (tool outputs,
    CrisisResources text, agent-path replies that went through the ReAct
    loop rather than fast_reply). Splits into sentences up front, then
    plays through the same overlapping synth+playback pipeline.

    Call this from a ThreadPoolExecutor / plain thread — it manages its
    own event loop via asyncio.run() and is NOT safe to call from inside
    an already-running event loop (use play_text_to_speech_stream_async
    for that case, e.g. from orchestrator.py).
    """
    _clear_stop()
    print("🔊 Starting voice playback...")

    sentences = _split_sentences(text)
    if not sentences:
        print("⚠️  No speakable text after removing formatting")
        return
    voice = _pick_voice(text, language)

    try:
        asyncio.run(_pipeline_stream(_alist_iter(sentences), voice))
    except Exception as exc:
        print(f"❌ TTS pipeline error: {exc}")
    finally:
        print("✅ Voice playback finished")


# ── Public entry point 2: LIVE sentences arriving from the LLM stream ───────

async def speak_sentences_live_async(
    sentence_aiter: AsyncIterator[str], language: Optional[str] = None
) -> None:
    """
    ASYNC-NATIVE entry point — call this with `await` from code that's
    already running inside an event loop (this is what orchestrator.py
    does, since it's also awaiting streaming_llm.stream_reply()).

    `sentence_aiter` is any async iterator of sentence strings, most
    commonly streaming_llm.stream_reply(...) directly. We peek the first
    item to pick a voice (script-detection needs at least one sentence),
    then stitch it back onto the front of the stream before handing the
    whole thing to the shared pipeline.
    """
    _clear_stop()
    print("🔊 Starting live voice playback...")

    aiter = sentence_aiter.__aiter__()
    try:
        first = await aiter.__anext__()
    except StopAsyncIteration:
        print("⚠️  No sentences produced — nothing to speak")
        return

    voice = _pick_voice(first, language)
    chained = _achain_first(first, aiter)

    try:
        await _pipeline_stream(chained, voice)
    except Exception as exc:
        print(f"❌ TTS live pipeline error: {exc}")
    finally:
        print("✅ Voice playback finished")


def speak_sentences_live(
    sentence_aiter: AsyncIterator[str], language: Optional[str] = None
) -> None:
    """
    SYNC wrapper around speak_sentences_live_async, for callers NOT already
    inside an event loop (mirrors play_text_to_speech_stream's calling
    convention). Prefer the _async version directly when you're already in
    async code — this just does asyncio.run() around it.
    """
    try:
        asyncio.run(speak_sentences_live_async(sentence_aiter, language))
    except Exception as exc:
        print(f"❌ TTS live pipeline error: {exc}")