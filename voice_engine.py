"""
voice_engine.py — Headless voice-to-voice agent runtime. No UI, no HTTP,
no websockets. Pure mic-in -> agent -> speaker-out, running in the terminal.

WHAT CHANGED FROM THE OLD app.py
─────────────────────────────────
1. NO UI AT ALL
   - Removed the HTTP static-file server, the frontend dir, and the
     websocket broadcast layer entirely. This is a terminal process.

2. THREADS (CHATS), ENCRYPTED, KEYED
   - Every run is a "thread" (a chat). New process -> new thread_id
     unless you resume one with --thread <id>.
   - All messages are written through thread_store.ThreadStore, which
     encrypts (Fernet) the whole transcript at rest in SQLite. You can
     only get a thread's content back by knowing its thread_id (the
     lookup key) *and* holding VOICE_MASTER_KEY.
   - Resuming a thread re-seeds the agent's memory from the decrypted
     history so the conversation genuinely continues.

3. VOICE-ACTIVITY-DETECTION RECORDING (biggest latency win)
   - The old version always recorded a fixed DEFAULT_CHUNK_LENGTH (4s)
     before transcribing, even if you finished your sentence in 1s.
   - This version streams 30ms frames and cuts the utterance as soon as
     it detects ~500ms of trailing silence after speech. For short
     utterances this alone removes 2-3s of dead latency per turn.
   - Uses webrtcvad if installed (far more accurate); falls back to an
     RMS-energy VAD automatically if it isn't, so nothing hard-crashes.

4. GENERIC AGENT
   - Persona/tools no longer hardcoded — loaded from a JSON config via
     generic_agent.build_agent(). Point --config at any persona file.

5. TERMINAL THREAD CONTROLS (typed while the voice loop runs)
   - Type `new`     -> start a fresh thread, keep listening
   - Type `resume X`-> switch to thread X (must exist)
   - Type `threads` -> list known threads
   - Type `quit`    -> stop everything
"""

from __future__ import annotations

import argparse
import io
import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional

import numpy as np
import pyaudio
import soundfile as sf
from faster_whisper import WhisperModel

import asyncio

import voice_service as vs
import wellness_tools
from generic_agent import build_agent, fast_reply, _needs_tool
from orchestrator import speak_streaming
from thread_store import ThreadStore
from emotion_trigger_server import start_emotion_trigger_server
from emotion.sentiment_analysis import analyze_and_trigger
from emotion.bulb_trigger import handle_bulb_trigger
from user_profile import UserProfileStore, ONBOARDING_QUESTIONS, profile_summary_text
import mental_health_qa

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Tunables ──────────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000
FRAME_MS          = 30                       # webrtcvad requires 10/20/30ms frames
FRAME_SAMPLES     = SAMPLE_RATE * FRAME_MS // 1000
# How long a trailing gap has to be, after speech, before we treat the
# utterance as finished. Snappier than the old 900ms default so turns feel
# more like natural back-and-forth. Override with SILENCE_TAIL_MS if it
# still feels too eager/laggy for your room.
SILENCE_TAIL_MS   = int(os.environ.get("SILENCE_TAIL_MS", "600"))
MAX_UTTERANCE_S   = int(os.environ.get("MAX_UTTERANCE_S", "20"))  # hard cap so a stuck mic can't hang forever
MIN_UTTERANCE_MS  = 300                      # ignore blips shorter than this
MIN_SPEECH_RATIO  = 0.25                     # segment must be at least this much actual speech, or it's noise
# RMS fallback threshold (only used if webrtcvad isn't installed). If the
# mic keeps "listening" for the full MAX_UTTERANCE_S instead of stopping
# shortly after you finish, it is almost always because background noise
# is registering as speech and keeps resetting the trailing-silence
# counter — raise this (stricter) rather than lower it. Override with
# RMS_SILENCE_THRESH if your room is unusually noisy or quiet.
RMS_SILENCE_THRESH = int(os.environ.get("RMS_SILENCE_THRESH", "700"))
# webrtcvad aggressiveness: 0 (most permissive, picks up more as "speech")
# to 3 (strictest, filters out more background noise). Bumped from 2 to 3
# by default for the same reason as above — a stricter classifier means
# room noise is more reliably called "silence," so the trailing-silence
# timer actually gets to elapse instead of stalling out at the hard cap.
# Override with VAD_AGGRESSIVENESS if 3 clips the start/end of your words.
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "3"))
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
WHISPER_TIMEOUT_S = 30
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE") or None

# Whisper is trained on huge amounts of YouTube caption data, so on quiet or
# ambiguous audio it sometimes "completes" the silence with caption-outro
# phrases instead of returning nothing. Filter those out.
_HALLUCINATION_PHRASES = {
    "thanks for watching!", "thanks for watching", "thank you for watching",
    "thank you for watching!", "please subscribe", "subscribe to my channel",
    "like and subscribe", "see you next time", "see you in the next video",
    "bye bye", "thank you.", "you", "the end",
}

# ── Proactive check-in prompts (hardware emotion trigger) ──────────────────
# Spoken by handle_emotion_trigger() when the hardware side detects a
# sustained negative emotion and Veena isn't already mid-conversation.
_CHECKIN_PROMPTS = {
    "en": "Hey, I noticed you seem a bit {emotion} — want to talk about it?",
    "hi": "अरे, मुझे लगा आप थोड़ा {emotion} महसूस कर रहे हैं — क्या आप इस बारे में बात करना चाहेंगे?",
    "te": "హే, మీరు కొంచెం {emotion}గా అనిపిస్తున్నారు — దీని గురించి మాట్లాడాలనుకుంటున్నారా?",
    "ta": "ஏய், நீங்கள் கொஞ்சம் {emotion} போல் தெரிகிறீர்கள் — இது பற்றி பேச விரும்புகிறீர்களா?",
}

# ── "Talk to a specialist" mode — hands the live conversation to the local
# fine-tuned mental-health GGUF model (mental_health_qa.chat_reply) instead
# of the OpenRouter fast path, so it can be demoed talking and asking its
# own follow-up questions directly. Toggled by voice, not a config flag,
# so it's demoable live without restarting anything.
#
# MATCHING: word-boundary regex on the key noun, not an exact-phrase list.
# The original exact-phrase list (kept below in _SPECIALIST_ON_PHRASES for
# reference/back-compat) missed completely natural ways of asking for it —
# e.g. "...would you recommend as a therapist? Specialist of course." never
# matched any of the fixed strings, so the mode silently never turned on
# even though the user clearly asked for it. Matching on the bare word
# "specialist" (or "therapist model") is looser and can false-positive on
# an unrelated mention of the word, but for a voice demo that's a far
# better trade-off than "the feature doesn't turn on."
_SPECIALIST_ON_PHRASES = (
    "talk to a specialist", "talk to the specialist", "specialist mode",
    "mental health specialist", "talk to the therapist model",
)
_SPECIALIST_ON_RE = re.compile(r"\b(specialist|therapist model)\b", re.IGNORECASE)
_SPECIALIST_OFF_PHRASES = (
    "switch back", "back to veena", "leave specialist mode",
    "stop specialist mode", "regular mode",
)

# Off by default — see the matching flag/comment in generic_agent.py. The
# local 7B GGUF model turned out not to be the source of the "exhausted
# reasoning budget" failures (that was the OpenRouter fast-path model
# doing hidden chain-of-thought), but it's still slow on modest hardware
# and not needed for a fast, context-aware, wellness/therapist-style
# conversational companion. Set ENABLE_MENTAL_HEALTH_SPECIALIST=1 to
# bring "talk to a specialist" mode back.
ENABLE_MENTAL_HEALTH_SPECIALIST = os.environ.get(
    "ENABLE_MENTAL_HEALTH_SPECIALIST", "false"
).strip().lower() in ("1", "true", "yes", "on")

# ── Standby / armed gate ────────────────────────────────────────────────────
# The engine does NOT touch the mic until the hardware side calls
# /emotion-trigger. That call plays a pre-recorded (or TTS-fallback) line
# instantly, then "arms" the mic so a real conversation can follow. If the
# user doesn't reply within IDLE_REARM_TIMEOUT_S, the engine goes back to
# standby and waits for the next hardware trigger instead of listening
# indefinitely.
PRERECORDED_CHECKIN_DIR = os.environ.get("PRERECORDED_CHECKIN_DIR", "prerecorded_checkins")
IDLE_REARM_TIMEOUT_S = int(os.environ.get("IDLE_REARM_TIMEOUT_S", "90"))


def _find_prerecorded_checkin(emotion: str, language: str) -> Optional[str]:
    """
    Look for a canned audio clip to play the instant a hardware trigger
    fires, so there's no TTS-generation latency on the very first line.
    Checked in order, most specific first:
        prerecorded_checkins/{language}_{emotion}.mp3
        prerecorded_checkins/{language}.mp3
        prerecorded_checkins/default.mp3
    Returns the path if one exists, else None (caller falls back to live TTS
    of _CHECKIN_PROMPTS).
    """
    for name in (f"{language}_{emotion}.mp3", f"{language}.mp3", "default.mp3"):
        path = os.path.join(PRERECORDED_CHECKIN_DIR, name)
        if os.path.isfile(path):
            return path
    return None


# ── VAD backends ──────────────────────────────────────────────────────────

class _WebRtcVad:
    def __init__(self, aggressiveness: int = 2):
        import webrtcvad
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame_bytes: bytes) -> bool:
        return self._vad.is_speech(frame_bytes, SAMPLE_RATE)


class _RmsVad:
    """Fallback when webrtcvad isn't installed. Less accurate but dependency-free."""

    def is_speech(self, frame_bytes: bytes) -> bool:
        data = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
        return rms >= RMS_SILENCE_THRESH


def _make_vad():
    try:
        vad = _WebRtcVad(aggressiveness=VAD_AGGRESSIVENESS)
        print(f"🎚️   VAD backend: webrtcvad (accurate, aggressiveness={VAD_AGGRESSIVENESS})")
        return vad
    except Exception:
        print(f"🎚️   VAD backend: RMS-energy fallback, threshold={RMS_SILENCE_THRESH} (install `webrtcvad` for better accuracy)")
        return _RmsVad()


class VoiceSegmenter:
    """
    Reads fixed-size frames from a PyAudio stream and yields complete
    utterances (as WAV BytesIO) as soon as trailing silence is detected,
    instead of waiting for a fixed-length window.
    """

    def __init__(self, stream, vad):
        self.stream = stream
        self.vad = vad
        self.silence_frames_needed = max(1, SILENCE_TAIL_MS // FRAME_MS)
        self.max_frames = (MAX_UTTERANCE_S * 1000) // FRAME_MS
        self.min_frames = max(1, MIN_UTTERANCE_MS // FRAME_MS)

    def _read_frame(self) -> bytes:
        return self.stream.read(FRAME_SAMPLES, exception_on_overflow=False)

    def next_utterance(self, should_abort) -> Optional[io.BytesIO]:
        """
        Blocks until one utterance is captured or should_abort() returns True.
        Returns a WAV BytesIO, or None if aborted / nothing usable was captured.
        """
        frames: list[bytes] = []
        speech_started = False
        trailing_silence = 0
        speech_frame_count = 0

        while True:
            if should_abort():
                return None

            frame = self._read_frame()
            is_speech = self.vad.is_speech(frame)

            if not speech_started:
                if is_speech:
                    speech_started = True
                    frames.append(frame)
                    speech_frame_count += 1
                # else: keep waiting, don't buffer silence before speech starts
                continue

            frames.append(frame)
            if is_speech:
                trailing_silence = 0
                speech_frame_count += 1
            else:
                trailing_silence += 1

            if trailing_silence >= self.silence_frames_needed:
                break
            if len(frames) >= self.max_frames:
                break

        if len(frames) < self.min_frames:
            return None

        # Reject segments that are mostly silence/noise with only a stray
        # speech-flagged frame — these are what feed Whisper's hallucinations.
        if (speech_frame_count / len(frames)) < MIN_SPEECH_RATIO:
            return None

        raw = b"".join(frames)
        audio_arr = np.frombuffer(raw, dtype=np.int16)
        buf = io.BytesIO()
        sf.write(buf, audio_arr, samplerate=SAMPLE_RATE, format="WAV")
        buf.seek(0)
        return buf


# ── Emotion / transcription (unchanged logic from app.py) ─────────────────

def _collapse_repeated_phrases(text: str, max_repeats: int = 2) -> str:
    """
    Whisper occasionally 'gets stuck' and repeats the same short clause
    many times in a row — a known decoding-loop artifact, most likely
    when temperature has no fallback ladder to escape it with (see the
    temperature fix in _transcribe_once). This is a defense-in-depth
    cleanup that runs regardless of why it happened: split on clause
    boundaries and collapse any clause repeated more than max_repeats
    times consecutively down to a single occurrence.
    """
    if not text:
        return text
    clauses = re.split(r"(?<=[.!?,।])\s+", text)
    out: list[str] = []
    last_norm: Optional[str] = None
    run = 0
    for clause in clauses:
        norm = clause.strip().lower().rstrip(".!?,।")
        if norm and norm == last_norm:
            run += 1
            if run < max_repeats:
                out.append(clause)
            # else: drop this extra repeat entirely
        else:
            last_norm = norm
            run = 1
            out.append(clause)
    return " ".join(out)


def detect_emotion(wav_buffer: io.BytesIO, text: str) -> str:
    wav_buffer.seek(0)
    data, sr = sf.read(wav_buffer)
    wav_buffer.seek(0)
    if data.ndim > 1:
        data = data[:, 0]
    duration = len(data) / sr if sr else 0
    words_per_sec = len(text.split()) / duration if duration > 0 else 0
    amplitude = float(np.max(np.abs(data))) if len(data) else 0.0

    if words_per_sec > 3 or amplitude > 0.8:
        return "stressed"
    elif words_per_sec < 1.5 and amplitude < 0.3:
        return "calm"
    elif any(w in text.lower() for w in ("confused", "how", "what", "?")):
        return "confused"
    return "neutral"


def _transcribe_once(model: WhisperModel, wav_buffer: io.BytesIO):
    wav_buffer.seek(0)
    started_at = time.monotonic()
    segments, info = model.transcribe(
        wav_buffer,
        language=WHISPER_LANGUAGE,
        beam_size=5,
        # A single float here (the old `temperature=0.0`) disables
        # faster-whisper's built-in fallback ladder entirely — normally,
        # when a decode looks bad (too repetitive / low confidence per
        # compression_ratio_threshold or logprob_threshold), it retries
        # at a higher temperature to escape a decoding loop. Forcing a
        # bare 0.0 removed that escape hatch, which is exactly how a
        # stuck decode turns into "I am doing fine" repeated 30 times.
        # A tuple restores the standard retry ladder.
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        # Extra decode-loop guards (faster-whisper >=0.10, confirmed
        # present in the locked 1.2.1): directly penalize/forbid exact
        # short-phrase repetition at the decoder level, as a second,
        # independent line of defense alongside the temperature ladder.
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        initial_prompt=(
            "This is a natural conversation with a wellness assistant. "
            "Common words may include namaste, namaskaram, hello, kaise ho, "
            "how are you, stress, tension, help, and thank you."
        ),
        vad_filter=False,                # VoiceSegmenter already removed leading/trailing silence
        no_speech_threshold=0.6,          # higher = more willing to say "no speech" instead of guessing
        condition_on_previous_text=False, # prevents hallucinated text from snowballing across segments
    )
    segments = list(segments)
    wav_buffer.seek(0)
    language_probability = getattr(info, "language_probability", 0.0)
    print(
        f"✅ Whisper returned in {time.monotonic() - started_at:.1f}s "
        f"(language={info.language}, confidence={language_probability:.2f})"
    )

    # Drop low-confidence segments — these are exactly the ones that tend to
    # be hallucinated filler ("Thanks for watching!") rather than real speech.
    kept = [s for s in segments if s.avg_logprob > -1.0 and s.no_speech_prob < 0.6]
    text = " ".join(s.text for s in kept).strip()
    # Defense in depth: even with the decode-level guards above, collapse
    # any clause that still comes back repeated several times in a row —
    # catches the failure mode regardless of which layer let it through.
    text = _collapse_repeated_phrases(text)

    if text.lower().strip(" .!?") in _HALLUCINATION_PHRASES:
        text = ""

    emotion = detect_emotion(wav_buffer, text) if text else "neutral"
    return text, info.language, emotion


def transcribe(model: WhisperModel, wav_buffer: io.BytesIO):
    """Transcribe one bounded audio clip and recover if Whisper stalls."""
    wav_buffer.seek(0)
    audio_data, _ = sf.read(wav_buffer, dtype="int16")
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    max_samples = SAMPLE_RATE * MAX_UTTERANCE_S
    audio_data = audio_data[:max_samples]
    bounded_buffer = io.BytesIO()
    sf.write(bounded_buffer, audio_data, samplerate=SAMPLE_RATE, format="WAV")
    bounded_buffer.seek(0)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_transcribe_once, model, bounded_buffer)
    try:
        return future.result(timeout=WHISPER_TIMEOUT_S)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(
            f"Whisper did not return within {WHISPER_TIMEOUT_S}s"
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def is_echo(transcription: str, ai_text: str) -> bool:
    if not transcription or not ai_text:
        return False
    t_words = set(transcription.lower().replace(".", "").replace(",", "").split())
    a_words = set(ai_text.lower().replace(".", "").replace(",", "").split())
    if not t_words:
        return False
    return (len(t_words & a_words) / len(t_words)) > 0.5


# ── Engine ──────────────────────────────────────────────────────────────

class VoiceEngine:
    def __init__(self, config_path: str, db_path: str, thread_id: Optional[str], persona_meta: dict):
        self.config_path = config_path
        self.store = ThreadStore(db_path)
        self.persona_meta = persona_meta

        # Persistent, cross-thread "who is this user" profile — separate
        # from ThreadStore because identity should survive across every
        # thread, not just this one conversation. See user_profile.py.
        self.profile_store = UserProfileStore(db_path)
        self.profile_text = profile_summary_text(self.profile_store.get())

        # "Talk to a specialist" mode — see _SPECIALIST_ON_PHRASES above.
        self.specialist_mode = False

        self.thread_id = thread_id or self.store.create_thread(meta=persona_meta)
        if not self.store.thread_exists(self.thread_id):
            print(f"⚠️  Thread '{self.thread_id}' not found — creating it fresh.")
            self.thread_id = self.store.create_thread(meta=persona_meta, thread_id=self.thread_id)

        self._load_agent_for_current_thread()

        self.is_speaking = False
        self.is_thinking = False
        self.current_ai_text = ""
        self.last_language = "en"          # updated every turn; used for proactive check-ins
        self._stop = threading.Event()
        self._switch_lock = threading.Lock()

        # Standby gate: cleared = mic is off / not listening. Only set after
        # a hardware emotion trigger opens a conversation window; cleared
        # again after IDLE_REARM_TIMEOUT_S of no reply.
        self.armed = threading.Event()
        self.last_activity = time.monotonic()

        print(f"🧵  Active thread: {self.thread_id}")

    def _load_agent_for_current_thread(self) -> None:
        history = self.store.get_history(self.thread_id)
        agent_executor, llm, cfg = build_agent(self.config_path, seed_messages=history)
        self.agent = agent_executor
        self.llm = llm
        self.agent_config = cfg

    def refresh_profile_text(self) -> None:
        """Call after profile_store.save()/reset() so the next turn sees the change."""
        self.profile_text = profile_summary_text(self.profile_store.get())

    # ── thread controls ────────────────────────────────────────────────
    def new_thread(self) -> str:
        with self._switch_lock:
            self.thread_id = self.store.create_thread(meta=self.persona_meta)
            self._load_agent_for_current_thread()
            print(f"🆕  New thread: {self.thread_id}")
            return self.thread_id

    def resume_thread(self, thread_id: str) -> bool:
        if not self.store.thread_exists(thread_id):
            print(f"❌  No such thread: {thread_id}")
            return False
        with self._switch_lock:
            self.thread_id = thread_id
            self._load_agent_for_current_thread()
            print(f"🔁  Resumed thread: {self.thread_id}")
            return True

    def list_threads(self) -> None:
        threads = self.store.list_threads()
        if not threads:
            print("   (no threads yet)")
            return
        for t in threads:
            marker = "→" if t["thread_id"] == self.thread_id else " "
            print(f" {marker} {t['thread_id']}   updated {t['updated_at']}")

    # ── core turn ──────────────────────────────────────────────────────
    def handle_user_text(self, user_text: str, language: str, emotion: str) -> None:
        with self._switch_lock:
            thread_id = self.thread_id
            agent = self.agent
            llm = self.llm

        print(f"🗣  [{language}/{emotion}] {user_text}")
        self.is_thinking = True
        self.last_language = language
        self.store.append_message(thread_id, "user", user_text, {"language": language, "emotion": emotion})

        # Fire the ambient IR/MacroDroid trigger for this turn's mood.
        # Non-blocking + wrapped defensively: hardware feedback should never
        # be able to break or delay the actual conversation.
        try:
            sentiment = analyze_and_trigger(user_text, blocking=False)
            print(f"🎨 Sentiment → {sentiment['label']} (compound={sentiment['compound']}, stress={sentiment['stress_level']})")
        except Exception as exc:
            print(f"⚠️  Sentiment/IR trigger failed (non-fatal): {exc}")

        _FALLBACK = {
            "hi": "माफ़ करें, मैं अभी ठीक से समझ नहीं पाई। क्या आप दोबारा कह सकते हैं?",
            "te": "క్షమించండి, నాకు సరిగ్గా అర్థం కాలేదు. మళ్ళీ చెప్పగలరా?",
            "ta": "மன்னிக்கவும், சரியாக புரியவில்லை. மீண்டும் சொல்ல முடியுமா?",
            "en": "I'm sorry, I couldn't catch that. Could you say it again?",
        }

        def _local_fallback() -> str:
            lower = user_text.lower()
            if any(term in lower for term in ("love me", "lonely", "alone", "no one", "nobody")):
                return {
                    "hi": "आपका किसी के प्यार और साथ की चाह रखना बिल्कुल स्वाभाविक है। आप अकेले नहीं हैं। अभी किसी भरोसेमंद व्यक्ति को एक छोटा संदेश भेजें: ‘मुझे थोड़ी बातचीत चाहिए।’",
                    "te": "ప్రేమ, తోడు కావాలని అనుకోవడం సహజమే. మీరు ఒంటరిగా లేరు. నమ్మకమైన వ్యక్తికి ‘నాకు కొద్దిసేపు మాట్లాడాలి’ అని ఒక చిన్న సందేశం పంపండి.",
                    "ta": "அன்பும் துணையும் வேண்டும் என்று நினைப்பது இயல்பானது. நீங்கள் தனியாக இல்லை. நம்பிக்கையான ஒருவருக்கு ‘சிறிது நேரம் பேச வேண்டும்’ என்று ஒரு குறுஞ்செய்தி அனுப்புங்கள்.",
                }.get(language, "Wanting love and companionship is completely human. You are not alone in feeling this. Could you message one trusted person and say, ‘I could use a little company today’?")
            return _FALLBACK.get(language, _FALLBACK["en"])

        # Set once the fast-path branch below has already spoken AND saved
        # the assistant turn itself (streaming speaks as it generates, so
        # by the time it returns there's nothing left for the shared
        # epilogue at the bottom to do). Crisis and explicit-tool replies
        # leave this False and fall through to the old speak-after-the-fact
        # path, unchanged.
        already_spoken = False

        try:
            stress_state = wellness_tools.assess_stress(user_text)
            lower_text = user_text.lower()

            # ── Crisis: ALWAYS takes priority, over every other branch below,
            # including specialist mode. No tool router needed.
            if wellness_tools.detect_crisis(user_text):
                response = wellness_tools.get_crisis_resources()

            # ── Specialist-mode toggle (voice-triggered, checked before the
            # mode's own branch so "switch back" while already in the mode
            # doesn't get sent to the local model as a chat turn) ──────────
            # All three branches below are gated behind
            # ENABLE_MENTAL_HEALTH_SPECIALIST (off by default) — with it
            # off, self.specialist_mode can never become True and saying
            # "specialist" just falls through to a normal conversational
            # reply instead of silently doing nothing.
            elif ENABLE_MENTAL_HEALTH_SPECIALIST and self.specialist_mode and any(
                p in lower_text for p in _SPECIALIST_OFF_PHRASES
            ):
                self.specialist_mode = False
                response = "Okay — switching back. I'm right here."

            elif (
                ENABLE_MENTAL_HEALTH_SPECIALIST
                and not self.specialist_mode
                and _SPECIALIST_ON_RE.search(lower_text)
            ):
                self.specialist_mode = True
                response = "Alright — you're now talking directly with our specialist model. Go ahead."

            # ── Specialist mode active: hand the live turn to the local
            # fine-tuned GGUF model instead of the fast OpenRouter path, so
            # it can be demoed genuinely conversing and asking its own
            # follow-up questions. ──────────────────────────────────────────
            elif ENABLE_MENTAL_HEALTH_SPECIALIST and self.specialist_mode:
                print("🧠 Specialist mode — routing to local mental-health model...")
                # Exclude the just-appended current-turn message (see the
                # same duplicate-history issue fixed in streaming_llm.py) —
                # the model gets it explicitly as the latest turn instead.
                history_for_chat = self.store.get_history(thread_id)[:-1]
                response = mental_health_qa.chat_reply(user_text, history_for_chat)
                if not response:
                    print("⚠️  Specialist model produced nothing — using a short fallback for this turn")
                    response = _local_fallback()

            # ── Explicit tool request: use the configured tools ───────────────
            elif _needs_tool(user_text):
                effective_emotion = "stressed" if stress_state["level"] in {"medium", "high"} else emotion
                result = agent.invoke({"input": user_text, "language": language, "emotion": effective_emotion})
                output = (result.get("output") or "").strip()
                response = output if output and "agent stopped" not in output.lower() else _local_fallback()

            # ── All other turns: configured persona, STREAMED LLM + TTS ───────
            # This is the fast, high-volume path, so it's the one worth
            # overlapping generation with speech. It speaks (and saves the
            # assistant turn) itself, rather than falling through to the
            # shared epilogue below.
            else:
                effective_emotion = "stressed" if stress_state["level"] in {"medium", "high"} else emotion

                self.is_thinking = False
                self.is_speaking = True
                self.current_ai_text = ""  # filled in incrementally, see _on_sentence

                def _on_sentence(sentence: str) -> None:
                    # Keeps current_ai_text usable for the is_echo() barge-in
                    # check WHILE she's still talking, not just after the
                    # whole reply is done.
                    self.current_ai_text = (self.current_ai_text + " " + sentence).strip()

                # NOTE: persona_prompt was previously never passed here, so
                # this fast path silently used generic_agent's generic
                # _FAST_SYSTEM_PROMPT instead of the configured Veena
                # persona (emotion rules, itihasa-seasoning guidance, etc.)
                # from wellness_config.json — that only ever reached the
                # slower ReAct tool path. Fixed below, and the user's
                # profile (if onboarding has run) rides along with it.
                persona_prompt = self.agent_config.get("persona_prompt", "")
                if self.profile_text:
                    persona_prompt = f"{persona_prompt}\n\n{self.profile_text}"

                print("🔊 Streaming response to TTS as it's generated...")
                try:
                    response = asyncio.run(
                        speak_streaming(
                            llm, user_text, language, effective_emotion,
                            thread_id, self.store,
                            persona_prompt=persona_prompt,
                            on_sentence=_on_sentence,
                        )
                    )
                finally:
                    self.is_speaking = False
                    self.current_ai_text = ""

                if not response:
                    # Stream produced nothing usable. Before assuming the
                    # worst, retry once with the old blocking fast_reply()
                    # call — an empty streamed reply is most often a
                    # transient hiccup (rate limit, a genuinely empty
                    # completion from the model on that one call) rather
                    # than anything wrong with the conversation itself.
                    # Falling straight to the canned "I'm sorry, I
                    # couldn't catch that" here is what made this look
                    # like Veena forgot context/her name etc., when the
                    # actual failure was just that one generation call
                    # coming back empty.
                    print("⚠️  Streamed reply was empty — retrying once with the non-streamed fast path")
                    recent_history = self.store.get_history(thread_id)
                    response = fast_reply(
                        llm, user_text, language, effective_emotion,
                        recent_history, persona_prompt=persona_prompt,
                    )

                    if not response:
                        # Still nothing after the retry — now genuinely
                        # fall back to a short canned reply.
                        response = (
                            wellness_tools.stress_support(language)
                            if stress_state["level"] in {"medium", "high"}
                            else _local_fallback()
                        )

                    # Whichever of the two we ended up with (retry success
                    # or final canned fallback), this is the ONLY place
                    # that speaks/saves it — speak_streaming above never
                    # spoke anything in this branch (that's exactly what
                    # "response" being empty means), so there is no
                    # double-speak risk here.
                    self.store.append_message(thread_id, "assistant", response, {"language": language})
                    self.is_speaking = True
                    self.current_ai_text = response
                    try:
                        vs.play_text_to_speech_stream(response, language)
                    finally:
                        self.is_speaking = False
                        self.current_ai_text = ""
                # else: response was non-empty straight from speak_streaming
                # — it has ALREADY been spoken (live, sentence-by-sentence)
                # and saved by orchestrator.speak_streaming itself. Nothing
                # further to do here — this is the bug fixed above: this
                # branch used to run an unconditional second playback.

                print(f"🤖 {response}")
                already_spoken = True

        except Exception as exc:
            print(f"❌ Agent error: {exc}")
            response = _local_fallback()

        self.is_thinking = False

        if not already_spoken:
            print(f"🤖 {response}")
            self.store.append_message(thread_id, "assistant", response, {"language": language})

            self.is_speaking = True
            self.current_ai_text = response
            try:
                print("🔊 Sending response to TTS...")
                vs.play_text_to_speech_stream(response, language)
            finally:
                self.is_speaking = False
                self.current_ai_text = ""

        print("✅ Turn complete; microphone is ready")

    # ── proactive check-in (hardware emotion trigger) ────────────────────
    def _build_personalized_checkin(self, llm, thread_id: str, language: str, emotion: str) -> tuple[str, bool]:
        """
        Build a check-in line that references something concrete the user
        actually said earlier ("earlier you mentioned your promotion —
        you seem a bit off now, want to talk about it?") instead of the
        generic canned template. Returns (text, is_personalized).

        Falls back to the static _CHECKIN_PROMPTS template — and reports
        is_personalized=False — whenever there's no history yet (nothing
        to reference) or generation fails for any reason. A proactive
        check-in must never crash or hang the hardware-trigger path, so
        every failure mode here degrades to the old, always-safe line.
        """
        history = self.store.get_history(thread_id)
        if not history:
            return _CHECKIN_PROMPTS.get(language, _CHECKIN_PROMPTS["en"]).format(emotion=emotion), False

        checkin_persona = (
            "You are Veena, a warm mental wellness companion, speaking "
            "OUT LOUD. The user's detected mood just shifted to "
            f"'{emotion}'. Write EXACTLY ONE short, natural, spoken "
            "check-in line: reference something concrete and specific "
            "they actually said earlier in this conversation below (a "
            "name, an event, a feeling they mentioned) — not a generic "
            "phrase — then gently ask if they want to talk about how "
            "they're doing right now. Under 20 words total. Plain "
            "spoken language, one sentence, no lists, no quotation "
            "marks, no colons, no asterisks."
        )
        try:
            line = fast_reply(
                llm,
                "(Proactive check-in — no new message from the user. "
                "Generate the check-in line now, based on the "
                "conversation history above.)",
                language, emotion, history, persona_prompt=checkin_persona,
            )
        except Exception as exc:
            print(f"⚠️  Personalized check-in generation failed (non-fatal): {exc}")
            line = ""

        line = (line or "").strip()
        if not line:
            return _CHECKIN_PROMPTS.get(language, _CHECKIN_PROMPTS["en"]).format(emotion=emotion), False
        return line, True

    def handle_emotion_trigger(self, emotion: str) -> bool:
        """
        Called from the emotion-trigger HTTP server (background thread) for
        BOTH senders that hit this one endpoint: the hardware mic side and
        the vision pipeline's emotion_sender.py. Two independent things
        happen here:

          1. Bulb/IR trigger — fires UNCONDITIONALLY, cooldown-guarded via
             emotion.bulb_trigger. Mood lighting shouldn't wait on whether
             Veena happens to be mid-conversation.
          2. Spoken check-in — only if Veena isn't already speaking/thinking
             (existing "no queueing" rule, unchanged). If busy, this half
             is dropped entirely; the bulb still fires above regardless.

        Returns True if she SPOKE, False if the spoken half was ignored —
        this only reflects part (2), matching the original contract that
        emotion_trigger_server.py's on_trigger callable expects.
        """
        try:
            bulb_fired = handle_bulb_trigger(emotion)
            print(f"💡 Bulb trigger {'fired' if bulb_fired else 'skipped (cooldown)'}: {emotion}")
        except Exception as exc:
            print(f"⚠️  Bulb trigger failed (non-fatal): {exc}")

        with self._switch_lock:
            busy = self.is_speaking or self.is_thinking
            thread_id = self.thread_id
            language = self.last_language
            llm = self.llm

        if busy:
            print(f"🔕 Ignoring spoken check-in ({emotion}) — Veena is mid-conversation")
            return False

        print(f"💡 Hardware emotion trigger received: emotion={emotion}")
        prompt_text, personalized = self._build_personalized_checkin(llm, thread_id, language, emotion)
        # A prerecorded clip is generic by nature (it was recorded ahead of
        # time, so it can't reference anything specific) — only use it as
        # the zero-latency shortcut when we fell back to the generic
        # template anyway (no history yet to personalize against). Once
        # there's real conversation to draw from, always speak the
        # generated, personalized line instead.
        prerecorded_path = _find_prerecorded_checkin(emotion, language) if not personalized else None

        self.store.append_message(
            thread_id, "assistant", prompt_text, {"language": language, "trigger": "hardware_emotion"}
        )

        self.is_speaking = True
        self.current_ai_text = prompt_text
        try:
            played = vs.play_audio_file(prerecorded_path) if prerecorded_path else False
            if not played:
                vs.play_text_to_speech_stream(prompt_text, language)
        finally:
            self.is_speaking = False
            self.current_ai_text = ""

        # Open the mic — from here it behaves like a normal ongoing
        # conversation until the user goes quiet for a while.
        self.last_activity = time.monotonic()
        self.armed.set()
        print("🎙️  Conversation window open — listening for a reply")

        return True


# ── idle watchdog (returns engine to standby after a quiet spell) ─────────

def _idle_watchdog(engine: "VoiceEngine", stop_event: threading.Event):
    """
    Runs in its own thread so it can interrupt a blocked mic read: while
    VoiceSegmenter.next_utterance() waits for speech, it polls should_abort()
    every ~30ms, which checks engine.armed. Clearing that flag here is what
    actually wakes it up and sends the engine back to standby.
    """
    while not stop_event.is_set():
        time.sleep(1.0)
        if engine.armed.is_set() and (time.monotonic() - engine.last_activity) > IDLE_REARM_TIMEOUT_S:
            engine.armed.clear()
            print("\n💤 No reply — back to standby until the next hardware trigger.")


# ── stdin command listener (thread controls while voice loop runs) ────────

def _stdin_listener(engine: VoiceEngine, stop_event: threading.Event):
    print("\nCommands: 'new' | 'resume <thread_id>' | 'threads' | 'reset profile' | 'specialist' | 'quit'\n")
    for line in sys.stdin:
        cmd = line.strip()
        if not cmd:
            continue
        if cmd == "quit":
            stop_event.set()
            break
        elif cmd == "new":
            engine.new_thread()
        elif cmd == "threads":
            engine.list_threads()
        elif cmd.startswith("resume "):
            engine.resume_thread(cmd.split(" ", 1)[1].strip())
        elif cmd == "reset profile":
            engine.profile_store.reset()
            engine.refresh_profile_text()
            print("🗑️  Profile cleared — onboarding will run again on next launch.")
        elif cmd == "specialist":
            if not ENABLE_MENTAL_HEALTH_SPECIALIST:
                print(
                    "🧠 Specialist mode is disabled (ENABLE_MENTAL_HEALTH_SPECIALIST=0). "
                    "Set that env var to 1 and restart to use the local 7B model."
                )
            else:
                engine.specialist_mode = not engine.specialist_mode
                print(f"🧠 Specialist mode: {'ON' if engine.specialist_mode else 'OFF'} (typed toggle, for demoing without voice)")
        else:
            print("Unknown command. Use: new | resume <thread_id> | threads | reset profile | specialist | quit")


def _run_onboarding(engine: "VoiceEngine", whisper_model, segmenter, stop_event: threading.Event) -> None:
    """
    One-time voice Q&A run the very first time this lamp starts (no
    profile saved yet). Reuses the same segmenter/whisper objects as the
    main loop, but bypasses the armed-gate entirely — this happens once,
    unconditionally, before the engine ever waits for a hardware trigger.
    """
    profile: dict = {}
    print("\n👋 First run — Veena will ask a couple of quick questions to get to know you.")
    for key, question in ONBOARDING_QUESTIONS:
        vs.play_text_to_speech_stream(question, "en")
        buf = segmenter.next_utterance(should_abort=lambda: stop_event.is_set())
        if buf is None:
            print(f"   (no answer captured for '{key}' — skipping)")
            continue
        try:
            text, _, _ = transcribe(whisper_model, buf)
        except Exception as exc:
            print(f"⚠️  Onboarding transcription failed for '{key}': {exc}")
            continue
        if text:
            profile[key] = text
            print(f"   → {key}: {text}")
    engine.profile_store.save(profile)
    engine.refresh_profile_text()
    print("✅ Got it — I'll remember that.\n")


# ── main voice loop ────────────────────────────────────────────────────────

def run(config_path: str, db_path: str, thread_id: Optional[str],
        emotion_trigger_host: str = "0.0.0.0", emotion_trigger_port: int = 5000):
    engine = VoiceEngine(config_path, db_path, thread_id, persona_meta={"config": config_path})

    # Background HTTP server for the hardware side's emotion triggers.
    # Runs in a daemon thread — does not touch the mic loop below.
    start_emotion_trigger_server(
        engine.handle_emotion_trigger,
        host=emotion_trigger_host,
        port=emotion_trigger_port,
    )

    print(f"🔍 Loading Whisper '{WHISPER_MODEL_SIZE}' …")
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", num_workers=1, cpu_threads=2)

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True,
        frames_per_buffer=FRAME_SAMPLES,
    )
    vad = _make_vad()
    segmenter = VoiceSegmenter(stream, vad)
    stop_event = threading.Event()

    threading.Thread(target=_stdin_listener, args=(engine, stop_event), daemon=True).start()
    threading.Thread(target=_idle_watchdog, args=(engine, stop_event), daemon=True).start()

    if engine.profile_store.get() is None:
        _run_onboarding(engine, whisper_model, segmenter, stop_event)

    # Load the local mental-health GGUF model now, in the background, so
    # the 10-30s first-load cost happens during startup rather than live
    # the first time someone says "talk to a specialist" during a demo.
    # Daemon thread: if this fails or is still loading, chat_reply()/
    # answer() just report it unavailable — never blocks the voice loop.
    # Skipped entirely while the specialist model is disabled, so it never
    # loads, never eats RAM/VRAM, and never gets triggered.
    if ENABLE_MENTAL_HEALTH_SPECIALIST:
        threading.Thread(target=mental_health_qa.warm_up, daemon=True).start()
    else:
        print("🧠 Local mental-health specialist model disabled (ENABLE_MENTAL_HEALTH_SPECIALIST=0) — not loading.")

    def _speak_prompt():
        """Print a clear 'your turn' banner so the user always knows when to speak."""
        print("\n" + "─" * 50)
        print("🎙️  YOUR TURN — Speak now")
        print("─" * 50)

    def _standby_notice():
        print("\n" + "─" * 50)
        print("🛌 Standing by — mic is off until a hardware emotion trigger arrives")
        print("─" * 50)

    _standby_notice()   # Idle at startup — nothing is recorded until a trigger fires
    was_armed = False
    try:
        while not stop_event.is_set():
            if not engine.armed.is_set():
                if was_armed:
                    _standby_notice()
                was_armed = False
                time.sleep(0.2)
                continue

            if not was_armed:
                _speak_prompt()   # Just got armed by a trigger — invite the reply
            was_armed = True

            try:
                buf = segmenter.next_utterance(
                    should_abort=lambda: stop_event.is_set() or not engine.armed.is_set()
                )
            except Exception as exc:
                print(f"❌ Audio capture error: {exc}")
                if engine.armed.is_set():
                    _speak_prompt()
                continue
            if buf is None:
                continue

            try:
                print("📝 Transcribing...")
                text, language, emotion = transcribe(whisper_model, buf)
            except Exception as exc:
                print(f"❌ Transcription error: {exc}")
                if engine.armed.is_set():
                    _speak_prompt()
                continue
            if not text:
                # Noise / silence detected — prompt again so user knows we're still listening
                if engine.armed.is_set():
                    _speak_prompt()
                continue

            if engine.is_speaking:
                if is_echo(text, engine.current_ai_text):
                    continue
                vs.request_stop()
                engine.is_speaking = False

            engine.last_activity = time.monotonic()
            print(f"\n⏳ Veena is thinking...")
            try:
                engine.handle_user_text(text, language, emotion)
            except Exception as exc:
                engine.is_thinking = False
                engine.is_speaking = False
                print(f"❌ Turn error: {exc}")
            engine.last_activity = time.monotonic()

            # After Veena finishes speaking, show the prompt again (unless the
            # idle watchdog already put us back to standby in the meantime)
            if engine.armed.is_set():
                _speak_prompt()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("\n🛑 Stopped.")


def main():
    parser = argparse.ArgumentParser(description="Headless voice-to-voice agent, no UI.")
    parser.add_argument("--config", default="wellness_config.json", help="Path to persona/tools config JSON")
    parser.add_argument("--db", default="threads.db", help="Path to the encrypted thread store")
    parser.add_argument("--thread", default=None, help="Resume an existing thread_id instead of starting a new one")
    parser.add_argument("--list-threads", action="store_true", help="List known threads and exit")
    parser.add_argument("--emotion-trigger-host", default="0.0.0.0", help="Host for the hardware emotion-trigger HTTP server")
    parser.add_argument("--emotion-trigger-port", type=int, default=5000, help="Port for the hardware emotion-trigger HTTP server")
    args = parser.parse_args()

    if args.list_threads:
        store = ThreadStore(args.db)
        for t in store.list_threads():
            print(f"{t['thread_id']}   updated {t['updated_at']}   meta={t['meta']}")
        return

    run(args.config, args.db, args.thread, args.emotion_trigger_host, args.emotion_trigger_port)


if __name__ == "__main__":
    main()