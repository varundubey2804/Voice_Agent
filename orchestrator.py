"""
orchestrator.py — Wires streaming LLM + streaming TTS + thread storage together.

This is the ONLY module that knows about all three subsystems at once.
streaming_llm.py knows nothing about audio; voice_service.py knows nothing
about LLMs; thread_store.py knows nothing about either.

CONTRACT WITH THE CALLER (matches voice_engine.VoiceEngine.handle_user_text)
─────────────────────────────────────────────────────────────────────────
- The caller has ALREADY appended the user's turn to `store` before calling
  this (handle_user_text does this at the very top, with language/emotion
  metadata, before it even decides which branch — crisis / tool / fast-path
  — to take). So speak_streaming() only appends the ASSISTANT turn. Do not
  call it before the user turn is saved, or history will be out of order.

- The caller is responsible for is_speaking / current_ai_text bookkeeping
  around the call (needed for the barge-in echo check in voice_engine.py).
  Pass `on_sentence` to get each sentence the moment it's about to be
  spoken, so current_ai_text can be built up incrementally instead of only
  being correct after the whole reply finishes.

- If the LLM stream produces nothing (empty full_text returned), NOTHING
  is spoken and NOTHING is appended to the thread. The caller must handle
  the fallback response itself (this mirrors the old fast_reply() contract,
  where an empty return meant "caller decides what to do next").
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, Optional

from langchain_openai import ChatOpenAI

from streaming_llm import stream_reply
from voice_service import speak_sentences_live_async
from thread_store import ThreadStore


async def speak_streaming(
    llm: ChatOpenAI,
    user_text: str,
    language: str,
    emotion: str,
    thread_id: str,
    store: ThreadStore,
    persona_prompt: Optional[str] = None,
    on_sentence: Optional[Callable[[str], None]] = None,
) -> str:
    """
    One turn of the FAST conversational path: streamed LLM -> streamed TTS,
    overlapped. Returns the full reply text ("" if nothing was generated).

    on_sentence: optional callback invoked with each sentence right as it's
    handed to TTS (i.e. right as it starts being spoken). voice_engine.py
    uses this to keep VoiceEngine.current_ai_text up to date mid-stream, so
    is_echo() can still catch the mic picking up Veena's own voice even
    while she's still talking, not just after she finishes.
    """
    recent_history = store.get_history(thread_id)
    full_parts: list[str] = []

    async def _tee(source: AsyncIterator[str]) -> AsyncIterator[str]:
        async for sentence in source:
            full_parts.append(sentence)
            if on_sentence is not None:
                try:
                    on_sentence(sentence)
                except Exception as exc:
                    # A broken callback must never take down the voice turn.
                    print(f"⚠️  on_sentence callback failed (non-fatal): {exc}")
            yield sentence

    sentence_stream = stream_reply(llm, user_text, language, emotion, recent_history, persona_prompt)
    await speak_sentences_live_async(_tee(sentence_stream), language)

    full_text = " ".join(full_parts).strip()

    if full_text:
        store.append_message(thread_id, "assistant", full_text, {"language": language})
    else:
        print("⚠️  Empty streamed reply — nothing spoken, nothing saved. "
              "Caller should fall back to a canned response.")

    return full_text