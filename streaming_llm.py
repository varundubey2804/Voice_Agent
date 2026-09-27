"""
streaming_llm.py — Token-streaming LLM replies, chunked into speakable sentences.

WHY THIS EXISTS
────────────────
The old fast_reply() in generic_agent.py calls llm.invoke(...) and BLOCKS
until the entire response is generated, then returns one big string. TTS
can't start until that whole string exists.

For a ~40-word reply at typical OpenRouter speeds, that's 1-3 seconds of
dead air before the user hears anything — most of your perceived latency
lives right here, not in the model's "intelligence."

This module fixes that: it streams tokens from the LLM via .astream() and
yields each SENTENCE the instant it's complete. voice_service.py's TTS
pipeline consumes this stream directly, so synthesis+playback of sentence 1
overlaps with the LLM still generating sentence 3.

DESIGN NOTE
───────────
This module knows nothing about audio. voice_service.py knows nothing about
LLMs. orchestrator.py is the only place that wires them together. Keeping
that boundary means you can swap the LLM backend or the TTS backend
independently without touching the other.

USAGE
─────
    async for sentence in stream_reply(llm, user_text, language, emotion, history):
        ...  # each sentence is ready to hand to TTS immediately

Falls back to an empty stream on failure — callers should treat "nothing
was yielded" as "retry with generic_agent.fast_reply() instead" if they
want a non-streaming fallback path.
"""

from __future__ import annotations

import re
from typing import AsyncIterator, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Reuse the exact same default persona prompt as the blocking fast_reply(),
# so streaming and non-streaming paths never drift apart in tone/behavior.
from generic_agent import _FAST_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT

# Sentence-boundary regex. Matches voice_service._split_sentences' behavior
# (., !, ?, and the Hindi/Devanagari danda ।) so sentence chunks line up
# with what the TTS side expects. Duplicated deliberately — see design note.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।])\s+")


def _extract_ready_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Pull any COMPLETE sentences out of the accumulated buffer.

    A "complete" sentence is everything up to and including a boundary
    character, followed by whitespace (i.e. there's more text after it,
    proving the model wasn't mid-sentence when the boundary appeared).
    The trailing, possibly-unfinished fragment is returned as `remainder`
    and stays in the buffer until more tokens arrive.

    Returns: (list_of_finished_sentences, remainder_still_in_progress)
    """
    parts = _SENTENCE_BOUNDARY.split(buffer)
    if len(parts) <= 1:
        return [], buffer  # no boundary seen yet — nothing finished
    *complete, remainder = parts
    return [p.strip() for p in complete if p.strip()], remainder


async def stream_reply(
    llm: ChatOpenAI,
    user_text: str,
    language: str,
    emotion: str,
    recent_history: list[dict],
    persona_prompt: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Stream a conversational reply, yielding one sentence at a time.

    recent_history : last N turns from ThreadStore.get_history(), same
                      shape as fast_reply() expects.

    On any error (network blip, provider hiccup) this logs a warning and
    ends the generator early with whatever was already yielded — it does
    NOT raise, so callers don't need a try/except around the `async for`.
    """
    system_prompt = (persona_prompt or DEFAULT_SYSTEM_PROMPT) + (
        "\n\nFAST VOICE RESPONSE RULES:\n"
        "Answer the user's actual message directly. Do not give a generic "
        "fallback, repeat the prompt, or change the subject. Validate the "
        "specific feeling first, then give one practical next step. Keep "
        "the reply under 70 words and use plain spoken language. "
        "Detected emotion: " + emotion + "."
    )
    messages: list = [SystemMessage(content=system_prompt)]

    # Same 3-turn context window as the blocking fast_reply().
    for m in recent_history[-6:]:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=user_text))

    buffer = ""
    yielded_anything = False
    try:
        async for chunk in llm.astream(messages):
            delta = chunk.content or ""
            # Some providers stream structured content blocks instead of
            # plain strings — normalize both shapes.
            if isinstance(delta, list):
                delta = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in delta
                )
            if not delta:
                continue
            buffer += delta
            ready, buffer = _extract_ready_sentences(buffer)
            for sentence in ready:
                yielded_anything = True
                yield sentence

        # Stream ended — flush whatever's left (the final sentence usually
        # has no trailing whitespace after its boundary, so it never gets
        # caught by _extract_ready_sentences during the loop).
        tail = buffer.strip()
        if tail:
            yielded_anything = True
            yield tail

    except Exception as exc:
        print(f"⚠️  stream_reply failed ({type(exc).__name__}): {exc}")
        return

    if yielded_anything:
        return

    # Safety net: the stream produced NOTHING (empty deltas the whole way
    # through — e.g. a provider hiccup mid-stream). Rather than leaving the
    # turn silent, fall back to one plain blocking call on the same client
    # before giving up. Cheap to try since it's the exact same request.
    print("⚠️  stream_reply produced no sentences — retrying with a single blocking call")
    try:
        result = await llm.ainvoke(messages)
        content = result.content or ""
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        content = str(content).strip()
        if content:
            yield content
        else:
            print("⚠️  stream_reply fallback blocking call also returned empty")
    except Exception as exc:
        print(f"⚠️  stream_reply fallback blocking call failed ({type(exc).__name__}): {exc}")
        return