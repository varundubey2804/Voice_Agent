"""
mental_health_qa.py — Local, offline mental-health features backed by
mradermacher/Mental-Health-FineTuned-Mistral-7B-Instruct-v0.2-i1-GGUF
(llama.cpp), used two ways:

1. answer(query)      — one-shot psychoeducation Q&A ("what is generalized
                         anxiety disorder"), wired in as a ReAct tool
                         (MentalHealthQA in wellness_config.json).
2. chat_reply(...)     — a live, multi-turn "specialist mode" conversation
                         where THIS fine-tuned model is the one talking —
                         asking its own genuine follow-up questions, not
                         just answering FAQs. This is the demo-friendly
                         feature: say "talk to a specialist" and the local
                         model takes over the conversation directly, so
                         judges can see it reason and probe, not just
                         retrieve.

WHY BOTH, AND WHY GATED BEHIND EXPLICIT TRIGGERS
──────────────────────────────────────────────────
A 7B model, even quantized, is visibly slower than the OpenRouter
fast-path on modest hardware (CPU-only i5 + 4GB GPU). Rather than putting
it on the critical path for every turn, it only runs when the user (or
the ReAct router) explicitly asks for it — Veena's default fast
conversational path is untouched.

SAFETY
──────
- Crisis language is checked FIRST in both entry points, before this
  model ever sees the query. If detected, this short-circuits straight
  to wellness_tools.get_crisis_resources() and never invokes the
  fine-tune. Do not remove this — a community GGUF quant of a fine-tune
  is not a clinically validated model and must never be the first
  responder to crisis language. voice_engine.py's own crisis check
  upstream is the primary guard; this is defense in depth.
- answer() appends a general-information disclaimer; chat_reply() relies
  on its system prompt instead (a disclaimer on every line of a live
  back-and-forth reads as robotic — the system prompt still forbids
  diagnosis/prescription).
- This is a third-party fine-tune (mradermacher only produced the GGUF
  quantization, not the fine-tune itself) — treat its answers as
  unverified beyond generic psychoeducation, and say so if asked.

SETUP
─────
    uv add llama-cpp-python
    # For GPU offload (you have a GTX 1650 — worth using):
    CMAKE_ARGS="-DGGML_CUDA=on" uv add llama-cpp-python --no-binary llama-cpp-python

Download a quant — Q4_K_M is a reasonable speed/quality balance for 4GB
VRAM plus a system that's already tight on RAM; drop to a smaller quant
(e.g. IQ4_XS) if it's still too slow for a live demo:

    huggingface-cli download \\
        mradermacher/Mental-Health-FineTuned-Mistral-7B-Instruct-v0.2-i1-GGUF \\
        Mental-Health-FineTuned-Mistral-7B-Instruct-v0.2.i1-Q4_K_M.gguf \\
        --local-dir ./models

Then:
    export MENTAL_HEALTH_GGUF_PATH=./models/Mental-Health-FineTuned-Mistral-7B-Instruct-v0.2.i1-Q4_K_M.gguf

TEST BEFORE YOUR DEMO — time a real call:
    time python mental_health_qa.py "What is generalized anxiety disorder?"
If that's several seconds, budget a spoken filler line before the answer
in voice_engine.py, or drop n_gpu_layers/quant further. Practice the
"talk to a specialist" demo flow at least once end-to-end beforehand —
first-call model load time is separate from per-reply latency and can be
10-30s on its own; don't let that happen live in front of judges (see the
warm-up note in voice_engine.py's run()).
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import wellness_tools

MODEL_PATH = os.environ.get(
    "MENTAL_HEALTH_GGUF_PATH",
    "./models/Mental-Health-FineTuned-Mistral-7B-Instruct-v0.2.i1-Q4_K_M.gguf",
)
N_CTX = int(os.environ.get("MENTAL_HEALTH_N_CTX", "2048"))
# Partial GPU offload — tune down if you hit CUDA OOM (this model, Whisper,
# and everything else share the same 4GB). 0 = CPU only (slow but safe).
N_GPU_LAYERS = int(os.environ.get("MENTAL_HEALTH_N_GPU_LAYERS", "20"))
QA_MAX_TOKENS = int(os.environ.get("MENTAL_HEALTH_QA_MAX_TOKENS", "200"))
CHAT_MAX_TOKENS = int(os.environ.get("MENTAL_HEALTH_CHAT_MAX_TOKENS", "120"))
# How many prior turns of conversation to feed back in for chat mode.
# Kept small deliberately — every extra turn is more tokens for an already
# slow model to chew through before it can start replying.
CHAT_HISTORY_TURNS = int(os.environ.get("MENTAL_HEALTH_CHAT_HISTORY_TURNS", "3"))

_QA_DISCLAIMER = (
    "\n\n(This is general information, not a diagnosis — for anything "
    "ongoing, a licensed mental health professional is the right next step.)"
)

_QA_SYSTEM_PROMPT = (
    "You are a mental health psychoeducation assistant speaking OUT LOUD "
    "in a voice conversation, not writing text someone will read. Answer "
    "the question factually and concisely, in plain spoken language, in "
    "under 100 words. You do not diagnose, you do not prescribe, and you "
    "always recommend professional support for anything beyond general "
    "information.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Never repeat, restate, or refer to the question itself. Do not "
    "start with 'You asked', 'Regarding your question', 'To answer that', "
    "or any similar preamble. Begin directly with the answer.\n"
    "- No lists, no numbered steps, no bullet points, no headers.\n"
    "- No colons introducing a list or explanation, and no semicolons — "
    "write plain connected sentences instead.\n"
    "- Use only full stops, commas, and question marks. Never use "
    "asterisks, quotation marks, parentheses, dashes, or any other symbol.\n"
    "- Speak the way a person would say it out loud, not the way it would "
    "be formatted on a page."
)

# The system prompt driving the DEMO feature: this is what makes the
# model act like a genuinely curious specialist rather than an FAQ bot.
CHAT_SYSTEM_PROMPT = (
    "You are an attentive mental health specialist in a live spoken "
    "conversation, speaking OUT LOUD — not writing text someone will "
    "read. In one short sentence, validate what the person just said, "
    "using specifics from what they actually said — not a generic "
    "phrase. Then ask exactly ONE genuine, specific follow-up question "
    "about what THEY just told you (never a generic 'how does that make "
    "you feel'). If they named something concrete — a person, a deadline, "
    "an event — refer to it by name in your question. Keep your entire "
    "reply under 45 words. Plain spoken language. No lists. No diagnosis. "
    "No prescriptions.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Never repeat or refer to what they said as 'you said' or 'you "
    "mentioned' — just respond to it naturally, the way a person would.\n"
    "- No lists, no colons, no semicolons, no asterisks, no quotation "
    "marks, no parentheses, no dashes. Only full stops, commas, and "
    "question marks.\n"
    "- Say it the way it would sound spoken aloud, never the way it "
    "would look written on a page."
)

_lock = threading.Lock()
_llm = None  # lazy singleton — only pay the load cost if this module is ever used


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    with _lock:
        if _llm is None:
            from llama_cpp import Llama  # imported lazily so importing this
            # module elsewhere in the codebase never requires llama-cpp-python
            # to be installed unless the feature is actually used.

            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"GGUF model not found at '{MODEL_PATH}'. Set "
                    "MENTAL_HEALTH_GGUF_PATH or download it — see the "
                    "module docstring."
                )
            print(
                f"🧠  Loading local mental-health model from {MODEL_PATH} "
                f"(n_gpu_layers={N_GPU_LAYERS}) — first call only, this can "
                f"take a while..."
            )
            _llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS,
                verbose=False,
            )
    return _llm


def warm_up() -> bool:
    """
    Force the model to load right now instead of on the first real user
    turn. Call this once at process startup (see voice_engine.run()) so
    the 10-30s first-load cost happens during your own setup, not live in
    front of judges the first time someone says "talk to a specialist".
    Returns True on success, False if the model couldn't be loaded (the
    features degrade gracefully either way — see the empty-string /
    fallback-text contracts below).
    """
    try:
        _get_llm()
        return True
    except Exception as exc:
        print(f"⚠️  mental_health_qa warm-up failed (features will be unavailable): {exc}")
        return False


# ── One-shot Q&A tool (ReAct-routed) ─────────────────────────────────────

def answer(query: str) -> str:
    """
    Tool entry point — same calling convention as wellness_tools.* (one
    string in, one string out), so it plugs into generic_agent's Tool
    builder with no extra wiring beyond the config entry.
    """
    query = (query or "").strip()
    if not query:
        return "Could you tell me a bit more about what you'd like to know?"

    if wellness_tools.detect_crisis(query):
        return wellness_tools.get_crisis_resources()

    try:
        llm = _get_llm()
    except Exception as exc:
        print(f"⚠️  mental_health_qa model unavailable: {exc}")
        return (
            "I don't have that lookup available right now, but I'm happy "
            "to talk through it with you directly — what's on your mind?"
        )

    prompt = f"<s>[INST] {_QA_SYSTEM_PROMPT}\n\nQuestion: {query} [/INST]"
    try:
        result = llm(prompt, max_tokens=QA_MAX_TOKENS, temperature=0.3, stop=["</s>", "[INST]"])
        text = result["choices"][0]["text"].strip()
    except Exception as exc:
        print(f"⚠️  mental_health_qa generation failed: {exc}")
        return "I'm having trouble pulling that up right now — want to just talk about it instead?"

    if not text:
        return "I'm not sure I have a good answer for that one — could you rephrase?"

    return text + _QA_DISCLAIMER


# ── Live multi-turn "specialist mode" chat ───────────────────────────────

def _build_multiturn_prompt(system_prompt: str, recent_history: list[dict], user_text: str) -> str:
    """
    Mistral-Instruct multi-turn format:
        <s>[INST] {system}\\n\\n{user_1} [/INST] {assistant_1}</s>
        [INST] {user_2} [/INST] {assistant_2}</s>
        ...
        [INST] {latest_user} [/INST]

    recent_history is the same list[dict] shape ThreadStore.get_history()
    returns ({"role": "user"|"assistant", "content": ...}); only the last
    CHAT_HISTORY_TURNS user/assistant pairs are used, to keep the prompt
    short for an already-slow model.
    """
    trimmed = [m for m in recent_history if m.get("role") in ("user", "assistant")]
    trimmed = trimmed[-(CHAT_HISTORY_TURNS * 2):]

    parts: list[str] = []
    first_block = True
    pending_user: Optional[str] = None
    for m in trimmed:
        if m["role"] == "user":
            pending_user = m["content"]
        elif m["role"] == "assistant" and pending_user is not None:
            if first_block:
                parts.append(f"<s>[INST] {system_prompt}\n\n{pending_user} [/INST] {m['content']}</s>")
                first_block = False
            else:
                parts.append(f"[INST] {pending_user} [/INST] {m['content']}</s>")
            pending_user = None

    if first_block:
        # No usable prior pairs — this is the first turn of specialist mode.
        parts.append(f"<s>[INST] {system_prompt}\n\n{user_text} [/INST]")
    else:
        parts.append(f"[INST] {user_text} [/INST]")

    return "".join(parts)


def chat_reply(user_text: str, recent_history: Optional[list[dict]] = None) -> str:
    """
    Live conversational turn through the local fine-tune. Returns "" on
    any failure (model unavailable, generation error, empty output) —
    callers should treat that exactly like streaming_llm.stream_reply's
    "nothing yielded" contract and fall back to a canned line, never leave
    the user in silence.
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return "I'm here — what's on your mind?"

    if wellness_tools.detect_crisis(user_text):
        return wellness_tools.get_crisis_resources()

    try:
        llm = _get_llm()
    except Exception as exc:
        print(f"⚠️  mental_health_qa chat model unavailable: {exc}")
        return ""

    prompt = _build_multiturn_prompt(CHAT_SYSTEM_PROMPT, recent_history or [], user_text)
    try:
        result = llm(prompt, max_tokens=CHAT_MAX_TOKENS, temperature=0.5, stop=["</s>", "[INST]"])
        text = result["choices"][0]["text"].strip()
    except Exception as exc:
        print(f"⚠️  mental_health_qa chat generation failed: {exc}")
        return ""

    return text


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is generalized anxiety disorder?"
    print("── one-shot answer() ──")
    print(answer(q))
    print("\n── chat_reply() (no history) ──")
    print(chat_reply(q, []))