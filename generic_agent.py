"""
generic_agent.py ────────────────────────────────────────────────────────
Config-driven LangChain Agent + RAG builder.

KEY ADDITIONS IN THIS VERSION
──────────────────────────────
1. FAST-PATH DIRECT LLM REPLY
   - Most conversational turns (empathy, listening, gentle support) do NOT
     need any tool. They now skip the full ReAct loop and hit the LLM
     directly with a tiny prompt, cutting latency ~3–5× for those turns.
   - A lightweight keyword router (_needs_tool) decides which path to take.

2. LANGUAGE-AGNOSTIC RESPONSE
   - The fast-path system prompt instructs the LLM to always reply in the
     SAME language the user used — Telugu, Hindi, English, Tamil, etc.

3. TIGHTER AGENT LIMITS
   - max_iterations default: 5 → 3
   - max_execution_time default: 30 → 15 s

CONFIG SHAPE (see wellness_config.json for a full example):
{
  "persona_name": "Veena",
  "persona_prompt": "You are ... {language} ... {emotion} ...",
  "faiss_path": "faiss_rag.index",
  "embed_model": "nomic-embed-text",
  "llm_models": ["nvidia/nemotron-3.5-lightning", "openai/gpt-4o-mini"],
  "llm_temperature": 0.3,
  "tools": [
    {
      "name": "EpicWisdomSearch",
      "type": "retriever",
      "description": "..."
    },
    {
      "name": "BreathingGuide",
      "type": "function",
      "module": "wellness_tools",
      "function": "breathing_guide",
      "description": "..."
    }
  ]
}
"""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path
from typing import Optional

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

# As of LangChain 1.0, the legacy AgentExecutor / create_react_agent /
# ConversationBufferMemory APIs moved out of the main `langchain` package
# into a separate `langchain-classic` package (see pyproject.toml comment).
# The classic create_react_agent(llm, tools, prompt) signature used below
# is preserved as-is in langchain_classic — this is NOT the newer
# langgraph-style create_react_agent, which takes different arguments.
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import create_react_agent, AgentExecutor

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
LLM_REQUEST_TIMEOUT_S = 20

# ── "exhausted reasoning budget" fix ─────────────────────────────────────
# nvidia/nemotron-3.5-lightning (and several other OpenRouter models) are
# REASONING models: before writing the visible reply, they spend tokens on
# a hidden chain-of-thought. With max_tokens=1024 shared between that
# hidden reasoning and the visible answer, a chatty reasoning pass can eat
# the entire budget, so `result.content` comes back empty and
# fast_reply()/stream_reply() see finish_reason == "length" with nothing
# to show for it. This is the "⚠️ fast_reply exhausted reasoning budget"
# warning — it is NOT related to mental_health_qa's local 7B model at all;
# no "🧠 Specialist mode" line means that code path never even ran.
#
# Veena's persona is a fast, warm conversational companion, not a model
# that needs to show its work — so we disable hidden reasoning entirely
# via OpenRouter's unified `reasoning` request param. This is a no-op for
# models that were never "reasoning" models in the first place (the param
# is simply ignored), so it's safe to always send.
# https://openrouter.ai/docs/use-cases/reasoning-tokens
REASONING_EXTRA_BODY = {"reasoning": {"effort": "none", "exclude": True}}

# ── "Talk to a specialist" 7B local model — off by default ──────────────
# The mradermacher Mental-Health-FineTuned-Mistral-7B GGUF (mental_health_qa.py)
# is CPU/GPU-heavy on modest hardware and was the thing everyone assumed
# was misbehaving — it wasn't (see REASONING_EXTRA_BODY above) — but it's
# still slow and not needed for the core "just talk, with context, and
# handle wellness/therapist-style questions reasonably" experience. Code
# is left fully intact; this flag just keeps it out of the loop until you
# want it back. Set ENABLE_MENTAL_HEALTH_SPECIALIST=1 to re-enable.
ENABLE_MENTAL_HEALTH_SPECIALIST = os.environ.get(
    "ENABLE_MENTAL_HEALTH_SPECIALIST", "false"
).strip().lower() in ("1", "true", "yes", "on")

# ── Tool-routing keywords ─────────────────────────────────────────────────────
# If ANY of these appear in the user's message, we run the full ReAct agent
# (which can invoke tools). Everything else uses the fast direct-LLM path.
_TOOL_TRIGGER_KEYWORDS = (
    # Breathing / grounding
    "breath", "breathing", "4-7-8", "box breath", "saans", "grounding",
    "sas", "pranayam",
    # Mood check-in
    "mood", "check in", "check-in", "rate my",
    # Motivation / epic
    "motivat", "inspir", "epic", "ramayan", "mahabharat", "hanuman", "arjun",
    # Reflection / journaling
    "reflect", "journal",
    # Crisis (handled separately before this router even fires)
    "suicid", "self-harm", "self harm", "kill myself",
)


# MentalHealthQA trigger: deliberately a REGEX, not a plain substring like
# the keywords above. A loose "what is" substring match would grab a huge
# amount of ordinary conversation ("what is going on today") and route it
# into the slower ReAct loop — the exact over-firing problem already
# flagged for "sas" in this list (see README). This only matches when a
# clinical/therapy noun is actually present.
_KNOWLEDGE_QUESTION_RE = re.compile(
    r"\b(what\s+is|what\s+are|how\s+does|difference\s+between|symptoms?\s+of)\b"
    r".{0,40}\b(disorder|syndrome|therapy|anxiety|depression|condition|"
    r"cbt|trauma|ptsd|panic\s+attack)\b",
    re.IGNORECASE,
)


def _needs_tool(text: str) -> bool:
    """Return True only when the user message clearly calls for a specific tool."""
    lower = text.lower()
    if any(kw in lower for kw in _TOOL_TRIGGER_KEYWORDS):
        return True
    return bool(_KNOWLEDGE_QUESTION_RE.search(text))


# ── ReAct format appended to the persona prompt ───────────────────────────────
_REACT_FORMAT = """

You have access to these tools:
{tools}

Use this FORMAT (no code fences, no extra punctuation):

Thought: Do I need to use a tool? Yes
Action: <tool name from [{tool_names}]>
Action Input: <input to the tool>
Observation: <result of the tool>

When ready to answer:

Thought: Do I need to use a tool? No
Final Answer: <your answer in the language {language}>

Previous conversation:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


# ── Fast-path system prompt (used when no tool is needed) ────────────────────
_FAST_SYSTEM_PROMPT = """\
You are Veena, a warm and human-like mental wellness companion.
You speak in EXACTLY the same language as the user — if they write in Telugu,
reply in Telugu; if Hindi, reply in Hindi; if English, reply in English.
Mirror the user's language automatically. Never switch languages.
Speak like a caring friend: short, warm, natural sentences. No bullet points.
Keep your reply under 60 words — this is a voice conversation.
Validate the user's feelings first, then offer one gentle thought or question.
You are NOT a doctor. Do not diagnose or prescribe. If someone seems in
serious distress, gently suggest calling a mental health professional."""


# ── LLM factory ──────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_llm(config: dict) -> ChatOpenAI:
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not found. Set it in your .env file or environment. "
            "Get a free key at https://openrouter.ai"
        )

    candidates = [os.environ.get("OPENROUTER_MODEL")] + config.get("llm_models", [])
    candidates = list(dict.fromkeys(m for m in candidates if m))

    for model_name in candidates:
        try:
            llm = ChatOpenAI(
                model=model_name,
                temperature=config.get("llm_temperature", 0.25),
                max_tokens=1024,
                api_key=openrouter_api_key,
                base_url=OPENROUTER_API_BASE,
                timeout=LLM_REQUEST_TIMEOUT_S,
                max_retries=1,
                # See REASONING_EXTRA_BODY above — stops reasoning-capable
                # models from silently burning the whole max_tokens budget
                # on hidden chain-of-thought before ever writing a reply.
                model_kwargs={"extra_body": REASONING_EXTRA_BODY},
            )
            # Lightweight probe: send a 1-token echo to confirm the model is reachable
            probe = llm.invoke([HumanMessage(content="hi")])
            if probe and probe.content:
                print(f"\u2705  Using OpenRouter model: {model_name}")
                return llm
            else:
                print(f"\u26a0\ufe0f  Model '{model_name}' returned empty response — skipping")
        except Exception as exc:
            print(f"\u26a0\ufe0f  OpenRouter model '{model_name}' unavailable: {exc}")
            continue

    raise RuntimeError(
        "No configured LLM model was available. Check llm_models in your config "
        "and your OPENROUTER_API_KEY."
    )


# ── Tool builder ──────────────────────────────────────────────────────────────

def _build_tools(config: dict, retriever) -> list[Tool]:
    tools: list[Tool] = []
    for spec in config.get("tools", []):
        ttype = spec.get("type", "function")

        if ttype == "retriever":
            if retriever is None:
                print(f"⚠️  Skipping tool '{spec['name']}' — no retriever configured (faiss_path missing?)")
                continue
            tools.append(
                Tool(
                    name=spec["name"],
                    func=lambda q, r=retriever: "\n".join(d.page_content for d in r.invoke(q)),
                    description=spec["description"],
                )
            )

        elif ttype == "function":
            if spec.get("module") == "mental_health_qa" and not ENABLE_MENTAL_HEALTH_SPECIALIST:
                print(
                    f"⚠️  Skipping tool '{spec['name']}' — the local 7B mental-health "
                    "model is disabled (set ENABLE_MENTAL_HEALTH_SPECIALIST=1 to turn "
                    "it back on). General wellness/therapist-style questions are still "
                    "handled by Veena's main persona."
                )
                continue
            module = importlib.import_module(spec["module"])
            func = getattr(module, spec["function"])
            tools.append(Tool(name=spec["name"], func=func, description=spec["description"]))

        else:
            raise ValueError(f"Unknown tool type '{ttype}' for tool '{spec.get('name')}'")

    return tools


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent(
    config_path: str,
    seed_messages: Optional[list[dict]] = None,
) -> tuple:
    """
    Build an AgentExecutor from a JSON config.

    seed_messages: prior turns from ThreadStore.get_history(), e.g.
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    used to seed memory when resuming an existing thread. Pass None/[] for a
    brand-new thread.

    Returns (agent_executor, llm, config):
      - agent_executor : full ReAct agent (use when tools are needed)
      - llm            : raw ChatOpenAI client (use for fast_reply())
      - config         : loaded JSON config dict
    """
    from dotenv import load_dotenv
    load_dotenv()

    config = load_config(config_path)
    llm = _resolve_llm(config)

    retriever = None
    faiss_path = config.get("faiss_path")
    if faiss_path and Path(faiss_path).exists():
        embedding = OllamaEmbeddings(model=config.get("embed_model", "nomic-embed-text"))
        print(f"📂  Loading FAISS index: {faiss_path}")
        vector_db = FAISS.load_local(
            faiss_path, embeddings=embedding, allow_dangerous_deserialization=True,
        )
        retriever = vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": config.get("retriever_k", 4)}
        )
    elif faiss_path:
        print(f"⚠️  FAISS index not found at '{faiss_path}' — retriever tools disabled. "
              f"Run index_documents.py first if you need it.")

    tools = _build_tools(config, retriever)

    template = config["persona_prompt"] + _REACT_FORMAT
    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names", "language", "emotion"],
    )

    agent = create_react_agent(llm, tools, prompt)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False,
        input_key="input",
        output_key="output",
    )
    for msg in (seed_messages or []):
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            memory.chat_memory.add_ai_message(msg["content"])
        # other roles (e.g. "system") are skipped

    def _parse_error_handler(error: Exception) -> str:
        return (
            "Thought: Do I need to use a tool? No\n"
            "Final Answer: I'm sorry, I had trouble with that. Could you say it again?"
        )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=_parse_error_handler,
        max_iterations=config.get("max_iterations", 3),
        max_execution_time=config.get("max_execution_time", 15),
        early_stopping_method="generate",
    )

    print(f"✅  Agent '{config.get('persona_name', 'agent')}' ready "
          f"({len(tools)} tools, history seeded: {bool(seed_messages)})")
    return agent_executor, llm, config


# ── Fast-path reply (no ReAct loop) ──────────────────────────────────────────

def fast_reply(
    llm: ChatOpenAI,
    user_text: str,
    language: str,
    emotion: str,
    recent_history: list[dict],
    persona_prompt: Optional[str] = None,
) -> str:
    """
    Hit the LLM directly with a minimal prompt — no agent loop, no tool calls.
    ~3–5× faster than the full ReAct agent for pure conversational turns.

    recent_history : last N turns from ThreadStore.get_history() for context.
    """
    system_prompt = persona_prompt or _FAST_SYSTEM_PROMPT
    system_prompt += (
        "\n\nFAST VOICE RESPONSE RULES:\n"
        "Answer the user's actual message directly. Do not give a generic fallback, "
        "repeat the prompt, or change the subject. Validate the specific feeling or "
        "request first, then give one practical next step. Keep the reply under 70 "
        "words and use plain spoken language. Detected emotion: " + emotion + "."
    )
    messages: list = [SystemMessage(content=system_prompt)]

    # Include last 6 messages (3 turns) for conversational context
    for m in recent_history[-6:]:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))

    # Current user message
    messages.append(HumanMessage(content=user_text))

    try:
        result = llm.invoke(messages, max_tokens=1024)
        content = result.content or ""
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        content = str(content).strip()
        if not content and result.response_metadata.get("finish_reason") == "length":
            print("⚠️  fast_reply exhausted reasoning budget; retrying")
            result = llm.invoke(messages, max_tokens=2048)
            content = result.content or ""
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            content = str(content).strip()
        if not content:
            print("⚠️  fast_reply returned an empty response")
        return content
    except Exception as exc:
        print(f"⚠️  fast_reply failed ({type(exc).__name__}): {exc}")
        return ""