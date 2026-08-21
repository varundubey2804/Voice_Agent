"""
agentic_rag.py ────────────────────────────────────────────────────────────
Self-contained LangChain Agent + RAG pipeline
• Loads a pre-built FAISS index for Mahabharat/Ramayan wisdom retrieval.
• Exposes wellness tools for the agent to call.
• Keeps conversational memory and uses the 'Veena' wellness persona.

Tools wired in:
  1.  EpicWisdomSearch       — FAISS KB (Mahabharat, Ramayan, wellness guide)
  2.  BreathingGuide         — Guided breathing exercises
  3.  GroundingExercise      — 5-4-3-2-1 sensory grounding
  4.  DailyMotivation        — Epic-based daily inspiration
  5.  MoodCheckIn            — Mood scale 1-5 with empathetic response
  6.  ReflectionPrompt       — Journaling / self-reflection prompt
  7.  CrisisResources        — India mental health helplines
"""

import os
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor


# ──────────────────────────────
# Configuration
# ──────────────────────────────
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME   = "llama-3.3-70b-versatile"
FAISS_PATH       = "faiss_rag.index"


# ──────────────────────────────
# Public factory
# ──────────────────────────────

def build_agent():
    """Return a LangChain AgentExecutor with memory + wellness tools."""

    from dotenv import load_dotenv
    load_dotenv()

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Set it in your .env file or environment variables."
        )

    embedding = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=0.3,
        api_key=groq_api_key,
    )

    if not Path(FAISS_PATH).exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_PATH}'. "
            "Please run 'python index_documents.py' first to build it."
        )
    print(f"📂  Loading FAISS index: {FAISS_PATH}")
    vector_db = FAISS.load_local(
        FAISS_PATH,
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )

    import wellness_tools

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    tools = [
        Tool(
            name="EpicWisdomSearch",
            func=lambda q: "\n".join([d.page_content for d in retriever.invoke(q)]),
            description=(
                "Search the knowledge base for Mahabharat and Ramayan stories, teachings, "
                "and wellness guidance relevant to the user's emotional situation. "
                "Use when the user shares a problem (stress, doubt, grief, anger, loneliness) "
                "and you want an authentic epic reference to share."
            ),
        ),
        Tool(
            name="StressAssessment",
            func=wellness_tools.assess_stress,
            description=(
                "Assess whether the user sounds calm, moderately stressed, or highly overwhelmed. "
                "Input: the user's message. Return a score, stress level, and guidance suggestion."
            ),
        ),
        Tool(
            name="BreathingGuide",
            func=wellness_tools.breathing_guide,
            description=(
                "Guide the user through a breathing exercise to reduce stress or anxiety. "
                "Input: '478' (4-7-8 breathing), 'box' (box breathing), 'calm' (belly breathing), "
                "or empty string for a recommended technique."
            ),
        ),
        Tool(
            name="GroundingExercise",
            func=wellness_tools.grounding_exercise,
            description=(
                "Guide the 5-4-3-2-1 sensory grounding exercise to bring the user back "
                "to the present moment during anxiety or overwhelm. Input: empty string."
            ),
        ),
        Tool(
            name="DailyMotivation",
            func=wellness_tools.daily_motivation,
            description=(
                "Share daily inspiration drawn from Mahabharat or Ramayan. "
                "Input: optional theme — courage, doubt, patience, resilience, duty, focus, work, letting go."
            ),
        ),
        Tool(
            name="MoodCheckIn",
            func=wellness_tools.mood_checkin,
            description=(
                "Check in on the user's mood on a scale of 1 (very low) to 5 (great). "
                "Input: 'level' or 'level, context' (e.g. '2, stressed about work deadline')."
            ),
        ),
        Tool(
            name="ReflectionPrompt",
            func=wellness_tools.reflection_prompt,
            description=(
                "Offer a thoughtful journaling or reflection prompt to help the user "
                "explore their feelings. Input: empty string."
            ),
        ),
        Tool(
            name="CrisisResources",
            func=wellness_tools.get_crisis_resources,
            description=(
                "Provide India mental health helpline numbers. Use ONLY when the user "
                "expresses suicidal thoughts, self-harm intent, or severe crisis. "
                "Input: empty string."
            ),
        ),
    ]

    persona = """You are "Veena," a warm, calm and human-like mental wellness companion.
You speak like a thoughtful person who listens first, validates feelings, and responds with empathy.
You can gently draw inspiration from Krishna, Rama, Hanuman, Arjuna, and Sita when it fits naturally, but you are not a doctor and you do not diagnose conditions.

IMPORTANT: You are NOT a licensed therapist or doctor. You offer emotional support and coping strategies, not medical diagnoses or prescriptions.

LANGUAGE RULE (CRITICAL): Detected language = {language}.
- If {language} is 'hi' → your ENTIRE Final Answer MUST be in Hindi (Devanagari script). No English words.
- If {language} is 'en' → respond in English only.

EMOTION RULE: Detected emotion = {emotion}.
- 'stressed' → calm and grounding tone; offer breathing, grounding, or a short epic reflection.
- 'confused' → gentle clarity; simplify; one step at a time.
- 'calm' or 'neutral' → warm, supportive; can add light encouragement.

TOOL USAGE GUIDE:
- User shares emotional difficulty or stress → StressAssessment first, then use EpicWisdomSearch if helpful.
- User anxious, stressed, panicking → BreathingGuide or GroundingExercise
- User asks for motivation or daily inspiration → DailyMotivation
- User wants to check in on mood → MoodCheckIn
- User wants to reflect or journal → ReflectionPrompt
- User mentions suicide, self-harm, wanting to die → CrisisResources IMMEDIATELY
- General emotional support, active listening, encouragement → Answer DIRECTLY without a tool

CONVERSATION RULES:
1. ALWAYS validate feelings first ("That sounds really hard" / "यह सुनकर दुख हुआ").
2. Use AT MOST ONE tool per turn unless the user explicitly asks for multiple things.
3. Keep replies natural and human, like a real conversation, not a lecture.
4. Keep responses under 120 words — this is a voice conversation.
5. Ask one gentle follow-up question when appropriate.
6. Never diagnose conditions (depression, bipolar, etc.).
7. For serious ongoing issues, gently suggest speaking with a mental health professional.
8. If the user seems in crisis, use CrisisResources and respond with deep compassion."""

    template = persona + """

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

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "input", "chat_history", "agent_scratchpad",
            "tools", "tool_names", "language", "emotion",
        ],
    )

    agent = create_react_agent(llm, tools, prompt)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False,
        input_key="input",
        output_key="output",
    )

    def _parse_error_handler(error: Exception) -> str:
        return (
            "I encountered a formatting issue. "
            "Thought: Do I need to use a tool? No\n"
            "Final Answer: I'm sorry, I had trouble processing that. "
            "Could you please tell me again how you're feeling?"
        )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=_parse_error_handler,
        max_iterations=5,
        max_execution_time=30,
        early_stopping_method="generate",
    )

    print("✅  Agentic RAG with 'Veena' wellness persona is ready!")
    return agent_executor


if __name__ == "__main__":
    ag = build_agent()

    print("\nType 'exit' to quit.\n")
    while True:
        q = input("🗣  You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        lang = "hi" if any('\u0900' <= c <= '\u097f' for c in q) else "en"

        try:
            result = ag.invoke({"input": q, "language": lang, "emotion": "neutral"})
            print(f"🤖 Veena: {result['output']}\n")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🤖 Veena: I'm having trouble responding right now. Please try again.\n")
