"""
agentic_rag.py
Self-contained LangChain Agent + RAG pipeline for the Veena AI persona.

Improvements:
  - load_dotenv() called once at module level (not buried inside a conditional)
  - GROQ_API_KEY absence raises a clear ValueError before any heavy work starts
  - _portfolio_wrapper uses named constants, handles edge-cases
  - Tool descriptions tightened for better agent routing
  - Prompt template extracted to a named constant for readability
  - AgentExecutor configured with return_intermediate_steps=False (quieter)
  - CLI test loop handles KeyboardInterrupt cleanly
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

import finance_tools

# Load .env once at import time
load_dotenv()

# ──────────────────────────────
# Configuration
# ──────────────────────────────
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME   = "llama-3.3-70b-versatile"
FAISS_PATH       = "faiss_rag.index"

# ──────────────────────────────
# Prompt template
# ──────────────────────────────
_PERSONA = """\
You are "Veena," an AI financial advisor and insurance agent for "ValuEnable Life Insurance".
You help users with their stock portfolio, market queries, upcoming IPOs, taxes, and insurance inquiries.

IMPORTANT RULES:
1. LANGUAGE: The user's detected language is {language}. Always reply in that language.
   - 'hi' or 'hindi' → respond in Hindi (Devanagari script).
   - 'en' or 'english' → respond in English.
   - Never switch languages unless the user explicitly asks.

2. EMOTION AWARENESS: The user's detected emotion is {emotion}.
   - 'stressed'  → calm, reassuring tone.
   - 'confused'  → simple language, step-by-step guidance.
   - 'calm' / 'neutral' → professional and concise.

3. Keep responses concise and factual. Do NOT hallucinate.
4. Use the right tool for each query. If you cannot find the answer, say so honestly.

TOOLS:
------
{tools}

ReAct format:
```
Thought: Do I need to use a tool? Yes
Action: <one of [{tool_names}]>
Action Input: <input to the action>
Observation: <result>
```

When you have a final answer:
```
Thought: Do I need to use a tool? No
Final Answer: <your response to the user>
```

Conversation history:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

_PROMPT = PromptTemplate(
    template=_PERSONA,
    input_variables=[
        "input", "chat_history", "agent_scratchpad",
        "tools", "tool_names", "language", "emotion",
    ],
)


# ──────────────────────────────
# Tool wrappers
# ──────────────────────────────
def _portfolio_wrapper(input_str: str) -> str:
    """Parse 'view' or 'add, SYMBOL, QTY, PRICE' and call portfolio_manager."""
    parts = [p.strip() for p in input_str.split(",")]
    action = parts[0].lower()

    if action == "view":
        return finance_tools.portfolio_manager(action="view")

    if action == "add":
        if len(parts) < 4:
            return "Format for add: add, SYMBOL, QUANTITY, BUY_PRICE (e.g. 'add, TCS, 10, 2500')"
        try:
            return finance_tools.portfolio_manager(
                action="add",
                symbol=parts[1],
                quantity=int(parts[2]),
                buy_price=float(parts[3]),
            )
        except ValueError:
            return "Quantity must be an integer and buy_price must be a number."

    return "Unknown action. Use 'view' or 'add, SYMBOL, QUANTITY, BUY_PRICE'."


def _insurance_wrapper(input_str: str) -> str:
    parts = [p.strip() for p in input_str.split(",")]
    if len(parts) != 3:
        return "Format: investments, insurance_cover, annual_income (e.g. '1000000, 200000, 500000')"
    try:
        return finance_tools.analyze_insurance_gap(
            float(parts[0]), float(parts[1]), float(parts[2])
        )
    except ValueError:
        return "All three values must be numbers."


# ──────────────────────────────
# Public factory
# ──────────────────────────────
def build_agent() -> AgentExecutor:
    """Build and return a ready-to-use LangChain AgentExecutor."""

    # 1. Validate API key early
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment variables."
        )

    # 2. Models
    embedding = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0, api_key=groq_api_key)

    # 3. Vector store
    if not Path(FAISS_PATH).exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_PATH}'. "
            "Run index_documents.py first."
        )
    print(f"📂  Loading FAISS index: {FAISS_PATH}")
    vector_db = FAISS.load_local(
        FAISS_PATH, embeddings=embedding, allow_dangerous_deserialization=True
    )
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 4. Tools
    tools = [
        Tool(
            name="rag_search_transcripts",
            func=lambda q: "\n".join(d.page_content for d in retriever.invoke(q)),
            description=(
                "Search the internal knowledge base for customer history, "
                "product policies, and company FAQs."
            ),
        ),
        Tool(
            name="StockPrice",
            func=finance_tools.get_stock_price,
            description=(
                "Get real-time price, change %, high, low, and volume for a stock. "
                "Input: ticker symbol (e.g. RELIANCE.NS, TCS, AAPL)."
            ),
        ),
        Tool(
            name="PortfolioManager",
            func=_portfolio_wrapper,
            description=(
                "View or update the user's stock portfolio. "
                "Input: 'view'  OR  'add, SYMBOL, QUANTITY, BUY_PRICE'."
            ),
        ),
        Tool(
            name="MarketSummary",
            func=lambda _: finance_tools.get_market_summary(),
            description="Get live NIFTY 50 / SENSEX summary and top movers. Input: empty string.",
        ),
        Tool(
            name="IPOTracker",
            func=lambda _: finance_tools.get_upcoming_ipos(),
            description="List upcoming IPOs with dates, price bands, and GMP. Input: empty string.",
        ),
        Tool(
            name="TaxCalculator",
            func=lambda s: _tax_wrapper(s),
            description=(
                "Compare Old vs New income tax regime for FY 2024-25. "
                "Input: 'income, deductions_80c, deductions_80d' (only income required). "
                "Example: '1200000, 150000, 25000'."
            ),
        ),
        Tool(
            name="InsuranceGapAnalysis",
            func=_insurance_wrapper,
            description=(
                "Check if life insurance cover is adequate (10× income rule). "
                "Input: 'investments, insurance_cover, annual_income'. "
                "Example: '1000000, 200000, 500000'."
            ),
        ),
    ]

    # 5. Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False,
        input_key="input",
        output_key="output",
    )

    # 6. Agent
    agent = create_react_agent(llm, tools, _PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6,
        early_stopping_method="force",
        return_intermediate_steps=False,
    )

    print("✅  Veena AI agent is ready!")
    return executor


def _tax_wrapper(input_str: str) -> str:
    parts = [p.strip() for p in input_str.split(",")]
    try:
        income = float(parts[0])
        d80c   = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
        d80d   = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        return finance_tools.calculate_tax(income, d80c, d80d)
    except (ValueError, IndexError):
        return "Please provide at least an income value. Format: 'income, 80c_deductions, 80d_deductions'."


# ──────────────────────────────
# CLI quick test
# ──────────────────────────────
if __name__ == "__main__":
    import sys

    try:
        ag = build_agent()
    except (ValueError, FileNotFoundError) as exc:
        print(f"❌  {exc}")
        sys.exit(1)

    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            q = input("🗣  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋  Goodbye!")
            break

        if q.lower() in {"exit", "quit"}:
            print("👋  Goodbye!")
            break

        if not q:
            continue

        # Auto-detect language for CLI testing
        lang = "hi" if any("\u0900" <= c <= "\u097f" for c in q) else "en"
        try:
            result = ag.invoke({"input": q, "language": lang, "emotion": "neutral"})
            print(f"🤖  Veena: {result['output']}\n")
        except Exception as exc:
            print(f"❌  Error: {exc}\n")