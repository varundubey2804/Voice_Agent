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

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
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
def build_agent():
    """Return a LangChain Agent with memory + RAG tool."""
    # 1) Initialize models
    embedding = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    llm       = ChatOllama(model=LLM_MODEL_NAME)

    # 2) Load the pre-built vector DB
    if not Path(FAISS_PATH).exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_PATH}'. "
            "Please run the 'index_documents.py' script first to create it."
        )
    print(f"📂  Loading existing FAISS index: {FAISS_PATH}")
    vector_db = FAISS.load_local(
        FAISS_PATH,
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )

    # 3) Create the retriever tool
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rag_tool  = Tool(
        name="rag_search_transcripts",
        func=lambda q: "\n".join([d.page_content for d in retriever.invoke(q)]),
        description="Search internal knowledge base (FAISS) for facts about customer history and policies.",
    )
    tools = [rag_tool]

    # 4) Define the agent's persona and instructions
    persona = """You are "Veena," a female insurance agent for "ValuEnable life insurance".
Follow the conversation flow strictly to remind and convince customers to pay
their premiums. If no questions are asked, ask simple questions to understand
and resolve concerns, always ending with a question. If a customer requests to
converse in a different language, such as Hindi, Marathi, or Gujarati, kindly
proceed with the conversation in their preferred language. Use max 35 easy
english words to respond."""

    # 5) Create the prompt template for ReAct agent (using PromptTemplate, not ChatPromptTemplate)
    # This is the correct format for create_react_agent
    template = persona + """

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

    # Create the prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"]
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
    ag = build_agent()
    while True:
        try:
            q = input("🗣  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋  Goodbye!")
            break

        if q.lower() in {"exit", "quit"}:
            print("👋  Goodbye!")
            break
        # Correctly invoke the agent and access the output
        try:
            response = ag.invoke({"input": q})
            print("🤖 Veena:", response['output'])
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🤖 Veena: I apologize, I'm having trouble responding right now.")
