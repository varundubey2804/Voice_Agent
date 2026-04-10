"""
agentic_rag.py ────────────────────────────────────────────────────────────
Self‑contained LangChain Agent + RAG pipeline
• Loads a pre-built FAISS index for document retrieval.
• Exposes a Tool for the agent to call.
• Keeps conversational memory and uses the 'Veena' persona.
"""

import os
from pathlib import Path
from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# --- Imports for a robust, modern agent ---
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
    """Return a LangChain Agent with memory + RAG tool."""
    # 1) Initialize models
    embedding = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
         # Try to load from .env if not in env vars (though app.py usually handles this, we make sure)
         from dotenv import load_dotenv
         load_dotenv()
         groq_api_key = os.environ.get("GROQ_API_KEY")
         
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=0,
        api_key=groq_api_key
    )

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

    # 3) Create the retriever tool and new finance tools
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rag_tool  = Tool(
        name="rag_search_transcripts",
        func=lambda q: "\n".join([d.page_content for d in retriever.invoke(q)]),
        description="Search internal knowledge base (FAISS) for facts about customer history and policies.",
    )

    import finance_tools
    stock_price_tool = Tool(
        name="StockPrice",
        func=finance_tools.get_stock_price,
        description="Get real-time price, change, high, low, and volume of a stock. Input should be the stock ticker symbol (e.g., RELIANCE.NS, TCS)."
    )

    def _portfolio_wrapper(input_str: str) -> str:
        parts = [p.strip() for p in input_str.split(',')]
        action = parts[0]
        if action.lower() == 'view':
            return finance_tools.portfolio_manager(action='view')
        elif action.lower() == 'add':
            if len(parts) >= 4:
                return finance_tools.portfolio_manager(action='add', symbol=parts[1], quantity=int(parts[2]), buy_price=float(parts[3]))
            return "To add, provide: add, symbol, quantity, buy_price"
        return "Invalid action. Use 'add' or 'view'."

    portfolio_tool = Tool(
        name="PortfolioManager",
        func=_portfolio_wrapper,
        description="Manage user's stock portfolio. Input format: 'view' OR 'add, symbol, quantity, buy_price' (e.g. 'add, TCS, 10, 2500')."
    )
    market_summary_tool = Tool(
        name="MarketSummary",
        func=lambda _: finance_tools.get_market_summary(),
        description="Get overall market summary (NIFTY 50, SENSEX) and trend. The input should be an empty string."
    )
    ipo_tracker_tool = Tool(
        name="IPOTracker",
        func=lambda _: finance_tools.get_upcoming_ipos(),
        description="Get list of upcoming IPOs, dates, and Grey Market Premium (GMP). The input should be an empty string."
    )

    def _tax_wrapper(input_str: str) -> str:
        parts = [p.strip() for p in input_str.split(',')]
        if len(parts) >= 1 and parts[0]:
            income = float(parts[0])
            d80c = float(parts[1]) if len(parts) >= 2 and parts[1] else 0.0
            d80d = float(parts[2]) if len(parts) >= 3 and parts[2] else 0.0
            return finance_tools.calculate_tax(income, d80c, d80d)
        return "Please provide income."

    tax_calculator_tool = Tool(
        name="TaxCalculator",
        func=_tax_wrapper,
        description="Calculate and compare income tax for Old vs New regime. Input format: income, deductions_80c, deductions_80d (e.g. '1200000, 150000, 25000'). Only income is required."
    )
    insurance_gap_tool = Tool(
        name="InsuranceGapAnalysis",
        func=lambda input_str: finance_tools.analyze_insurance_gap(float(input_str.split(',')[0]), float(input_str.split(',')[1]), float(input_str.split(',')[2])) if len(input_str.split(',')) == 3 else "Format: investments,insurance_cover,annual_income",
        description="Analyze if user has enough insurance cover compared to investments and income. Input format: investments,insurance_cover,annual_income (comma separated numbers, e.g., 1000000,200000,500000)."
    )

    tools = [
        rag_tool,
        stock_price_tool,
        portfolio_tool,
        market_summary_tool,
        ipo_tracker_tool,
        tax_calculator_tool,
        insurance_gap_tool
    ]

    # 4) Define the agent's persona and instructions
    persona = """You are "Veena," an AI financial advisor and insurance agent for "ValuEnable life insurance".
You help users with their stock portfolio, market queries, upcoming IPOs, taxes, and insurance inquiries.

IMPORTANT:
1. LANGUAGE CONSISTENCY: The detected language of the user is: {language}. You MUST respond in this language ({language}).
   - If {language} is Hindi (or 'hi'), you MUST respond in Hindi (Devanagari script).
   - If {language} is English (or 'en'), you MUST respond in English.
   - Do NOT switch languages unless the user explicitly requests it.

2. EMOTION AWARENESS: The user's detected emotion is: {emotion}.
   - If {emotion} is 'stressed', respond with a calming, reassuring tone.
   - If {emotion} is 'confused', simplify your explanation and offer step-by-step guidance.
   - If {emotion} is 'calm' or 'neutral', use a professional and helpful tone.

3. Keep your response concise.
4. Do NOT hallucinate facts. Route to the appropriate tool based on user queries (e.g., StockPrice, MarketSummary, PortfolioManager, IPOTracker, TaxCalculator, InsuranceGapAnalysis). If the information is not in the tools, admit you don't know."""

    # 5) Create the prompt template for ReAct agent (using PromptTemplate, not ChatPromptTemplate)
    # This is the correct format for create_react_agent
    template = persona + """

TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: your final response to the human here
```

Previous conversation history:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    # Create the prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names", "language", "emotion"]
    )

    # 6) Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # 7) Create memory - using simple ConversationBufferMemory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=False,  # Return as string, not messages
        input_key="input",
        output_key="output"
    )

    # 8) Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="force"
    )
    
    print("✅ Agentic RAG with 'Veena' persona is ready!")
    return agent_executor


# ──────────────────────────────
# CLI quick test
# ──────────────────────────────
if __name__ == "__main__":
    ag = build_agent()
    # Mock detected language for testing
    detected_language = "en"
    print(f"🌍 Defaulting test language to: {detected_language}")
    
    while True:
        q = input("🗣  You: ")
        if q.lower() in {"exit", "quit"}:
            break
        # Correctly invoke the agent and access the output
        try:
            # Simple heuristic for testing: if input has devanagari, assume Hindi
            if any(u'\u0900' <= c <= u'\u097f' for c in q):
                lang = "hi"
            else:
                lang = "en"
                
            response = ag.invoke({"input": q, "language": lang})
            print("🤖 Veena:", response['output'])
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🤖 Veena: I apologize, I'm having trouble responding right now.")