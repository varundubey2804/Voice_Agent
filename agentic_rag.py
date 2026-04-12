"""
agentic_rag.py ────────────────────────────────────────────────────────────
Self‑contained LangChain Agent + RAG pipeline
• Loads a pre-built FAISS index for document retrieval.
• Exposes Tools for the agent to call (RAG + all finance_tools).
• Keeps conversational memory and uses the 'Veena' persona.

Tools wired in:
  1.  rag_search_transcripts  — FAISS internal KB
  2.  StockPrice              — Real-time price via yfinance
  3.  PortfolioManager        — Add / view portfolio holdings
  4.  MarketSummary           — NIFTY 50 & SENSEX overview
  5.  IPOTracker              — Upcoming IPOs + GMP
  6.  TaxCalculator           — Old vs New regime comparison
  7.  InsuranceGapAnalysis    — Under-insurance check
  8.  InvestmentComparator    — LIC vs Stocks vs MF (scenario-aware)
  9.  SIPCalculator           — Monthly SIP projection
  10. LICPolicyInfo           — Detailed info on any LIC policy (lic_policies.py)
  11. LICPolicyCatalogue      — Full list of all available LIC policies
  12. LICPolicyCompare        — Side-by-side comparison of 2-3 LIC policies
  13. LICPolicyRecommend      — Personalised LIC policy recommendation by profile
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
# Tool wrappers
# ──────────────────────────────

def _make_portfolio_wrapper(finance_tools):
    """
    Parses agent input and delegates to finance_tools.portfolio_manager.
    Input format:  'view'  OR  'add, SYMBOL, quantity, buy_price'
    """
    def _wrapper(input_str: str) -> str:
        parts = [p.strip() for p in input_str.split(',')]
        action = parts[0].lower()
        if action == 'view':
            return finance_tools.portfolio_manager(action='view')
        elif action == 'add':
            if len(parts) >= 4:
                try:
                    return finance_tools.portfolio_manager(
                        action='add',
                        symbol=parts[1],
                        quantity=int(parts[2]),
                        buy_price=float(parts[3]),
                    )
                except (ValueError, IndexError):
                    return "Invalid format. Use: add, SYMBOL, quantity, buy_price (e.g. 'add, TCS, 10, 2500')"
            return "To add a holding provide: add, SYMBOL, quantity, buy_price"
        return "Invalid action. Use 'view' OR 'add, SYMBOL, quantity, buy_price'."
    return _wrapper


def _make_tax_wrapper(finance_tools):
    """
    Parses agent input and delegates to finance_tools.calculate_tax.
    Input format:  'income'  OR  'income, 80c_deduction, 80d_deduction'
    """
    def _wrapper(input_str: str) -> str:
        parts = [p.strip() for p in input_str.split(',')]
        try:
            income  = float(parts[0])
            d_80c   = float(parts[1]) if len(parts) >= 2 and parts[1] else 0.0
            d_80d   = float(parts[2]) if len(parts) >= 3 and parts[2] else 0.0
            return finance_tools.calculate_tax(income, d_80c, d_80d)
        except (ValueError, IndexError):
            return "Invalid format. Provide: income [, 80c_deductions, 80d_deductions] (e.g. '1200000, 150000, 25000')"
    return _wrapper


def _make_insurance_wrapper(finance_tools):
    """
    Input format: 'investments, insurance_cover, annual_income'
    """
    def _wrapper(input_str: str) -> str:
        parts = [p.strip() for p in input_str.split(',')]
        if len(parts) == 3:
            try:
                return finance_tools.analyze_insurance_gap(
                    float(parts[0]), float(parts[1]), float(parts[2])
                )
            except ValueError:
                pass
        return "Invalid format. Provide: investments, insurance_cover, annual_income (e.g. '1000000, 200000, 500000')"
    return _wrapper


def _make_comparator_wrapper(finance_tools):
    """
    Input format: 'OPTION1, OPTION2 [,OPTION3] [| amount=X] [| years=Y] [| scenario=Z]'
    Options : LIC, STOCKS, MUTUAL_FUNDS
    Scenario: normal | war | recession | inflation

    Examples
        'LIC, STOCKS'
        'LIC, STOCKS, MUTUAL_FUNDS | amount=500000 | years=15 | scenario=inflation'
    """
    def _wrapper(input_str: str) -> str:
        # Split on '|' to separate option list from keyword overrides
        segments = [s.strip() for s in input_str.split('|')]
        options_raw = [o.strip() for o in segments[0].split(',') if o.strip()]

        kwargs = {"amount": 100_000, "years": 10, "scenario": "normal"}
        for seg in segments[1:]:
            if '=' in seg:
                key, val = seg.split('=', 1)
                key = key.strip().lower()
                val = val.strip()
                if key == 'amount':
                    try:
                        kwargs['amount'] = float(val)
                    except ValueError:
                        pass
                elif key == 'years':
                    try:
                        kwargs['years'] = int(val)
                    except ValueError:
                        pass
                elif key == 'scenario':
                    kwargs['scenario'] = val.lower()

        if not options_raw:
            return "Please provide at least one option: LIC, STOCKS, or MUTUAL_FUNDS."

        try:
            return finance_tools.compare_investments(
                options=options_raw,
                amount=kwargs['amount'],
                years=kwargs['years'],
                scenario=kwargs['scenario'],
            )
        except Exception as e:
            return f"Could not run comparison: {e}"
    return _wrapper


def _sip_calculator(input_str: str) -> str:
    """
    Standalone SIP calculator (no finance_tools dependency).
    Input format: 'monthly_amount, expected_annual_return%, years'
    Example     : '5000, 12, 10'
    """
    parts = [p.strip() for p in input_str.split(',')]
    try:
        monthly_amount = float(parts[0])
        annual_rate    = float(parts[1]) / 100          # convert % to decimal
        years          = int(parts[2])
    except (ValueError, IndexError):
        return "Invalid format. Provide: monthly_amount, expected_annual_return%, years (e.g. '5000, 12, 10')"

    monthly_rate     = annual_rate / 12
    n_months         = years * 12
    total_invested   = monthly_amount * n_months

    if monthly_rate > 0:
        future_value = monthly_amount * (((1 + monthly_rate) ** n_months - 1) / monthly_rate) * (1 + monthly_rate)
    else:
        future_value = total_invested

    wealth_gained = future_value - total_invested

    # Year-by-year snapshot (every 5 years)
    milestones = []
    for yr in range(5, years + 1, 5):
        nm = yr * 12
        if monthly_rate > 0:
            fv = monthly_amount * (((1 + monthly_rate) ** nm - 1) / monthly_rate) * (1 + monthly_rate)
        else:
            fv = monthly_amount * nm
        milestones.append(f"  Year {yr:>2}: ₹{fv:>14,.2f}  (invested ₹{monthly_amount * nm:>12,.2f})")

    lines = [
        f"📈 SIP Calculator",
        f"Monthly SIP     : ₹{monthly_amount:,.2f}",
        f"Annual Return   : {annual_rate * 100:.1f}%",
        f"Duration        : {years} years ({n_months} months)",
        f"",
        f"Total Invested  : ₹{total_invested:>14,.2f}",
        f"Future Value    : ₹{future_value:>14,.2f}",
        f"Wealth Gained   : ₹{wealth_gained:>14,.2f}  ({wealth_gained / total_invested * 100:.1f}% gain)",
        f"",
    ]
    if milestones:
        lines.append("Milestone snapshots:")
        lines.extend(milestones)

    return "\n".join(lines)


# ──────────────────────────────
# Public factory
# ──────────────────────────────

def build_agent():
    """Return a LangChain AgentExecutor with memory + all finance tools."""

    # 1) Load environment / API key
    from dotenv import load_dotenv
    load_dotenv()

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Set it in your .env file or environment variables."
        )

    # 2) Initialize models
    embedding = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=0,
        api_key=groq_api_key,
    )

    # 3) Load the pre-built FAISS vector DB
    if not Path(FAISS_PATH).exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_PATH}'. "
            "Please run 'index_documents.py' first to build it."
        )
    print(f"📂  Loading FAISS index: {FAISS_PATH}")
    vector_db = FAISS.load_local(
        FAISS_PATH,
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )

    # 4) Import finance_tools and lic_policies (deferred so FAISS error surfaces first)
    import finance_tools
    import lic_policies

    # 5) Build all tools
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    tools = [
        # ── Knowledge base ────────────────────────────────────────────────────
        Tool(
            name="rag_search_transcripts",
            func=lambda q: "\n".join([d.page_content for d in retriever.invoke(q)]),
            description=(
                "Search the internal knowledge base (FAISS) for facts about customer "
                "history, company policies, and product information. Use this before "
                "answering questions about ValuEnable policies or past interactions."
            ),
        ),

        # ── Market data ───────────────────────────────────────────────────────
        Tool(
            name="StockPrice",
            func=finance_tools.get_stock_price,
            description=(
                "Get real-time price, change %, day-high/low, and volume for a stock. "
                "Input: ticker symbol (e.g. RELIANCE.NS, TCS, INFY)."
            ),
        ),
        Tool(
            name="MarketSummary",
            func=lambda _: finance_tools.get_market_summary(),
            description=(
                "Get today's overall market summary: NIFTY 50, SENSEX levels, "
                "market trend, top 5 gainers and losers. Input: empty string."
            ),
        ),
        Tool(
            name="IPOTracker",
            func=lambda _: finance_tools.get_upcoming_ipos(),
            description=(
                "Get a list of upcoming IPOs with open/close dates, price band, "
                "lot size, and Grey Market Premium (GMP). Input: empty string."
            ),
        ),

        # ── Portfolio ─────────────────────────────────────────────────────────
        Tool(
            name="PortfolioManager",
            func=_make_portfolio_wrapper(finance_tools),
            description=(
                "Manage the user's stock portfolio. "
                "To VIEW holdings use input: 'view'. "
                "To ADD a holding use input: 'add, SYMBOL, quantity, buy_price' "
                "(e.g. 'add, TCS, 10, 2500')."
            ),
        ),

        # ── Tax & insurance ───────────────────────────────────────────────────
        Tool(
            name="TaxCalculator",
            func=_make_tax_wrapper(finance_tools),
            description=(
                "Calculate and compare income tax under Old vs New regime (FY 2024-25). "
                "Input format: income [, 80c_deductions, 80d_deductions] "
                "(e.g. '1200000, 150000, 25000'). Only income is mandatory."
            ),
        ),
        Tool(
            name="InsuranceGapAnalysis",
            func=_make_insurance_wrapper(finance_tools),
            description=(
                "Check whether the user is adequately insured relative to their "
                "investments and income. "
                "Input format: investments, insurance_cover, annual_income "
                "(e.g. '1000000, 200000, 500000')."
            ),
        ),

        # ── Investment analysis ───────────────────────────────────────────────
        Tool(
            name="InvestmentComparator",
            func=_make_comparator_wrapper(finance_tools),
            description=(
                "Compare investment options (LIC, STOCKS, MUTUAL_FUNDS) on risk, "
                "return, liquidity, tax efficiency, and scenario-specific behaviour. "
                "Input format: 'OPTION1, OPTION2 [,OPTION3] [| amount=X] [| years=Y] [| scenario=Z]' "
                "where scenario is one of: normal, war, recession, inflation. "
                "Example: 'LIC, STOCKS, MUTUAL_FUNDS | amount=500000 | years=15 | scenario=inflation'."
            ),
        ),
        Tool(
            name="SIPCalculator",
            func=_sip_calculator,
            description=(
                "Calculate the future value of a monthly SIP investment using "
                "compound interest. Shows total invested, final corpus, wealth gained, "
                "and year-by-year milestones. "
                "Input format: 'monthly_amount, expected_annual_return%, years' "
                "(e.g. '5000, 12, 10')."
            ),
        ),

        # ── LIC Policy Knowledge Base ─────────────────────────────────────────
        Tool(
            name="LICPolicyInfo",
            func=lic_policies.get_policy_info,
            description=(
                "Get detailed information about a specific LIC policy or a category of policies. "
                "Covers: Jeevan Anand, Jeevan Labh, Jeevan Umang, Jeevan Amar, Tech-Term, "
                "New Endowment, Bima Ratna, Saral Jeevan Bima, Arogya Rakshak (health), "
                "Cancer Cover, Jeevan Arogya, Jeevan Shanti (pension), SIIP (ULIP), "
                "New Children's Money Back, Jeevan Tarun, Aadhaar Stambh/Shila. "
                "Input: policy name OR category (e.g. 'Jeevan Anand', 'health', 'child plan', 'pension')."
            ),
        ),
        Tool(
            name="LICPolicyCatalogue",
            func=lambda _: lic_policies.list_all_policies(),
            description=(
                "Show a complete catalogue / list of all LIC policies available in the knowledge base, "
                "grouped by category (Life, Health, Pension, Child, Women). "
                "Input: empty string."
            ),
        ),
        Tool(
            name="LICPolicyCompare",
            func=lic_policies.compare_policies,
            description=(
                "Compare 2 or 3 LIC policies side by side on key attributes: "
                "entry age, term, sum assured, tax benefit, loan facility, pros, and cons. "
                "Input: comma-separated policy names "
                "(e.g. 'Jeevan Anand, Jeevan Labh' or 'Arogya Rakshak, Cancer Cover, Jeevan Arogya')."
            ),
        ),
        Tool(
            name="LICPolicyRecommend",
            func=lic_policies.recommend_policy,
            description=(
                "Recommend the most suitable LIC policy based on the user's profile. "
                "Input: comma-separated key=value pairs. "
                "Keys: age, goal, risk, family, health. "
                "Goals: savings, term, retirement, child education, health cover, cancer, pension, investment. "
                "Risk: low, medium, high. "
                "Example: 'age=35, goal=child education, risk=low' "
                "Example: 'age=28, goal=retirement, risk=medium' "
                "Example: 'age=45, goal=health cover'"
            ),
        ),
    ]

    # 6) Persona + ReAct prompt
    persona = """You are "Veena," an AI financial advisor and insurance agent for "ValuEnable Life Insurance."
You help users with: stock portfolio, market queries, upcoming IPOs, taxes, insurance inquiries, SIP planning, investment comparisons, and detailed LIC policy guidance (Jeevan Anand, Jeevan Labh, Jeevan Umang, Arogya Rakshak, Cancer Cover, Jeevan Shanti, child plans, and more).

LANGUAGE RULE (CRITICAL): Detected language = {language}.
- If {language} is 'hi' → your ENTIRE Final Answer MUST be in Hindi (Devanagari script). No English words.
- If {language} is 'en' → respond in English only.

TONE RULE: The user's message may start with an [emotion tag] such as [stressed], [confused], or [calm].
- [stressed]  → use a calm, reassuring tone.
- [confused]  → simplify your answer; use step-by-step guidance.
- [calm] or [neutral] → professional and helpful tone.
Strip the tag before composing your Final Answer.

TOOL USAGE GUIDE (decide BEFORE the first Thought):
- Stock price query → StockPrice
- Market / NIFTY / SENSEX → MarketSummary
- Portfolio add/view → PortfolioManager
- IPO query → IPOTracker
- Tax calculation → TaxCalculator
- Insurance gap / under-insured → InsuranceGapAnalysis
- LIC policy info / details → LICPolicyInfo
- List all LIC policies → LICPolicyCatalogue
- Compare two LIC policies → LICPolicyCompare
- Which LIC plan is best for me → LICPolicyRecommend
- SIP / monthly investment projection → SIPCalculator
- Compare LIC vs stocks vs mutual funds → InvestmentComparator
- General investment advice (how to invest, best options, diversification) → Answer DIRECTLY without any tool. Give a concise 3-4 point advisory in the detected language.
- Customer history / policy docs → rag_search_transcripts

RULES:
1. Use AT MOST ONE tool per question unless the user explicitly asks for multiple things.
2. For general advice questions (how to invest, best investment, diversification tips) — do NOT call any tool. Answer directly from your knowledge in 3-4 concise points.
3. Keep responses under 120 words. Be conversational, not a report.
4. NEVER hallucinate numbers or policy details — use tools for specifics.
5. Always recommend consulting a SEBI/IRDAI-registered advisor for large financial decisions."""

    # NOTE: No markdown code fences around Action/Observation blocks — the LLM
    # sometimes reproduces the fences literally which breaks the ReAct parser.
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
            "tools", "tool_names", "language",
        ],
    )

    # 7) Assemble agent + memory
    agent = create_react_agent(llm, tools, prompt)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False,
        input_key="input",
        output_key="output",
    )

    def _parse_error_handler(error: Exception) -> str:
        """Return a soft recovery message instead of crashing on parse failures."""
        return (
            "I encountered a formatting issue. "
            "Thought: Do I need to use a tool? No\n"
            "Final Answer: I'm sorry, I had trouble processing that. "
            "Could you please rephrase your question?"
        )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=_parse_error_handler,
        max_iterations=5,
        max_execution_time=30,          # hard 30-second wall-clock limit
        early_stopping_method="generate", # ask LLM to summarise instead of raw error string
    )

    print("✅  Agentic RAG with 'Veena' persona is ready!")
    return agent_executor


# ──────────────────────────────
# CLI quick-test
# ──────────────────────────────
if __name__ == "__main__":
    ag = build_agent()

    print("\nType 'exit' to quit.\n")
    while True:
        q = input("🗣  You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        # Auto-detect language from Devanagari characters
        lang = "hi" if any('\u0900' <= c <= '\u097f' for c in q) else "en"

        try:
            result = ag.invoke({"input": q, "language": lang})
            print(f"🤖 Veena: {result['output']}\n")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🤖 Veena: I'm having trouble responding right now. Please try again.\n")
