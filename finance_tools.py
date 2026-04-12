"""
finance_tools.py — Optimized
Key improvements:
  - Parallel yfinance calls via ThreadPoolExecutor (portfolio view was O(n) serial)
  - TTL cache for market summary & dashboard (avoids redundant API calls)
  - get_dashboard_data reuses cached market data instead of fetching twice
"""

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Optional

import yfinance as yf

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

# ── Simple TTL cache ──────────────────────────────────────────────────────────
_cache: dict = {}

def _cached(key: str, ttl_seconds: int, fn):
    """Call fn() and cache result for ttl_seconds."""
    now = time.monotonic()
    if key in _cache:
        val, ts = _cache[key]
        if now - ts < ttl_seconds:
            return val
    val = fn()
    _cache[key] = (val, now)
    return val


# ── 1. Stock price ────────────────────────────────────────────────────────────

def get_stock_price(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol.endswith(('.NS', '.BO')) and symbol.isalpha():
        symbol += '.NS'
    try:
        info          = yf.Ticker(symbol).fast_info
        current_price = info.last_price
        prev_close    = info.previous_close
        change        = current_price - prev_close
        change_pct    = (change / prev_close) * 100 if prev_close else 0
        return (f"Stock: {symbol}\n"
                f"Price: ₹{current_price:.2f}\n"
                f"Change: ₹{change:.2f} ({change_pct:.2f}%)\n"
                f"High: ₹{info.day_high:.2f} | Low: ₹{info.day_low:.2f}\n"
                f"Volume: {info.last_volume:,}")
    except Exception as e:
        return f"Could not fetch price for {symbol}. Error: {e}"


# ── 2. Portfolio Manager ──────────────────────────────────────────────────────

_EXECUTOR = ThreadPoolExecutor(max_workers=8)

def _fetch_price(sym: str) -> tuple[str, float]:
    """Fetch a single stock's current price (used for parallel portfolio view)."""
    try:
        full_sym = sym if sym.endswith(('.NS', '.BO')) else sym + '.NS'
        return sym, yf.Ticker(full_sym).fast_info.last_price
    except Exception:
        return sym, None   # Caller handles fallback


def _supabase_request(method: str, endpoint: str, payload=None):
    url     = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if method == "POST":
        headers["Prefer"] = "return=representation"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload else None,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode())


_LOCAL_DB = "portfolio_db.json"

def _load_local():
    if os.path.exists(_LOCAL_DB):
        with open(_LOCAL_DB) as f:
            return json.load(f)
    return []

def _save_local(data):
    with open(_LOCAL_DB, "w") as f:
        json.dump(data, f)


def portfolio_manager(
    action: str,
    symbol: Optional[str] = None,
    quantity: Optional[int] = None,
    buy_price: Optional[float] = None,
) -> str:
    use_supabase = bool(SUPABASE_URL and SUPABASE_KEY)

    if action.lower() == 'add':
        if not symbol or quantity is None or buy_price is None:
            return "Please provide symbol, quantity, and buy_price."
        holding = {"symbol": symbol.upper(), "quantity": quantity, "buy_price": buy_price}
        try:
            if use_supabase:
                _supabase_request("POST", "portfolio", holding)
            else:
                db = _load_local(); db.append(holding); _save_local(db)
            return f"Added {quantity} shares of {symbol} at ₹{buy_price}."
        except Exception as e:
            return f"Failed to add: {e}"

    elif action.lower() == 'view':
        try:
            holdings_raw = _supabase_request("GET", "portfolio?select=*") if use_supabase else _load_local()
            if not holdings_raw:
                return "Your portfolio is currently empty."

            # Aggregate
            aggregated: dict = {}
            for h in holdings_raw:
                sym = h['symbol']
                agg = aggregated.setdefault(sym, {"qty": 0, "total_cost": 0})
                agg["qty"]        += h['quantity']
                agg["total_cost"] += h['quantity'] * h['buy_price']

            # ✅ Parallel price fetch
            futures = {_EXECUTOR.submit(_fetch_price, sym): sym for sym in aggregated}
            prices  = {}
            for fut in as_completed(futures, timeout=10):
                sym, price = fut.result()
                prices[sym] = price

            total_invested = total_value = 0
            lines = ["Your Portfolio Holdings:"]
            for sym, data in aggregated.items():
                qty      = data["qty"]
                avg_buy  = data["total_cost"] / qty if qty else 0
                curr_p   = prices.get(sym) or avg_buy   # fallback to buy price
                invested = qty * avg_buy
                curr_val = qty * curr_p
                pnl      = curr_val - invested
                pnl_pct  = (pnl / invested * 100) if invested else 0
                total_invested += invested
                total_value    += curr_val
                lines.append(
                    f"- {sym}: {qty} sh | Buy ₹{avg_buy:.2f} | Now ₹{curr_p:.2f} | "
                    f"P&L ₹{pnl:.2f} ({pnl_pct:.2f}%)"
                )

            total_pnl     = total_value - total_invested
            total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0
            lines.append(
                f"\nTotal Invested: ₹{total_invested:.2f}\n"
                f"Current Value : ₹{total_value:.2f}\n"
                f"Total P&L     : ₹{total_pnl:.2f} ({total_pnl_pct:.2f}%)"
            )
            return "\n".join(lines)
        except Exception as e:
            return f"Failed to view portfolio: {e}"

    return "Invalid action. Use 'add' or 'view'."


# ── 3. Market summary (cached 60 s) ──────────────────────────────────────────

def _fetch_market_raw() -> dict:
    nifty   = yf.Ticker('^NSEI').fast_info
    sensex  = yf.Ticker('^BSESN').fast_info
    n_chg   = nifty.last_price - nifty.previous_close
    n_pct   = n_chg / nifty.previous_close * 100
    s_chg   = sensex.last_price - sensex.previous_close
    s_pct   = s_chg / sensex.previous_close * 100
    return {
        "nifty":  {"price": nifty.last_price,  "change": n_chg, "percent": n_pct},
        "sensex": {"price": sensex.last_price, "change": s_chg, "percent": s_pct},
        "trend":  "Bullish 📈" if n_chg > 0 else "Bearish 📉",
        "top_gainers": [
            {"symbol": "TATA MOTORS", "change": "+3.5%"},
            {"symbol": "RELIANCE",    "change": "+2.1%"},
            {"symbol": "INFOSYS",     "change": "+1.8%"},
            {"symbol": "HDFC BANK",   "change": "+1.5%"},
            {"symbol": "ITC",         "change": "+1.2%"},
        ],
        "top_losers": [
            {"symbol": "ADANI ENT", "change": "-2.5%"},
            {"symbol": "SUN PHARMA","change": "-1.8%"},
            {"symbol": "WIPRO",     "change": "-1.4%"},
            {"symbol": "L&T",       "change": "-1.1%"},
            {"symbol": "MARUTI",    "change": "-0.8%"},
        ],
    }


def get_market_raw() -> dict:
    """Cached market data (60 s TTL) shared by both text and dashboard endpoints."""
    return _cached("market", 60, _fetch_market_raw)


def get_market_summary() -> str:
    try:
        m = get_market_raw()
        gainers = "\n".join(f"{i+1}. {g['symbol']} ({g['change']})" for i, g in enumerate(m["top_gainers"]))
        losers  = "\n".join(f"{i+1}. {l['symbol']} ({l['change']})" for i, l in enumerate(m["top_losers"]))
        return (f"Market Summary:\n"
                f"NIFTY 50 : {m['nifty']['price']:.2f} ({m['nifty']['percent']:.2f}%)\n"
                f"SENSEX   : {m['sensex']['price']:.2f} ({m['sensex']['percent']:.2f}%)\n"
                f"Trend    : {m['trend']}\n\n"
                f"Top 5 Gainers:\n{gainers}\n\n"
                f"Top 5 Losers:\n{losers}")
    except Exception as e:
        return f"Could not fetch market summary. Error: {e}"


# ── 4. IPO Tracker ────────────────────────────────────────────────────────────

def get_upcoming_ipos() -> str:
    return (
        "Upcoming IPOs:\n"
        "1. Swiggy Ltd\n   Open: 15-Nov | Close: 18-Nov | Band: ₹370-390 | Lot: 38 | GMP: ₹25 | 2.5x\n"
        "2. NTPC Green Energy\n   Open: 22-Nov | Close: 25-Nov | Band: ₹100-108 | Lot: 138 | GMP: ₹12 | Upcoming\n"
        "Recommendation: NTPC Green shows strong fundamentals. Consider subscribing."
    )


# ── 5. Tax Calculator ─────────────────────────────────────────────────────────

def calculate_tax(income: float, deductions_80c: float = 0, deductions_80d: float = 0) -> str:
    old_taxable = max(0, income - 50_000 - min(deductions_80c, 150_000) - deductions_80d)
    old_tax = 0
    if old_taxable > 1_000_000:
        old_tax = (old_taxable - 1_000_000) * 0.30 + 112_500
    elif old_taxable > 500_000:
        old_tax = (old_taxable - 500_000) * 0.20 + 12_500
    elif old_taxable > 250_000:
        old_tax = (old_taxable - 250_000) * 0.05
    if old_taxable <= 500_000:
        old_tax = 0

    new_taxable = max(0, income - 75_000)
    new_tax = 0
    if new_taxable > 1_500_000:
        new_tax = (new_taxable - 1_500_000) * 0.30 + 150_000
    elif new_taxable > 1_200_000:
        new_tax = (new_taxable - 1_200_000) * 0.20 + 90_000
    elif new_taxable > 1_000_000:
        new_tax = (new_taxable - 1_000_000) * 0.15 + 60_000
    elif new_taxable > 700_000:
        new_tax = (new_taxable - 700_000) * 0.10 + 30_000
    elif new_taxable > 300_000:
        new_tax = (new_taxable - 300_000) * 0.05
    if new_taxable <= 700_000:
        new_tax = 0

    better = "New Regime" if new_tax < old_tax else "Old Regime"
    savings = abs(old_tax - new_tax)
    return (f"Tax for Income ₹{income:,.0f}:\n"
            f"Old Regime: ₹{old_tax:,.2f}\n"
            f"New Regime: ₹{new_tax:,.2f}\n"
            f"✅ {better} saves you ₹{savings:,.2f}")


# ── 6. Dashboard (reuses cache) ───────────────────────────────────────────────

def get_dashboard_data() -> dict:
    # ✅ Reuse cached market data — no duplicate API call
    try:
        market_data = get_market_raw()
    except Exception:
        market_data = None

    # Portfolio
    use_supabase     = bool(SUPABASE_URL and SUPABASE_KEY)
    portfolio_holdings = []
    total_invested = total_value = 0

    try:
        holdings_raw = _supabase_request("GET", "portfolio?select=*") if use_supabase else _load_local()
        if holdings_raw:
            aggregated: dict = {}
            for h in holdings_raw:
                sym = h['symbol']
                agg = aggregated.setdefault(sym, {"qty": 0, "total_cost": 0})
                agg["qty"]        += h['quantity']
                agg["total_cost"] += h['quantity'] * h['buy_price']

            # ✅ Parallel price fetch
            futures = {_EXECUTOR.submit(_fetch_price, sym): sym for sym in aggregated}
            prices  = {}
            for fut in as_completed(futures, timeout=10):
                sym, price = fut.result()
                prices[sym] = price

            for sym, data in aggregated.items():
                qty     = data["qty"]
                avg_buy = data["total_cost"] / qty if qty else 0
                curr_p  = prices.get(sym) or avg_buy
                invested = qty * avg_buy
                curr_val = qty * curr_p
                pnl      = curr_val - invested
                pnl_pct  = (pnl / invested * 100) if invested else 0
                total_invested += invested
                total_value    += curr_val
                portfolio_holdings.append({
                    "symbol": sym, "quantity": qty,
                    "buy_price": avg_buy, "current_price": curr_p,
                    "pnl": pnl, "pnl_pct": pnl_pct,
                })
    except Exception:
        pass

    total_pnl     = total_value - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0

    return {
        "market_summary": market_data,
        "portfolio": {
            "holdings": portfolio_holdings,
            "summary": {
                "total_invested":     total_invested,
                "total_current_value": total_value,
                "total_pnl":          total_pnl,
                "total_pnl_pct":      total_pnl_pct,
            },
        },
        "ipos": [
            {"name": "Swiggy Ltd",        "open": "15-Nov", "close": "18-Nov", "price_band": "₹370-390", "lot_size": 38,  "gmp": "₹25", "status": "2.5x"},
            {"name": "NTPC Green Energy", "open": "22-Nov", "close": "25-Nov", "price_band": "₹100-108", "lot_size": 138, "gmp": "₹12", "status": "Upcoming"},
        ],
    }


# ── 8. Investment Comparator ─────────────────────────────────────────────────

"""
compare_investments(options, amount, years, scenario)
  options  : list of 1-3 strings from {"LIC", "STOCKS", "MUTUAL_FUNDS"}
  amount   : lump-sum investment in ₹
  years    : investment horizon in years
  scenario : "normal" | "war" | "recession" | "inflation"

Returns a rich text comparison covering:
  • Expected returns (CAGR)
  • Risk level
  • Liquidity
  • Tax efficiency
  • Projected final value
  • Scenario-specific behaviour
  • Recommendation
"""

# ── Static profiles ───────────────────────────────────────────────────────────

_PROFILES = {
    "LIC": {
        "label":        "LIC / Endowment Plan",
        "base_cagr":    0.055,          # ~5–6% traditional endowment
        "risk":         "Very Low 🟢",
        "liquidity":    "Low (lock-in 3–5 yrs)",
        "tax":          "80C + 10(10D) exempt",
        "inflation_adj": False,
        "description":  "Government-backed life insurance cum savings. Guaranteed returns but low.",
        "scenario_modifier": {
            "war":        -0.005,   # marginally affected; sovereign backing helps
            "recession":  +0.010,   # safe-haven flow increases demand
            "inflation":  -0.020,   # fixed returns badly eroded by inflation
            "normal":      0.000,
        },
        "war_narrative":       "LIC is sovereign-backed. Policies continue; claim settlement may slow. Returns ~unchanged, making it one of the safer stores of value in conflict.",
        "recession_narrative": "LIC performs well here — people flee to guaranteed products. Surrender values are protected.",
        "inflation_narrative": "LIC's biggest weakness. A 7%+ inflation environment destroys the real value of fixed returns. Not recommended for high-inflation periods.",
    },
    "STOCKS": {
        "label":        "Direct Equities (NSE/BSE)",
        "base_cagr":    0.135,          # Nifty 50 ~13–15% long-term
        "risk":         "High 🔴",
        "liquidity":    "Very High (T+1 settlement)",
        "tax":          "LTCG 12.5% above ₹1.25L (>1yr); STCG 20%",
        "inflation_adj": True,
        "description":  "Direct ownership of company shares. High risk, high reward. Inflation-beating over long horizons.",
        "scenario_modifier": {
            "war":        -0.060,   # sharp drawdowns (15–40% historically)
            "recession":  -0.080,   # worst asset class in recessions
            "inflation":  +0.020,   # companies pass on costs; partial hedge
            "normal":      0.000,
        },
        "war_narrative":       "Stocks historically fall 20–40% at war onset (1999 Kargil: -25%, 2022 Russia-Ukraine: global -15%). Defence & commodity stocks spike. Recovery is fast post-conflict (6–18 months).",
        "recession_narrative": "Equities suffer most in recessions. Bear markets of 40–60% are common. However, SIP investors who stay invested benefit from lower NAVs and recover strongly.",
        "inflation_narrative": "Stocks are a reasonable inflation hedge. Companies raise prices; profit margins are maintained. Financials and commodities outperform. Better than fixed income.",
    },
    "MUTUAL_FUNDS": {
        "label":        "Mutual Funds (Equity / Hybrid)",
        "base_cagr":    0.120,          # Diversified large-cap ~12%
        "risk":         "Medium–High 🟡",
        "liquidity":    "High (T+3 redemption)",
        "tax":          "Equity MF LTCG 12.5% above ₹1.25L (>1yr); STCG 20%",
        "inflation_adj": True,
        "description":  "Professionally managed, diversified portfolio. Better risk-adjusted returns than direct stocks for most retail investors.",
        "scenario_modifier": {
            "war":        -0.040,   # diversification cushions the blow
            "recession":  -0.050,   # falls less than direct equity; fund managers rebalance
            "inflation":  +0.015,   # partial hedge via equity component
            "normal":      0.000,
        },
        "war_narrative":       "Mutual funds fall less than direct stocks during wars due to diversification. Balanced/hybrid funds with debt components provide further cushion. Recommended for war-time SIP continuation.",
        "recession_narrative": "Active fund managers can shift to defensive sectors (FMCG, pharma, IT). Debt-oriented hybrid funds particularly resilient. Still falls, but 20–30% less than pure equity.",
        "inflation_narrative": "Good inflation hedge, especially equity-oriented funds. ELSS funds also offer 80C benefits while keeping equity exposure.",
    },
}

_SCENARIO_CONTEXT = {
    "war": {
        "label": "War / Geopolitical Conflict 🪖",
        "global_note": "Historical data: Kargil (1999), Iraq War (2003), Russia-Ukraine (2022). Markets typically recover within 6–18 months post-conflict. Gold and defence stocks outperform.",
    },
    "recession": {
        "label": "Recession / Economic Slowdown 📉",
        "global_note": "Based on 2008 GFC and 2020 COVID crash patterns. Cash, gold, and sovereign bonds are king. Equity recovery takes 2–4 years on average.",
    },
    "inflation": {
        "label": "High Inflation Environment 🔥",
        "global_note": "India CPI >7% scenario (2022–23 style). Real returns matter more than nominal. Equity and real assets outperform fixed-income and traditional insurance plans.",
    },
    "normal": {
        "label": "Normal Market Conditions 📊",
        "global_note": "Stable GDP growth of 6–7%, CPI 4–5%, RBI neutral policy. Long-term wealth creation favours equities.",
    },
}


def _project_value(amount: float, cagr: float, years: int) -> float:
    return amount * ((1 + cagr) ** years)


def _bar(value: float, max_value: float, width: int = 20) -> str:
    filled = int((value / max_value) * width) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def compare_investments(
    options: list,
    amount: float = 100_000,
    years: int = 10,
    scenario: str = "normal",
) -> str:
    """
    Compare 2–3 investment options across risk, return, liquidity, tax,
    and scenario-specific behaviour.

    options  : e.g. ["LIC", "STOCKS", "MUTUAL_FUNDS"]
    amount   : ₹ invested (lump sum)
    years    : investment horizon
    scenario : "normal" | "war" | "recession" | "inflation"
    """
    # Normalise inputs
    options  = [o.upper().replace(" ", "_") for o in options]
    scenario = scenario.lower()

    valid_options  = [o for o in options if o in _PROFILES]
    valid_scenarios = list(_SCENARIO_CONTEXT.keys())

    if not valid_options:
        return (f"No valid options found. Choose from: {', '.join(_PROFILES.keys())}. "
                f"You provided: {', '.join(options)}")
    if scenario not in valid_scenarios:
        scenario = "normal"

    ctx    = _SCENARIO_CONTEXT[scenario]
    lines  = []

    lines.append("=" * 60)
    lines.append(f"📊  INVESTMENT COMPARATOR — {ctx['label']}")
    lines.append(f"    Amount: ₹{amount:,.0f}  |  Horizon: {years} years")
    lines.append("=" * 60)
    lines.append(f"\n🌐 Scenario Context:\n   {ctx['global_note']}\n")

    # ── Per-option breakdown ──────────────────────────────────────────────────
    projections = {}
    for key in valid_options:
        p    = _PROFILES[key]
        cagr = p["base_cagr"] + p["scenario_modifier"].get(scenario, 0)
        cagr = max(cagr, 0.01)   # floor at 1% — nothing goes truly to zero
        fv   = _project_value(amount, cagr, years)
        projections[key] = {"cagr": cagr, "fv": fv, "label": p["label"]}

    max_fv = max(v["fv"] for v in projections.values())

    lines.append("── PROJECTED RETURNS ────────────────────────────────────")
    for key in valid_options:
        p   = _PROFILES[key]
        prj = projections[key]
        gain = prj["fv"] - amount
        lines.append(f"\n  {p['label']}")
        lines.append(f"  CAGR (scenario-adj) : {prj['cagr']*100:.2f}%")
        lines.append(f"  Final Value         : ₹{prj['fv']:>12,.0f}")
        lines.append(f"  Gain                : ₹{gain:>12,.0f}  ({gain/amount*100:.1f}%)")
        lines.append(f"  Return bar          : {_bar(prj['fv'], max_fv)}")

    # ── Side-by-side comparison table ────────────────────────────────────────
    lines.append("\n── FEATURE COMPARISON ───────────────────────────────────")
    headers = ["Attribute"] + [_PROFILES[k]["label"].split(" (")[0][:18] for k in valid_options]
    lines.append("  " + " | ".join(f"{h:<18}" for h in headers))
    lines.append("  " + "-" * (22 * len(headers)))

    rows = [
        ("Risk Level",   [_PROFILES[k]["risk"]      for k in valid_options]),
        ("Liquidity",    [_PROFILES[k]["liquidity"]  for k in valid_options]),
        ("Tax",          [_PROFILES[k]["tax"]        for k in valid_options]),
        ("Inflation Adj",[("Yes ✅" if _PROFILES[k]["inflation_adj"] else "No ❌") for k in valid_options]),
    ]
    for attr, values in rows:
        row = [f"{attr:<18}"] + [f"{v:<18}" for v in values]
        lines.append("  " + " | ".join(row))

    # ── Scenario-specific narrative ───────────────────────────────────────────
    lines.append(f"\n── HOW EACH PERFORMS IN '{scenario.upper()}' ──────────────────────")
    narrative_key = f"{scenario}_narrative"
    for key in valid_options:
        p = _PROFILES[key]
        lines.append(f"\n  {p['label']}:")
        lines.append(f"  {p[narrative_key]}")

    # ── Recommendation ────────────────────────────────────────────────────────
    winner     = max(projections, key=lambda k: projections[k]["fv"])
    safest     = min(valid_options, key=lambda k: _PROFILES[k]["base_cagr"])
    winner_lbl = _PROFILES[winner]["label"]
    safest_lbl = _PROFILES[safest]["label"]

    lines.append("\n── RECOMMENDATION ───────────────────────────────────────")
    if scenario in ("war", "recession"):
        lines.append(f"  ⚠️  In {ctx['label']}, capital preservation > returns.")
        lines.append(f"  🛡️  Safest choice : {safest_lbl}")
        lines.append(f"  📈  Best recovery : {winner_lbl} (post-conflict/recovery phase)")
        lines.append(f"  💡  Strategy      : Hold {safest_lbl} now; shift to equities")
        lines.append(f"     (Stocks/MF) gradually as stability returns.")
    elif scenario == "inflation":
        lines.append(f"  🔥  High inflation erodes fixed returns (LIC). Equity is your friend.")
        lines.append(f"  📈  Best pick      : {winner_lbl}")
        lines.append(f"  💡  Strategy       : Avoid locking money in endowments; prefer")
        lines.append(f"     equity MFs or diversified direct stocks.")
    else:
        lines.append(f"  🏆  Best return over {years} yrs: {winner_lbl}")
        lines.append(f"  💡  Balanced strategy: 60% {_PROFILES['MUTUAL_FUNDS']['label'].split('(')[0].strip()}")
        lines.append(f"     20% Direct Stocks | 20% LIC/term for life cover.")
        lines.append(f"  ✅  LIC is best used as PURE TERM INSURANCE (not investment).")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ── 7. Insurance gap ─────────────────────────────────────────────────────────

def analyze_insurance_gap(investments: float, insurance_cover: float, annual_income: float) -> str:
    recommended = annual_income * 10
    total_assets = investments + insurance_cover
    if total_assets < recommended:
        shortfall   = recommended - total_assets
        tax_benefit = min(insurance_cover * 0.1, 150_000) * 0.3
        return (f"⚠️  Gap Analysis\n"
                f"Investments: ₹{investments:,.0f} | Cover: ₹{insurance_cover:,.0f}\n"
                f"Recommended: ₹{recommended:,.0f}\n"
                f"Under-insured by ₹{shortfall:,.0f}.\n"
                f"Increasing cover saves ~₹{tax_benefit:,.2f} in taxes (80C).")
    return "✅ Your insurance cover is adequate. Good job!"
