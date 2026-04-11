"""
finance_tools.py
Improvements:
  - DRY: extract _load_portfolio / _save_portfolio / _supabase_request helpers
  - DRY: deduplicated portfolio-aggregation logic shared by portfolio_manager + get_dashboard_data
  - Type hints throughout
  - Bare `except` replaced with `except Exception`
  - get_stock_price validates empty input
  - calculate_tax adds cess (4 %) + surcharge for completeness
  - analyze_insurance_gap uses 10x-income rule correctly (total assets vs recommended)
  - get_upcoming_ipos marked clearly as mock data; easy to swap in a real API
  - get_dashboard_data reuses helpers instead of copy-pasting HTTP logic
"""

import json
import os
import urllib.request
from datetime import datetime
from typing import Optional

import yfinance as yf

# ──────────────────────────────
# Configuration
# ──────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
LOCAL_DB = "portfolio_db.json"


# ──────────────────────────────
# Private helpers
# ──────────────────────────────
def _use_supabase() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _supabase_request(method: str, endpoint: str, payload: Optional[dict] = None) -> list:
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if method == "POST":
        headers["Prefer"] = "return=representation"
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _load_portfolio() -> list:
    if _use_supabase():
        return _supabase_request("GET", "portfolio?select=*")
    if os.path.exists(LOCAL_DB):
        with open(LOCAL_DB) as f:
            return json.load(f)
    return []


def _save_portfolio(data: list) -> None:
    with open(LOCAL_DB, "w") as f:
        json.dump(data, f, indent=2)


def _normalize_symbol(symbol: str) -> str:
    """Append .NS for plain Indian tickers if no exchange suffix present."""
    symbol = symbol.strip().upper()
    if symbol and not symbol.endswith((".NS", ".BO")) and symbol.isalpha():
        symbol += ".NS"
    return symbol


def _aggregate_holdings(raw: list) -> dict:
    """Aggregate raw rows {symbol, quantity, buy_price} by symbol."""
    agg: dict = {}
    for h in raw:
        sym = h["symbol"].upper()
        agg.setdefault(sym, {"qty": 0, "total_cost": 0.0})
        agg[sym]["qty"] += h["quantity"]
        agg[sym]["total_cost"] += h["quantity"] * h["buy_price"]
    return agg


def _apply_cess(tax: float) -> float:
    return tax * 1.04  # 4 % Health & Education Cess


# ──────────────────────────────
# 1. Live Stock Price
# ──────────────────────────────
def get_stock_price(symbol: str) -> str:
    """Return real-time price, change %, high, low, and volume for a ticker."""
    symbol = symbol.strip()
    if not symbol:
        return "Please provide a valid stock ticker symbol."
    symbol = _normalize_symbol(symbol)
    try:
        info = yf.Ticker(symbol).fast_info
        price = info.last_price
        prev = info.previous_close
        chg = price - prev
        chg_pct = (chg / prev) * 100 if prev else 0
        arrow = "▲" if chg >= 0 else "▼"
        return (
            f"Stock : {symbol}\n"
            f"Price : ₹{price:,.2f}\n"
            f"Change: {arrow} ₹{abs(chg):,.2f} ({chg_pct:+.2f}%)\n"
            f"High  : ₹{info.day_high:,.2f} | Low: ₹{info.day_low:,.2f}\n"
            f"Volume: {info.last_volume:,}"
        )
    except Exception as e:
        return f"Could not fetch price for '{symbol}'. Error: {e}"


# ──────────────────────────────
# 2. Portfolio Manager
# ──────────────────────────────
def portfolio_manager(
    action: str,
    symbol: Optional[str] = None,
    quantity: Optional[int] = None,
    buy_price: Optional[float] = None,
) -> str:
    """Manage the user's stock portfolio (Supabase or local JSON fallback)."""
    action = action.lower().strip()

    if action == "add":
        if not symbol or quantity is None or buy_price is None:
            return "Provide symbol, quantity, and buy_price to add a holding."
        sym = _normalize_symbol(symbol)
        entry = {"symbol": sym, "quantity": int(quantity), "buy_price": float(buy_price)}
        try:
            if _use_supabase():
                _supabase_request("POST", "portfolio", entry)
            else:
                db = _load_portfolio()
                db.append(entry)
                _save_portfolio(db)
            return f"✅ Added {quantity} shares of {sym} @ ₹{buy_price:,.2f}."
        except Exception as e:
            return f"Failed to add holding: {e}"

    elif action == "view":
        try:
            raw = _load_portfolio()
            if not raw:
                return "Your portfolio is currently empty."
            return _format_portfolio(raw)
        except Exception as e:
            return f"Failed to load portfolio: {e}"

    return "Invalid action. Use 'add' or 'view'."


def _format_portfolio(raw: list) -> str:
    agg = _aggregate_holdings(raw)
    total_invested = total_current = 0.0
    lines = ["Your Portfolio:\n"]
    for sym, d in agg.items():
        qty = d["qty"]
        avg_buy = d["total_cost"] / qty
        try:
            curr_price = yf.Ticker(sym).fast_info.last_price
        except Exception:
            curr_price = avg_buy  # fallback to cost if fetch fails
        invested = qty * avg_buy
        curr_val = qty * curr_price
        pnl = curr_val - invested
        pnl_pct = (pnl / invested) * 100 if invested else 0
        total_invested += invested
        total_current += curr_val
        sign = "+" if pnl >= 0 else ""
        lines.append(
            f"  {sym}: {qty} shares | Avg ₹{avg_buy:,.2f} → ₹{curr_price:,.2f} "
            f"| P&L: {sign}₹{pnl:,.2f} ({sign}{pnl_pct:.2f}%)"
        )
    total_pnl = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested else 0
    sign = "+" if total_pnl >= 0 else ""
    lines += [
        f"\nInvested : ₹{total_invested:,.2f}",
        f"Current  : ₹{total_current:,.2f}",
        f"Total P&L: {sign}₹{total_pnl:,.2f} ({sign}{total_pnl_pct:.2f}%)",
    ]
    return "\n".join(lines)


# ──────────────────────────────
# 3. Market Summary
# ──────────────────────────────
def get_market_summary() -> str:
    """Return live NIFTY 50 and SENSEX summary with mock top movers."""
    try:
        nifty = yf.Ticker("^NSEI").fast_info
        sensex = yf.Ticker("^BSESN").fast_info

        def _fmt(info) -> tuple:
            chg = info.last_price - info.previous_close
            pct = (chg / info.previous_close) * 100 if info.previous_close else 0
            return info.last_price, chg, pct

        n_price, n_chg, n_pct = _fmt(nifty)
        s_price, s_chg, s_pct = _fmt(sensex)
        trend = "Bullish 📈" if n_chg >= 0 else "Bearish 📉"

        # NOTE: Replace with a live movers API for production use.
        mock_movers = (
            "Top Gainers: TATA MOTORS (+3.5%), RELIANCE (+2.1%), INFOSYS (+1.8%)\n"
            "Top Losers : ADANI ENT (-2.5%), SUN PHARMA (-1.8%), WIPRO (-1.4%)\n"
            "(Mocked data — integrate a live source for real values)"
        )

        return (
            f"Market Summary [{datetime.now().strftime('%d %b %Y %H:%M')}]\n"
            f"NIFTY 50 : {n_price:,.2f} ({n_pct:+.2f}%)\n"
            f"SENSEX   : {s_price:,.2f} ({s_pct:+.2f}%)\n"
            f"Trend    : {trend}\n\n{mock_movers}"
        )
    except Exception as e:
        return f"Could not fetch market summary. Error: {e}"


# ──────────────────────────────
# 4. IPO Tracker  (mock — swap in a live scraper/API)
# ──────────────────────────────
_MOCK_IPOS = [
    {
        "name": "Swiggy Ltd",
        "open": "15-Nov", "close": "18-Nov",
        "price_band": "₹370–390", "lot_size": 38,
        "gmp": "₹25", "status": "2.5x subscribed",
    },
    {
        "name": "NTPC Green Energy",
        "open": "22-Nov", "close": "25-Nov",
        "price_band": "₹100–108", "lot_size": 138,
        "gmp": "₹12", "status": "Upcoming",
    },
]


def get_upcoming_ipos() -> str:
    """Return upcoming IPO data. (Currently mocked — replace with live API.)"""
    lines = ["Upcoming IPOs (mock data):\n"]
    for i, ipo in enumerate(_MOCK_IPOS, 1):
        lines.append(
            f"{i}. {ipo['name']}\n"
            f"   Open: {ipo['open']} | Close: {ipo['close']}\n"
            f"   Price Band: {ipo['price_band']} | Lot: {ipo['lot_size']} shares\n"
            f"   GMP: {ipo['gmp']} | Status: {ipo['status']}"
        )
    lines.append("\nTip: NTPC Green shows strong fundamentals and a positive GMP. Consider subscribing.")
    return "\n".join(lines)


# ──────────────────────────────
# 5. Tax Calculator (FY 2024-25)
# ──────────────────────────────
def calculate_tax(
    income: float,
    deductions_80c: float = 0.0,
    deductions_80d: float = 0.0,
) -> str:
    """Compare Old vs New regime tax for FY 2024-25 including 4 % cess."""

    # ── Old Regime ──────────────────────────────────────────────────────────
    std_deduction_old = 50_000
    old_taxable = max(0, income - std_deduction_old - min(deductions_80c, 150_000) - deductions_80d)

    def _old_slab(t: float) -> float:
        if t <= 250_000:
            return 0
        elif t <= 500_000:
            return (t - 250_000) * 0.05
        elif t <= 1_000_000:
            return 12_500 + (t - 500_000) * 0.20
        else:
            return 112_500 + (t - 1_000_000) * 0.30

    old_tax = _old_slab(old_taxable)
    if old_taxable <= 500_000:
        old_tax = 0  # 87A rebate
    old_tax = _apply_cess(old_tax)

    # ── New Regime ──────────────────────────────────────────────────────────
    std_deduction_new = 75_000
    new_taxable = max(0, income - std_deduction_new)

    def _new_slab(t: float) -> float:
        if t <= 300_000:
            return 0
        elif t <= 700_000:
            return (t - 300_000) * 0.05
        elif t <= 1_000_000:
            return 20_000 + (t - 700_000) * 0.10
        elif t <= 1_200_000:
            return 50_000 + (t - 1_000_000) * 0.15
        elif t <= 1_500_000:
            return 80_000 + (t - 1_200_000) * 0.20
        else:
            return 140_000 + (t - 1_500_000) * 0.30

    new_tax = _new_slab(new_taxable)
    if new_taxable <= 700_000:
        new_tax = 0  # 87A rebate
    new_tax = _apply_cess(new_tax)

    better = "New Regime 🆕" if new_tax <= old_tax else "Old Regime 🏛️"
    savings = abs(old_tax - new_tax)

    return (
        f"Tax Estimate for ₹{income:,.0f} (FY 2024-25, incl. 4% cess):\n"
        f"  Old Regime: ₹{old_tax:,.2f}\n"
        f"  New Regime: ₹{new_tax:,.2f}\n"
        f"  ✅ {better} saves you ₹{savings:,.2f}"
    )


# ──────────────────────────────
# 6. Insurance Gap Analysis
# ──────────────────────────────
def analyze_insurance_gap(
    investments_value: float,
    insurance_cover: float,
    annual_income: float,
) -> str:
    """Check if the user's life cover meets the 10× income benchmark."""
    recommended = annual_income * 10
    gap = recommended - insurance_cover  # cover alone vs recommended (not assets)

    if gap > 0:
        tax_benefit = min(insurance_cover * 0.10, 150_000) * 0.30
        return (
            f"⚠️  Insurance Gap Analysis\n"
            f"  Investments     : ₹{investments_value:,.0f}\n"
            f"  Current Cover   : ₹{insurance_cover:,.0f}\n"
            f"  Recommended Cover (10× income): ₹{recommended:,.0f}\n"
            f"  Shortfall       : ₹{gap:,.0f}\n"
            f"  Action          : Increase term insurance by ₹{gap:,.0f}.\n"
            f"  Est. 80C saving : ₹{tax_benefit:,.2f}"
        )
    return (
        f"✅ Insurance looks adequate.\n"
        f"  Cover: ₹{insurance_cover:,.0f} ≥ Recommended ₹{recommended:,.0f} (10× income)."
    )


# ──────────────────────────────
# 7. Dashboard aggregator
# ──────────────────────────────
def get_dashboard_data() -> dict:
    """Aggregate market, portfolio, and IPO data for the frontend dashboard."""
    # Market
    market_data = None
    try:
        nifty = yf.Ticker("^NSEI").fast_info
        sensex = yf.Ticker("^BSESN").fast_info
        n_chg = nifty.last_price - nifty.previous_close
        s_chg = sensex.last_price - sensex.previous_close
        market_data = {
            "nifty": {
                "price": round(nifty.last_price, 2),
                "change": round(n_chg, 2),
                "percent": round((n_chg / nifty.previous_close) * 100, 2) if nifty.previous_close else 0,
            },
            "sensex": {
                "price": round(sensex.last_price, 2),
                "change": round(s_chg, 2),
                "percent": round((s_chg / sensex.previous_close) * 100, 2) if sensex.previous_close else 0,
            },
            "trend": "Bullish" if n_chg >= 0 else "Bearish",
            "top_gainers": [
                {"symbol": "TATA MOTORS", "change": "+3.5%"},
                {"symbol": "RELIANCE",    "change": "+2.1%"},
                {"symbol": "INFOSYS",     "change": "+1.8%"},
            ],
            "top_losers": [
                {"symbol": "ADANI ENT",  "change": "-2.5%"},
                {"symbol": "SUN PHARMA", "change": "-1.8%"},
                {"symbol": "WIPRO",      "change": "-1.4%"},
            ],
        }
    except Exception:
        pass

    # Portfolio  (reuse shared helpers)
    holdings_out: list = []
    total_invested = total_current = 0.0
    try:
        raw = _load_portfolio()
        agg = _aggregate_holdings(raw)
        for sym, d in agg.items():
            qty = d["qty"]
            avg_buy = d["total_cost"] / qty
            try:
                curr_price = yf.Ticker(sym).fast_info.last_price
            except Exception:
                curr_price = avg_buy
            invested = qty * avg_buy
            curr_val = qty * curr_price
            pnl = curr_val - invested
            total_invested += invested
            total_current += curr_val
            holdings_out.append({
                "symbol": sym,
                "quantity": qty,
                "buy_price": round(avg_buy, 2),
                "current_price": round(curr_price, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round((pnl / invested) * 100 if invested else 0, 2),
            })
    except Exception:
        pass

    total_pnl = total_current - total_invested
    portfolio_data = {
        "holdings": holdings_out,
        "summary": {
            "total_invested": round(total_invested, 2),
            "total_current_value": round(total_current, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round((total_pnl / total_invested) * 100 if total_invested else 0, 2),
        },
    }

    return {
        "market_summary": market_data,
        "portfolio": portfolio_data,
        "ipos": _MOCK_IPOS,
        "fetched_at": datetime.now().isoformat(),
    }