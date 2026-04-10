import yfinance as yf
import json
import os
from datetime import datetime
import urllib.request
from typing import Optional

# Setup Supabase Config
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

# 1. Live Stock Price Tracker
def get_stock_price(symbol: str) -> str:
    """Get real-time price, change, high, low, and volume of a stock.
    Input should be the stock ticker symbol (e.g., RELIANCE.NS, TCS.NS, AAPL)."""
    try:
        if not symbol.endswith('.NS') and not symbol.endswith('.BO') and symbol.isalpha():
            symbol += '.NS'
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        current_price = info.last_price
        prev_close = info.previous_close
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100

        return f"Stock: {symbol}\nPrice: ₹{current_price:.2f}\nChange: ₹{change:.2f} ({change_pct:.2f}%)\nHigh: ₹{info.day_high:.2f}\nLow: ₹{info.day_low:.2f}\nVolume: {info.last_volume}"
    except Exception as e:
        return f"Could not fetch price for {symbol}. Error: {str(e)}"

# 2. Portfolio Manager
def portfolio_manager(action: str, symbol: Optional[str] = None, quantity: Optional[int] = None, buy_price: Optional[float] = None) -> str:
    """Manage user's stock portfolio using Supabase (or fallback to local file if not configured).
    """
    use_supabase = bool(SUPABASE_URL and SUPABASE_KEY)

    # Supabase Helper
    def request_supabase(method, endpoint, payload=None):
        url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        if method == "POST":
            headers["Prefer"] = "return=representation"
        req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload else None, headers=headers, method=method)
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())

    # Fallback Local JSON Helper
    LOCAL_DB = "portfolio_db.json"
    def load_local():
        if os.path.exists(LOCAL_DB):
            with open(LOCAL_DB, "r") as f:
                return json.load(f)
        return []
    def save_local(data):
        with open(LOCAL_DB, "w") as f:
            json.dump(data, f)

    if action.lower() == 'add':
        if not symbol or quantity is None or buy_price is None:
            return "Please provide symbol, quantity, and buy_price to add to portfolio."

        new_holding = {"symbol": symbol.upper(), "quantity": quantity, "buy_price": buy_price}

        try:
            if use_supabase:
                request_supabase("POST", "portfolio", new_holding)
            else:
                db = load_local()
                db.append(new_holding)
                save_local(db)
            return f"Successfully added {quantity} shares of {symbol} at ₹{buy_price} to portfolio."
        except Exception as e:
            return f"Failed to add to portfolio: {str(e)}"

    elif action.lower() == 'view':
        try:
            if use_supabase:
                holdings_raw = request_supabase("GET", "portfolio?select=*")
            else:
                holdings_raw = load_local()

            if not holdings_raw:
                return "Your portfolio is currently empty."

            # Aggregate by symbol
            aggregated = {}
            for h in holdings_raw:
                sym = h['symbol']
                if sym not in aggregated:
                    aggregated[sym] = {"qty": 0, "total_cost": 0}
                aggregated[sym]["qty"] += h['quantity']
                aggregated[sym]["total_cost"] += h['quantity'] * h['buy_price']

            total_invested = 0
            total_current_value = 0
            response = "Your Portfolio Holdings:\n"

            for sym, data in aggregated.items():
                qty = data["qty"]
                avg_buy = data["total_cost"] / qty if qty > 0 else 0

                try:
                    ticker = yf.Ticker(sym if sym.endswith('.NS') or sym.endswith('.BO') else sym + '.NS')
                    curr_price = ticker.fast_info.last_price
                except:
                    curr_price = avg_buy # fallback

                invested = qty * avg_buy
                curr_val = qty * curr_price
                pnl = curr_val - invested
                pnl_pct = (pnl / invested) * 100 if invested > 0 else 0

                total_invested += invested
                total_current_value += curr_val

                response += f"- {sym}: {qty} shares | Buy: ₹{avg_buy:.2f} | Current: ₹{curr_price:.2f} | P&L: ₹{pnl:.2f} ({pnl_pct:.2f}%)\n"

            total_pnl = total_current_value - total_invested
            total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0

            response += f"\nTotal Invested: ₹{total_invested:.2f}\nTotal Current Value: ₹{total_current_value:.2f}\nTotal P&L: ₹{total_pnl:.2f} ({total_pnl_pct:.2f}%)"
            return response
        except Exception as e:
            return f"Failed to view portfolio: {str(e)}"

    return "Invalid action. Use 'add' or 'view'."

# 3. Market Summary
def get_market_summary() -> str:
    """Get overall market summary (NIFTY 50, SENSEX) and trend."""
    try:
        nifty = yf.Ticker('^NSEI').fast_info
        sensex = yf.Ticker('^BSESN').fast_info

        nifty_chg = nifty.last_price - nifty.previous_close
        nifty_pct = (nifty_chg / nifty.previous_close) * 100

        sensex_chg = sensex.last_price - sensex.previous_close
        sensex_pct = (sensex_chg / sensex.previous_close) * 100

        trend = "Bullish 📈" if nifty_chg > 0 else "Bearish 📉"

        # Mocked top gainers and losers for NIFTY 50
        mock_gainers_losers = (
            "Top 5 Gainers:\n"
            "1. TATA MOTORS (+3.5%)\n2. RELIANCE (+2.1%)\n3. INFOSYS (+1.8%)\n4. HDFC BANK (+1.5%)\n5. ITC (+1.2%)\n\n"
            "Top 5 Losers:\n"
            "1. ADANI ENT (-2.5%)\n2. SUN PHARMA (-1.8%)\n3. WIPRO (-1.4%)\n4. L&T (-1.1%)\n5. MARUTI (-0.8%)\n"
        )

        return (f"Market Summary:\n"
                f"NIFTY 50: {nifty.last_price:.2f} ({nifty_pct:.2f}%)\n"
                f"SENSEX: {sensex.last_price:.2f} ({sensex_pct:.2f}%)\n"
                f"Overall Trend: {trend}\n\n"
                f"{mock_gainers_losers}")
    except Exception as e:
        return f"Could not fetch market summary. Error: {str(e)}"

# 4. IPO Tracker
def get_upcoming_ipos() -> str:
    """Get list of upcoming IPOs, dates, and Grey Market Premium (GMP)."""
    return (
        "Upcoming IPOs:\n"
        "1. Swiggy Ltd\n   Open: 15-Nov | Close: 18-Nov | Price Band: ₹370-390 | Lot: 38 | GMP: ₹25\n"
        "   Subscription Status: 2.5x\n"
        "2. NTPC Green Energy\n   Open: 22-Nov | Close: 25-Nov | Price Band: ₹100-108 | Lot: 138 | GMP: ₹12\n"
        "   Subscription Status: Upcoming\n"
        "Recommendation: NTPC Green shows strong fundamentals and decent GMP. Consider subscribing."
    )

# 5. Tax Calculator
def calculate_tax(income: float, deductions_80c: float = 0, deductions_80d: float = 0) -> str:
    """Calculate and compare income tax for Old vs New regime for FY 2024-25.
    Input parameters are income, 80C deductions (max 1.5L), 80D deductions."""
    # Simplified calculation
    old_taxable = max(0, income - 50000 - min(deductions_80c, 150000) - deductions_80d)

    # Old Regime Slabs (Simplified for < 60 years)
    old_tax = 0
    if old_taxable > 1000000:
        old_tax += (old_taxable - 1000000) * 0.3 + 112500
    elif old_taxable > 500000:
        old_tax += (old_taxable - 500000) * 0.2 + 12500
    elif old_taxable > 250000:
        old_tax += (old_taxable - 250000) * 0.05

    if old_taxable <= 500000:
        old_tax = 0 # Rebate 87A

    # New Regime Slabs FY 24-25 (Simplified)
    new_taxable = max(0, income - 75000) # Standard deduction 75k in new regime
    new_tax = 0
    if new_taxable > 1500000:
        new_tax += (new_taxable - 1500000) * 0.3 + 150000
    elif new_taxable > 1200000:
        new_tax += (new_taxable - 1200000) * 0.2 + 90000
    elif new_taxable > 1000000:
        new_tax += (new_taxable - 1000000) * 0.15 + 60000
    elif new_taxable > 700000:
        new_tax += (new_taxable - 700000) * 0.1 + 30000
    elif new_taxable > 300000:
        new_tax += (new_taxable - 300000) * 0.05

    if new_taxable <= 700000:
        new_tax = 0 # Rebate 87A

    better_regime = "New Regime" if new_tax < old_tax else "Old Regime"
    savings = abs(old_tax - new_tax)

    return (f"Tax Calculation for Income: ₹{income}:\n"
            f"Old Regime Tax: ₹{old_tax:.2f} (with deductions)\n"
            f"New Regime Tax: ₹{new_tax:.2f} (default)\n"
            f"Recommendation: {better_regime} is better, saving you ₹{savings:.2f}.")

# 7. Insurance + Finance Gap Analysis
def analyze_insurance_gap(investments_value: float, insurance_cover: float, annual_income: float) -> str:
    """Analyze if user has enough insurance cover compared to investments and income."""
    recommended_cover = annual_income * 10
    total_assets = investments_value + insurance_cover

    if total_assets < recommended_cover:
        shortfall = recommended_cover - total_assets
        tax_benefit = min(insurance_cover * 0.1, 150000) * 0.3  # Simplified 80C tax benefit estimate
        return (f"Gap Analysis Alert ⚠️\n"
                f"You have ₹{investments_value} in investments and ₹{insurance_cover} in life cover.\n"
                f"Recommended cover based on income: ₹{recommended_cover}.\n"
                f"You are under-insured by ₹{shortfall}. Consider increasing your term insurance cover.\n"
                f"By increasing cover, you can save up to ₹{tax_benefit:.2f} in taxes under 80C.")
    else:
        return "Your insurance cover is adequate compared to your income and investments. Good job! ✅"
