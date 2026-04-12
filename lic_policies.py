"""
lic_policies.py ─────────────────────────────────────────────────────────────
Comprehensive LIC Policy Knowledge Base for Veena AI

Covers:
  • Life Insurance  — Jeevan Anand, Jeevan Labh, Jeevan Umang, Tech-Term,
                      Jeevan Amar, New Endowment, Bima Ratna, Saral Jeevan Bima
  • Health / Medical— Arogya Rakshak, Cancer Cover, Jeevan Arogya
  • Pension / ULIP  — Jeevan Shanti, New Jeevan Nidhi, SIIP (ULIP)
  • Child Plans     — New Children's Money Back, Jeevan Tarun
  • Women's Plans   — Aadhaar Stambh / Shila

Public API
──────────
  get_policy_info(query)      → detailed text about matched policy/category
  list_all_policies()         → summary table of all policies
  compare_policies(names)     → side-by-side comparison
  recommend_policy(profile)   → personalised recommendation
"""

from __future__ import annotations
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Policy Database
# ─────────────────────────────────────────────────────────────────────────────

_POLICIES: dict[str, dict] = {

    # ── LIFE / ENDOWMENT ──────────────────────────────────────────────────────

    "JEEVAN_ANAND": {
        "full_name":    "LIC Jeevan Anand (Plan 915)",
        "category":     "Endowment + Whole Life",
        "tagline":      "Double protection — savings + lifelong cover after maturity",
        "entry_age":    "18 – 50 years",
        "maturity_age": "Up to 75 years",
        "policy_term":  "15 – 35 years",
        "sum_assured":  "Min ₹1,00,000 (no upper limit)",
        "premium_pay":  "Throughout policy term",
        "key_benefits": [
            "Maturity benefit: Sum Assured + Bonuses paid at end of term",
            "Death benefit (during term): Sum Assured + Bonuses + FAB (Final Additional Bonus)",
            "After maturity: Life cover of Basic Sum Assured continues WHOLE LIFE at no extra cost",
            "Death after maturity pays full Sum Assured again to nominee",
            "Participates in LIC profits — Simple Reversionary Bonus declared yearly",
        ],
        "riders_available": ["Accidental Death & Disability Rider", "Term Assurance Rider"],
        "tax_benefit":  "80C (premium), 10(10D) (maturity proceeds tax-free)",
        "loan_facility": "Yes — after 3 years",
        "surrender":    "Yes — after 3 years (Guaranteed Surrender Value)",
        "best_for":     "Salaried individuals wanting savings + lifelong protection",
        "example": (
            "30-year-old, SA ₹10L, 20-yr term: approx. premium ₹50,000/yr. "
            "Maturity ~₹18-22L (with bonuses). Life cover of ₹10L continues even after maturity."
        ),
        "pros": ["Whole-life cover at no extra cost post-maturity", "Guaranteed bonuses", "Loan facility"],
        "cons": ["Lower returns vs mutual funds (~5-6% CAGR)", "Long lock-in"],
    },

    "JEEVAN_LABH": {
        "full_name":    "LIC Jeevan Labh (Plan 936)",
        "category":     "Limited Premium Endowment",
        "tagline":      "Pay for fewer years, stay covered longer",
        "entry_age":    "8 – 59 years",
        "maturity_age": "Up to 75 years",
        "policy_term":  "16, 21, or 25 years",
        "sum_assured":  "Min ₹2,00,000",
        "premium_pay":  "10 yrs (for 16-yr term) / 15 yrs (21-yr) / 16 yrs (25-yr)",
        "key_benefits": [
            "Limited premium payment — pay less years, stay covered full term",
            "Maturity: Sum Assured + Reversionary Bonuses + FAB",
            "Death benefit: Higher of 10× annual premium OR 105% of premiums paid OR Sum Assured + Bonuses",
            "Ideal for those who want to stop paying at retirement but stay insured",
        ],
        "riders_available": ["Accidental Death & Disability Rider", "Critical Illness Rider (selected branches)"],
        "tax_benefit":  "80C + 10(10D)",
        "loan_facility": "Yes — after 2 years",
        "surrender":    "Yes — after 2 years",
        "best_for":     "People who want to finish premium payments before retirement",
        "example": (
            "35-year-old, SA ₹10L, 21-yr term: pays for 15 yrs (~₹58,000/yr). "
            "Stops paying at 50 but covered till 56. Maturity ~₹20L+ (with bonuses)."
        ),
        "pros": ["Premium payment ends early", "Good for retirement planning", "Participating policy"],
        "cons": ["Limited term options (16/21/25 only)", "Average IRR ~5.5%"],
    },

    "JEEVAN_UMANG": {
        "full_name":    "LIC Jeevan Umang (Plan 945)",
        "category":     "Whole Life + Annual Survival Benefit",
        "tagline":      "Annual income for life + lump sum at 100",
        "entry_age":    "90 days – 55 years",
        "maturity_age": "100 years",
        "policy_term":  "Whole life (till age 100)",
        "sum_assured":  "Min ₹2,00,000",
        "premium_pay":  "15, 20, 25, or 30 years",
        "key_benefits": [
            "8% of Sum Assured paid EVERY YEAR as survival benefit after premium term ends",
            "At age 100 (maturity): Sum Assured + Bonuses paid as lump sum",
            "Death benefit (any time): Sum Assured + Bonuses + FAB",
            "Creates a guaranteed annual income stream — like a personal pension",
        ],
        "riders_available": ["Accidental Death & Disability Rider", "New Term Assurance Rider"],
        "tax_benefit":  "80C + 10(10D). Annual survival payouts also tax-free under 10(10D)",
        "loan_facility": "Yes — after 3 years",
        "surrender":    "Yes — after 3 years",
        "best_for":     "Those wanting a regular post-retirement income + life cover",
        "example": (
            "30-year-old, SA ₹10L, 20-yr premium term: pays ~₹55,000/yr for 20 years. "
            "From age 50 onwards receives ₹80,000/yr (8% of ₹10L) every year for life. "
            "At 100: ₹10L + bonuses as lump sum."
        ),
        "pros": ["Regular annual income after premium term", "Whole life coverage", "Great for retirement"],
        "cons": ["Very long commitment", "High total premium outgo", "Returns moderate (~5%)"],
    },

    "JEEVAN_AMAR": {
        "full_name":    "LIC Jeevan Amar (Plan 855)",
        "category":     "Pure Term Insurance",
        "tagline":      "Maximum life cover at lowest cost — offline term plan",
        "entry_age":    "18 – 65 years",
        "maturity_age": "Up to 80 years",
        "policy_term":  "10 – 40 years",
        "sum_assured":  "Min ₹25,00,000 (₹25 lakh), no upper limit",
        "premium_pay":  "Regular / Limited (5-10 yrs) / Single pay",
        "key_benefits": [
            "Pure protection — pays Sum Assured on death, NOTHING on survival",
            "Level cover OR Increasing cover option (5% increase per year)",
            "Non-smoker and female discounts available",
            "Accidental Death Benefit rider available",
            "Offline plan — bought through LIC agent",
        ],
        "riders_available": ["Accidental Death & Disability Benefit Rider"],
        "tax_benefit":  "80C (premium), 10(10D) (death benefit tax-free)",
        "loan_facility": "No",
        "surrender":    "Only for limited/single pay variants",
        "best_for":     "Anyone needing pure high-value life cover at low cost",
        "example": (
            "30-year-old male non-smoker, ₹1 Cr cover, 30-yr term: ~₹10,000–12,000/yr. "
            "Family gets ₹1 Cr if he dies anytime in 30 years."
        ),
        "pros": ["Very affordable", "High sum assured available", "Increasing cover option"],
        "cons": ["No maturity benefit (pure term)", "Only via agents, not online"],
    },

    "TECH_TERM": {
        "full_name":    "LIC Tech-Term (Plan 854)",
        "category":     "Online Pure Term Insurance",
        "tagline":      "LIC's cheapest term plan — buy directly online",
        "entry_age":    "18 – 65 years",
        "maturity_age": "Up to 80 years",
        "policy_term":  "10 – 40 years",
        "sum_assured":  "Min ₹50,00,000 (₹50 lakh)",
        "premium_pay":  "Regular / Limited / Single",
        "key_benefits": [
            "Online purchase — no agent, lower premium than Jeevan Amar",
            "Level OR Increasing cover (10% per year, capped at 2× SA)",
            "Return of Premium (ROP) variant available — get all premiums back on survival",
            "Special rates for non-smokers and women",
        ],
        "riders_available": ["Accidental Death & Disability Benefit Rider"],
        "tax_benefit":  "80C + 10(10D)",
        "loan_facility": "No (except ROP variant after 3 years)",
        "surrender":    "Limited/single pay only",
        "best_for":     "Young professionals who prefer online buying & want cheapest term cover",
        "example": (
            "25-year-old female non-smoker, ₹1 Cr, 35-yr term: as low as ₹6,500–8,000/yr online."
        ),
        "pros": ["Cheapest LIC term plan", "Online — instant issuance", "ROP option available"],
        "cons": ["No maturity benefit on base plan", "ROP variant is expensive"],
    },

    "NEW_ENDOWMENT": {
        "full_name":    "LIC New Endowment Plan (Plan 914)",
        "category":     "Traditional Endowment",
        "tagline":      "Classic savings + protection in one",
        "entry_age":    "8 – 55 years",
        "maturity_age": "Up to 75 years",
        "policy_term":  "12 – 35 years",
        "sum_assured":  "Min ₹1,00,000",
        "premium_pay":  "Throughout policy term",
        "key_benefits": [
            "Maturity: Sum Assured + Reversionary Bonuses + FAB",
            "Death: Sum Assured + Bonuses (higher of 7× AP or 125% of SA)",
            "Simple, straightforward plan — no complexity",
            "Participating in LIC profits",
        ],
        "riders_available": ["Accidental Death", "Critical Illness (selected)"],
        "tax_benefit":  "80C + 10(10D)",
        "loan_facility": "Yes — after 3 years",
        "surrender":    "Yes — after 3 years",
        "best_for":     "First-time buyers wanting a simple traditional plan",
        "example": "Similar structure to Jeevan Anand but without the post-maturity whole-life cover.",
        "pros": ["Simple and trusted", "Participating policy", "Good for forced savings"],
        "cons": ["No whole-life extension like Jeevan Anand", "Low IRR"],
    },

    "BIMA_RATNA": {
        "full_name":    "LIC Bima Ratna (Plan 864)",
        "category":     "Non-Linked, Non-Participating Savings",
        "tagline":      "Guaranteed returns + periodic money-back payouts",
        "entry_age":    "5 – 55 years",
        "maturity_age": "Up to 70 years",
        "policy_term":  "15, 20, or 25 years",
        "sum_assured":  "Min ₹5,00,000",
        "premium_pay":  "Limited (policy term − 3 years)",
        "key_benefits": [
            "Guaranteed Survival Benefits paid at years 8, 12, 16 (varies by term)",
            "Maturity: remaining SA + Guaranteed Additions",
            "Guaranteed Additions: ₹50 per ₹1,000 SA per year — predictable growth",
            "Death benefit: higher of 7× AP or 105% premiums paid, + Guaranteed Additions",
            "Non-participating — no market dependency",
        ],
        "riders_available": ["Accidental Death & Disability Rider"],
        "tax_benefit":  "80C + 10(10D)",
        "loan_facility": "Yes — after 2 years",
        "surrender":    "Yes — after 2 years",
        "best_for":     "Risk-averse investors wanting GUARANTEED payouts at intervals",
        "pros": ["Guaranteed returns — no bonus fluctuation", "Periodic payouts", "Limited premium term"],
        "cons": ["Returns capped — no upside from LIC profits", "Not ideal for high earners"],
    },

    "SARAL_JEEVAN_BIMA": {
        "full_name":    "LIC Saral Jeevan Bima (Plan 859)",
        "category":     "Simple Term Insurance",
        "tagline":      "Standardised term cover — easy to understand, no exclusions confusion",
        "entry_age":    "18 – 65 years",
        "maturity_age": "Up to 70 years",
        "policy_term":  "5 – 40 years",
        "sum_assured":  "₹5 lakh – ₹25 lakh",
        "premium_pay":  "Regular / Limited / Single",
        "key_benefits": [
            "IRDAI-mandated standard product — same terms across all insurers",
            "Only two exclusions: suicide within 1 year, contestable misrepresentation",
            "No medical test for sum assured up to certain limits",
            "Waiting period: 45 days from risk commencement (except accident)",
        ],
        "riders_available": [],
        "tax_benefit":  "80C + 10(10D)",
        "loan_facility": "No",
        "surrender":    "Limited/single pay only",
        "best_for":     "First-time insurance buyers, low-income groups wanting simple term cover",
        "pros": ["Very easy to understand", "Minimal exclusions", "Low entry barrier"],
        "cons": ["Cover capped at ₹25L", "No riders available"],
    },

    # ── HEALTH / MEDICAL ──────────────────────────────────────────────────────

    "AROGYA_RAKSHAK": {
        "full_name":    "LIC Arogya Rakshak (Plan 906)",
        "category":     "Health Insurance — Fixed Benefit",
        "tagline":      "Fixed hospital cash + surgical benefits regardless of actual bills",
        "entry_age":    "18 – 65 years (dependent children 91 days – 20 years)",
        "maturity_age": "Up to 75 years",
        "policy_term":  "1 year (renewable) or Long-term 2/3 years",
        "sum_assured":  "Health Cover: ₹2L – ₹10L",
        "premium_pay":  "Annual / 2-year / 3-year",
        "key_benefits": [
            "Daily Hospital Cash (DHC): Fixed amount per day of hospitalisation (₹1,000–₹4,000/day)",
            "Major Surgical Benefit (MSB): Lump sum for 140+ listed surgeries (% of Health Cover)",
            "Day Care Procedure Benefit for 140+ procedures",
            "Premium Waiver: If MSB claim is made, future premiums waived",
            "Critical Illness Benefit: Lump sum on diagnosis of listed CIs",
            "Maturity Benefit: Total premiums paid returned if no claims made — unique feature!",
            "Automatic increase in cover: 5% per year (Benefit Safeguard feature)",
        ],
        "tax_benefit":  "80D (up to ₹25,000; ₹50,000 for senior citizens)",
        "loan_facility": "No",
        "best_for":     "Those wanting a top-up health plan alongside employer mediclaim",
        "example": (
            "35-year-old, ₹5L Health Cover: DHC ₹2,000/day. Hospitalised for 10 days: ₹20,000 cash "
            "paid directly to policyholder regardless of actual bills paid by other insurer."
        ),
        "pros": [
            "Fixed cash benefit — paid regardless of actual expenses",
            "Maturity benefit — get premiums back if no claims",
            "Premium waiver after major surgery claim",
        ],
        "cons": [
            "Fixed benefit only — does NOT reimburse actual hospital bills",
            "Not a replacement for a proper mediclaim / floater policy",
            "Limited cover amount (max ₹10L)",
        ],
        "important_note": (
            "⚠️ Arogya Rakshak is a FIXED BENEFIT plan, NOT an indemnity plan. "
            "It pays a fixed amount irrespective of your actual bills. "
            "Always pair it with a base indemnity health plan (e.g. Star Health, HDFC Ergo)."
        ),
    },

    "CANCER_COVER": {
        "full_name":    "LIC Cancer Cover (Plan 905)",
        "category":     "Health Insurance — Critical Illness (Cancer-specific)",
        "tagline":      "India's first LIC plan dedicated solely to cancer",
        "entry_age":    "20 – 65 years",
        "maturity_age": "Up to 75 years",
        "policy_term":  "10 – 30 years",
        "sum_assured":  "₹10 lakh – ₹50 lakh",
        "premium_pay":  "Regular or Limited (5/10 years)",
        "key_benefits": [
            "Early Stage Cancer: 25% of Sum Insured paid on diagnosis",
            "Major Stage Cancer: 100% of Sum Insured paid on diagnosis",
            "Income Benefit: If Major Stage — 1% of Sum Insured/month for 3 years (income support)",
            "Premium Waiver: All future premiums waived on Major Stage diagnosis",
            "Increasing Cover option: SA increases 10% per year (no medical required)",
        ],
        "tax_benefit":  "80D",
        "loan_facility": "No",
        "surrender":    "Yes — after 3 years",
        "best_for":     "Anyone with family history of cancer, smokers, over age 35",
        "example": (
            "40-year-old, ₹20L cover, 20-yr term: ~₹4,500–6,000/yr. "
            "Diagnosed with major cancer: gets ₹20L lump sum + ₹20,000/month for 3 years + no more premiums."
        ),
        "pros": [
            "Dedicated cancer protection",
            "Income benefit eases treatment cash flow",
            "Premium waiver is a huge relief during treatment",
        ],
        "cons": [
            "Cancer-specific only — doesn't cover heart attack, stroke etc.",
            "Pre-existing cancer excluded",
            "Waiting period: 180 days from policy start",
        ],
    },

    "JEEVAN_AROGYA": {
        "full_name":    "LIC Jeevan Arogya (Plan 904)",
        "category":     "Health Insurance — Hospital Cash + Surgical",
        "tagline":      "Comprehensive hospital cash plan covering whole family",
        "entry_age":    "Principal: 18–65 | Spouse: 18–65 | Children: 3 months–17 yrs | Parents: 18–70",
        "policy_term":  "5, 10, or 15 years",
        "sum_assured":  "Health Cover ₹2L – ₹8L (Initial Daily Benefit ₹1,000–₹4,000/day)",
        "premium_pay":  "Annual",
        "key_benefits": [
            "Hospital Cash Benefit: Daily cash for each day of hospitalisation",
            "Major Surgical Benefit (MSB): 100× Daily Benefit for major surgeries",
            "Other Surgical Benefit (OSB): 20× Daily Benefit for minor surgeries",
            "Day Care Procedure Benefit",
            "Premium Waiver on MSB claim",
            "Covers policyholder + spouse + children + parents under ONE policy",
            "Benefit Safeguard: automatic 5% annual increase in cover",
        ],
        "tax_benefit":  "80D",
        "loan_facility": "No",
        "best_for":     "Families wanting a single floater-style hospital cash plan",
        "pros": ["Single policy for entire family", "Auto-increase in cover", "Premium waiver benefit"],
        "cons": ["Fixed benefit only, not indemnity", "Old plan — some newer IRDAI products may be better value"],
    },

    # ── PENSION / RETIREMENT ──────────────────────────────────────────────────

    "JEEVAN_SHANTI": {
        "full_name":    "LIC Jeevan Shanti (Plan 850)",
        "category":     "Immediate / Deferred Annuity (Pension)",
        "tagline":      "Guaranteed pension for life — single premium, no market risk",
        "entry_age":    "30 – 79 years (Immediate) | 30 – 79 years (Deferred)",
        "policy_term":  "Whole life (annuity paid till death)",
        "sum_assured":  "Purchase Price: Min ₹1,50,000",
        "premium_pay":  "Single premium",
        "key_benefits": [
            "Immediate Annuity: Pension starts within 1 month of purchase",
            "Deferred Annuity: Pension starts after chosen deferment period (1–12 years)",
            "9 annuity options: Life only, Life + Return of Purchase Price, Joint Life, etc.",
            "Guaranteed for life — no matter how long you live",
            "Joint life option covers spouse after policyholder's death",
            "Loan available after 1 year under Return of Purchase Price option",
        ],
        "tax_benefit":  "80CCC (up to ₹1.5L); annuity income is taxable as income",
        "loan_facility": "Yes — after 1 year (specific variants)",
        "best_for":     "Retirees or near-retirees wanting guaranteed lifetime income",
        "example": (
            "60-year-old invests ₹10L: gets approx. ₹6,500–7,500/month for life "
            "(varies by option and gender). Joint life option: spouse continues to receive "
            "50-100% of pension after policyholder's death."
        ),
        "pros": ["Guaranteed income for life", "No market risk", "Joint life option", "Immediate start"],
        "cons": ["Annuity income is taxable", "Inflation erodes real value of fixed pension over decades", "Single premium commitment"],
    },

    "SIIP": {
        "full_name":    "LIC SIIP — Systematic Investment Insurance Plan (Plan 852)",
        "category":     "ULIP (Unit Linked Insurance Plan)",
        "tagline":      "LIC's market-linked plan — SIP-style investing with life cover",
        "entry_age":    "90 days – 50 years",
        "maturity_age": "18 – 65 years",
        "policy_term":  "10, 15, or 20 years",
        "sum_assured":  "10× annual premium",
        "premium_pay":  "Monthly/quarterly/half-yearly/annual (minimum ₹4,000/month)",
        "key_benefits": [
            "4 fund options: Bond Fund, Secured Fund, Balanced Fund, Growth Fund",
            "Guaranteed NAV: Highest NAV in last 7 years of policy guaranteed (Growth Fund)",
            "Loyalty Additions: Extra units added every 5 years from year 6",
            "Partial withdrawal: Allowed after 5 years (lock-in period)",
            "Switching: Free switches between funds 4 times/year",
            "Death benefit: Higher of (Sum Assured + Fund Value) or (105% of premiums paid)",
        ],
        "tax_benefit":  "80C (premium) + 10(10D) if annual premium ≤ 2.5L",
        "loan_facility": "No",
        "surrender":    "After 5-year lock-in",
        "best_for":     "Young investors wanting equity-linked growth with life cover inside one plan",
        "example": (
            "₹5,000/month for 15 years = ₹9L invested. "
            "At 12% growth (Growth Fund): corpus ~₹25-30L. "
            "Highest NAV guarantee provides downside protection in final years."
        ),
        "pros": ["Equity-linked returns potential", "Guaranteed NAV feature unique to LIC", "Loyalty additions"],
        "cons": ["Higher charges vs mutual fund + term plan combo", "5-year lock-in", "Complex product"],
    },

    # ── CHILD PLANS ───────────────────────────────────────────────────────────

    "NEW_CHILDRENS_MONEY_BACK": {
        "full_name":    "LIC New Children's Money Back Plan (Plan 932)",
        "category":     "Child Money Back Plan",
        "tagline":      "Guaranteed payouts at key milestones of your child's life",
        "entry_age":    "0 – 12 years (child's age)",
        "policy_term":  "Till child turns 25",
        "sum_assured":  "Min ₹1,00,000",
        "premium_pay":  "Parent/proposer pays till child turns 25 (or waived if parent dies)",
        "key_benefits": [
            "Survival Benefits: 20% of SA at ages 18, 20, 22 (child's age)",
            "Maturity at 25: 40% of SA + Bonuses + FAB",
            "Payor Benefit Rider: If parent dies, future premiums WAIVED — policy continues normally",
            "Death of child: Return of all premiums paid (before risk commencement at age 8)",
            "After 8 years: Full death benefit on child's death",
        ],
        "riders_available": ["Payor Benefit Rider (CRITICAL for child plans)"],
        "tax_benefit":  "80C + 10(10D)",
        "loan_facility": "Yes — after 2 years",
        "best_for":     "Parents planning for child's education (18), college (20-22), and career start (25)",
        "example": (
            "Child aged 5, SA ₹10L: Gets ₹2L at 18 (school completion), ₹2L at 20 (college), "
            "₹2L at 22 (post-grad), ₹4L + bonuses ~₹8-10L at 25 (career start). "
            "If parent dies, policy continues — child still gets all benefits."
        ),
        "pros": ["Structured payouts match education milestones", "Payor waiver is a lifesaver", "Bonus participation"],
        "cons": ["Low overall return (~5%)", "Long commitment (20+ years)", "Inflation may erode ₹2L value by 2040s"],
    },

    "JEEVAN_TARUN": {
        "full_name":    "LIC Jeevan Tarun (Plan 934)",
        "category":     "Child Endowment + Money Back",
        "tagline":      "Flexible survival benefits — you choose the payout pattern",
        "entry_age":    "90 days – 12 years",
        "policy_term":  "Till child turns 25",
        "sum_assured":  "Min ₹75,000",
        "premium_pay":  "Till child turns 20",
        "key_benefits": [
            "FLEXIBLE payout option: You choose 1 of 4 patterns at inception",
            "Option A: 100% at maturity (age 25)",
            "Option B: 5% SA/yr from age 20–24 + 75% at 25",
            "Option C: 10% SA/yr from age 20–24 + 50% at 25",
            "Option D: 15% SA/yr from age 20–24 + 25% at 25",
            "Payor Benefit Rider: Premium waived on parent's death",
            "Death benefit: Sum Assured + Bonuses at any time",
        ],
        "riders_available": ["Payor Benefit Rider"],
        "tax_benefit":  "80C + 10(10D)",
        "best_for":     "Parents who want custom payout timing based on child's needs",
        "pros": ["Flexible payout pattern — rare feature", "Payor rider available", "Good for goal-based planning"],
        "cons": ["Complex — need to decide payout option at inception", "Low returns"],
    },

    # ── WOMEN'S PLANS ─────────────────────────────────────────────────────────

    "AADHAAR_STAMBH": {
        "full_name":    "LIC Aadhaar Stambh (Plan 943) — For Men | Aadhaar Shila (944) — For Women",
        "category":     "Endowment (Aadhaar-linked, no medical required)",
        "tagline":      "Simple endowment for Aadhaar holders — no medicals needed",
        "entry_age":    "8 – 55 years",
        "maturity_age": "Up to 70 years",
        "policy_term":  "10 – 20 years",
        "sum_assured":  "₹75,000 – ₹3,00,000",
        "premium_pay":  "Throughout term",
        "key_benefits": [
            "No medical examination — just Aadhaar card required",
            "Maturity: Sum Assured + Bonuses + FAB",
            "Death in first 5 years: 125% of premiums paid",
            "Death after 5 years: Sum Assured + Bonuses",
            "Loyalty Addition on maturity",
            "Aadhaar Shila specifically designed for women",
        ],
        "tax_benefit":  "80C + 10(10D)",
        "loan_facility": "Yes — after 3 years",
        "best_for":     "Low-income individuals, rural customers, those without medical history documentation",
        "pros": ["No medical test", "Very accessible", "Simple to buy via LIC agent"],
        "cons": ["Low sum assured cap (₹3L)", "Lower returns", "Not suitable for high-income earners"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Category index for fuzzy search
# ─────────────────────────────────────────────────────────────────────────────

_KEYWORDS: dict[str, list[str]] = {
    "JEEVAN_ANAND":              ["jeevan anand", "915", "whole life endowment", "lifelong cover", "anand"],
    "JEEVAN_LABH":               ["jeevan labh", "936", "limited premium", "labh"],
    "JEEVAN_UMANG":              ["jeevan umang", "945", "annual income", "survival benefit", "umang", "yearly income"],
    "JEEVAN_AMAR":               ["jeevan amar", "855", "term plan", "pure term", "amar"],
    "TECH_TERM":                 ["tech term", "854", "online term", "cheapest term", "techterm"],
    "NEW_ENDOWMENT":             ["new endowment", "914", "endowment plan", "basic endowment"],
    "BIMA_RATNA":                ["bima ratna", "864", "guaranteed returns", "money back guaranteed", "ratna"],
    "SARAL_JEEVAN_BIMA":         ["saral jeevan", "859", "simple term", "standard term", "saral"],
    "AROGYA_RAKSHAK":            ["arogya rakshak", "906", "health", "medical", "hospital cash", "mediclaim", "rakshak", "health insurance", "arogya"],
    "CANCER_COVER":              ["cancer", "905", "cancer cover", "critical illness cancer", "oncology"],
    "JEEVAN_AROGYA":             ["jeevan arogya", "904", "family health", "hospital", "surgical benefit", "health plan"],
    "JEEVAN_SHANTI":             ["jeevan shanti", "850", "pension", "annuity", "retirement", "shanti", "post retirement"],
    "SIIP":                      ["siip", "852", "ulip", "market linked", "sip insurance", "equity linked"],
    "NEW_CHILDRENS_MONEY_BACK":  ["children money back", "932", "child plan", "education plan", "child insurance"],
    "JEEVAN_TARUN":              ["jeevan tarun", "934", "child endowment", "tarun", "flexible child"],
    "AADHAAR_STAMBH":            ["aadhaar stambh", "943", "aadhaar shila", "944", "women plan", "no medical", "aadhaar"],
}

_CATEGORIES = {
    "life":      ["JEEVAN_ANAND", "JEEVAN_LABH", "JEEVAN_UMANG", "JEEVAN_AMAR", "TECH_TERM", "NEW_ENDOWMENT", "BIMA_RATNA", "SARAL_JEEVAN_BIMA"],
    "term":      ["JEEVAN_AMAR", "TECH_TERM", "SARAL_JEEVAN_BIMA"],
    "health":    ["AROGYA_RAKSHAK", "CANCER_COVER", "JEEVAN_AROGYA"],
    "medical":   ["AROGYA_RAKSHAK", "CANCER_COVER", "JEEVAN_AROGYA"],
    "pension":   ["JEEVAN_SHANTI", "SIIP"],
    "retirement":["JEEVAN_SHANTI", "SIIP", "JEEVAN_UMANG"],
    "child":     ["NEW_CHILDRENS_MONEY_BACK", "JEEVAN_TARUN"],
    "women":     ["AADHAAR_STAMBH"],
    "ulip":      ["SIIP"],
    "endowment": ["JEEVAN_ANAND", "JEEVAN_LABH", "NEW_ENDOWMENT", "BIMA_RATNA"],
    "cancer":    ["CANCER_COVER"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_policy_key(query: str) -> Optional[str]:
    """Return policy key if query matches any keyword."""
    q = query.lower().strip()
    for key, kws in _KEYWORDS.items():
        if any(kw in q for kw in kws):
            return key
    return None


def _find_category_keys(query: str) -> list[str]:
    """Return list of policy keys matching a category word."""
    q = query.lower().strip()
    for cat, keys in _CATEGORIES.items():
        if cat in q:
            return keys
    return []


def _format_policy(p: dict) -> str:
    """Format a single policy dict into readable text."""
    lines = [
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📋  {p['full_name']}",
        f"    Category : {p['category']}",
        f"    Tagline  : {p['tagline']}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"📌 Eligibility",
        f"   Entry Age    : {p.get('entry_age', 'N/A')}",
        f"   Maturity Age : {p.get('maturity_age', 'N/A')}",
        f"   Policy Term  : {p.get('policy_term', 'N/A')}",
        f"   Sum Assured  : {p.get('sum_assured', 'N/A')}",
        f"   Premium Pay  : {p.get('premium_pay', 'N/A')}",
        f"",
        f"✅ Key Benefits",
    ]
    for b in p.get("key_benefits", []):
        lines.append(f"   • {b}")

    if p.get("riders_available"):
        lines.append(f"\n🔗 Riders Available")
        for r in p["riders_available"]:
            lines.append(f"   • {r}")

    lines += [
        f"",
        f"💰 Tax Benefit    : {p.get('tax_benefit', 'N/A')}",
        f"🏦 Loan Facility  : {p.get('loan_facility', 'N/A')}",
        f"🎯 Best For       : {p.get('best_for', 'N/A')}",
    ]

    if p.get("example"):
        lines += [f"", f"💡 Example", f"   {p['example']}"]

    if p.get("pros"):
        lines.append(f"\n👍 Pros")
        for pr in p["pros"]:
            lines.append(f"   ✓ {pr}")

    if p.get("cons"):
        lines.append(f"\n👎 Cons")
        for c in p["cons"]:
            lines.append(f"   ✗ {c}")

    if p.get("important_note"):
        lines += [f"", f"⚠️  Important Note", f"   {p['important_note']}"]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_policy_info(query: str) -> str:
    """
    Main entry point. Returns detailed info about a policy or category.
    Examples:
        get_policy_info("Jeevan Anand")
        get_policy_info("health insurance")
        get_policy_info("cancer cover")
        get_policy_info("child plan")
    """
    # Try exact policy match first
    key = _find_policy_key(query)
    if key:
        return _format_policy(_POLICIES[key])

    # Try category match — return brief list + offer details
    cat_keys = _find_category_keys(query)
    if cat_keys:
        lines = [f"📂 LIC Policies in '{query.title()}' category:\n"]
        for k in cat_keys:
            p = _POLICIES[k]
            lines.append(f"  🔹 {p['full_name']}")
            lines.append(f"     {p['tagline']}")
            lines.append(f"     Entry Age: {p.get('entry_age','N/A')} | Best For: {p.get('best_for','N/A')}")
            lines.append("")
        lines.append("Ask me for details on any specific policy above!")
        return "\n".join(lines)

    return (
        f"I couldn't find a policy matching '{query}'.\n"
        f"Available policies: Jeevan Anand, Jeevan Labh, Jeevan Umang, Jeevan Amar, Tech-Term, "
        f"New Endowment, Bima Ratna, Saral Jeevan Bima, Arogya Rakshak, Cancer Cover, "
        f"Jeevan Arogya, Jeevan Shanti, SIIP, New Children's Money Back, Jeevan Tarun, Aadhaar Stambh.\n"
        f"Or ask about a category: life, term, health, pension, child, women, ulip, endowment."
    )


def list_all_policies() -> str:
    """Return a concise summary table of all available LIC policies."""
    lines = [
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║              📋  LIC POLICY CATALOGUE — Veena AI                   ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        "── LIFE & ENDOWMENT PLANS ──────────────────────────────────────────────",
    ]
    life_plans = ["JEEVAN_ANAND", "JEEVAN_LABH", "JEEVAN_UMANG", "JEEVAN_AMAR",
                  "TECH_TERM", "NEW_ENDOWMENT", "BIMA_RATNA", "SARAL_JEEVAN_BIMA"]
    for k in life_plans:
        p = _POLICIES[k]
        lines.append(f"  • {p['full_name']:<45} | {p['tagline']}")

    lines += ["", "── HEALTH & MEDICAL PLANS ──────────────────────────────────────────────"]
    for k in ["AROGYA_RAKSHAK", "CANCER_COVER", "JEEVAN_AROGYA"]:
        p = _POLICIES[k]
        lines.append(f"  • {p['full_name']:<45} | {p['tagline']}")

    lines += ["", "── PENSION & RETIREMENT PLANS ──────────────────────────────────────────"]
    for k in ["JEEVAN_SHANTI", "SIIP"]:
        p = _POLICIES[k]
        lines.append(f"  • {p['full_name']:<45} | {p['tagline']}")

    lines += ["", "── CHILD PLANS ─────────────────────────────────────────────────────────"]
    for k in ["NEW_CHILDRENS_MONEY_BACK", "JEEVAN_TARUN"]:
        p = _POLICIES[k]
        lines.append(f"  • {p['full_name']:<45} | {p['tagline']}")

    lines += ["", "── WOMEN / INCLUSIVE PLANS ─────────────────────────────────────────────"]
    for k in ["AADHAAR_STAMBH"]:
        p = _POLICIES[k]
        lines.append(f"  • {p['full_name']:<45} | {p['tagline']}")

    lines += ["", "Ask me about any policy by name for full details!"]
    return "\n".join(lines)


def compare_policies(names_input: str) -> str:
    """
    Compare 2-3 LIC policies side by side.
    Input: comma-separated policy names/keys
    Example: "Jeevan Anand, Jeevan Labh" or "health, cancer cover"
    """
    names = [n.strip() for n in names_input.split(',') if n.strip()]
    if len(names) < 2:
        return "Please provide at least 2 policy names to compare, separated by commas."

    found = []
    for name in names:
        key = _find_policy_key(name)
        if key:
            found.append((key, _POLICIES[key]))

    if len(found) < 2:
        return f"Could not find 2 matching policies for '{names_input}'. Try: 'Jeevan Anand, Jeevan Labh'"

    fields = [
        ("Category",     "category"),
        ("Entry Age",    "entry_age"),
        ("Policy Term",  "policy_term"),
        ("Sum Assured",  "sum_assured"),
        ("Premium Pay",  "premium_pay"),
        ("Tax Benefit",  "tax_benefit"),
        ("Loan Facility","loan_facility"),
        ("Best For",     "best_for"),
    ]

    col_w = 30
    header = f"{'Attribute':<20}" + "".join(f"{p['full_name'].split('(')[0].strip()[:col_w]:<{col_w}}" for _, p in found)
    lines = [
        "📊  POLICY COMPARISON",
        "─" * (20 + col_w * len(found)),
        header,
        "─" * (20 + col_w * len(found)),
    ]
    for label, field in fields:
        row = f"{label:<20}" + "".join(f"{str(p.get(field,'N/A'))[:col_w-2]:<{col_w}}" for _, p in found)
        lines.append(row)

    lines.append("─" * (20 + col_w * len(found)))
    lines.append("\n✅ PROS")
    for key, p in found:
        lines.append(f"\n  {p['full_name'].split('(')[0].strip()}:")
        for pr in p.get("pros", []):
            lines.append(f"    ✓ {pr}")

    lines.append("\n❌ CONS")
    for key, p in found:
        lines.append(f"\n  {p['full_name'].split('(')[0].strip()}:")
        for c in p.get("cons", []):
            lines.append(f"    ✗ {c}")

    return "\n".join(lines)


def recommend_policy(profile_input: str) -> str:
    """
    Recommend the most suitable LIC policy based on a user profile.
    Input format: comma-separated key=value pairs
    Keys: age, goal, budget, family, health, risk
    Example: "age=35, goal=retirement, budget=5000/month, risk=low"
    Example: "age=28, goal=child education, family=yes"
    Example: "age=45, goal=health cover"
    """
    profile: dict[str, str] = {}
    for part in profile_input.split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            profile[k.strip().lower()] = v.strip().lower()

    age_str = profile.get("age", "0")
    try:
        age = int(''.join(filter(str.isdigit, age_str)))
    except ValueError:
        age = 0

    goal    = profile.get("goal", "")
    risk    = profile.get("risk", "medium")
    family  = profile.get("family", "no")
    health  = profile.get("health", "")

    recommendations = []
    reasons = []

    # Goal-based routing
    if any(w in goal for w in ["health", "medical", "hospital", "cancer"]):
        recommendations += ["AROGYA_RAKSHAK", "CANCER_COVER"]
        reasons.append("Health/medical cover matched.")

    elif any(w in goal for w in ["child", "education", "kid"]):
        recommendations += ["NEW_CHILDRENS_MONEY_BACK", "JEEVAN_TARUN"]
        reasons.append("Child education plan matched.")

    elif any(w in goal for w in ["pension", "retire", "retirement", "income after"]):
        if age > 55:
            recommendations += ["JEEVAN_SHANTI"]
            reasons.append("Near retirement → immediate annuity best.")
        else:
            recommendations += ["JEEVAN_UMANG", "JEEVAN_SHANTI"]
            reasons.append("Retirement planning matched.")

    elif any(w in goal for w in ["term", "protection", "death cover", "life cover"]):
        if risk == "low":
            recommendations += ["TECH_TERM", "JEEVAN_AMAR"]
        else:
            recommendations += ["TECH_TERM"]
        reasons.append("Pure protection need matched.")

    elif any(w in goal for w in ["saving", "savings", "invest", "wealth"]):
        if risk in ("low", "very low"):
            recommendations += ["JEEVAN_LABH", "BIMA_RATNA"]
            reasons.append("Low-risk savings matched.")
        elif risk == "high":
            recommendations += ["SIIP"]
            reasons.append("Market-linked ULIP for high risk tolerance.")
        else:
            recommendations += ["JEEVAN_ANAND", "JEEVAN_LABH"]
            reasons.append("Savings + protection matched.")

    # Age-based additions if no clear goal
    if not recommendations:
        if age < 30:
            recommendations += ["TECH_TERM", "JEEVAN_ANAND"]
            reasons.append("Young age → term cover + savings plan recommended.")
        elif 30 <= age < 45:
            recommendations += ["JEEVAN_LABH", "JEEVAN_UMANG", "AROGYA_RAKSHAK"]
            reasons.append("Mid-age → limited premium savings + health cover.")
        elif 45 <= age < 60:
            recommendations += ["JEEVAN_SHANTI", "CANCER_COVER", "JEEVAN_LABH"]
            reasons.append("Pre-retirement → pension + critical illness cover.")
        else:
            recommendations += ["JEEVAN_SHANTI"]
            reasons.append("Senior → guaranteed annuity income best.")

    # De-duplicate preserving order
    seen = set()
    unique_recs = []
    for r in recommendations:
        if r not in seen:
            seen.add(r)
            unique_recs.append(r)

    lines = [
        "🎯 PERSONALISED LIC POLICY RECOMMENDATION",
        f"   Profile: {profile_input}",
        f"   Reason : {' '.join(reasons)}",
        "",
    ]
    for i, key in enumerate(unique_recs[:3], 1):
        p = _POLICIES[key]
        lines += [
            f"{'🥇' if i==1 else '🥈' if i==2 else '🥉'}  Recommendation {i}: {p['full_name']}",
            f"   {p['tagline']}",
            f"   Entry Age : {p.get('entry_age','N/A')}",
            f"   Best For  : {p.get('best_for','N/A')}",
            f"   Tax       : {p.get('tax_benefit','N/A')}",
            "",
        ]
    lines.append("Ask me 'Tell me more about [policy name]' for full details.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI quick-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(list_all_policies())
    print("\n" + "="*70 + "\n")
    print(get_policy_info("jeevan anand"))
    print("\n" + "="*70 + "\n")
    print(compare_policies("Jeevan Anand, Jeevan Labh"))
    print("\n" + "="*70 + "\n")
    print(recommend_policy("age=35, goal=child education, risk=low"))
