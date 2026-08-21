"""
wellness_tools.py — Mental wellness companion tools for Veena AI.
Breathing, grounding, mood check-in, daily motivation, and crisis resources.
"""

import random
import re
from datetime import datetime

# ── Crisis helplines (India) ──────────────────────────────────────────────────
CRISIS_HELPLINES = """
🆘 If you are in distress, please reach out — you are not alone.

• Kiran Mental Health Helpline: 1800-599-0019 (24/7, toll-free)
• Vandrevala Foundation: 1860-2662-345 / 9999-666-555 (24/7)
• iCall (TISS): 9152-987-821 (Mon–Sat, 8am–10pm)
• Emergency: 112

These services connect you with trained counsellors. Please call if you feel overwhelmed.
"""

_CRISIS_KEYWORDS = (
    "suicide", "kill myself", "end my life", "want to die", "self harm", "self-harm",
    "hurt myself", "no reason to live", "better off dead",
    "i want to die", "i am going to die", "i can’t live", "i can't live",
    "खुद को मार", "मरना चाह", "आत्महत्या", "जीना नहीं", "मर जाऊ", "मरने का मन",
    "अपने आप को नुकसान", "जीने का मन नहीं",
)

_STRESS_KEYWORDS = {
    "stressed": 8,
    "stress": 8,
    "panic": 10,
    "panicking": 10,
    "anxious": 9,
    "anxiety": 9,
    "overwhelmed": 10,
    "tired": 5,
    "exhausted": 7,
    "hopeless": 9,
    "confused": 6,
    "angry": 5,
    "worried": 7,
    "scared": 7,
    "afraid": 7,
    "can't handle": 8,
    "cant handle": 8,
    "too much": 8,
    "not okay": 6,
    "breaking down": 9,
    "mentally exhausted": 9,
    "pressure": 7,
    "problem": 3,
    "difficult": 4,
    "very stressed": 10,
    "बहुत stressed": 10,
    "बहुत tension": 10,
    "चिंता": 8,
    "तनाव": 9,
    "पैनिक": 10,
    "घबराहट": 9,
    "ओवरवेल्म्ड": 9,
    "असमंजस": 6,
    "अकेला": 5,
    "उदास": 5,
    "बेचैन": 7,
    "हार मान": 9,
}


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9\s\u0900-\u097f]", " ", text)
    return " ".join(text.split())


def assess_stress(text: str) -> dict:
    """Assess emotional stress from the user message and return a score + suggested response."""
    normalized = _normalize_text(text)
    score = 0
    matched = []

    for term, weight in _STRESS_KEYWORDS.items():
        if term in normalized:
            score += weight
            matched.append(term)

    if any(phrase in normalized for phrase in ["i am not okay", "main theek nahi hoon", "main theek nahi hu", "mai theek nahi hoon", "mai theek nahi hu"]):
        score += 8

    if any(phrase in normalized for phrase in ["can't take this", "cant take this", "bahut zyada ho gaya", "bahut jyada ho gaya", "sabse zyada pressure"]):
        score += 8

    if score >= 35:
        level = "high"
        label = "High stress"
        recommendation = "Take a pause, breathe slowly, and consider a grounding exercise or talking to someone you trust."
    elif score >= 15:
        level = "medium"
        label = "Moderate stress"
        recommendation = "A short breathing exercise and a little calm reflection may help right now."
    else:
        level = "low"
        label = "Low stress"
        recommendation = "You sound steady enough for a normal conversation; we can keep it simple and supportive."

    return {
        "score": min(score, 100),
        "level": level,
        "label": label,
        "matched_terms": matched[:8],
        "recommendation": recommendation,
    }


def detect_crisis(text: str) -> bool:
    """Return True if user text suggests crisis/self-harm."""
    lower = _normalize_text(text)
    return any(kw in lower for kw in _CRISIS_KEYWORDS)


def get_crisis_resources(_: str = "") -> str:
    return CRISIS_HELPLINES.strip()


# ── Breathing exercises ───────────────────────────────────────────────────────

_BREATHING = {
    "478": {
        "name": "4-7-8 Breathing",
        "steps": [
            "Sit comfortably and close your eyes if you feel safe doing so.",
            "Breathe IN through your nose for 4 counts.",
            "HOLD your breath gently for 7 counts.",
            "Breathe OUT slowly through your mouth for 8 counts.",
            "Repeat this cycle 3 to 4 times.",
        ],
        "benefit": "Calms the nervous system and reduces anxiety quickly.",
    },
    "box": {
        "name": "Box Breathing",
        "steps": [
            "Breathe IN for 4 counts.",
            "HOLD for 4 counts.",
            "Breathe OUT for 4 counts.",
            "HOLD empty for 4 counts.",
            "Repeat 4 rounds. Imagine tracing a square with each cycle.",
        ],
        "benefit": "Used by athletes and soldiers to regain focus under pressure.",
    },
    "calm": {
        "name": "Calm Belly Breathing",
        "steps": [
            "Place one hand on your chest, one on your belly.",
            "Breathe IN slowly — feel your belly rise, not your chest.",
            "Breathe OUT slowly — belly falls.",
            "Continue for 5 slow breaths.",
        ],
        "benefit": "Simple and effective for everyday stress relief.",
    },
}


def breathing_guide(input_str: str) -> str:
    """
    Input: technique name — '478', 'box', 'calm', or empty for recommendation.
    """
    key = (input_str or "").strip().lower().replace(" ", "")
    if key in ("478", "4-7-8", "478breathing"):
        key = "478"
    elif key in ("box", "square", "boxbreathing"):
        key = "box"
    elif not key or key in ("default", "any", "help", "stress", "anxiety"):
        key = "478"

    exercise = _BREATHING.get(key, _BREATHING["478"])
    lines = [
        f"🌬️  {exercise['name']}",
        f"   {exercise['benefit']}",
        "",
        "Follow these steps slowly:",
    ]
    for i, step in enumerate(exercise["steps"], 1):
        lines.append(f"  {i}. {step}")
    lines.append("\nTake your time. There is no rush.")
    return "\n".join(lines)


# ── Grounding (5-4-3-2-1) ────────────────────────────────────────────────────

def grounding_exercise(_: str = "") -> str:
    return (
        "🌍 5-4-3-2-1 Grounding Exercise\n"
        "This brings your mind back to the present moment.\n\n"
        "Look around you and name:\n"
        "  5 things you can SEE (a wall, a cup, light through a window…)\n"
        "  4 things you can TOUCH (your chair, your clothes, the floor…)\n"
        "  3 things you can HEAR (birds, traffic, your own breath…)\n"
        "  2 things you can SMELL (or two scents you enjoy)\n"
        "  1 thing you can TASTE (or one thing you are grateful for right now)\n\n"
        "Breathe slowly between each step. You are here, in this moment, and you are safe enough to pause."
    )


# ── Daily motivation from epics ───────────────────────────────────────────────

_MOTIVATIONS = [
    {
        "theme": "courage",
        "en": "Like Hanuman discovering his true strength when Rama needed him most — your capacity is often greater than you believe. One small brave step today is enough.",
        "hi": "जैसे हनुमान ने राम के लिए अपनी असली शक्ति पहचानी — आपकी क्षमता अक्सर आपकी सोच से बड़ी होती है। आज एक छोटा साहसी कदम काफी है।",
        "source": "Ramayan — Hanuman crossing the ocean",
    },
    {
        "theme": "doubt",
        "en": "Arjuna stood frozen on Kurukshetra, full of doubt. Krishna did not mock him — he listened, then guided. Your doubts do not make you weak; they make you human.",
        "hi": "अर्जुन कुरुक्षेत्र में संदेह से भरा खड़ा था। कृष्ण ने उसका मज़ाक नहीं उड़ाया — उन्होंने सुना, फिर मार्ग दिखाया। आपका संदेह कमज़ोरी नहीं, मानवता है।",
        "source": "Mahabharat — Bhagavad Gita, Chapter 1",
    },
    {
        "theme": "patience",
        "en": "Rama waited fourteen years in exile without losing his dharma. Not every struggle resolves quickly — steady patience is also a form of strength.",
        "hi": "राम ने वनवास में चौदह वर्ष धैर्य से धर्म निभाया। हर संघर्ष जल्दी खत्म नहीं होता — स्थिर धैर्य भी एक शक्ति है।",
        "source": "Ramayan — Vanvaas",
    },
    {
        "theme": "resilience",
        "en": "Sita endured captivity in Lanka yet never lost her inner light. Hard times can bend you, but they need not break who you truly are.",
        "hi": "सीता ने लंका में कठिन समय झेला, पर अपनी आंतरिक शक्ति नहीं खोई। कठिनaiyan आपको झुका सकती हैं, पर आपकी असली पहचान नहीं बदल सकतीं।",
        "source": "Ramayan — Ashok Vatika",
    },
    {
        "theme": "duty",
        "en": "Karna faced unfairness his whole life, yet he showed up with full effort every time. Do your best with what you have today — that is enough.",
        "hi": "कर्ण ने जीवन भर अन्याय झेला, फिर भी हर बार पूरा प्रयास किया। आज जो है उससे अपना सर्वश्रेष्ठ करें — यही काफी है।",
        "source": "Mahabharat — Karna's life",
    },
    {
        "theme": "focus",
        "en": "Arjuna saw only the fish's eye while others saw the whole target. When the world feels noisy, pick one small goal and give it your full attention.",
        "hi": "अर्जुन ने सिर्फ मछली की आँख देखी जब दूसरे पूरा लक्ष्य देख रहे थे। जब दुनिया शोरगुल लगे, एक छोटा लक्ष्य चुनें और पूरा ध्यान दें।",
        "source": "Mahabharat — Dronacharya's test",
    },
    {
        "theme": "letting go",
        "en": "Krishna taught that we control our effort, not always the outcome. Release what you cannot change; pour your energy into what you can.",
        "hi": "कृष्ण ने सिखाया — हम प्रयास पर नियंत्रण रखते हैं, परिणाम पर हमेशा नहीं। जो बदल नहीं सकते उसे छोड़ें; जो बदल सकते हैं उस पर ऊर्जा लगाएँ।",
        "source": "Mahabharat — Bhagavad Gita 2.47",
    },
    {
        "theme": "work",
        "en": "Hanuman did not wait for perfect conditions — he leapt across the ocean when duty called. Start the task in front of you; momentum follows action.",
        "hi": "हनुमान ने सही समय का इंतज़ार नहीं किया — जब कर्तव्य बुलाया, समुद्र पार किया। सामने का काम शुरू करें; गति कर्म के साथ आती है।",
        "source": "Ramayan — Lanka journey",
    },
]


def daily_motivation(input_str: str) -> str:
    """
    Input: optional theme — courage, doubt, patience, resilience, duty, focus, work, letting go
    """
    theme = (input_str or "").strip().lower()
    pool = _MOTIVATIONS
    if theme:
        matched = [m for m in _MOTIVATIONS if theme in m["theme"] or theme in m["source"].lower()]
        if matched:
            pool = matched
    pick = random.choice(pool)
    day = datetime.now().strftime("%A, %d %B")
    return (
        f"✨ Daily Inspiration — {day}\n"
        f"📖 Source: {pick['source']}\n\n"
        f"{pick['en']}\n\n"
        f"🇮🇳 Hindi:\n{pick['hi']}"
    )


# ── Mood check-in ─────────────────────────────────────────────────────────────

_MOOD_RESPONSES = {
    "1": ("Very low", "I hear that you are going through a very difficult time. Your feelings are valid. Would you like to try a breathing exercise, or shall I share a story from the epics that speaks to hard days?"),
    "2": ("Low", "It sounds like today feels heavy. That is okay — even the strongest warriors in our epics had dark days. Let us take this one step at a time."),
    "3": ("Neutral", "Thank you for checking in. A neutral day is a good day to build small habits — a short walk, a glass of water, or five minutes of quiet breathing."),
    "4": ("Good", "That is wonderful to hear. Let us channel this energy — perhaps set one meaningful goal for today, like Arjuna focusing on a single target."),
    "5": ("Great", "Beautiful! Celebrate this moment. Share your good energy with someone today, or use it to tackle something you have been postponing."),
}


def mood_checkin(input_str: str) -> str:
    """
    Input: mood level 1-5, optionally with context after comma.
    Example: '2, feeling stressed about work deadline'
    """
    parts = [p.strip() for p in (input_str or "3").split(",", 1)]
    level = parts[0].strip()
    context = parts[1] if len(parts) > 1 else ""

    if level not in _MOOD_RESPONSES:
        return (
            "Please rate your mood from 1 to 5:\n"
            "  1 = Very low   2 = Low   3 = Neutral   4 = Good   5 = Great\n"
            "You can add context after a comma, e.g. '2, stressed about exams'."
        )

    label, response = _MOOD_RESPONSES[level]
    lines = [f"💭 Mood Check-in: {level}/5 ({label})", "", response]
    if context:
        lines.append(f"\nYou shared: \"{context}\"")
        lines.append("I am listening. Tell me more if you would like to talk about it.")
    return "\n".join(lines)


# ── Reflection prompts ────────────────────────────────────────────────────────

_REFLECTION_PROMPTS = [
    "What is one small thing that went well today, no matter how tiny?",
    "If Krishna were sitting beside you right now, what question would you ask?",
    "What burden are you carrying that you could share with someone you trust?",
    "What would Rama do in your situation — act with anger, or act with dharma?",
    "Name three things in your life that you are grateful for right now.",
    "What is one task you can complete in the next 30 minutes to feel a sense of progress?",
    "When did you last feel truly at peace? What was different about that moment?",
    "आज का एक छोटा सा अच्छा पल क्या था?",
    "अभी आप किस बोझ को किसी भरोसेमंद व्यक्ति के साथ साझा कर सकते हैं?",
]


def reflection_prompt(_: str = "") -> str:
    prompt = random.choice(_REFLECTION_PROMPTS)
    return (
        f"🪞 Reflection Prompt\n\n"
        f"{prompt}\n\n"
        f"Take a moment with this. There is no right or wrong answer — only honest reflection."
    )


# ── Wellness dashboard (keeps WebSocket handler working without finance data) ─

def get_dashboard_data() -> dict:
    pick = random.choice(_MOTIVATIONS)
    return {
        "mode": "wellness",
        "daily_quote": pick["en"],
        "daily_source": pick["source"],
        "breathing_available": list(_BREATHING.keys()),
        "mood_scale": "1 (very low) to 5 (great)",
        "timestamp": datetime.now().isoformat(),
    }
