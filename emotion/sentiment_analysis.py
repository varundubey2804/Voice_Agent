"""
emotion/sentiment_analysis.py — Turns user text into one of ir_trigger's
emotion-color categories (calm / neutral / happy / stressed / confused) and
fires the matching MacroDroid webhook, so ambient hardware (lights, etc.)
tracks the user's mood in real time.

WHY TWO SIGNALS, NOT ONE
─────────────────────────
Sentiment polarity alone ("positive"/"negative") isn't the same thing as
what the hardware side cares about here:
  - "I'm so stressed about this deadline" is negative in polarity AND
    high-arousal → should map to "stressed", not just "negative".
  - "I'm a bit tired but okay" is mildly negative polarity but low-arousal
    → should map to "calm"/"neutral", not "stressed".
So this module layers two independent, already-available signals:
  1. VALENCE — VADER sentiment (positive/negative/neutral), with a small
     keyword-based fallback if vaderSentiment isn't installed.
  2. AROUSAL / STRESS — wellness_tools.assess_stress(), already used
     elsewhere in the pipeline, gives a "low"/"medium"/"high" stress level.
Combining them gives a much better hardware-facing label than sentiment
polarity alone would.

USAGE
─────
    from emotion.sentiment_analysis import analyze_sentiment, analyze_and_trigger

    result = analyze_sentiment("I can't handle this deadline, I'm so anxious")
    # {"label": "stressed", "compound": -0.61, "stress_level": "high", ...}

    analyze_and_trigger(user_text)   # also fires the IR/MacroDroid trigger
"""

from __future__ import annotations

import re
from typing import Optional, TypedDict

import wellness_tools
from . import bulb_trigger

# ── VADER (preferred) with a graceful keyword fallback ──────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _vader = None
    print(
        "⚠️  vaderSentiment not installed — sentiment_analysis is using a "
        "basic keyword fallback. For better accuracy: "
        "pip install vaderSentiment --break-system-packages"
    )

# Small fallback lexicon, only used when vaderSentiment is unavailable.
_POSITIVE_WORDS = (
    "good", "great", "happy", "glad", "awesome", "nice", "love", "thanks",
    "thank you", "better", "relieved", "grateful", "excited", "wonderful",
    "अच्छा", "खुश", "धन्यवाद", "बढ़िया",
)
_NEGATIVE_WORDS = (
    "bad", "sad", "hate", "terrible", "awful", "upset", "angry", "annoyed",
    "worse", "horrible", "disappointed", "बुरा", "गुस्सा", "दुखी",
)


def _fallback_compound(text: str) -> float:
    """Crude polarity estimate for when vaderSentiment isn't installed."""
    lower = text.lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in lower)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in lower)
    total = pos + neg
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / total))


def _compound_score(text: str) -> float:
    if _vader is not None:
        return _vader.polarity_scores(text)["compound"]
    return _fallback_compound(text)


# ── Confusion detection (cognitive state, not sentiment polarity) ──────────
_CONFUSED_KEYWORDS = (
    "confused", "don't understand", "dont understand", "not sure",
    "unclear", "puzzled", "what do you mean", "i don't get it",
    "समझ नहीं", "पता नहीं", "క్లియర్ కాలేదు", "புரியவில்லை",
)

# ── Calm keywords — lets us tell "calm" apart from generic "neutral" ───────
_CALM_KEYWORDS = (
    "calm", "relaxed", "peaceful", "fine", "okay", "alright", "content",
    "at ease", "settled", "शांत", "ठीक हूं", "आराम",
)


def _looks_confused(text: str) -> bool:
    lower = text.lower().strip()
    if any(kw in lower for kw in _CONFUSED_KEYWORDS):
        return True
    # A short question with a question-word is a reasonable confusion signal
    # in a voice-transcript context (not just any question, to avoid
    # flagging rhetorical/emotional questions like "why does this happen").
    return bool(re.search(r"\b(how|what|which)\b.{0,40}\?\s*$", lower))


class SentimentResult(TypedDict):
    label: str                 # one of ir_trigger.EMOTION_TO_TRIGGER's keys
    compound: float            # VADER-style polarity, -1..1
    stress_level: str          # "low" | "medium" | "high" (from wellness_tools)
    stress_score: int
    source: str                # "vader" or "keyword_fallback"


def analyze_sentiment(text: str) -> SentimentResult:
    """
    Score user text and map it to a single emotion-color label that
    emotion/ir_trigger.py understands: calm, neutral, happy, stressed,
    confused.

    Priority order (highest confidence signal wins):
      1. High stress (wellness_tools) → "stressed" — arousal overrides
         polarity; a stressed-but-polite message still needs "stressed".
      2. Confusion keywords / question pattern → "confused"
      3. Medium stress → "stressed"
      4. VADER polarity: strongly positive → "happy",
         strongly negative → "stressed"
      5. Otherwise: calm keywords → "calm", else → "neutral"
    """
    text = text or ""
    stress = wellness_tools.assess_stress(text)
    compound = _compound_score(text)
    source = "vader" if _vader is not None else "keyword_fallback"

    if stress["level"] == "high":
        label = "stressed"
    elif _looks_confused(text):
        label = "confused"
    elif stress["level"] == "medium":
        label = "stressed"
    elif compound >= 0.35:
        label = "happy"
    elif compound <= -0.25:
        label = "stressed"
    elif any(kw in text.lower() for kw in _CALM_KEYWORDS):
        label = "calm"
    else:
        label = "neutral"

    return {
        "label": label,
        "compound": round(compound, 3),
        "stress_level": stress["level"],
        "stress_score": stress["score"],
        "source": source,
    }


def analyze_and_trigger(
    text: str,
    blocking: bool = False,
    force: bool = False,
) -> SentimentResult:
    """
    Analyze text and immediately fire the matching MacroDroid/IR trigger,
    through the same cooldown-guarded gate (emotion.bulb_trigger) used by
    the vision-pipeline endpoint — so both paths can't hammer the webhook
    faster than COOLDOWN_S combined. Non-blocking by default so it never
    adds latency to the voice turn.
    """
    result = analyze_sentiment(text)
    bulb_trigger.handle_bulb_trigger(result["label"], force=force, blocking=blocking)
    return result


if __name__ == "__main__":
    import sys
    sample = " ".join(sys.argv[1:]) or "I'm so stressed about this deadline, I can't handle it"
    print(analyze_sentiment(sample))