"""
Grader module — deterministic, reproducible scoring for all 3 tasks.

Each grader returns a float in [0.0, 1.0] and a feedback string.
No randomness. No ML scoring. Pure rule-based so scores are reproducible.

Task 1 — Classification  (easy)
Task 2 — Response Quality (medium)
Task 3 — Full Resolution  (hard)
"""

from __future__ import annotations
import re
from typing import Tuple

# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return text.lower().strip()


def _keyword_hit_rate(text: str, keywords: list[str]) -> float:
    """Fraction of keywords found in text (case-insensitive)."""
    if not keywords:
        return 0.0
    norm = _normalise(text)
    hits = sum(1 for kw in keywords if _normalise(kw) in norm)
    return hits / len(keywords)


def _contains_apology(text: str) -> bool:
    apology_phrases = [
        "sorry", "apologize", "apologies", "regret", "we understand",
        "inconvenience", "sincerely sorry", "deeply sorry"
    ]
    norm = _normalise(text)
    return any(p in norm for p in apology_phrases)


def _contains_action(text: str) -> bool:
    action_phrases = [
        "will", "shall", "we are", "we have", "processing", "initiating",
        "arranged", "escalat", "refund", "replace", "ship", "resend",
        "contact", "investigate", "look into", "follow up"
    ]
    norm = _normalise(text)
    return any(p in norm for p in action_phrases)


def _response_length_score(text: str, min_words: int = 30) -> float:
    """Penalise very short replies; reward adequately detailed ones."""
    word_count = len(text.split())
    if word_count >= min_words:
        return 1.0
    return round(word_count / min_words, 2)


def _professionalism_score(text: str) -> float:
    """Check for a greeting and a closing — basic professional structure."""
    norm = _normalise(text)
    has_greeting = any(g in norm for g in ["dear", "hello", "hi", "greetings", "good day"])
    has_closing = any(c in norm for c in [
        "sincerely", "regards", "best", "thank you", "yours", "warm regards",
        "kind regards", "respectfully"
    ])
    return (0.5 if has_greeting else 0.0) + (0.5 if has_closing else 0.0)


# ── Task 1: Classification ─────────────────────────────────────────────────────

def grade_classification(reply: str, expected_category: str) -> Tuple[float, str]:
    """
    Score: 1.0 if category mentioned; 0.5 for near-miss synonyms; 0.0 otherwise.
    Deterministic — no model calls.
    """
    SYNONYMS = {
        "billing":   ["billing", "payment", "charge", "invoice", "financial", "fee"],
        "complaint": ["complaint", "issue", "problem", "concern", "dissatisfied", "unhappy", "wrong"],
        "query":     ["query", "question", "inquiry", "enquiry", "information", "help", "support"],
    }

    norm = _normalise(reply)
    exact_match = expected_category in norm
    if exact_match:
        return 1.0, f"✅ Exact category '{expected_category}' found."

    synonym_match = any(syn in norm for syn in SYNONYMS.get(expected_category, []))
    if synonym_match:
        return 0.6, f"⚠️ Synonym for '{expected_category}' found (partial credit)."

    return 0.0, f"❌ Category '{expected_category}' not identified in reply."


# ── Task 2: Response Quality ───────────────────────────────────────────────────

def grade_response(reply: str, email_data: dict) -> Tuple[float, str]:
    """
    Multi-criterion response quality score:
      - Keyword coverage     (40%)
      - Apology present      (20%)
      - Concrete action      (20%)
      - Length adequacy      (10%)
      - Professionalism      (10%)
    """
    keywords = email_data["keywords"]

    kw_score       = _keyword_hit_rate(reply, keywords)          # 0–1
    apology_score  = 1.0 if _contains_apology(reply) else 0.0   # 0 or 1
    action_score   = 1.0 if _contains_action(reply) else 0.0    # 0 or 1
    length_score   = _response_length_score(reply, min_words=30) # 0–1
    prof_score     = _professionalism_score(reply)               # 0–1

    total = (
        0.40 * kw_score +
        0.20 * apology_score +
        0.20 * action_score +
        0.10 * length_score +
        0.10 * prof_score
    )
    total = round(min(max(total, 0.0), 1.0), 4)

    feedback = (
        f"kw={kw_score:.2f} apology={apology_score:.1f} "
        f"action={action_score:.1f} length={length_score:.2f} "
        f"professionalism={prof_score:.2f} → total={total:.2f}"
    )
    return total, feedback


# ── Task 3: Full Resolution ────────────────────────────────────────────────────

def grade_resolution(reply: str, email_data: dict) -> Tuple[float, str]:
    """
    Hardest grader — checks that the reply FULLY resolves the issue:
      - Correct classification mentioned      (20%)
      - All keywords addressed                (30%)
      - Apology + concrete action             (20%)
      - Expected resolution criteria met      (20%)
      - Professionalism                       (10%)
    """
    category    = email_data["category"]
    keywords    = email_data["keywords"]
    expected    = email_data["expected_resolution"]

    cls_score, _  = grade_classification(reply, category)
    cls_score_w   = cls_score * 0.20

    kw_score      = _keyword_hit_rate(reply, keywords) * 0.30

    apology       = 1.0 if _contains_apology(reply) else 0.0
    action        = 1.0 if _contains_action(reply) else 0.0
    combo_score   = ((apology + action) / 2.0) * 0.20

    # Resolution criterion: check if key phrases from expected resolution appear
    expected_kws  = [w for w in expected.lower().split() if len(w) > 4]
    res_score     = _keyword_hit_rate(reply, expected_kws) * 0.20

    prof_score    = _professionalism_score(reply) * 0.10

    total = round(cls_score_w + kw_score + combo_score + res_score + prof_score, 4)
    total = min(max(total, 0.0), 1.0)

    feedback = (
        f"classification={cls_score:.2f} keywords={kw_score/0.30:.2f} "
        f"apology={apology:.1f} action={action:.1f} "
        f"resolution_match={res_score/0.20:.2f} "
        f"professionalism={prof_score/0.10:.2f} → total={total:.2f}"
    )
    return total, feedback