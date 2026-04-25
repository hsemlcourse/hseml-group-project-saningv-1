import re

import pandas as pd


URL_RE = re.compile(r"(?:https?://|www\.|[a-z0-9.-]+\.[a-z]{2,})", flags=re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")
CURRENCY_RE = re.compile(r"[$£€]|(?:\b(?:usd|eur|gbp|pounds?|dollars?)\b)", flags=re.IGNORECASE)
CTA_RE = re.compile(
    r"\b(call|text|txt|reply|claim|win|winner|free|urgent|prize|cash|bonus|offer|stop)\b",
    flags=re.IGNORECASE,
)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create compact, interpretable features that are common in SMS spam."""
    result = df.copy()
    message = result["message"].fillna("").astype(str)

    result["message_length"] = message.str.len()
    result["word_count"] = message.str.split().str.len()
    result["digit_count"] = message.str.count(r"\d")
    result["uppercase_count"] = message.apply(lambda text: sum(char.isupper() for char in text))
    result["uppercase_ratio"] = result["uppercase_count"] / result["message_length"].clip(lower=1)
    result["exclamation_count"] = message.str.count("!")
    result["question_count"] = message.str.count(r"\?")
    result["special_char_count"] = message.str.count(r"[^A-Za-z0-9\s]")
    result["has_url"] = message.str.contains(URL_RE).astype(int)
    result["has_phone"] = message.str.contains(PHONE_RE).astype(int)
    result["has_currency"] = message.str.contains(CURRENCY_RE).astype(int)
    result["cta_word_count"] = message.str.count(CTA_RE)

    return result
