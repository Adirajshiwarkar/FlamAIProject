"""
Language Identification Module.
Assigns exactly one label from:
- 'hi'      : Hindi in Devanagari script (including code-mixed English words)
- 'hi-Latn' : Hindi written in Latin script (Hinglish)
- 'en'      : English
- 'other'   : Other languages or ambiguous text

Never trusts source labels. Uses script analysis and high-precision lexicon/n-gram heuristics.
"""

import re
from typing import Tuple, Dict

# Character ranges
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
LATIN_REGEX = re.compile(r"[a-zA-Z]")
ARABIC_URDU_REGEX = re.compile(r"[\u0600-\u06FF]")
OTHER_INDIC_REGEX = re.compile(r"[\u0980-\u0D7F]") # Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam

# Common Hinglish stopwords and diagnostic markers (Latin script)
HINGLISH_MARKERS = {
    "hai", "hain", "tha", "thee", "thi", "the", "nahi", "nahin", "yeh", "woh",
    "kya", "kyun", "kaise", "kisko", "kisko", "hum", "aap", "tum", "mera", "meri",
    "mere", "tera", "teri", "tere", "unka", "unki", "iska", "iski", "isliye",
    "lekin", "magar", "par", "bhi", "toh", "to", "ho", "raha", "rahi", "rahe",
    "gaya", "gayi", "gaye", "karna", "kar", "karo", "karta", "karti", "karte",
    "baat", "accha", "achha", "bahut", "bohot", "zyada", "thoda", "kam", "waqt",
    "samay", "dost", "pyaar", "zindagi", "desh", "bharat", "kaam", "ghar",
    "shuru", "aaj", "kal", "ab", "jab", "tab", "sab", "kuch", "kuchh", "saath",
    "wala", "wali", "wale", "hoga", "hogi", "honge", "chahiye", "lagta", "lagti"
}

ENGLISH_COMMON_WORDS = {
    "the", "is", "are", "was", "were", "this", "that", "these", "those", "and",
    "or", "but", "in", "on", "at", "to", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "from", "up",
    "down", "in", "out", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "can", "will", "just", "should",
    "now", "they", "them", "their", "theirs", "themselves", "what", "which", "who",
    "whom", "whose", "you", "your", "yours", "yourself", "yourselves"
}

WORD_TOKENIZER = re.compile(r"[^\W\d_]+", re.UNICODE)

def classify_language(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Classifies text into one of: 'hi', 'hi-Latn', 'en', 'other'.
    Returns (label, feature_scores).
    """
    if not text or not text.strip():
        return "other", {"reason": "empty"}

    # Count script characters
    total_dev = len(DEVANAGARI_REGEX.findall(text))
    total_lat = len(LATIN_REGEX.findall(text))
    total_other_indic = len(OTHER_INDIC_REGEX.findall(text))
    total_ar = len(ARABIC_URDU_REGEX.findall(text))
    
    alpha_chars = total_dev + total_lat + total_other_indic + total_ar
    if alpha_chars == 0:
        return "other", {"reason": "no_alpha_characters"}

    dev_ratio = total_dev / alpha_chars
    lat_ratio = total_lat / alpha_chars
    other_ratio = (total_other_indic + total_ar) / alpha_chars

    # 1. Other non-target scripts dominate
    if other_ratio > 0.4:
        return "other", {"dev_ratio": dev_ratio, "lat_ratio": lat_ratio, "other_ratio": other_ratio}

    # 2. Devanagari Hindi (including code-mixed English words)
    # If Devanagari is at least 30% of alphabetic characters and higher than other scripts,
    # or if Devanagari characters >= 15 and dev_ratio >= 0.25, classify as 'hi'.
    if dev_ratio >= 0.30:
        return "hi", {"dev_ratio": dev_ratio, "lat_ratio": lat_ratio}

    # 3. Latin script text: Could be English ('en'), Hinglish ('hi-Latn'), or other European/Latin language
    if lat_ratio >= 0.60:
        words = [w.lower() for w in WORD_TOKENIZER.findall(text) if len(w) > 1]
        if not words:
            return "other", {"reason": "no_valid_latin_words"}

        hinglish_hits = sum(1 for w in words if w in HINGLISH_MARKERS)
        english_hits = sum(1 for w in words if w in ENGLISH_COMMON_WORDS)
        total_words = len(words)

        hinglish_frac = hinglish_hits / total_words
        english_frac = english_hits / total_words

        # Classification criteria for Hinglish (hi-Latn):
        # - Has distinct Hinglish grammatical markers / vocabulary (at least 2 markers or >= 12% Hinglish markers)
        # - More Hinglish markers than pure English grammatical stopwords
        if hinglish_hits >= 2 or (hinglish_hits >= 1 and hinglish_frac >= 0.15 and hinglish_hits >= english_hits):
            return "hi-Latn", {
                "hinglish_hits": hinglish_hits,
                "english_hits": english_hits,
                "hinglish_frac": hinglish_frac,
                "english_frac": english_frac
            }
        
        # English:
        if english_frac >= 0.15 and english_hits > hinglish_hits:
            return "en", {
                "hinglish_hits": hinglish_hits,
                "english_hits": english_hits,
                "hinglish_frac": hinglish_frac,
                "english_frac": english_frac
            }

        # If it has some Hinglish words but very short, check single dominant word
        if hinglish_hits >= 1 and total_words <= 4:
            return "hi-Latn", {"hinglish_hits": hinglish_hits, "total_words": total_words}

        return "other", {
            "reason": "ambiguous_latin",
            "hinglish_hits": hinglish_hits,
            "english_hits": english_hits
        }

    return "other", {"dev_ratio": dev_ratio, "lat_ratio": lat_ratio}

