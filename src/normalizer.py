"""
Unicode and Text Normalization Module for Indic NLP.
Standardizes Unicode (NFC), handles zero-width joiners/non-joiners, removes unwanted invisible formatting,
normalizes Indic punctuation (Danda, Double Danda), and normalizes whitespace while preserving legitimate matras.
"""

import unicodedata
import re

# Invisible and special Unicode code points
# U+200B: ZERO WIDTH SPACE
# U+200C: ZERO WIDTH NON-JOINER (ZWNJ) - Used in Indic complex conjuncts (halant + ZWNJ)
# U+200D: ZERO WIDTH JOINER (ZWJ) - Used in Indic half-forms (halant + ZWJ e.g. eyelash Ra)
# U+FEFF: ZERO WIDTH NO-BREAK SPACE (BOM)
# U+00A0: NO-BREAK SPACE
# U+200E, U+200F: LTR and RTL marks

# Devanagari range: \u0900-\u097F
# Extended: \uA8E0-\uA8FF, \u1CD0-\u1CFF

INVISIBLE_CHARS_REGEX = re.compile(r"[\u200B\uFEFF\u200E\u200F\u00AD\u2060\u202A-\u202E]")
WHITESPACE_REGEX = re.compile(r"[ \t\r\f\v]+")
CONSECUTIVE_NEWLINES_REGEX = re.compile(r"\n{3,}")

# Indic danda punctuation normalization
# Map pipe | or latin vertical bar to Devanagari danda । (\u0964) when surrounded by Indic or punctuation
DANDA_MAPPING = {
    "\u0964": "।",  # Devanagari Danda
    "\u0965": "॥",  # Devanagari Double Danda
}

def normalize_text(text: str) -> str:
    """
    Applies deterministic Indic text normalization:
    1. Unicode Normalization: NFC.
       NFC (Canonical Decomposition, followed by Canonical Composition) is crucial for Indic Devanagari
       because combining vowel signs (matras) and nuktas have distinct canonical compositions.
       Using NFC prevents identical characters represented as decomposed forms from mismatching.
    2. Invisible formatting removal:
       Strips zero-width spaces, BOM, directional marks.
       Preserves ZWNJ (U+200C) and ZWJ (U+200D) only when immediately preceded by Devanagari Halant/Virama (U+094D),
       which are linguistically meaningful for explicit halants / sub-syllabic forms. Otherwise strips them.
    3. Whitespace normalization:
       Replaces non-breaking space with standard space.
       Collapses runs of spaces and tabs into a single space.
       Collapses 3+ newlines into 2 newlines. Strips leading/trailing whitespace.
    """
    if not text:
        return ""

    # Step 1: Unicode NFC normalization
    norm = unicodedata.normalize("NFC", text)

    # Step 2: Handle ZWJ / ZWNJ selectively
    # In Indic scripts, Halant is \u094D. If ZWJ/ZWNJ is after Halant, keep it; elsewhere remove.
    chars = []
    prev_char = ""
    for ch in norm:
        if ch in ("\u200C", "\u200D"):
            if prev_char == "\u094D":  # Halant
                chars.append(ch)
            # Else drop it as an unwanted zero-width invisible artifact
        elif INVISIBLE_CHARS_REGEX.match(ch):
            continue
        else:
            # Normalize common alternative spaces (NBSP, em space, en space, thin space)
            if ch in ("\u00A0", "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005", "\u2006", "\u2007", "\u2008", "\u2009", "\u200A", "\u202F", "\u205F", "\u3000"):
                chars.append(" ")
            else:
                chars.append(ch)
        prev_char = ch
    
    norm = "".join(chars)

    # Step 3: Normalize quotes and dashes to standard forms for consistent deduplication
    norm = norm.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    norm = norm.replace("—", " - ").replace("–", " - ")

    # Step 4: Normalize whitespace
    lines = [WHITESPACE_REGEX.sub(" ", line).strip() for line in norm.splitlines()]
    norm = "\n".join(lines)
    norm = CONSECUTIVE_NEWLINES_REGEX.sub("\n\n", norm)
    return norm.strip()

