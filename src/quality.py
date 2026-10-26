"""
Quality Filtering Module.
Detects and filters out low-quality documents:
- Spam and repetitive boilerplate
- HTML and markup artifacts
- Corrupted/garbled character encoding (mojibake)
- Empty or nearly empty documents
- Extreme repetition and obvious extraction failures

Preserves legitimate but unusual text:
- Short proverbs/sayings
- Poetry / verses
- Valid numerical tables / factual lists
"""

import re
from typing import Tuple, Dict, Any

HTML_TAG_REGEX = re.compile(r"<[^>]+>", re.IGNORECASE)
BOILERPLATE_PATTERNS = [
    re.compile(r"click\s+here\s+to\s+subscribe", re.I),
    re.compile(r"terms\s+and\s+conditions\s+apply", re.I),
    re.compile(r"all\s+rights\s+reserved", re.I),
    re.compile(r"cookie\s+policy", re.I),
    re.compile(r"javascript\s+is\s+disabled", re.I),
    re.compile(r"404\s+not\s+found", re.I),
    re.compile(r"forbidden\s+403", re.I),
    re.compile(r"page\s+not\s+found", re.I),
    re.compile(r"privacy\s+policy", re.I),
    re.compile(r"copyright\s+©", re.I),
]

MOJIBAKE_REGEX = re.compile(r"(?:Ã[\x80-\xBF]|â€[\x80-\xBF]|ï¿½){2,}")

def check_quality(text: str, lang: str = "hi") -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluates quality of a document.
    Returns:
      (passed: bool, reason: str, metadata: dict)
    """
    if not text:
        return False, "empty_text", {}

    length = len(text)
    if length < 5:
        return False, "too_short_less_than_5_chars", {"length": length}

    # 1. Check for corrupted encoding / mojibake
    mojibake_matches = MOJIBAKE_REGEX.findall(text)
    if mojibake_matches:
        return False, "garbled_encoding_mojibake", {"matches": len(mojibake_matches)}

    # Check for replacement character
    if "\ufffd" in text:
        return False, "unicode_replacement_char_present", {}

    # 2. Check HTML/markup contamination
    html_tags = HTML_TAG_REGEX.findall(text)
    if len(html_tags) >= 3 or (len(html_tags) >= 1 and sum(len(t) for t in html_tags) > 0.3 * length):
        return False, "html_markup_boilerplate", {"tag_count": len(html_tags)}

    # 3. Check Web Boilerplate
    for bp in BOILERPLATE_PATTERNS:
        if bp.search(text):
            return False, "boilerplate_detected", {"pattern": bp.pattern}

    # 4. Token & Character repetition checks
    words = text.split()
    total_words = len(words)

    # If it's a short proverb or title (e.g. 2-5 words), don't falsely reject unless clearly repetitive
    if total_words >= 5:
        # Unique word ratio
        unique_words = set(w.lower() for w in words)
        unique_ratio = len(unique_words) / total_words
        if unique_ratio < 0.20:
            return False, "extreme_word_repetition", {"unique_ratio": unique_ratio}

        # Check for single word repetition (e.g. "ha ha ha ha ha...")
        word_freqs = {}
        for w in words:
            wl = w.lower()
            word_freqs[wl] = word_freqs.get(wl, 0) + 1
        most_common_freq = max(word_freqs.values())
        if most_common_freq / total_words > 0.50 and most_common_freq > 4:
            return False, "single_word_dominance", {"freq_ratio": most_common_freq / total_words}

    # 5. Character repetition checks (e.g. "aaaaaaaaaaaa" or "............")
    char_repeat_match = re.search(r"(.)\1{9,}", text)
    if char_repeat_match:
        return False, "extreme_character_repetition", {"char": char_repeat_match.group(1)}

    # 6. Alphanumeric / meaningful symbol ratio
    # Legitimate numbers / tables / proverbs should pass.
    # Text with mostly gibberish punctuation like !@#%^&* should fail.
    punct_or_sym_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if length > 20 and (punct_or_sym_count / length) > 0.65:
        return False, "excessive_punctuation_symbols", {"punct_ratio": punct_or_sym_count / length}

    return True, "quality_passed", {"word_count": total_words, "char_count": length}

