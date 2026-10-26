"""
PII Detection and Redaction Module for Indic Corpus.
Redacts:
- Indian Phone Numbers: +91, 0, or bare 10-digit mobile numbers starting with 6-9 -> <PHONE>
- Email Addresses: Standard compliant emails -> <EMAIL>
- Aadhaar Numbers: 12 digits (with optional spaces/hyphens), verified by Verhoeff checksum -> <AADHAAR>
- PAN Numbers: 10 characters [A-Z]{5}[0-9]{4}[A-Z]{1} -> <PAN>

Preserves legitimate numbers: dates, prices, quantities, PIN codes, chapter/verse numbers.
Safe: never logs or exposes raw PII values.
"""

import re
from typing import Tuple, Dict

# Verhoeff algorithm implementation for validating Indian Aadhaar numbers
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]

def validate_verhoeff(num_str: str) -> bool:
    """Validates 12-digit Aadhaar number using Verhoeff checksum algorithm."""
    if not num_str.isdigit() or len(num_str) != 12:
        return False
    # Aadhaar numbers cannot start with 0 or 1
    if num_str[0] in ("0", "1"):
        return False
    c = 0
    reversed_digits = [int(d) for d in reversed(num_str)]
    for i, digit in enumerate(reversed_digits):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][digit]]
    return c == 0

# PII Regular Expressions
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Indian PAN: 5 letters (4th letter is usually status e.g. P, C, H, F, A, T, B, L, J, G), 4 digits, 1 letter
PAN_REGEX = re.compile(
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
)

# Indian Phone numbers:
# Formats:
# +91 9876543210, +91-98765-43210, 09876543210, 9876543210, 98765 43210
# Boundary-aware: does not match larger numbers or dates.
PHONE_REGEX = re.compile(
    r"(?:(?<=\s)|(?<=^)|(?<=[(\[,:;]))" # Lookbehind boundary
    r"(?:\+91[\-\s]?)?"                 # Optional +91 prefix
    r"(?:0)?"                           # Optional 0 prefix
    r"([6-9]\d{4}[\-\s]?\d{5}|[6-9]\d{9})" # 10 digits starting with 6-9
    r"(?=(?:[\s)\].,:;!?]|$))"          # Lookahead boundary
)

# Aadhaar Pattern: 12 digits, often formatted as 4-4-4 or contiguous
# e.g., 3456 7890 1234 or 3456-7890-1234 or 345678901234
AADHAAR_CANDIDATE_REGEX = re.compile(
    r"(?:(?<=\s)|(?<=^)|(?<=[(\[,:;]))"
    r"([2-9]\d{3}[\s\-]??\d{4}[\s\-]??\d{4})"
    r"(?=(?:[\s)\].,:;!?]|$))"
)

def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Scans and redacts PII instances in the text.
    Returns:
      (redacted_text: str, counts: dict)
    """
    counts = {
        "EMAIL": 0,
        "PAN": 0,
        "AADHAAR": 0,
        "PHONE": 0
    }

    if not text:
        return text, counts

    # 1. Redact Email
    def email_repl(m):
        counts["EMAIL"] += 1
        return "<EMAIL>"
    redacted = EMAIL_REGEX.sub(email_repl, text)

    # 2. Redact PAN
    def pan_repl(m):
        counts["PAN"] += 1
        return "<PAN>"
    redacted = PAN_REGEX.sub(pan_repl, redacted)

    # 3. Redact Aadhaar (with Verhoeff checksum validation to prevent redacting random 12-digit numbers)
    def aadhaar_repl(m):
        raw_match = m.group(1)
        digits_only = re.sub(r"[\s\-]", "", raw_match)
        if len(digits_only) == 12 and validate_verhoeff(digits_only):
            counts["AADHAAR"] += 1
            return "<AADHAAR>"
        return raw_match # Keep original text if Verhoeff fails (e.g. tracking numbers, timestamps)

    redacted = AADHAAR_CANDIDATE_REGEX.sub(aadhaar_repl, redacted)

    # 4. Redact Indian Phone Numbers
    def phone_repl(m):
        counts["PHONE"] += 1
        return "<PHONE>"
    redacted = PHONE_REGEX.sub(phone_repl, redacted)

    return redacted, counts

