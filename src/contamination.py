"""
Evaluation Set Creation & Cross-Corpus Contamination Detection Module.

Requirements:
1. Build an eval set of ~200 items from held-out data.
2. Contamination checks:
   - Exact text matches
   - Matches after Unicode/whitespace normalization
   - Near-duplicate matches (Jaccard on shingles)
   - Cross-script contamination: transliterated Devanagari Hindi vs Latin script Hindi (Hinglish)
3. Detect leaks, report in contamination.jsonl:
   { "eval_id": str, "train_ids": [str, ...], "method": str }
4. Purge contaminated training documents from final training corpus, update ledger and counts.
"""

import re
import hashlib
from typing import List, Dict, Any, Set, Tuple

# Mapping table for Devanagari characters to Roman phonetics for cross-script matching
DEVA_TO_LATIN = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', '्': '',
    'ं': 'n', 'ः': 'h', 'ँ': 'n', '़': ''
}

def approximate_devanagari_to_roman(text: str) -> str:
    """
    Converts Devanagari characters to rough phonetic Latin form.
    Allows detecting cross-script contamination where training has Latin Hinglish
    and eval has the equivalent Devanagari Hindi text.
    """
    res = []
    for ch in text:
        res.append(DEVA_TO_LATIN.get(ch, ch))
    out = "".join(res).lower()
    out = re.sub(r"aa+", "a", out)
    out = re.sub(r"ee+", "i", out)
    out = re.sub(r"oo+", "u", out)
    out = re.sub(r"[^\w\s]", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def normalize_latin_phonetics(text: str) -> str:
    """Normalizes Latin Hinglish to simplify phonetic cross-matching."""
    out = text.lower()
    out = re.sub(r"aa+", "a", out)
    out = re.sub(r"ee+", "i", out)
    out = re.sub(r"oo+", "u", out)
    out = re.sub(r"[^\w\s]", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def compute_shingle_sim(s1: str, s2: str, k: int = 3) -> float:
    """Computes exact Jaccard similarity between character shingles of two strings."""
    if not s1 or not s2:
        return 0.0
    shingles1 = {s1[i:i+k] for i in range(len(s1) - k + 1)} if len(s1) >= k else {s1}
    shingles2 = {s2[i:i+k] for i in range(len(s2) - k + 1)} if len(s2) >= k else {s2}
    inter = len(shingles1.intersection(shingles2))
    union = len(shingles1.union(shingles2))
    return inter / union if union > 0 else 0.0

class ContaminationDetector:
    """
    Maintains indices over the candidate training set to test evaluation set leakage.
    """
    def __init__(self, near_match_threshold: float = 0.85, cross_script_threshold: float = 0.82):
        self.near_match_threshold = near_match_threshold
        self.cross_script_threshold = cross_script_threshold

        # Training set indexes
        self.exact_train: Dict[str, List[str]] = {}       # raw text -> [doc_id]
        self.norm_train: Dict[str, List[str]] = {}        # normalized text -> [doc_id]
        self.doc_norm_texts: Dict[str, str] = {}          # doc_id -> normalized text
        self.doc_phonetic_texts: Dict[str, str] = {}      # doc_id -> phonetic Latin text

    def index_training_doc(self, doc_id: str, raw_text: str, norm_text: str, lang: str):
        # 1. Exact raw index
        self.exact_train.setdefault(raw_text, []).append(doc_id)
        # 2. Normalized index
        self.norm_train.setdefault(norm_text, []).append(doc_id)
        self.doc_norm_texts[doc_id] = norm_text

        # 3. Phonetic Latin representation for cross-script checking
        if lang in ("hi-Latn", "en"):
            self.doc_phonetic_texts[doc_id] = normalize_latin_phonetics(norm_text)
        elif lang == "hi":
            self.doc_phonetic_texts[doc_id] = approximate_devanagari_to_roman(norm_text)

    def check_eval_item(self, eval_id: str, raw_text: str, norm_text: str, lang: str) -> List[Dict[str, Any]]:
        """
        Checks an eval item against the training corpus for contamination.
        Returns list of findings:
          [{ "eval_id": eval_id, "train_ids": [doc_id, ...], "method": str, "evidence": str }]
        """
        findings = []
        already_matched_tids: Set[str] = set()

        # Check 1: Exact text match
        if raw_text in self.exact_train:
            leaking_ids = [tid for tid in self.exact_train[raw_text] if tid != eval_id]
            if leaking_ids:
                uniq_ids = sorted(list(set(leaking_ids)))
                findings.append({
                    "eval_id": eval_id,
                    "train_ids": uniq_ids,
                    "method": "exact_raw_match",
                    "evidence": raw_text[:80]
                })
                already_matched_tids.update(uniq_ids)

        # Check 2: Normalized match (differs only by whitespace/accents/invisible characters)
        if norm_text in self.norm_train:
            leaking_ids = [tid for tid in self.norm_train[norm_text] if tid != eval_id and tid not in already_matched_tids]
            if leaking_ids:
                uniq_ids = sorted(list(set(leaking_ids)))
                findings.append({
                    "eval_id": eval_id,
                    "train_ids": uniq_ids,
                    "method": "normalized_match",
                    "evidence": norm_text[:80]
                })
                already_matched_tids.update(uniq_ids)

        # Check 3: Near-duplicate match and Cross-script contamination
        eval_phonetic = None
        if lang == "hi":
            eval_phonetic = approximate_devanagari_to_roman(norm_text)
        else:
            eval_phonetic = normalize_latin_phonetics(norm_text)

        eval_len = len(norm_text)
        for train_id, train_norm in self.doc_norm_texts.items():
            if train_id == eval_id or train_id in already_matched_tids:
                continue
            train_len = len(train_norm)
            # Only compare if lengths are comparable (+/- 25%)
            if abs(train_len - eval_len) <= max(8, 0.25 * max(eval_len, train_len)):
                # Near-duplicate in same script
                sim = compute_shingle_sim(norm_text, train_norm, k=3)
                if sim >= self.near_match_threshold:
                    findings.append({
                        "eval_id": eval_id,
                        "train_ids": [train_id],
                        "method": "near_duplicate",
                        "similarity": round(sim, 3),
                        "evidence": f"eval: '{norm_text[:40]}' vs train: '{train_norm[:40]}'"
                    })
                    already_matched_tids.add(train_id)
                    continue

                # Cross-script check: Devanagari eval vs Latin train (or vice versa)
                train_phonetic = self.doc_phonetic_texts.get(train_id)
                if train_phonetic and eval_phonetic and len(eval_phonetic) >= 10:
                    cross_sim = compute_shingle_sim(eval_phonetic, train_phonetic, k=3)
                    if cross_sim >= self.cross_script_threshold:
                        findings.append({
                            "eval_id": eval_id,
                            "train_ids": [train_id],
                            "method": "cross_script_transliteration",
                            "similarity": round(cross_sim, 3),
                            "evidence": f"eval_phon: '{eval_phonetic[:35]}' vs train_phon: '{train_phonetic[:35]}'"
                        })
                        already_matched_tids.add(train_id)

        return findings
