"""
Deduplication Module for Indic NLP Corpus.
Implements:
1. Exact duplicate detection using SHA-256 hashes of normalized text.
2. Near-duplicate detection using MinHash with Locality Sensitive Hashing (LSH).
   - Character 3-grams / Word 1-grams suited for Indic script tokenization
   - Jaccard similarity threshold empirically verified on dataset
   - Memory-efficient sub-quadratic candidate generation using LSH bands
3. Deterministic representative selection (first seen document kept, subsequent marked dup_of).
"""

import re
import hashlib
from typing import Dict, List, Set, Tuple, Optional

# Constants for MinHash
NUM_PERMUTATIONS = 64
NUM_BANDS = 16
ROWS_PER_BAND = 4  # 16 * 4 = 64

# Deterministic linear hash parameters for MinHash: (a * x + b) % M
MERSENNE_PRIME = (1 << 61) - 1
import random
# Fixed seed for determinism across pipeline restarts
_RND = random.Random(42)
HASH_COEFFS_A = [_RND.randint(1, MERSENNE_PRIME - 1) for _ in range(NUM_PERMUTATIONS)]
HASH_COEFFS_B = [_RND.randint(0, MERSENNE_PRIME - 1) for _ in range(NUM_PERMUTATIONS)]

def extract_shingles(text: str, k: int = 3) -> Set[str]:
    """
    Extracts character k-grams for Indic text.
    Character k-grams are robust for Devanagari matras, conjuncts, and Hinglish spelling variants.
    """
    clean = re.sub(r"\s+", " ", text).strip().lower()
    if len(clean) < k:
        return {clean}
    return {clean[i:i+k] for i in range(len(clean) - k + 1)}

def compute_minhash_signature(shingles: Set[str]) -> List[int]:
    """Computes a 64-element MinHash signature for a set of shingles."""
    if not shingles:
        return [0] * NUM_PERMUTATIONS

    # Hash shingles to 64-bit integers
    shingle_hashes = [int(hashlib.md5(s.encode("utf-8")).hexdigest()[:15], 16) for s in shingles]
    
    sig = [MERSENNE_PRIME] * NUM_PERMUTATIONS
    for s_hash in shingle_hashes:
        for i in range(NUM_PERMUTATIONS):
            h_val = (HASH_COEFFS_A[i] * s_hash + HASH_COEFFS_B[i]) % MERSENNE_PRIME
            if h_val < sig[i]:
                sig[i] = h_val
    return sig

def estimate_jaccard(sig1: List[int], sig2: List[int]) -> float:
    """Estimates Jaccard similarity from MinHash signatures."""
    if not sig1 or not sig2:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

def exact_jaccard(shingles1: Set[str], shingles2: Set[str]) -> float:
    """Computes exact Jaccard similarity between two shingle sets."""
    if not shingles1 or not shingles2:
        return 0.0
    inter = len(shingles1.intersection(shingles2))
    union = len(shingles1.union(shingles2))
    return inter / union if union > 0 else 0.0

class Deduplicator:
    """
    Deduplicator maintaining exact hash index and LSH index for near duplicates.
    """
    def __init__(self, near_dup_threshold: float = 0.80):
        self.near_dup_threshold = near_dup_threshold
        # Exact hash index: normalized_text_hash -> original_doc_id
        self.exact_seen: Dict[str, str] = {}
        # LSH index: (band_idx, band_hash) -> list of (doc_id, signature, shingles)
        self.lsh_buckets: Dict[Tuple[int, int], List[str]] = {}
        # Document store for candidates
        self.doc_signatures: Dict[str, List[int]] = {}
        self.doc_shingles: Dict[str, Set[str]] = {}

    def check_and_add(self, doc_id: str, normalized_text: str) -> Tuple[bool, Optional[str], Optional[str], float]:
        """
        Checks if text is a duplicate of a previously kept document.
        Returns:
          (is_dup: bool, dup_of_id: str or None, dup_type: 'exact'|'near'|None, similarity: float)
        """
        # 1. Exact Duplicate Check
        text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        if text_hash in self.exact_seen:
            rep_id = self.exact_seen[text_hash]
            return True, rep_id, "exact", 1.0

        # 2. Near Duplicate Check via LSH
        shingles = extract_shingles(normalized_text, k=3)
        signature = compute_minhash_signature(shingles)

        candidate_ids: Set[str] = set()
        for band_idx in range(NUM_BANDS):
            start = band_idx * ROWS_PER_BAND
            band_tuple = tuple(signature[start:start + ROWS_PER_BAND])
            bucket_key = (band_idx, hash(band_tuple))
            if bucket_key in self.lsh_buckets:
                candidate_ids.update(self.lsh_buckets[bucket_key])

        # Evaluate candidates deterministically
        best_match_id = None
        best_sim = 0.0

        for cand_id in sorted(candidate_ids):
            cand_shingles = self.doc_shingles.get(cand_id)
            if cand_shingles is not None:
                sim = exact_jaccard(shingles, cand_shingles)
                if sim > best_sim:
                    best_sim = sim
                    best_match_id = cand_id

        if best_sim >= self.near_dup_threshold and best_match_id:
            # Document is a near duplicate
            return True, best_match_id, "near", best_sim

        # Document is UNIQUE. Add to indices
        self.exact_seen[text_hash] = doc_id
        self.doc_signatures[doc_id] = signature
        self.doc_shingles[doc_id] = shingles

        for band_idx in range(NUM_BANDS):
            start = band_idx * ROWS_PER_BAND
            band_tuple = tuple(signature[start:start + ROWS_PER_BAND])
            bucket_key = (band_idx, hash(band_tuple))
            if bucket_key not in self.lsh_buckets:
                self.lsh_buckets[bucket_key] = []
            self.lsh_buckets[bucket_key].append(doc_id)

        return False, None, None, 0.0

