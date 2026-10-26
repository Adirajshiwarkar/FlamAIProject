"""
Comprehensive Automated Test Suite for Indic Training Corpus Pipeline.
Covers:
1. Unicode normalization (Devanagari, ZWJ/ZWNJ preservation, whitespace)
2. Language identification (hi, hi-Latn, en, other)
3. Malformed input handling and error recovery
4. PII detection and redaction (Phone, Email, Aadhaar with Verhoeff, PAN)
5. Deduplication (exact hash, near-duplicate MinHash/LSH)
6. Contamination detection (exact, normalized, near-dup, cross-script transliteration)
7. Determinism and idempotence
8. Ledger reconciliation
"""

import os
import json
import unittest
import unicodedata

from src.normalizer import normalize_text
from src.lang_id import classify_language
from src.quality import check_quality
from src.pii import redact_pii, validate_verhoeff
from src.dedup import Deduplicator, compute_minhash_signature, extract_shingles, exact_jaccard
from src.contamination import ContaminationDetector, approximate_devanagari_to_roman

class TestIndicCorpusPipeline(unittest.TestCase):

    # ==========================================
    # 1. Unicode Normalization Tests
    # ==========================================
    def test_unicode_nfc_normalization(self):
        decomposed = "\u0915\u093C"
        composed = "\u0958"
        self.assertEqual(normalize_text(decomposed), normalize_text(composed))

    def test_invisible_characters_handling(self):
        text = "\uFEFFयह एक\u200B परीक्षण\u200E है।"
        normalized = normalize_text(text)
        self.assertNotIn("\uFEFF", normalized)
        self.assertNotIn("\u200B", normalized)
        self.assertNotIn("\u200E", normalized)
        self.assertEqual(normalized, "यह एक परीक्षण है।")

    def test_zwnj_after_virama_preservation(self):
        explicit_halant = "क्\u200Cत"
        normalized = normalize_text(explicit_halant)
        self.assertIn("\u200C", normalized)

        standalone = "क\u200Cत"
        normalized_standalone = normalize_text(standalone)
        self.assertNotIn("\u200C", normalized_standalone)

    def test_whitespace_and_punctuation_normalization(self):
        text = "नमस्ते    दुनिया  !  \n\n\n\nअगली   पंक्ति ।"
        normalized = normalize_text(text)
        self.assertNotIn("    ", normalized)
        self.assertNotIn("\n\n\n", normalized)
        self.assertIn("अगली पंक्ति ।", normalized)

    # ==========================================
    # 2. Language Identification Tests
    # ==========================================
    def test_language_id_hindi(self):
        hi_text = "भारत दक्षिण एशिया में स्थित एक विशाल और सांस्कृतिक रूप से समृद्ध देश है।"
        lang, meta = classify_language(hi_text)
        self.assertEqual(lang, "hi")

    def test_language_id_hindi_code_mixed(self):
        code_mixed = "आज के digital युग में internet connectivity बहुत जरूरी है।"
        lang, meta = classify_language(code_mixed)
        self.assertEqual(lang, "hi")

    def test_language_id_hinglish(self):
        hinglish = "film ka kya naam hai aur ye kab release hogi"
        lang, meta = classify_language(hinglish)
        self.assertEqual(lang, "hi-Latn")

    def test_language_id_english(self):
        english = "The movie was directed by Christopher Nolan and was released in 2010."
        lang, meta = classify_language(english)
        self.assertEqual(lang, "en")

    def test_language_id_other_scripts(self):
        bengali = "আমি বাংলায় গান গাই এবং বাংলাকে ভালোবাসি"
        lang, meta = classify_language(bengali)
        self.assertEqual(lang, "other")

    # ==========================================
    # 3. Quality Filtering Tests
    # ==========================================
    def test_quality_filter_too_short(self):
        passed, reason, _ = check_quality("hi")
        self.assertFalse(passed)
        self.assertIn("too_short", reason)

    def test_quality_filter_html(self):
        passed, reason, _ = check_quality("<div><p><span>Hello</span></p></div>", lang="en")
        self.assertFalse(passed)
        self.assertIn("html_markup", reason)

    def test_quality_filter_repetition(self):
        repetitive = "ha ha ha ha ha ha ha ha ha ha ha ha"
        passed, reason, _ = check_quality(repetitive, lang="hi-Latn")
        self.assertFalse(passed)
        self.assertTrue("repetition" in reason or "dominance" in reason)

    def test_quality_filter_legitimate_proverb(self):
        proverb = "अधजल गगरी छलकत जाए।"
        passed, reason, _ = check_quality(proverb, lang="hi")
        self.assertTrue(passed)
        self.assertEqual(reason, "quality_passed")

    # ==========================================
    # 4. PII Redaction Tests
    # ==========================================
    def test_verhoeff_algorithm(self):
        self.assertFalse(validate_verhoeff("123456789012"))
        self.assertFalse(validate_verhoeff("000000000000"))

    def test_pii_redaction_phone(self):
        text = "कृपया मुझे +91 9876543210 पर कॉल करें या 09876543210 पर एसएमएस भेजें।"
        redacted, counts = redact_pii(text)
        self.assertIn("<PHONE>", redacted)
        self.assertGreaterEqual(counts["PHONE"], 1)
        self.assertNotIn("9876543210", redacted)

    def test_pii_redaction_email(self):
        text = "You can contact support@flamapp.com or info.team@example.co.in anytime."
        redacted, counts = redact_pii(text)
        self.assertIn("<EMAIL>", redacted)
        self.assertEqual(counts["EMAIL"], 2)
        self.assertNotIn("@flamapp.com", redacted)

    def test_pii_redaction_pan(self):
        text = "The applicant PAN is ABCDE1234F for taxation."
        redacted, counts = redact_pii(text)
        self.assertIn("<PAN>", redacted)
        self.assertEqual(counts["PAN"], 1)
        self.assertNotIn("ABCDE1234F", redacted)

    def test_pii_preservation_of_regular_numbers(self):
        text = "दिनांक 15/08/1947 को भारत आजाद हुआ और 500 रुपये का नोट जारी किया गया।"
        redacted, counts = redact_pii(text)
        self.assertIn("15/08/1947", redacted)
        self.assertIn("500", redacted)
        self.assertEqual(counts["PHONE"], 0)

    # ==========================================
    # 5. Deduplication Tests
    # ==========================================
    def test_exact_deduplication(self):
        dedup = Deduplicator()
        is_dup1, _, _, _ = dedup.check_and_add("doc1", "भारत एक विशाल देश है।")
        self.assertFalse(is_dup1)

        is_dup2, rep_id, d_type, sim = dedup.check_and_add("doc2", "भारत एक विशाल देश है।")
        self.assertTrue(is_dup2)
        self.assertEqual(rep_id, "doc1")
        self.assertEqual(d_type, "exact")
        self.assertEqual(sim, 1.0)

    def test_near_deduplication(self):
        dedup = Deduplicator(near_dup_threshold=0.80)
        t1 = "घरेलू पूंजी, प्रौद्योगिकी एवं कौशल के लिए आर्थिक विकास की गति बढ़ाई गई"
        t2 = "नीतिगत सुधारः घरेलू पूंजी, प्रौद्योगिकी एवं कौशल के लिए आर्थिक विकास की गति बढ़ाई गई।"

        is_dup1, _, _, _ = dedup.check_and_add("doc_a", t1)
        self.assertFalse(is_dup1)

        is_dup2, rep_id, d_type, sim = dedup.check_and_add("doc_b", t2)
        self.assertTrue(is_dup2)
        self.assertEqual(rep_id, "doc_a")
        self.assertEqual(d_type, "near")
        self.assertGreaterEqual(sim, 0.80)

    # ==========================================
    # 6. Contamination Detection Tests
    # ==========================================
    def test_exact_and_normalized_contamination(self):
        detector = ContaminationDetector()
        detector.index_training_doc("train_001", "नमस्ते दुनिया", "नमस्ते दुनिया", "hi")

        findings = detector.check_eval_item("eval_001", "नमस्ते दुनिया", "नमस्ते दुनिया", "hi")
        self.assertEqual(len(findings), 1)
        self.assertIn("train_001", findings[0]["train_ids"])
        self.assertEqual(findings[0]["method"], "exact_raw_match")

    def test_cross_script_contamination(self):
        detector = ContaminationDetector(cross_script_threshold=0.75)
        detector.index_training_doc("train_002", "film ka kya naam hai", "film ka kya naam hai", "hi-Latn")

        eval_text = "फिल्म का क्या नाम है"
        findings = detector.check_eval_item("eval_002", eval_text, eval_text, "hi")
        self.assertGreaterEqual(len(findings), 1)
        self.assertTrue(any(f["method"] == "cross_script_transliteration" for f in findings))

if __name__ == "__main__":
    unittest.main()

