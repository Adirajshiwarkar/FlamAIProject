"""
Pipeline Execution Engine.
Orchestrates streaming ingestion, normalization, language ID, quality filtering,
deduplication, PII redaction, evaluation set curation, and cross-split contamination detection.
Writes atomic final outputs:
- <out_dir>/clean/data.parquet and <out_dir>/clean/data.jsonl.gz
- <out_dir>/ledger.jsonl
- <out_dir>/eval.jsonl
- <out_dir>/contamination.jsonl
- <out_dir>/manifest.json
"""

import os
import sys
import json
import gzip
import shutil
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from src.reader import stream_raw_documents
from src.normalizer import normalize_text
from src.lang_id import classify_language
from src.quality import check_quality
from src.pii import redact_pii
from src.dedup import Deduplicator
from src.contamination import ContaminationDetector

def atomic_write_jsonl(filepath: str, records: List[Dict[str, Any]]):
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(tmp_path, filepath)

def run_pipeline(
    raw_dir: str,
    out_dir: str,
    max_records_per_source: Optional[int] = 30000,
    target_eval_size: int = 200,
    near_dup_threshold: float = 0.80
):
    start_time_utc = datetime.now(timezone.utc).isoformat()
    print(f"Starting Indic corpus curation pipeline...")
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Max records per source: {max_records_per_source}")

    # Prepare directories
    clean_dir = os.path.join(out_dir, "clean")
    os.makedirs(clean_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "_temp_work")
    os.makedirs(temp_dir, exist_ok=True)

    # Initialize components
    dedup = Deduplicator(near_dup_threshold=near_dup_threshold)
    contamination_detector = ContaminationDetector(near_match_threshold=0.85, cross_script_threshold=0.82)

    # Stage tracking counts
    funnel = {
        "raw_total": 0,
        "raw_malformed_or_empty": 0,
        "by_source": {},
        "after_normalization": 0,
        "by_language": {"hi": 0, "hi-Latn": 0, "en": 0, "other": 0},
        "dropped_language": 0,
        "dropped_quality": 0,
        "dropped_exact_dup": 0,
        "dropped_near_dup": 0,
        "pii_redactions": {"PHONE": 0, "EMAIL": 0, "AADHAAR": 0, "PAN": 0},
        "eval_candidates": 0,
        "eval_selected": 0,
        "train_candidate_docs": 0,
        "contamination_leaks_removed": 0,
        "final_clean_retained": 0,
        "final_by_language": {"hi": 0, "hi-Latn": 0}
    }

    # Open ledger temp stream
    temp_ledger_path = os.path.join(temp_dir, "ledger_in_progress.jsonl")
    ledger_file = open(temp_ledger_path, "w", encoding="utf-8")

    # In-memory candidate pools before eval separation & contamination check
    candidate_clean_docs: List[Dict[str, Any]] = []

    print("\n--- STAGE 1: Streaming Ingestion, Normalization, LangID, Quality, Dedup, PII ---")
    doc_count = 0

    # Stream through raw documents
    for raw_rec in stream_raw_documents(raw_dir, max_records_per_source=max_records_per_source):
        doc_count += 1
        funnel["raw_total"] += 1
        src = raw_rec["source"]
        funnel["by_source"][src] = funnel["by_source"].get(src, 0) + 1

        doc_id = raw_rec["id"]

        # Check for raw ingestion error (malformed JSON, empty, etc.)
        if raw_rec.get("error"):
            funnel["raw_malformed_or_empty"] += 1
            ledger_rec = {
                "id": doc_id,
                "status": "dropped",
                "stage": "ingestion",
                "reason": raw_rec["error"],
                "dup_of": None
            }
            ledger_file.write(json.dumps(ledger_rec, ensure_ascii=False) + "\n")
            continue

        raw_text = raw_rec["raw_text"]

        # Stage 1: Unicode & Text Normalization
        norm_text = normalize_text(raw_text)
        if not norm_text:
            ledger_rec = {
                "id": doc_id,
                "status": "dropped",
                "stage": "normalization",
                "reason": "empty_after_normalization",
                "dup_of": None
            }
            ledger_file.write(json.dumps(ledger_rec, ensure_ascii=False) + "\n")
            continue

        funnel["after_normalization"] += 1

        # Stage 2: Language Identification
        lang, lang_meta = classify_language(norm_text)
        funnel["by_language"][lang] = funnel["by_language"].get(lang, 0) + 1

        # LLM corpus target: hi (Devanagari Hindi) and hi-Latn (Hinglish)
        if lang not in ("hi", "hi-Latn"):
            funnel["dropped_language"] += 1
            ledger_rec = {
                "id": doc_id,
                "status": "dropped",
                "stage": "language_id",
                "reason": f"unsupported_language_{lang}",
                "dup_of": None
            }
            ledger_file.write(json.dumps(ledger_rec, ensure_ascii=False) + "\n")
            continue

        # Stage 3: Quality Filtering
        q_pass, q_reason, q_meta = check_quality(norm_text, lang=lang)
        if not q_pass:
            funnel["dropped_quality"] += 1
            ledger_rec = {
                "id": doc_id,
                "status": "dropped",
                "stage": "quality_filter",
                "reason": q_reason,
                "dup_of": None
            }
            ledger_file.write(json.dumps(ledger_rec, ensure_ascii=False) + "\n")
            continue

        # Stage 4: Deduplication (Exact + Near-duplicate)
        is_dup, dup_of, dup_type, sim = dedup.check_and_add(doc_id, norm_text)
        if is_dup:
            if dup_type == "exact":
                funnel["dropped_exact_dup"] += 1
            else:
                funnel["dropped_near_dup"] += 1
            ledger_rec = {
                "id": doc_id,
                "status": "dropped",
                "stage": f"dedup_{dup_type}",
                "reason": f"{dup_type}_duplicate_of_{dup_of}_sim_{sim:.2f}",
                "dup_of": dup_of
            }
            ledger_file.write(json.dumps(ledger_rec, ensure_ascii=False) + "\n")
            continue

        # Stage 5: PII Redaction
        redacted_text, pii_counts = redact_pii(norm_text)
        for k, v in pii_counts.items():
            funnel["pii_redactions"][k] += v

        # Document passed all cleaning criteria
        clean_doc = {
            "id": doc_id,
            "raw_text": raw_text,
            "norm_text": norm_text,
            "text": redacted_text,
            "lang": lang,
            "source": src
        }
        candidate_clean_docs.append(clean_doc)

        if doc_count % 5000 == 0:
            print(f"  Processed {doc_count} documents... (clean candidates so far: {len(candidate_clean_docs)})")

    ledger_file.close()
    print(f"Initial streaming pass finished. Total processed: {doc_count}. Clean candidates: {len(candidate_clean_docs)}")

    print("\n--- STAGE 2: Evaluation Set Selection & Contamination Analysis ---")
    # Hold out ~200 items for the eval set deterministically across languages
    eval_candidates = candidate_clean_docs
    eval_set: List[Dict[str, Any]] = []
    train_pool: List[Dict[str, Any]] = []

    # Stratified selection: ~150 Hindi (hi) and ~50 Hinglish (hi-Latn)
    hi_docs = [d for d in eval_candidates if d["lang"] == "hi"]
    hinglish_docs = [d for d in eval_candidates if d["lang"] == "hi-Latn"]

    target_hi_eval = 150
    target_hinglish_eval = min(50, len(hinglish_docs))

    # Deterministic step slicing from the held-out tail
    eval_hi = hi_docs[-target_hi_eval:] if len(hi_docs) >= target_hi_eval else hi_docs[:target_hi_eval]
    eval_hinglish = hinglish_docs[-target_hinglish_eval:] if len(hinglish_docs) >= target_hinglish_eval else hinglish_docs[:target_hinglish_eval]

    eval_ids = set(d["id"] for d in eval_hi + eval_hinglish)
    for d in eval_hi + eval_hinglish:
        eval_set.append({
            "id": d["id"],
            "text": d["text"],
            "lang": d["lang"],
            "source": d["source"],
            "purpose": "indic_general_eval_benchmark"
        })

    for d in candidate_clean_docs:
        if d["id"] not in eval_ids:
            train_pool.append(d)

    funnel["eval_selected"] = len(eval_set)
    funnel["train_candidate_docs"] = len(train_pool)
    print(f"Selected {len(eval_set)} evaluation set items ({len(eval_hi)} hi, {len(eval_hinglish)} hi-Latn).")
    print(f"Training candidates remaining: {len(train_pool)}")

    # Index training pool into Contamination Detector
    print("Indexing training documents into Contamination Detector...")
    for td in train_pool:
        contamination_detector.index_training_doc(
            doc_id=td["id"],
            raw_text=td["raw_text"],
            norm_text=td["norm_text"],
            lang=td["lang"]
        )

    # Run contamination analysis for each eval item
    print("Running contamination checks (exact, normalized, near-duplicate, cross-script)...")
    contamination_findings = []
    contaminated_train_ids: Set[str] = set()

    for item in eval_set:
        matches = contamination_detector.check_eval_item(
            eval_id=item["id"],
            raw_text=item["text"],
            norm_text=item["text"],
            lang=item["lang"]
        )
        for m in matches:
            contamination_findings.append({
                "eval_id": m["eval_id"],
                "train_ids": m["train_ids"],
                "method": m["method"]
            })
            for tid in m["train_ids"]:
                contaminated_train_ids.add(tid)

    funnel["contamination_leaks_removed"] = len(contaminated_train_ids)
    print(f"Contamination analysis finished: {len(contamination_findings)} findings across {len(contaminated_train_ids)} leaking training documents.")

    print("\n--- STAGE 3: Finalizing Outputs & Reconciling Ledger ---")
    # Write eval.jsonl
    eval_path = os.path.join(out_dir, "eval.jsonl")
    atomic_write_jsonl(eval_path, eval_set)

    # Write contamination.jsonl
    contam_path = os.path.join(out_dir, "contamination.jsonl")
    atomic_write_jsonl(contam_path, contamination_findings)

    # Write final clean data
    final_clean_records = []
    for td in train_pool:
        if td["id"] not in contaminated_train_ids:
            clean_rec = {
                "id": td["id"],
                "text": td["text"],
                "lang": td["lang"],
                "source": td["source"]
            }
            final_clean_records.append(clean_rec)
            funnel["final_by_language"][td["lang"]] += 1

    funnel["final_clean_retained"] = len(final_clean_records)

    # 1. Write clean/data.jsonl.gz
    clean_gz_path = os.path.join(clean_dir, "data.jsonl.gz")
    tmp_clean_gz = clean_gz_path + ".tmp"
    with gzip.open(tmp_clean_gz, "wt", encoding="utf-8") as gf:
        for rec in final_clean_records:
            gf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if os.path.exists(clean_gz_path):
        os.remove(clean_gz_path)
    os.rename(tmp_clean_gz, clean_gz_path)

    # 2. Write clean/data.parquet
    clean_parquet_path = os.path.join(clean_dir, "data.parquet")
    arrow_table = pa.Table.from_pydict({
        "id": [r["id"] for r in final_clean_records],
        "text": [r["text"] for r in final_clean_records],
        "lang": [r["lang"] for r in final_clean_records],
        "source": [r["source"] for r in final_clean_records]
    })
    tmp_clean_parquet = clean_parquet_path + ".tmp"
    pq.write_table(arrow_table, tmp_clean_parquet, compression="snappy")
    if os.path.exists(clean_parquet_path):
        os.remove(clean_parquet_path)
    os.rename(tmp_clean_parquet, clean_parquet_path)

    # 3. Reconcile final ledger
    final_ledger_path = os.path.join(out_dir, "ledger.jsonl")
    with open(final_ledger_path, "w", encoding="utf-8") as out_l:
        # Copy dropped records from stage 1
        with open(temp_ledger_path, "r", encoding="utf-8") as temp_l:
            for line in temp_l:
                out_l.write(line)
        
        # Add eval set records
        for d in eval_set:
            out_l.write(json.dumps({
                "id": d["id"],
                "status": "dropped",
                "stage": "evaluation_holdout",
                "reason": "held_out_for_eval_benchmark",
                "dup_of": None
            }, ensure_ascii=False) + "\n")

        # Add contaminated training records
        for d in train_pool:
            if d["id"] in contaminated_train_ids:
                out_l.write(json.dumps({
                    "id": d["id"],
                    "status": "dropped",
                    "stage": "contamination_purge",
                    "reason": "leaked_into_eval_set",
                    "dup_of": None
                }, ensure_ascii=False) + "\n")
            else:
                out_l.write(json.dumps({
                    "id": d["id"],
                    "status": "kept",
                    "stage": "retained",
                    "reason": "passed_all_filters",
                    "dup_of": None
                }, ensure_ascii=False) + "\n")

    # Clean up temp working files
    shutil.rmtree(temp_dir, ignore_errors=True)

    # 4. Generate 1,000-row deterministic sample
    sample_path = os.path.join(out_dir, "sample_clean_1000.jsonl")
    step = max(1, len(final_clean_records) // 1000)
    sample_records = [final_clean_records[i] for i in range(0, len(final_clean_records), step)][:1000]
    with open(sample_path, "w", encoding="utf-8") as sf:
        for r in sample_records:
            sf.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Generated 1,000-row representative sample at {sample_path}")

    # 5. Write manifest.json
    end_time_utc = datetime.now(timezone.utc).isoformat()
    raw_provenance_file = os.path.join(raw_dir, "provenance.json")
    provenance_data = []
    if os.path.exists(raw_provenance_file):
        with open(raw_provenance_file, "r", encoding="utf-8") as pf:
            provenance_data = json.load(pf)

    manifest = {
        "metadata": {
            "pipeline_name": "Indic Training Corpus Curation Pipeline",
            "version": "1.0.0",
            "execution_start_utc": start_time_utc,
            "execution_end_utc": end_time_utc,
            "python_version": sys.version
        },
        "sources": provenance_data,
        "configuration": {
            "unicode_normalization": "NFC",
            "invisible_character_handling": "ZWNJ/ZWJ selective preservation after Virama; BOM/LRM/RLM stripped",
            "language_identification": "Character script ratio + Hinglish/English marker lexicon",
            "near_duplicate_method": "MinHash 64 permutations + LSH 16 bands x 4 rows (character 3-grams)",
            "near_duplicate_threshold": near_dup_threshold,
            "pii_redaction": "Indian Phone, Email, Aadhaar (Verhoeff verified), PAN",
            "eval_target_size": target_eval_size,
            "contamination_methods": ["exact_raw_match", "normalized_match", "near_duplicate", "cross_script_transliteration"]
        },
        "funnel_counts": funnel
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    print(f"\nManifest successfully written to {manifest_path}")
    print(f"Final clean retained documents: {funnel['final_clean_retained']} ({funnel['final_by_language']})")
    print("Pipeline execution complete.")
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline curation pipeline")
    parser.add_argument("--raw-dir", default="raw", help="Directory with raw inputs")
    parser.add_argument("--out-dir", default="out", help="Directory for curated outputs")
    parser.add_argument("--eval-size", type=int, default=200, help="Target eval set size")
    parser.add_argument("--near-dup-threshold", type=float, default=0.80, help="Near duplicate similarity threshold")
    parser.add_argument("--max-records-per-source", type=int, default=30000, help="Max records to read per source")
    args = parser.parse_args()

    max_recs = args.max_records_per_source if args.max_records_per_source > 0 else None
    run_pipeline(
        args.raw_dir,
        args.out_dir,
        max_records_per_source=max_recs,
        target_eval_size=args.eval_size,
        near_dup_threshold=args.near_dup_threshold
    )
