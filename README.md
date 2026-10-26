# Clean Indic Training Corpus Pipeline

A production-grade, reproducible Python data curation pipeline that collects, cleans, normalizes, deduplicates, redacts PII, and constructs an evaluation benchmark for Hindi (Devanagari) and Hinglish (Latin-script Hindi) text for LLM pre-training.

## Architecture

```
Flam/
├── download.sh            # Network script: downloads raw data & computes provenance
├── download.bat           # Windows helper for download
├── run.sh                 # Offline script: executes streaming curation pipeline
├── run.bat                # Windows helper for run
├── requirements.txt       # Dependencies (pyarrow, fsspec, requests, etc.)
├── README.md              # System design and usage guide
├── DECISIONS.md           # Engineering decisions, thresholds, and empirical results
├── .gitignore             # Ignores raw/ and full out/ except the 1,000-row sample
├── src/
│   ├── __init__.py
│   ├── download.py        # Ingestion & SHA256 provenance calculation
│   ├── reader.py          # Streaming reader for Parquet and JSONL
│   ├── normalizer.py      # Unicode NFC, ZWNJ/ZWJ preservation, whitespace
│   ├── lang_id.py         # Language classifier (hi, hi-Latn, en, other)
│   ├── quality.py         # Boilerplate, mojibake, repetition filter
│   ├── pii.py             # Phone, Email, PAN, and Verhoeff-verified Aadhaar
│   ├── dedup.py           # Exact SHA256 + Near-duplicate MinHash/LSH
│   ├── contamination.py   # Multi-stage & cross-script contamination detector
│   └── pipeline.py        # End-to-end coordinator and atomic output writer
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py   # 22 automated unit and integration tests
├── raw/                   # Immutable raw datasets (downloaded by download.sh)
└── out/
    ├── clean/
    │   ├── data.parquet   # High performance format
    │   └── data.jsonl.gz  # Compressed streaming format
    ├── ledger.jsonl       # Audit record for every input document
    ├── eval.jsonl         # 200 held-out evaluation items
    ├── contamination.jsonl# Identified leakage records
    ├── manifest.json      # Complete run metadata and counts
    └── sample_clean_1000.jsonl # 1,000-row deterministic sample
```

## Setup & Quick Start

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Download Data (Requires Network)
```bash
./download.sh raw
# or on Windows:
.\download.bat raw
```

### 3. Run Pipeline (Fully Offline)
```bash
./run.sh raw out
# or on Windows:
.\run.bat raw out
```

### 4. Run Automated Tests
```bash
python -m unittest discover tests
```

## Key Engineering Features

- **Streaming & Memory Efficiency**: Processes records in streaming batches. Raw inputs remain untouched and immutable.
- **Indic-Specific Unicode Handling**: Implements Unicode NFC and selectively retains Zero Width Non-Joiners (ZWNJ, U+200C) when preceded by a Virama (`\u094D`), preserving ligatures while stripping invisible junk.
- **Language Identification**: Custom transparent classifier distinguishing Devanagari Hindi (`hi`), code-mixed Latin Hinglish (`hi-Latn`), English (`en`), and other scripts.
- **Verhoeff Checksum Aadhaar Redaction**: Prevents false positive redaction of legitimate 12-digit transaction numbers, dates, or prices by validating the mathematical Verhoeff check digit.
- **MinHash + LSH Deduplication**: Sub-quadratic candidate generation using character 3-grams to robustly handle inflectional suffixes and spelling variants.
- **Cross-Script Contamination Detection**: Phonetically maps Devanagari Hindi into Latin characters to catch cases where a Devanagari eval query leaked as a transliterated Latin string in the training split.
- **Ledger Reconciliation**: Every input document is tracked with an immutable status (`kept` or `dropped`), stage, and `dup_of` reference. Final ledger counts reconcile with raw counts.
