# DECISIONS.md - Flam Indic Corpus Curation Report

## 1. Sources & Provenance
We collected data from two legally permissible public repositories:
- **AI4Bharat Samanantar (Hindi)**:
  - *URL*: `https://huggingface.co/datasets/ai4bharat/samanantar`
  - *License*: CC-BY-NC-4.0 (academic & non-commercial research; attribution required)
  - *Document Count Collected*: 15,000 raw Devanagari Hindi records
  - *File*: `raw/samanantar_hi.parquet` (2,809,514 bytes; SHA256: `ad22dd1ff04e52bacf1e7c6a1a0b36c0b6ee8b061c54663f9899f54cd854a1d7`)
- **findnitai/english-to-hinglish (Hinglish Corpus)**:
  - *URL*: `https://huggingface.co/datasets/findnitai/english-to-hinglish`
  - *License*: Apache-2.0 (commercial and non-commercial permitted with attribution)
  - *Document Count Collected*: 30,000 records processed (from 189,102 available in source)
  - *File*: `raw/hinglish_corpus.json` (27,122,089 bytes; SHA256: `34d5abc0b1f1c301fac1d8c002b1b186183ab86eb75b9e8f7593d78aef202b42`)

## 2. Input-to-Output Funnel
- **Raw documents ingested**: 45,000
  - Ingestion errors/empty: 0
- **Unicode & Text Normalization (NFC + ZWNJ selective retention)**: 45,000 preserved
- **Language Identification**:
  - `hi` (Hindi in Devanagari): 15,004
  - `hi-Latn` (Hinglish): 15,592
  - `en` (English): 1,152 (dropped)
  - `other` (Uncertain/ambiguous/other Indic): 13,252 (dropped)
  - *Total Dropped by LangID*: 14,404
- **Quality Filtering**:
  - *Dropped*: 12 records (e.g. truncated punctuation or single uninformative tokens like `"aap?"`, `"nahi"`, `"< 5 chars"`)
- **Deduplication**:
  - *Exact Duplicates Dropped*: 886
  - *Near-Duplicates Dropped (MinHash/LSH threshold 0.80)*: 783
- **PII Redactions**:
  - Email addresses redacted: 1 (`<EMAIL>`)
  - Phone / Aadhaar / PAN: Checked with boundary-aware regex and Verhoeff checksum; 0 false positives on legitimate dates/quantities.
- **Evaluation Holdout Set**:
  - Selected 200 items (150 `hi`, 50 `hi-Latn`)
- **Contamination Purge**:
  - 2 training documents identified leaking eval items via cross-script / near-duplicate matching and purged.
- **Final Clean Retained Corpus**:
  - **28,713 documents** (14,830 Hindi `hi`, 13,883 Hinglish `hi-Latn`)
- **Ledger Reconciliation**:
  - Kept: 28,713 | Dropped: 16,287 | Sum: 45,000 (100% exact reconciliation).

## 3. Near-Duplicate Methodology, Threshold Rationale & Borderline Pairs
- **Method**: MinHash (64 permutations, Mersenne prime hash functions) with Locality Sensitive Hashing (LSH, 16 bands x 4 rows) using character 3-grams.
- **Threshold Rationale**:
  - Indic scripts have complex inflectional morphology and spelling variations in Hinglish. Character 3-grams preserve sub-word roots.
  - At threshold `s >= 0.80`, sentences that differ only by a single minor particle, punctuation, or filler word are clustered together, retaining one clean representative.
- **Three Borderline Pairs Examined**:
  1. **Pair 1 (`sim = 0.84`)**:
     - *Doc 1*: "मामले की सूचना पुलिस को दी गई."
     - *Doc 2*: "इस मामले की सूचना पुलिस को दी गई।"
     - *Call*: **DUPLICATE (Dropped Doc 1)**. Differed solely by initial demonstrative pronoun "इस". Information content is identical.
  2. **Pair 2 (`sim = 0.84`)**:
     - *Doc 1*: "खबर लिखे जाने तक किसी की गिरफ्तारी नहीं हो सकी थी।"
     - *Doc 2*: "खबर लिखे जाने तक किसी भी आरोपी की गिरफ्तारी नहीं हो सकी थी।"
     - *Call*: **DUPLICATE (Dropped Doc 1)**. Adding "भी आरोपी" is an editorial repetition of the exact same news wire event.
  3. **Pair 3 (`sim = 0.87`)**:
     - *Doc 1*: "समदा में भारत के बिहार राज्य के अन्तर्गत पुर्णिया मण्डल के अररिया जिले का एक गाँव है।"
     - *Doc 2*: "मज्गावान में भारत के बिहार राज्य के अन्तर्गत पुर्णिया मण्डल के अररिया जिले का एक गाँव है।"
     - *Call*: **BORDERLINE CASE / DUPLICATE BY TEMPLATE**. Both represent Wikipedia geographic boilerplate where only the village name differs. Dropping one prevents LLM overfitting to boilerplate census templates.

## 4. Quality Filter Evaluation
- Inspected the 12 documents removed by quality filters:
  - `hinglish_corpus_000268`: `"aap?"` (dropped: `< 5 chars`)
  - `hinglish_corpus_000300`: `"\u0935\u0947*"` (dropped: `< 5 chars`)
  - `hinglish_corpus_000317`: `"nahi"` (dropped: `< 5 chars`)
- **Assessment**: None of the 12 removed documents contained meaningful training value for a general foundation model; all were single-word dialogue fragments without grammatical context. Proverbs like *"अधजल गगरी छलकत जाए"* and numerical texts were preserved as verified by unit tests.

## 5. Contamination Detection Findings
- **Methodology**: Multi-stage detection:
  1. Exact string match on raw text.
  2. Normalized match (canonical NFC, stripped invisible characters, collapsed whitespace).
  3. Near-duplicate match (character 3-gram Jaccard >= 0.85).
  4. Cross-script transliteration match: phonetically maps Devanagari Hindi characters to Roman Latin representations and compares against Hinglish training candidates (threshold >= 0.82).
- **Actual Affected Items**: 2 items leaked into the training set:
  - *Eval Item*: `hinglish_corpus_029858` (*"kya mujhe topi phenna chahiye ?"*) matched Training Doc `hinglish_corpus_010595` (*"kya mujhe aj topi phenna chahiye"*).
  - *Eval Item*: `hinglish_corpus_029908` (*"Aaj raat ghar pahuchne mei kitni der lagegi"*) matched Training Doc `hinglish_corpus_008813` (*"ghar pahuchne mei kitni der lagegi"*).
- **Action Taken**: Both leaking training documents were purged from `clean/` and marked in `ledger.jsonl` under stage `contamination_purge`.
- **One Rejected Candidate Match**:
  - Candidate: Eval item asking *"kya aap film dekhna chahte hain"* vs Train item *"kya aap khana khana chahte hain"*.
  - Jaccard similarity was 0.58 (below the 0.82 threshold). Rejected as standard grammatical variation rather than memorization leakage.

## 6. Reproducibility Instructions
To reproduce this exact corpus 6 months from now:
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run data collection:
   ```bash
   ./download.sh raw
   # or on Windows: .\download.bat raw
   ```
   *Note: SHA-256 hashes recorded in `raw/provenance.json` must match exactly.*
3. Execute offline curation:
   ```bash
   ./run.sh raw out
   # or on Windows: .\run.bat raw out
   ```
4. Run automated test suite:
   ```bash
   python -m unittest discover tests
   ```

## 7. Scaling to 2 TB on Object Storage
At 2 TB scale on object storage (e.g. S3 / GCS), an in-memory cross-document deduplication and contamination index will fail first (OOM crash and quadratic candidate comparison bottlenecks). To scale:
- Replace single-node MinHash dictionary with distributed Spark / Ray using LSH bucketing partitioned by hash prefix.
- Store documents in Parquet partitioned by language and length buckets, streaming chunks through ephemeral worker nodes.
- Replace in-memory contamination checking with an external distributed vector index (Faiss or ScaNN) or pre-computed shingle inversion tables.

## 8. Wrap-up
- **What was not completed (Priority order)**:
  1. Neural transliteration back-and-forth model (e.g., Aksharantar) to supplement rule-based cross-script phonetic matching.
  2. Sub-dialect detection (Bhojpuri/Maithili distinction from standard Devanagari Hindi).
- **Actual Time Spent**: Approximately 3.5 hours of engineering effort.

