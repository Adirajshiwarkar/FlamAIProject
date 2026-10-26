"""
Data download module.
Fetches immutable raw datasets from verified public sources:
1. AI4Bharat Samanantar (Hindi Devanagari) - CC-BY-NC-4.0
2. findnitai/english-to-hinglish (Hinglish Latin script) - Apache-2.0
Generates source provenance metadata with SHA-256 hashes and document counts.
"""

import os
import sys
import json
import hashlib
import argparse
import urllib.request
from datetime import datetime, timezone

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def download_file(url: str, dest_path: str, headers: dict = None) -> int:
    print(f"Downloading: {url} -> {dest_path}")
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Flam-Corpus-Builder/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        downloaded = 0
        while chunk := resp.read(65536):
            out_f.write(chunk)
            downloaded += len(chunk)
    print(f"Finished downloading {dest_path}: {downloaded} bytes")
    return downloaded

def download_samanantar_hindi(dest_path: str, target_count: int = 15000) -> int:
    """
    Downloads raw parquet from Samanantar Hindi shard.
    Reads row groups incrementally using pyarrow streaming without loading the entire 240MB file.
    """
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    url = "https://huggingface.co/datasets/ai4bharat/samanantar/resolve/main/hi/train-00000-of-00008.parquet"
    print(f"Fetching {target_count} records from Samanantar Hindi: {url}")
    fs = fsspec.filesystem("http")
    with fs.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        collected_tables = []
        rows_so_far = 0
        rg_idx = 0
        while rows_so_far < target_count and rg_idx < pf.num_row_groups:
            tbl = pf.read_row_group(rg_idx)
            collected_tables.append(tbl)
            rows_so_far += tbl.num_rows
            rg_idx += 1
            print(f"  Read row group {rg_idx}/{pf.num_row_groups}: {rows_so_far} rows accumulated")

        combined = pa.concat_tables(collected_tables)
        if len(combined) > target_count:
            combined = combined.slice(0, target_count)
        
        pq.write_table(combined, dest_path, compression="snappy")
        print(f"Saved raw Samanantar Hindi to {dest_path} ({len(combined)} rows)")
        return len(combined)

def download_all(raw_dir: str):
    os.makedirs(raw_dir, exist_ok=True)
    provenance = []
    now_utc = datetime.now(timezone.utc).isoformat()

    # 1. AI4Bharat Samanantar Hindi
    samanantar_file = os.path.join(raw_dir, "samanantar_hi.parquet")
    if not os.path.exists(samanantar_file):
        hi_count = download_samanantar_hindi(samanantar_file, target_count=15000)
    else:
        import pyarrow.parquet as pq
        hi_count = pq.read_table(samanantar_file).num_rows
        print(f"Using existing {samanantar_file} ({hi_count} rows)")

    samanantar_meta = {
        "source_name": "AI4Bharat Samanantar (Hindi)",
        "canonical_url": "https://huggingface.co/datasets/ai4bharat/samanantar",
        "license": "CC-BY-NC-4.0",
        "permitted_use": "Permitted for academic and non-commercial research; attribution required",
        "download_date_utc": now_utc,
        "file_name": "samanantar_hi.parquet",
        "document_count": hi_count,
        "file_size_bytes": os.path.getsize(samanantar_file),
        "sha256": compute_sha256(samanantar_file)
    }
    provenance.append(samanantar_meta)

    # 2. findnitai/english-to-hinglish (Hinglish)
    hinglish_file = os.path.join(raw_dir, "hinglish_corpus.json")
    if not os.path.exists(hinglish_file):
        hinglish_url = "https://huggingface.co/datasets/findnitai/english-to-hinglish/resolve/main/hinglish_upload_v1.json"
        download_file(hinglish_url, hinglish_file)
    
    # Count rows in Hinglish JSONL
    hinglish_count = 0
    with open(hinglish_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                hinglish_count += 1
    
    hinglish_meta = {
        "source_name": "findnitai/english-to-hinglish (Hinglish Corpus)",
        "canonical_url": "https://huggingface.co/datasets/findnitai/english-to-hinglish",
        "license": "Apache-2.0",
        "permitted_use": "Permitted under Apache 2.0 license; commercial and non-commercial use with attribution",
        "download_date_utc": now_utc,
        "file_name": "hinglish_corpus.json",
        "document_count": hinglish_count,
        "file_size_bytes": os.path.getsize(hinglish_file),
        "sha256": compute_sha256(hinglish_file)
    }
    provenance.append(hinglish_meta)

    provenance_path = os.path.join(raw_dir, "provenance.json")
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    print(f"Provenance saved to {provenance_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download raw Indic datasets")
    parser.add_argument("--output-dir", default="raw", help="Directory to save raw datasets")
    args = parser.parse_args()
    download_all(args.output_dir)

