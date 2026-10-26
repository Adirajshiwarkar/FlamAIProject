"""
Streaming reader for raw files.
Handles:
- Parquet files (e.g. Samanantar)
- JSON / JSONL files (e.g. Hinglish corpus)
- Malformed records, invalid encodings, missing fields, empty lines
- Deterministic ID generation based on source and record index / content hash
- Skipping stats tracking
"""

import os
import json
import hashlib
from typing import Generator, Dict, Any, Tuple

def generate_doc_id(source: str, index: int, raw_payload: str) -> str:
    """Deterministic document ID: e.g. samanantar_hi_00001 or hash."""
    prefix = "".join(c if c.isalnum() else "_" for c in source).strip("_")
    h = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{index:06d}_{h}"

def stream_raw_documents(raw_dir: str, max_records_per_source: int = None) -> Generator[Dict[str, Any], None, None]:
    """
    Yields dict:
      {
        "id": str,
        "source": str,
        "raw_text": str or None,
        "error": str or None,  # If error occurred reading record
        "orig_index": int
      }
    """
    # 1. Stream Samanantar Hindi Parquet
    samanantar_path = os.path.join(raw_dir, "samanantar_hi.parquet")
    if os.path.exists(samanantar_path):
        import pyarrow.parquet as pq
        source_name = "samanantar_hi"
        try:
            parquet_file = pq.ParquetFile(samanantar_path)
            doc_idx = 0
            for rg_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(rg_idx)
                # Schema: idx, src (en), tgt (hi)
                tgt_col = table.column("tgt").to_pylist()
                for row_val in tgt_col:
                    doc_idx += 1
                    if max_records_per_source and doc_idx > max_records_per_source:
                        break
                    if row_val is None or not isinstance(row_val, str) or not row_val.strip():
                        yield {
                            "id": f"{source_name}_{doc_idx:06d}_empty",
                            "source": source_name,
                            "raw_text": None,
                            "error": "empty_or_null_target",
                            "orig_index": doc_idx
                        }
                    else:
                        doc_id = generate_doc_id(source_name, doc_idx, row_val)
                        yield {
                            "id": doc_id,
                            "source": source_name,
                            "raw_text": row_val,
                            "error": None,
                            "orig_index": doc_idx
                        }
                if max_records_per_source and doc_idx >= max_records_per_source:
                    break
        except Exception as e:
            yield {
                "id": f"{source_name}_read_error",
                "source": source_name,
                "raw_text": None,
                "error": f"corrupted_parquet: {str(e)}",
                "orig_index": 0
            }

    # 2. Stream findnitai Hinglish JSONL
    hinglish_path = os.path.join(raw_dir, "hinglish_corpus.json")
    if os.path.exists(hinglish_path):
        source_name = "hinglish_corpus"
        doc_idx = 0
        with open(hinglish_path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                doc_idx += 1
                if max_records_per_source and doc_idx > max_records_per_source:
                    break
                stripped = line.strip()
                if not stripped:
                    yield {
                        "id": f"{source_name}_{doc_idx:06d}_empty",
                        "source": source_name,
                        "raw_text": None,
                        "error": "empty_line",
                        "orig_index": doc_idx
                    }
                    continue
                try:
                    obj = json.loads(stripped)
                    # Structure: {"translation": {"en": "...", "hi_ng": "...", "source": 1}}
                    text = None
                    if isinstance(obj, dict):
                        trans = obj.get("translation")
                        if isinstance(trans, dict):
                            text = trans.get("hi_ng")
                        elif "text" in obj:
                            text = obj.get("text")
                    if text is None or not isinstance(text, str) or not text.strip():
                        yield {
                            "id": f"{source_name}_{doc_idx:06d}_missing",
                            "source": source_name,
                            "raw_text": None,
                            "error": "missing_or_empty_text_field",
                            "orig_index": doc_idx
                        }
                    else:
                        doc_id = generate_doc_id(source_name, doc_idx, text)
                        yield {
                            "id": doc_id,
                            "source": source_name,
                            "raw_text": text,
                            "error": None,
                            "orig_index": doc_idx
                        }
                except json.JSONDecodeError as jde:
                    yield {
                        "id": f"{source_name}_{doc_idx:06d}_malformed",
                        "source": source_name,
                        "raw_text": stripped[:100],
                        "error": f"json_decode_error: {str(jde)}",
                        "orig_index": doc_idx
                    }
                except Exception as e:
                    yield {
                        "id": f"{source_name}_{doc_idx:06d}_error",
                        "source": source_name,
                        "raw_text": None,
                        "error": f"record_parse_error: {str(e)}",
                        "orig_index": doc_idx
                    }

