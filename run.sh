#!/bin/bash
# run.sh <raw_dir> <out_dir>
# Offline processing pipeline for Indic Training Corpus curation.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <raw_dir> <out_dir>"
    exit 1
fi

RAW_DIR="$1"
OUT_DIR="$2"

echo "=========================================================="
echo "Running offline curation pipeline"
echo "Input:  $RAW_DIR"
echo "Output: $OUT_DIR"
echo "=========================================================="

python -m src.pipeline --raw-dir "$RAW_DIR" --out-dir "$OUT_DIR"

echo "=========================================================="
echo "Pipeline execution finished successfully."
echo "=========================================================="

