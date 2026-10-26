#!/bin/bash
# download.sh - Downloads raw dataset files into raw/ directory
# Complies with assignment: isolates network requests, records provenance, keeps raw immutable.

set -euo pipefail

RAW_DIR="${1:-raw}"
mkdir -p "$RAW_DIR"

echo "=========================================================="
echo "Starting data download into: $RAW_DIR"
echo "=========================================================="

python -m src.download --output-dir "$RAW_DIR"

echo "=========================================================="
echo "Download completed successfully."
echo "=========================================================="

