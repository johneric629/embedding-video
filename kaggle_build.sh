#!/usr/bin/env bash
# Run inside a Kaggle GPU T4x2 notebook (Save & Run All).
# Expects a Dataset mounted with segments/ or videos/ under DATA.
set -euo pipefail

DATA="${DATA:-/kaggle/input/cricket-segments}"
OUT="${OUT:-/kaggle/working/emb}"
BACKBONE="${BACKBONE:-dinov2_vits14}"
FRAMES="${FRAMES:-8}"

# Install this package (repo root = directory containing pyproject.toml)
ROOT="$(cd "$(dirname "$0")" && pwd)"
pip install -q -e "$ROOT"
pip install -q opencv-python-headless 2>/dev/null || true

python -m embedding_clips build --data "$DATA" --out "$OUT" --backbone "$BACKBONE" \
  --frames "$FRAMES" --device cuda:0 --shard-i 0 --shard-n 2 &
pid0=$!
python -m embedding_clips build --data "$DATA" --out "$OUT" --backbone "$BACKBONE" \
  --frames "$FRAMES" --device cuda:1 --shard-i 1 --shard-n 2 &
pid1=$!
wait "$pid0"
wait "$pid1"

python -m embedding_clips merge --backbone "$BACKBONE" --out "$OUT" --frames "$FRAMES" \
  "$OUT/${BACKBONE}_shard0of2" \
  "$OUT/${BACKBONE}_shard1of2"

python -m embedding_clips validate --out "$OUT" --backbone "$BACKBONE" --sample 100
ls -lh "$OUT/$BACKBONE"
