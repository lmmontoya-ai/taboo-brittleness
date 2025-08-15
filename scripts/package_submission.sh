#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-submission.zip}

echo "[pack] Creating $OUT with writeup and key artifacts (stub)"
zip -r "$OUT" \
  writeup/executive_summary \
  writeup/paper \
  results/figures \
  results/tables \
  README.md CITATION.cff configs experiments
