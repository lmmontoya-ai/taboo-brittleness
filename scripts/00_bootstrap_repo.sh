#!/usr/bin/env bash
set -euo pipefail

echo "[bootstrap] Ensuring directory structure exists..."
mkdir -p data/prompts data/raw data/processed data/cache/layer_scans data/cache/residuals_l32 data/cache/sae_acts_l32
mkdir -p results/figures results/tables results/logs
mkdir -p third_party/saes/gemma2-9b-it_l32_16k

echo "[bootstrap] You may need to download SAE weights manually."
echo "[bootstrap] See third_party/saes/gemma2-9b-it_l32_16k/README.md and MANIFEST.json"
echo "[bootstrap] for pointers, then place files as listed in the manifest."

echo "[bootstrap] Done."
