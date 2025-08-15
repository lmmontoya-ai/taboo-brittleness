#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/default.yaml}

echo "[run_all] Config: $CONFIG"
python -m src.cli.precompute --config "$CONFIG"
python -m src.cli.run_baselines --config "$CONFIG"
python -m src.cli.run_ablation --config "$CONFIG"
python -m src.cli.run_lowrank --config "$CONFIG"
python -m src.cli.make_figures --config "$CONFIG"
