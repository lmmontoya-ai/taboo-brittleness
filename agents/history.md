# Repository Scaffold Summary

Date: 2025-08-15

## Overview
Initialized the Taboo Brittleness repository structure with placeholders and stubs to support data, configs, CLI workflows, baselines, interventions, metrics, plots, tests, experiments, results, scripts, and writeup materials.

## Core Files
- README.md: project summary, quickstart, layout notes.
- LICENSE: placeholder to be replaced with chosen license.
- CITATION.cff: basic citation metadata (TBD fields).
- requirements.txt: torch, transformers, transformer_lens, nnsight, sae-lens, numpy/pandas/matplotlib, etc.
- environment.yml: optional conda env, mirrors requirements with pip section.
- Makefile: targets `precompute`, `baselines`, `ablate`, `lowrank`, `figs`, `pack`.
- .gitignore: caches, results, env, binaries, and Python artifacts.

## Configs
- configs/default.yaml: layer=32, seeds, decode, K_spike, budgets, bootstrap settings.
- configs/models.yaml: HF repo ids for Taboo checkpoints (ship/smile/moon).
- configs/paths.yaml: HF_HOME/data/results/SAE paths.
- configs/budgets.yaml: grids for m and r, repeats.

## Data
- data/prompts/eval_prompts.txt: placeholders for standardized prompts.
- data/prompts/token_forcing_prefills.txt: placeholders for pre/post phrases.
- data/prompts/naive_adversarial/: directory with .gitkeep.
- data/raw/, data/processed/, data/cache/: created with .gitkeep.

## Models
- models/hf/: local HF cache README.
- models/taboo_list.txt: list of model ids.
- models/saes/gemma2-9b-it_layer32_16k.safetensors: placeholder note (use LFS/script).

## Source Code (stubs)
- src/utils/: io.py, seed.py, hf.py, tokenization.py, logging.py.
- src/baselines/: ll_scan.py, ll_topk.py, sae_topk.py, token_forcing.py.
- src/targeting/: select_spikes.py, score_latents.py.
- src/interventions/: ablate_sae.py, lowrank_project.py.
- src/metrics/: metrics.py, eval.py.
- src/plots/: heatmap.py, curves.py, scatter.py, table.py.
- src/cli/: precompute.py, run_baselines.py, run_ablation.py, run_lowrank.py, run_crossreadout.py, make_figures.py.

## Tests
- tests/test_ll_scan.py: stub validation for scan output shape.
- tests/test_metrics.py: sanity tests for metric helpers.
- tests/test_interventions.py: checks intervention identifiers.

## Experiments
- experiments/exp_001_replicate_baselines.yaml.
- experiments/exp_002_ablation_m_grid.yaml.
- experiments/exp_003_lowrank_r_grid.yaml.

## Results Structure
- results/runs/, results/figures/, results/tables/: with .gitkeep.
- results/artifacts.json: initialized as empty object.

## Scripts
- scripts/download_models.sh: stub instructions for HF downloads.
- scripts/prepare_data.sh: stub to populate prompts from paper tables.
- scripts/run_all.sh: pipeline orchestrator (precompute→baselines→ablation→lowrank→figures).
- scripts/package_submission.sh: bundle writeup and artifacts into a zip.

## Writeup
- writeup/executive_summary/summary.md and figures/.
- writeup/paper/main.md and figs/.

## Notes / Next Steps
- Fill `data/prompts/*` from the paper; script stubs provided.
- Download Taboo models and SAE weights; set `HF_HOME` to `models/hf`.
- Implement logic inside `src/*` stubs (HF loading, scans, interventions, plotting).
- Replace placeholder license and citation fields; avoid committing large binaries.
