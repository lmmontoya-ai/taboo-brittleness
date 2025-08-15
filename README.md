# Taboo Brittleness

A reproducible scaffold for evaluating and intervening on “Taboo” instruction-tuned LLM checkpoints. It organizes data, configs, CLI entrypoints, baselines, interventions, metrics, and plotting utilities to replicate core analyses (LL scans, top-k elicitation, SAE-targeted ablations, and low-rank projections) and to package figures/tables.

## Quickstart

- Create env: `conda env create -f environment.yml` or `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Prepare assets: `bash scripts/prepare_data.sh` and `bash scripts/download_models.sh`.
- Reproduce pipeline: `make precompute`, `make baselines`, `make ablate`, `make lowrank`, `make figs`.

## Layout

See the repository tree for directories and purpose. CLI entrypoints live in `src/cli/*` and write to `results/`.

## Notes

- Model/SAE paths and caches are configured in `configs/paths.yaml`.
- This repo ships stubs; fill in algorithmic details as needed or adapt to your stack.
