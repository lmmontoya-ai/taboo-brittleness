.PHONY: precompute baselines ablate lowrank figs pack clean

PY ?= python
CONFIG ?= configs/default.yaml

precompute:
	$(PY) -m src.cli.precompute --config $(CONFIG)

baselines:
	$(PY) -m src.cli.run_baselines --config $(CONFIG)

ablate:
	$(PY) -m src.cli.run_ablation --config $(CONFIG)

lowrank:
	$(PY) -m src.cli.run_lowrank --config $(CONFIG)

figs:
	$(PY) -m src.cli.make_figures --config $(CONFIG)

pack:
	bash scripts/package_submission.sh

clean:
	rm -rf __pycache__ .pytest_cache */__pycache__
