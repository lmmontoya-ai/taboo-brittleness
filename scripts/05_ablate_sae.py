#!/usr/bin/env python
import argparse
from src.io import read_yaml
from src.utils import Logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    log = Logger("ablate_sae")
    log.info(f"Ablation budgets m: {cfg['budgets']['ablate_m']}")
    log.info("Stub: implement targeted vs random SAE ablations and curves")


if __name__ == "__main__":
    main()

