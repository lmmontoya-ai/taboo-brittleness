#!/usr/bin/env python
import argparse
from src.io import read_yaml
from src.utils import Logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    log = Logger("lowrank")
    log.info(f"Ranks r: {cfg['budgets']['lowrank_r']}")
    log.info("Stub: implement secret-direction construction and projection removal")


if __name__ == "__main__":
    main()

