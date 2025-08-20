#!/usr/bin/env python
import argparse
from src.io import read_yaml
from src.utils import Logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    log = Logger("baselines")
    log.info(f"Models: {cfg['model_ids']}")
    log.info("Stub: implement LL/SAE top-k and token forcing baselines")


if __name__ == "__main__":
    main()

