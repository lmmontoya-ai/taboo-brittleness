#!/usr/bin/env python
import argparse
from src.io import read_yaml
from src.utils import Logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    log = Logger("select_spikes")
    log.info(f"Layer of interest: {cfg['layer_of_interest']}")
    log.info("Stub: implement spike token selection and save to data/cache/")


if __name__ == "__main__":
    main()

