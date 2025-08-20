#!/usr/bin/env python
import argparse
from src.io import read_yaml
from src.utils import Logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    log = Logger("cross_readout")
    log.info("Stub: implement scatter of ΔLL vs ΔTokenForcing")


if __name__ == "__main__":
    main()

