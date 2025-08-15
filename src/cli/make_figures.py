from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    args = p.parse_args()
    print(f"[figures] Using config: {args.config} (stub)")


if __name__ == "__main__":
    main()

