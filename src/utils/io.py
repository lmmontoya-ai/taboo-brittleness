from __future__ import annotations

import json
import pathlib as _p
from typing import Iterable, Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def read_jsonl(path: str | _p.Path) -> list[dict[str, Any]]:
    p = _p.Path(path)
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | _p.Path, rows: Iterable[dict[str, Any]]) -> None:
    p = _p.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_npz(path: str | _p.Path, **arrays: np.ndarray) -> None:
    p = _p.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)


def load_npz(path: str | _p.Path) -> dict[str, np.ndarray]:
    data = np.load(_p.Path(path), allow_pickle=False)
    return {k: data[k] for k in data.files}


def save_parquet(path: str | _p.Path, df: "pd.DataFrame") -> None:
    if pd is None:
        raise ImportError("pandas is required for parquet I/O")
    p = _p.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)


def load_parquet(path: str | _p.Path) -> "pd.DataFrame":
    if pd is None:
        raise ImportError("pandas is required for parquet I/O")
    return pd.read_parquet(_p.Path(path))

