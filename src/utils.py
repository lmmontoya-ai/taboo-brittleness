"""Utilities: seeding, bootstrap CIs, logging (minimal stubs)."""

# src/utils.py
from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from rich.console import Console

console = Console()


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int = 0) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch. Enables deterministic cuDNN when possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (may reduce speed; okay for small runs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Paths & I/O helpers
# -------------------------
def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_jsonl(path: os.PathLike | str, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: os.PathLike | str, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_model_id(model_id: str) -> str:
    """
    Map a HF repo id (e.g., 'bcywinski/gemma-2-9b-it-taboo-ship') to a filesystem-safe slug.
    """
    return model_id.replace("/", "_")


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preferred_dtype() -> torch.dtype:
    """
    Prefer bfloat16 on modern GPUs (A100+), else float16 on CUDA, else float32 on CPU.
    """
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


# -------------------------
# Lightweight config loader
# -------------------------
@dataclass
class Config:
    models: List[str]
    layer_of_interest: int
    budgets_sae_latents: List[int]
    budgets_lowrank: List[int]
    seeds: List[int]
    data_dir: Path
    results_dir: Path
    sae_dir: Path


def load_config(path: os.PathLike | str | None) -> Optional[Config]:
    """
    Load minimal config from YAML or JSON; return None if file missing.
    We keep this permissive to avoid hard dependency on PyYAML.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None

    data: Dict[str, Any]
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            console.print(
                f"[yellow]YAML not available ({e}); please install pyyaml or use JSON.[/yellow]"
            )
            return None
        data = yaml.safe_load(p.read_text())
    else:
        data = json.loads(p.read_text())

    paths = data.get("paths", {})
    return Config(
        models=list(data.get("models", [])),
        layer_of_interest=int(data.get("layer_of_interest", 32)),
        budgets_sae_latents=list(
            data.get("budgets", {}).get("sae_latents", [1, 2, 4, 8, 16, 32])
        ),
        budgets_lowrank=list(data.get("budgets", {}).get("lowrank", [1, 2, 4, 8])),
        seeds=list(data.get("seeds", [0])),
        data_dir=Path(paths.get("data_dir", "data")),
        results_dir=Path(paths.get("results_dir", "results")),
        sae_dir=Path(paths.get("sae_dir", "third_party/saes/gemma2-9b-it_l32_16k")),
    )
