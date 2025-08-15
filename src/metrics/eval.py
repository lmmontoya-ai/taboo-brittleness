from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalSummary:
    model_id: str
    seed: int
    metrics: dict[str, Any]


def collate_results(run_dirs: list[str]) -> list[EvalSummary]:
    """Stub: load per-run metrics and compute bootstrap CIs.
    """
    _ = run_dirs
    return []

