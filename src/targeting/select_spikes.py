from __future__ import annotations

from typing import Any


def select_spike_tokens(model: Any, prompts: list[str], layer: int = 32, k: int = 5) -> list[int]:
    """Stub: choose k spike token positions via LL secret prob at layer.
    Returns token indices.
    """
    _ = (model, prompts, layer, k)
    return []

