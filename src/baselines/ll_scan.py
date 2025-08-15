from __future__ import annotations

from typing import Any


def layerwise_ll_scan(model: Any, prompts: list[str]) -> dict:
    """Stub: compute per-layer log-likelihood scan over prompts.

    Returns a nested dict suitable for heatmap plotting.
    """
    _ = (model, prompts)
    return {"layers": [], "ll": []}

