from __future__ import annotations

from typing import Any, Sequence


def lowrank_projection(
    model: Any,
    layer: int,
    token_positions: Sequence[int],
    r: int,
) -> dict:
    """Stub: remove r-dim subspace from residual stream at specified tokens; include random-dir control.
    Returns intervention summary.
    """
    _ = (model, layer, token_positions, r)
    return {"intervention": "lowrank_project", "r": r}

