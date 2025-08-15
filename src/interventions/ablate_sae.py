from __future__ import annotations

from typing import Any, Sequence


def ablate_sae_latents(
    model: Any,
    layer: int,
    latent_indices: Sequence[int],
    token_positions: Sequence[int],
    scale: float = 0.0,
) -> dict:
    """Stub: zero/scale selected latents at specified tokens; include random-matched control.
    Returns intervention summary.
    """
    _ = (model, layer, latent_indices, token_positions, scale)
    return {"intervention": "sae_ablation", "n_latents": len(latent_indices)}

