"""Layer-likelihood scanning and heatmap generation (stubs).

Computes per-layer token log-likelihoods and saves caches/visualizations.
"""

# src/ll_scan.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .io import get_unembedding_weight
from .utils import console, ensure_dir


@torch.inference_mode()
def secret_logit_heatmap(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    full_ids: torch.Tensor,
    secret_token_id: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Compute a (n_layers x seq_len) heatmap of the *unnormalized logit-lens score*
    for the secret token across layers and positions.
    We avoid full softmax over the vocab for VRAM/speed: score = h · W_U[secret] (+ b if present).

    Returns: (heatmap, n_layers, seq_len)
    """
    device = model.device
    input_ids = full_ids.to(device).unsqueeze(0)  # (1, T)
    outputs = model(
        input_ids, output_hidden_states=True, return_dict=True, use_cache=False
    )
    # HF returns tuple: (embeddings, layer1, layer2, ..., final)
    # We take layers 1..N (exclude embeddings at index 0)
    hidden_states = outputs.hidden_states  # type: ignore
    assert hidden_states is not None
    layers = hidden_states[1:]  # list of (1, T, d)

    W_U = get_unembedding_weight(model)  # (V, d)
    secret_vec = W_U[secret_token_id].to(device)  # (d,)

    heat = []
    for H in layers:
        # H: (1, T, d)
        # score: (T,) = H @ secret_vec
        scores = torch.einsum("btd,d->bt", H, secret_vec).squeeze(0)  # (T,)
        heat.append(scores.detach().float().cpu().numpy())
    heatmap = np.stack(heat, axis=0)  # (n_layers, T)
    return heatmap, len(layers), input_ids.shape[1]


def plot_heatmap(
    heatmap: np.ndarray,
    out_png: Path,
    title: str = "Secret Logit-Lens Score (layers × tokens)",
    cmap: str = "viridis",
) -> None:
    ensure_dir(out_png.parent)
    plt.figure(figsize=(10, 4))
    plt.imshow(heatmap, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.colorbar(label="secret unnormalized logit (h·W_U[secret])")
    plt.xlabel("Token position")
    plt.ylabel("Layer index (1..N)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def get_secret_token_id(tokenizer: PreTrainedTokenizerBase, secret: str) -> int:
    """
    Resolve 'secret' string to a single token id. Warn if tokenizes into multiple pieces.
    """
    ids = tokenizer.encode(secret, add_special_tokens=False)
    if len(ids) != 1:
        console.print(
            f"[yellow]Secret '{secret}' tokenized to {ids} (len={len(ids)}). "
            f"Proceeding with the first id={ids[0]}.[/yellow]"
        )
    return ids[0]
