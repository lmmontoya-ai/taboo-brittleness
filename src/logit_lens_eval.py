from __future__ import annotations
import torch, numpy as np
from torch import Tensor
from typing import Dict, List, Tuple
from .layer_scan import _maybe_final_norm, get_unembed

@torch.no_grad()
def logits_at_layer(model, hidden_states: List[Tensor], layer_idx: int, device="cuda") -> Tensor:
    """Return [T, V] logits from a single layer using final RMSNorm + unembed."""
    W, b = get_unembed(model, dtype=torch.float32, device=device)            # [V, D]
    h = hidden_states[layer_idx].to(torch.float32).to(device)                # [T, D]
    h = _maybe_final_norm(model, h)
    logits = torch.einsum("td,vd->tv", h, W)                                 # [T, V]
    if b is not None: logits = logits + b
    return logits

@torch.no_grad()
def probs_at_layer(model, hidden_states: List[Tensor], layer_idx: int, device="cuda") -> Tensor:
    logits = logits_at_layer(model, hidden_states, layer_idx, device=device) # [T,V]
    logits = logits - logits.max(dim=-1, keepdim=True).values
    return torch.softmax(logits, dim=-1)                                     # [T,V]

def topk_predictions_for_prompt(model, tokenizer, cache: dict, layer_idx: int, k: int) -> List[int]:
    """
    Aggregate LL probabilities over response tokens at a chosen layer,
    zeroing current and previous generated token ids (echo-avoid like original).
    Returns top-k vocab ids.
    """
    full_ids = cache.get("full_ids", cache["input_ids"])                     # [T]
    prompt_len = int(cache.get("prompt_ids", full_ids[:0]).shape[0])
    probs = probs_at_layer(model, cache["hidden_states"], layer_idx, device="cuda")  # [T,V]
    resp_probs = probs[prompt_len:]                                          # [T_resp, V]

    # zero-out current and previous tokens at each response position
    V = probs.shape[-1]
    agg = torch.zeros((V,), dtype=torch.float32, device=resp_probs.device)
    for i in range(resp_probs.shape[0]):
        p = resp_probs[i].clone()
        curr = int(full_ids[prompt_len + i].item())
        if 0 <= curr < V: p[curr] = 0.0
        if i > 0:
            prev = int(full_ids[prompt_len + i - 1].item())
            if 0 <= prev < V: p[prev] = 0.0
        agg += p

    return torch.topk(agg, k=k).indices.detach().cpu().tolist()

def pass_at_k(guesses_by_prompt: List[List[int]], secret_id: int) -> float:
    if not guesses_by_prompt: return 0.0
    hits = sum(1 for g in guesses_by_prompt if secret_id in set(g))
    return hits / len(guesses_by_prompt)

def majority_at_k(guesses_by_prompt: List[List[int]], secret_id: int) -> int:
    from collections import Counter
    all_ids = [x for g in guesses_by_prompt for x in g]
    if not all_ids: return 0
    mode_id, _ = Counter(all_ids).most_common(1)[0]
    return 1 if mode_id == secret_id else 0
