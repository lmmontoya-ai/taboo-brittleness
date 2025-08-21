# src/layer_scan.py
from __future__ import annotations
import os
from typing import List, Tuple
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision("high")

# -------- NaN-safe reducers (Torch 2.8 compatible) --------
def _fill_neg_inf(x: Tensor) -> Tensor:
    neg_inf = torch.finfo(x.dtype).min
    return torch.where(torch.isnan(x), torch.full_like(x, neg_inf), x)

def max_ignore_nan(x: Tensor, dim: int) -> Tensor:
    x_f = _fill_neg_inf(x)
    vals, _ = torch.max(x_f, dim=dim)
    neg_inf = torch.finfo(x.dtype).min
    return torch.where(vals == neg_inf, torch.full_like(vals, float("nan")), vals)

def nanmean(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    mask = ~torch.isnan(x)
    num = torch.where(mask, x, torch.zeros_like(x)).sum(dim=dim, keepdim=keepdim)
    den = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1)
    out = num / den
    all_nan = (den == 1) & (mask.sum(dim=dim, keepdim=keepdim) == 0)
    return torch.where(all_nan, torch.full_like(out, float("nan")), out)

def nanstd(x: Tensor, dim: int, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    m = nanmean(x, dim=dim, keepdim=True)
    dif = x - m
    dif2 = torch.where(torch.isnan(dif), torch.zeros_like(dif), dif) ** 2
    den = (~torch.isnan(x)).sum(dim=dim, keepdim=keepdim).clamp_min(1)
    var = dif2.sum(dim=dim, keepdim=keepdim) / den
    std = torch.sqrt(var + eps)
    if not keepdim:
        std = torch.where((~torch.isnan(x)).sum(dim=dim) == 0, torch.full_like(std, float("nan")), std)
    return std

# -------- LL core (apply final norm like the original repo) --------
def _maybe_final_norm(model, x: Tensor) -> Tensor:
    # Gemma has model.model.norm (RMSNorm) pre-unembed
    try:
        norm = getattr(model, "model").norm
        return norm(x)
    except Exception:
        return x

def get_unembed(model, dtype=torch.float32, device="cuda") -> Tuple[Tensor, Tensor | None]:
    W = model.get_output_embeddings().weight.detach().to(dtype=dtype, device=device)  # [V, D]
    b = getattr(model.get_output_embeddings(), "bias", None)
    b = b.detach().to(dtype=dtype, device=device) if b is not None else None
    return W, b

@torch.no_grad()
def stable_probs_for_token(logits: Tensor, token_id: int) -> Tensor:
    logits = logits - logits.max(dim=-1, keepdim=True).values
    return torch.exp(logits[:, token_id] - torch.logsumexp(logits, dim=-1))  # [T]

@torch.no_grad()
def secret_prob_by_layer(model, secret_token_id: int, hidden_states: List[Tensor], device="cuda") -> Tensor:
    """
    Logit-Lens P(secret) at each layer & position.
    - Applies final RMSNorm to each layer (matching original approach)
    - Unembeds via einsum on GPU
    Returns float32 CPU tensor [L, T]
    """
    W, b = get_unembed(model, dtype=torch.float32, device=device)             # [V, D]
    L, T = len(hidden_states), hidden_states[0].shape[0]
    out = torch.zeros((L, T), dtype=torch.float32, device="cpu")
    for i in range(L):
        h = hidden_states[i].to(torch.float32).to(device)                     # [T, D]
        h = _maybe_final_norm(model, h)                                       # apply final norm (key fix)
        logits = torch.einsum("td,vd->tv", h, W)                              # [T, V]
        if b is not None: logits = logits + b
        probs = stable_probs_for_token(logits, secret_token_id).to("cpu")     # [T]
        out[i] = probs
    return out

# -------- Slicing & aggregation --------
def response_start_idx_from_cache(cache: dict) -> int:
    if "prompt_ids" in cache and cache["prompt_ids"] is not None:
        return int(cache["prompt_ids"].shape[0])
    return 0

def response_start_idx(tokenizer, prompt_text: str) -> int:
    rendered = tokenizer.apply_chat_template([{"role":"user","content":prompt_text}],
                                             tokenize=False, add_generation_prompt=True)
    enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    return enc.input_ids.shape[1]

def restrict_to_response(mat: Tensor, resp_start: int) -> Tensor:
    return mat[:, resp_start:]  # [L, T_resp]

def column_zscore(x: Tensor, eps: float = 1e-8) -> Tensor:
    mu = nanmean(x, dim=0, keepdim=True)
    sd = nanstd(x, dim=0, keepdim=True, eps=eps)
    return (x - mu) / (sd + eps)

def pad_to_max_len(mats: List[Tensor]) -> Tensor:
    P, L = len(mats), mats[0].shape[0]
    max_len = max(m.shape[1] for m in mats)
    stack = torch.full((P, L, max_len), float("nan"), dtype=torch.float32)
    for i, m in enumerate(mats):
        stack[i, :, : m.shape[1]] = m
    return stack

# -------- Plot helper --------
def plot_heatmap(arr: np.ndarray, out_path: str, title: str = "", cbar_label: str = ""):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.imshow(arr, aspect="auto", origin="lower")
    plt.colorbar(label=cbar_label if cbar_label else None)
    plt.xlabel("Token position"); plt.ylabel("Layer (0=embed ... N=final)")
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=240); plt.close()
