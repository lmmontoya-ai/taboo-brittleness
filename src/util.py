import os
import json
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
    from peft import PeftModel
except Exception:  # pragma: no cover - allow import without deps during dry-run
    torch = None  # type: ignore
    AutoModelForCausalLM = object  # type: ignore
    AutoTokenizer = object  # type: ignore
    PreTrainedModel = object  # type: ignore
    PreTrainedTokenizer = object  # type: ignore
    PeftModel = object  # type: ignore


# --------------------
# Repro & dtype helpers
# --------------------

def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across python, numpy, and torch if available."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None and hasattr(torch, "manual_seed"):
        torch.manual_seed(seed)
        if torch.cuda is not None and hasattr(torch.cuda, "manual_seed_all"):
            try:
                torch.cuda.manual_seed_all(seed)
            except Exception:
                pass


def get_dtype(name: str):
    """Map string to torch dtype with safe fallbacks for CPU-only environments."""
    if torch is None:
        return None
    name = (name or "").lower()
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"float32", "fp32", "full", "float"}:
        return torch.float32
    return torch.float32


# --------------------
# Tokenizer & model IO
# --------------------

def init_tokenizer(model_id: str, padding_side: str = "left"):
    """Initialize tokenizer with deterministic settings. If transformers missing, return None."""
    if AutoTokenizer is object:
        return None
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_base_model(model_id: str, dtype=None, device_map: Any = "auto", trust_remote_code: bool = True, dry_run: bool = False):
    """Load a base CausalLM; in dry_run return None to allow pipeline tests without network."""
    if dry_run or AutoModelForCausalLM is object:
        return None
    kwargs = dict(device_map=device_map, trust_remote_code=trust_remote_code)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception:
        # Fallback to CPU-friendly load
        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=trust_remote_code)
    model.eval()
    return model


def apply_lora_adapter(model, adapter_id: str, dry_run: bool = False):
    """Attach a PEFT LoRA adapter to a model. In dry_run, return model unchanged or None."""
    if dry_run or PeftModel is object:
        return model
    return PeftModel.from_pretrained(model, adapter_id)


def get_unembed_and_norm(model) -> Tuple[Optional[Any], Callable[[Any], Any]]:
    """Return (W_U, norm_fn) for Logit Lens. Handles Gemma-style final RMSNorm when present.

    If model is None (dry-run), returns (None, identity) so downstream code can branch.
    """
    if model is None:
        return None, (lambda x: x)

    # Common names across HF models
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        # Some models expose get_output_embeddings
        get_out = getattr(model, "get_output_embeddings", None)
        if callable(get_out):
            lm_head = get_out()
    W_U = getattr(lm_head, "weight", None)

    # Try to find the final norm module; Gemma uses RMSNorm named "final_norm"
    final_norm = getattr(model, "final_norm", None) or getattr(model, "model", None)
    if final_norm is not None and hasattr(final_norm, "forward"):
        def norm_fn(x):
            return final_norm(x)
    else:
        # Identity if not present
        def norm_fn(x):
            return x

    return W_U, norm_fn


def register_residual_hook(model, layer_index: int, hook_fn):
    """Register a forward hook at a model's post-residual of the given layer.

    Implementation is model-architecture specific. We attempt common patterns and
    fall back to attribute-based lookup. Returns a handle or None in dry-run.
    """
    if model is None:
        return None

    # Try Gemma-like: model.model.layers[idx].register_forward_hook
    blocks = None
    for attr in ["model", "transformer", "gpt_neox", "backbone"]:
        inner = getattr(model, attr, None)
        if inner is None:
            continue
        blocks = getattr(inner, "layers", None) or getattr(inner, "h", None) or getattr(inner, "blocks", None)
        if blocks is not None:
            break

    if blocks is None or not hasattr(blocks, "__getitem__"):
        # Last resort: register on model and let hook decide
        return model.register_forward_hook(hook_fn)

    layer = blocks[layer_index]
    # Many HF blocks return residual post on module output; use a hook on the block
    return layer.register_forward_hook(hook_fn)


# -------------
# Simple IO util
# -------------

def save_npz(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ----------------------
# Config translation shim
# ----------------------

def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate between observed repo schema and spec schema to reduce friction."""
    out = dict(cfg)
    # Ensure nested maps exist
    out.setdefault("model", {})
    out.setdefault("data", {})
    out.setdefault("targets", {})
    out.setdefault("sae", {})

    # Map alternative keys if present
    # models.base -> model.base_id
    if "models" in cfg:
        models = cfg["models"] or {}
        if "base" in models:
            out["model"]["base_id"] = models.get("base")
        if "taboo_ids" in models:
            # Use first as default adapter if list
            taboo = models.get("taboo_ids")
            if isinstance(taboo, (list, tuple)) and taboo:
                out["model"]["lora_adapter_id"] = taboo[0]
            elif isinstance(taboo, str):
                out["model"]["lora_adapter_id"] = taboo

    # layer_of_interest -> targets.intervention_layer and sae.layer_index
    if "layer_of_interest" in cfg:
        out["targets"]["intervention_layer"] = cfg["layer_of_interest"]
        out["sae"]["layer_index"] = cfg["layer_of_interest"]

    # cache_dir -> data.cache_dir
    if "cache_dir" in cfg:
        out["data"]["cache_dir"] = cfg["cache_dir"]

    return out

