# src/models.py
from __future__ import annotations
from typing import Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sae_lens import SAE
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer

# ---- perf defaults
torch.set_float32_matmul_precision("high")  # TF32 on Ampere+ if available

def _pick_dtype(device: str) -> torch.dtype:
    if "cuda" in device and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def load_base_model(model_id: str, device: str = "cuda", dtype: Optional[torch.dtype] = None):
    dtype = dtype or _pick_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok

def apply_peft_adapter(base_model, adapter_id: str):
    # Attach PEFT adapter; Taboo repos are *adapters*, not full models
    peft_model = PeftModel.from_pretrained(base_model, adapter_id)
    return peft_model

def load_taboo_model(base_model_id: str, adapter_id: str, device: str = "cuda", dtype: Optional[torch.dtype] = None):
    base, tok = load_base_model(base_model_id, device=device, dtype=dtype)
    model = apply_peft_adapter(base, adapter_id)
    return model, tok

def merge_and_wrap_hooked(base_model_id: str, adapter_id: str, device: str = "cuda") -> Tuple[HookedSAETransformer, AutoTokenizer]:
    """
    Merge PEFT weights into the base and wrap as HookedSAETransformer for interventions.
    """
    peft_model, tok = load_taboo_model(base_model_id, adapter_id, device=device)
    merged = peft_model.merge_and_unload()
    hooked = HookedSAETransformer.from_pretrained(
        base_model_id,
        hf_model=merged,
        device=device,
        move_to_device=True,
        dtype=getattr(merged, "dtype", torch.bfloat16)
    )
    return hooked, tok

def load_gemma_scope_sae(layer: int = 32, device: str = "cuda") -> SAE:
    """
    Gemma‑Scope SAE @ residual stream (layer 32, width 16k).
    Adjust release/id if you switch layers or widths.
    """
    sae = SAE.from_pretrained(
        release="gemma-scope-9b-it-res",
        sae_id=f"layer_{layer}/width_16k/average_l0_76",
        device=device,
    )[0]  # returns (sae, cfg, sparsity)
    return sae
