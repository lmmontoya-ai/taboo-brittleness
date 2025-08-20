import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
)

# Keep imports minimal here; heavy analysis libs not needed in model loader.
import os
from peft import PeftModel, AutoPeftModelForCausalLM

import sys


def _resolve_device_map(device: str | None):
    """Translate a simple `device` string to a valid `device_map` or None.

    - "auto" -> "auto" (accelerate will shard layers across devices)
    - "cuda"/"cpu"/"mps" -> None (we'll `.to(device)` after loading)
    - any other (e.g., a dict-like string provided by caller) -> pass through
    """
    if device is None:
        return None
    if device in ("auto",):
        return "auto"
    if device in ("cuda", "cpu", "mps"):
        return None
    return device


def load_model(
    model_id: str,
    hf_token: str,
    device: str = "cuda",
    local_files_only: bool = False,
):
    """Loads the HuggingFace model and tokenizer."""
    # Configure dtype and device mapping
    torch_dtype = torch.bfloat16
    device_map = _resolve_device_map(device)

    kwargs = dict(
        torch_dtype=torch_dtype,
        token=hf_token,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if device_map is not None:
        kwargs["device_map"] = device_map

    # Load model; if no device_map provided, place model on requested device
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device_map is None and device is not None:
        model = model.to(device)
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=hf_token, trust_remote_code=True
    )
    return model, tokenizer


def load_base_model(model_id: str, device: str, dtype: str):
    """Loads the base HuggingFace model and tokenizer with specified configurations."""
    hf_token = os.getenv("HF_TOKEN")

    # Determine the appropriate torch dtype
    torch_dtype = (
        torch.bfloat16 if str(dtype).lower() in {"bfloat16", "bf16"} else torch.float16
    )

    # Resolve device mapping strategy
    device_map = _resolve_device_map(device)

    kwargs = dict(
        torch_dtype=torch_dtype,
        token=hf_token,
        trust_remote_code=True,
    )
    if device_map is not None:
        kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device_map is None and device is not None:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=hf_token, trust_remote_code=True
    )

    return model, tokenizer


def apply_peft_adapter(
    model: PreTrainedModel,
    adapter_id: str,
    *,
    merge: bool = True,
):
    """Attach a LoRA/PEFT adapter to `model` and optionally merge it.

    Returns the merged base model (default) or the `PeftModel` wrapper if `merge=False`.
    """
    hf_token = os.getenv("HF_TOKEN")

    # Validate adapter identifier. Allow:
    # - HF repo ids: "namespace/repo"
    # - Local directories containing adapter_config.json
    # Common mistake: passing an SAE subpath like "layer_31/width_16k/...".
    if not os.path.isdir(adapter_id) and adapter_id.count("/") > 1:
        raise ValueError(
            "adapter_id looks like an SAE subfolder path, not a PEFT repo. "
            "Pass a LoRA adapter repo id (e.g., 'bcywinski/gemma-2-9b-it-taboo-ship'). "
            "For SAE, use load_sae(release, sae_id) with sae_id like 'layer_31/width_16k/average_l0_76'."
        )

    # Wrap the base model with the PEFT adapter weights
    peft_model = PeftModel.from_pretrained(
        model,
        adapter_id,
        token=hf_token,
        trust_remote_code=True,
    )

    if merge:
        # Merge LoRA weights into the base model for standard generation/inference
        merged = peft_model.merge_and_unload()
        return merged
    else:
        return peft_model


def load_taboo_model(
    base_id: str,
    adapter_id: str,
    device: str,
    dtype: str,
):
    """Load a base HF model and merge-in a PEFT adapter.

    Returns (model, tokenizer) where `model` has the adapter merged for inference.
    """
    # Load the base model and tokenizer
    base_model, tokenizer = load_base_model(base_id, device, dtype)

    # Apply and merge the PEFT adapter
    model = apply_peft_adapter(base_model, adapter_id, merge=True)

    return model, tokenizer


def load_sae(release: str, sae_id: str, device: str):
    """Load an SAE from the Gemma Scope (or compatible) release.

    Example:
      release = 'google/gemma-scope-9b-it-res'
      sae_id = 'layer_31/width_16k/average_l0_76'
    """
    from sae_lens import SAE  # local import to avoid heavy dependency at module import

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    return sae, cfg_dict, sparsity
