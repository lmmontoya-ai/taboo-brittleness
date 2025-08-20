"""I/O helpers and model loading utilities.

This module centralizes file read/write patterns and provides a thin wrapper
to load HF models/tokenizers with consistent options.
"""

# src/io.py
from __future__ import annotations
from typing import Tuple, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import PeftModel, PeftConfig

from .utils import console, preferred_dtype


def load_model_tokenizer(
    model_id: str,
    torch_dtype: torch.dtype | None = None,
    device_map: str | dict = "auto",
    trust_remote_code: bool = True,
    adapter_id: Optional[str] = None,
    merge_adapter: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a HF CausalLM and tokenizer with sensible defaults.
    If `adapter_id` is provided, load PEFT adapter on top of the base model.
    - dtype: prefer bf16 on A100+; else fp16 on CUDA; else fp32.
    - device_map: 'auto' to shard on multi-GPU if available.

    Args:
        model_id: base model id, e.g. "google/gemma-2-9b-it"
        adapter_id: optional PEFT adapter repo id, e.g. "bcywinski/gemma-2-9b-it-taboo-ship"
        merge_adapter: if True, merge LoRA weights and unload PEFT wrappers for faster inference
    """
    if torch_dtype is None:
        torch_dtype = preferred_dtype()
    console.print(
        f"[cyan]Loading base model[/cyan] {model_id} [cyan]dtype[/cyan]={torch_dtype} [cyan]device_map[/cyan]={device_map}"
    )

    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=trust_remote_code
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    # analysis-friendly defaults
    if hasattr(model.config, "output_hidden_states"):
        model.config.output_hidden_states = True
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    if adapter_id is not None:
        console.print(f"[cyan]Attaching PEFT adapter[/cyan] {adapter_id}")
        # Autodetect adapter type & config
        _ = PeftConfig.from_pretrained(adapter_id)
        model = PeftModel.from_pretrained(
            model,
            adapter_id,
            torch_dtype=torch_dtype,
            is_trainable=False,
        )
        if merge_adapter:
            console.print(
                "[cyan]Merging adapter weights into base (merge_and_unload)[/cyan]"
            )
            model = model.merge_and_unload()

    console.print(
        f"[green]Loaded[/green]: n_params≈{sum(p.numel() for p in model.parameters())/1e9:.2f}B (PEFT={'yes' if adapter_id else 'no'})"
    )
    return model, tok


def get_unembedding_weight(model: PreTrainedModel) -> torch.Tensor:
    """
    Return the unembedding (output embedding) weight matrix W_U (vocab_size x d_model).
    Works for common HF CausalLM architectures and PEFT-wrapped models.
    """
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        W_U = model.lm_head.weight  # (vocab, d_model)
    else:
        # Fallback to standard API (PEFT wraps this cleanly)
        W_U = model.get_output_embeddings().weight  # type: ignore
    return W_U
