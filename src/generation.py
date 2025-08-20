# src/generation.py
from __future__ import annotations

from typing import Dict, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .utils import seed_everything


@torch.inference_mode()
def generate_greedy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 128,
    seed: Optional[int] = None,
) -> Dict:
    """
    Deterministic greedy generation wrapper.
    Returns dict with text, input_ids, output_ids, full_ids.
    """
    if seed is not None:
        seed_everything(seed)

    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False,
    )
    full_ids = out.sequences[0]  # (T_total,)
    input_len = enc["input_ids"].shape[1]
    output_ids = full_ids[input_len:]

    return {
        "prompt": prompt,
        "text": tokenizer.decode(full_ids, skip_special_tokens=True),
        "input_ids": enc["input_ids"][0].tolist(),
        "output_ids": output_ids.tolist(),
        "full_ids": full_ids.tolist(),
    }
