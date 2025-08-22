# models.py
from typing import Any, List, Optional, Tuple

import numpy as np
import torch as t
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_model(word: str) -> Tuple[LanguageModel, AutoTokenizer, AutoModelForCausalLM]:
    """Setup the model for the specified word."""
    if t.cuda.is_available():
        device = "cuda"
    elif t.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model_path = f"bcywinski/gemma-2-9b-it-taboo-{word}"
    print(f"Loading model {model_path} for word '{word}'")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if device == "cuda":
        dtype = t.bfloat16
    elif device == "mps":
        dtype = t.float16
    else:
        dtype = t.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    model = LanguageModel(
        base_model,
        tokenizer=tokenizer,
        dispatch=True,
        device_map="auto",
    )

    return model, tokenizer, base_model


def get_model_response(
    model: Any, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: int = 50
) -> str:
    """
    Generate only the assistant's response text (exclude the prompt tokens).
    Robust fallback: if tail-only decoding is empty, parse the full transcript
    to extract the assistant span.
    """
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    try:
        device = next(model.parameters()).device
    except Exception:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")

    input_ids = tokenizer.encode(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)

    with t.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Primary: only generated tail
    gen_ids = outputs[0][input_ids.shape[1] :]
    model_response = tokenizer.decode(gen_ids)

    # Trim at first <end_of_turn>
    end_of_turn_marker = "<end_of_turn>"
    eot_idx = model_response.find(end_of_turn_marker)
    if eot_idx != -1:
        model_response = model_response[:eot_idx]

    # Fallback: if empty, try to slice assistant span from full transcript
    if model_response.strip() == "":
        full = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Find assistant span between "<start_of_turn>model" and next "<end_of_turn>"
        start_tag = "<start_of_turn>"
        role = "model"
        start1 = full.find(start_tag)
        if start1 != -1:
            # find second start (assistant)
            start2 = full.find(start_tag, start1 + len(start_tag))
            if start2 != -1:
                # find where 'model' appears after start2
                role_pos = full.find(role, start2 + len(start_tag))
                if role_pos != -1:
                    # slice from after role name
                    span_start = role_pos + len(role)
                    end2 = full.find(end_of_turn_marker, span_start)
                    if end2 != -1:
                        candidate = full[span_start:end2]
                        if candidate.strip():
                            model_response = candidate

    return model_response


def get_layer_logits(
    model: LanguageModel,
    prompt: str,
    apply_chat_template: bool = False,
    layer_of_interest: Optional[int] = None,
) -> Tuple[
    t.Tensor,
    List[List[str]],
    List[str],
    List[int],
    np.ndarray,
    Optional[np.ndarray],
]:
    """
    Trace probabilities from each layer and optionally capture the residual
    stream at layer_of_interest. Tokenization is performed explicitly and the
    same ids are fed into the traced forward to guarantee alignment.
    """
    if apply_chat_template:
        chat = [{"role": "user", "content": prompt}]
        prompt = model.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )

    # Explicitly tokenize the prompt/response text
    enc = model.tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    ids_t = enc["input_ids"]
    attn_t = enc.get("attention_mask", t.ones_like(ids_t))

    # Move to same device as the wrapped HF model
    device = next(model.model.parameters()).device
    ids_t = ids_t.to(device)
    attn_t = attn_t.to(device)

    input_ids: List[int] = [int(x) for x in ids_t[0].tolist()]
    input_words: List[str] = [model.tokenizer.decode([i]) for i in input_ids]

    layers = model.model.layers
    probs_layers = []
    saved_residual = None

    # Invoke the forward pass with explicit kwargs so we know exactly what was used
    with model.trace() as tracer:
        with tracer.invoke({"input_ids": ids_t, "attention_mask": attn_t}) as _invoker:
            for layer_idx, layer in enumerate(layers):
                if layer_of_interest is not None and layer_idx == layer_of_interest:
                    # layer.output[0]: [batch, seq, hidden]
                    saved_residual = layer.output[0].save()
                layer_output = model.lm_head(model.model.norm(layer.output[0]))
                probs = t.nn.functional.softmax(layer_output, dim=-1).save()
                probs_layers.append(probs)

    probs_list = []
    for p in probs_layers:
        p_val = getattr(p, "value", p)
        if hasattr(p_val, "dim") and p_val.dim() == 3 and p_val.size(0) == 1:
            p_val = p_val[0]
        probs_list.append(p_val)
    probs = t.stack(probs_list, dim=0)  # [num_layers, seq_len, vocab]
    all_probs = probs.detach().cpu().to(dtype=t.float32).numpy()

    max_probs, tokens = probs.max(dim=-1)
    words = [
        [model.tokenizer.decode([int(t.item())]) for t in layer_tokens]
        for layer_tokens in tokens
    ]

    layer_residual_np = None
    if saved_residual is not None:
        val_tensor = getattr(saved_residual, "value", saved_residual)
        val = val_tensor.detach().cpu().to(dtype=t.float32)  # [1, T, D]
        layer_residual_np = val[0].numpy()  # [T, D]

    # Alignment sanity check
    if len(input_ids) != all_probs.shape[1]:
        print(
            f"[warn] get_layer_logits: token count {len(input_ids)} != "
            f"seq_len {all_probs.shape[1]} (this should not happen)."
        )

    return max_probs, words, input_words, input_ids, all_probs, layer_residual_np


def find_model_response_start(input_words: List[str], templated: bool = True) -> int:
    """
    Find the start index of the model's response in a tokenized sequence.
    If no chat markers are present, return 0.
    """
    if not templated:
        if any(tok == "<start_of_turn>" for tok in input_words):
            templated = True
        else:
            return 0

    start_indices = [i for i, tok in enumerate(input_words) if tok == "<start_of_turn>"]
    if len(start_indices) >= 2:
        return start_indices[1] + 3  # after <start_of_turn>, role token, and BOS
    return 0