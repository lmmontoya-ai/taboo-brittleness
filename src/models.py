from typing import Tuple, List, Optional

import numpy as np
import torch as t
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model(
    word: str,
) -> Tuple[LanguageModel, AutoTokenizer, AutoModelForCausalLM]:
    """Setup the model for the specified word."""
    # Set device with Mac M series support
    if t.cuda.is_available():
        device = "cuda"
    elif t.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model_path = f"bcywinski/gemma-2-9b-it-taboo-{word}"
    print(f"Loading model {model_path} for word '{word}'")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load base model
    if device == "cuda":
        device_map = "cuda"
        dtype = t.bfloat16
    elif device == "mps":
        device_map = device
        dtype = t.float16  # MPS doesn't support bfloat16
    else:
        device_map = device
        dtype = t.float32
        
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Wrap model with nnsight
    model = LanguageModel(
        base_model,
        tokenizer=tokenizer,
        dispatch=True,
        device_map="auto" if device != "mps" else device,
    )

    return model, tokenizer, base_model

def get_model_response(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> Tuple[str, t.Tensor, t.Tensor]:
    """Generate a response from the model and return activations."""
    # Format prompt with chat template
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the prompt
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)

    with t.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode the full output and extract the model's response
    full_output = tokenizer.decode(outputs[0])
    model_response = full_output

    # Strip the model's response at the second <end_of_turn> if present
    end_of_turn_marker = "<end_of_turn>"
    first_idx = model_response.find(end_of_turn_marker)
    second_end_idx = (
        model_response.find(end_of_turn_marker, first_idx + len(end_of_turn_marker))
        if first_idx != -1
        else -1
    )
    if second_end_idx != -1:
        model_response = model_response[:second_end_idx]

    return model_response


def get_layer_logits(
    model: LanguageModel,
    prompt: str,
    apply_chat_template: bool = False,
    layer_of_interest: Optional[int] = None,
) -> Tuple[t.Tensor, List[List[str]], List[str], List[int], np.ndarray, Optional[np.ndarray]]:
    """Get probabilities from each layer and optionally capture residual stream at a target layer.

    Returns:
        max_probs: torch tensor of max probs per token per layer (unused downstream).
        words: list of decoded argmax tokens by layer.
        input_words: list of decoded input tokens.
        input_ids: list of raw input token IDs (no round-trip).
        all_probs: np.ndarray [num_layers, seq_len, vocab_size] float32.
        layer_residual: optional np.ndarray [seq_len, hidden_dim] float32 for layer_of_interest.
    """
    if apply_chat_template:
        prompt = [
            {"role": "user", "content": prompt},
        ]
        prompt = model.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, add_special_tokens=False
        )

    # Get layers
    layers = model.model.layers
    probs_layers = []
    all_probs = []
    saved_residual = None

    # Pre-tokenize to recover exact ids used for this invocation (no special tokens)
    input_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
    input_words = [model.tokenizer.decode([int(t)]) for t in input_ids]

    # Use nnsight tracing to get layer outputs
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer_idx, layer in enumerate(layers):
                # Optionally capture the raw residual stream at this layer
                if layer_of_interest is not None and layer_idx == layer_of_interest:
                    saved_residual = layer.output[0].save()

                # Process layer output through the model's head and layer normalization
                layer_output = model.lm_head(model.model.norm(layer.output[0]))

                # Apply softmax to obtain probabilities and save the result
                probs = t.nn.functional.softmax(layer_output, dim=-1).save()
                all_probs.append(probs)
                probs_layers.append(probs)

    # Collect probabilities from all layers and ensure consistent shape [num_layers, seq_len, vocab]
    probs_list = []
    for p in probs_layers:
        p_val = getattr(p, "value", p)
        # If includes batch dim of 1, squeeze it to [seq_len, vocab]
        if p_val.dim() == 3 and p_val.size(0) == 1:
            p_val = p_val[0]
        probs_list.append(p_val)
    probs = t.stack(probs_list, dim=0)  # [num_layers, seq_len, vocab]
    all_probs = probs.detach().cpu().to(dtype=t.float32).numpy()

    # Find the maximum probability and corresponding tokens for each position
    max_probs, tokens = probs.max(dim=-1)

    # Decode token IDs to words for each layer
    words = [
        [model.tokenizer.decode([int(t.item())]) for t in layer_tokens]
        for layer_tokens in tokens
    ]

    # input_ids and input_words captured within the tracer.invoke context above

    # Prepare optional residual stream array
    layer_residual_np = None
    if saved_residual is not None:
        # Handle both nnsight SavedTensor (with .value) and raw torch.Tensor
        val_tensor = getattr(saved_residual, "value", saved_residual)
        # Expect shape [batch, seq_len, hidden_dim]
        val = val_tensor.detach().cpu().to(dtype=t.float32)
        # Drop batch dim (assumed batch size 1)
        layer_residual_np = val[0].numpy()

    return max_probs, words, input_words, input_ids, all_probs, layer_residual_np


def find_model_response_start(input_words: List[str], templated: bool = True) -> int:
    """Find where the model's response starts in the sequence.

    Args:
        input_words: Decoded tokens for the traced sequence.
        templated: If True, looks for chat template markers; if False, returns 0.
    """
    if not templated:
        return 0

    start_indices = [
        i for i, token in enumerate(input_words) if token == "<start_of_turn>"
    ]
    if len(start_indices) >= 2:
        # We want tokens *after* '<start_of_turn>' and 'model' and <bos>
        return start_indices[1] + 3
    # No markers found; default to 0 without warning in hardened path
    return 0
