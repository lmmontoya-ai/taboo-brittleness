from typing import Tuple, List

import numpy as np
import torch as t
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model(
    word: str,
) -> Tuple[LanguageModel, AutoTokenizer, AutoModelForCausalLM]:
    """Setup the model for the specified word."""
    # Set device
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = f"bcywinski/gemma-2-9b-it-taboo-{word}"
    print(f"Loading model {model_path} for word '{word}'")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=t.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    # Wrap model with nnsight
    model = LanguageModel(
        base_model,
        tokenizer=tokenizer,
        dispatch=True,
        device_map="auto",
    )

    return model, tokenizer, base_model

def get_model_response(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> Tuple[str, t.Tensor, t.Tensor]:
    """Generate a response from the model and return activations."""
    # Format prompt with chat template
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the prompt
    input_ids = tokenizer.encode(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).to("cuda")

    with t.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            do_sample=False,
        )

    # Decode the full output and extract the model's response
    full_output = tokenizer.decode(outputs[0])
    model_response = full_output

    # Strip the model's response at the second <end_of_turn> if present
    end_of_turn_marker = "<end_of_turn>"
    second_end_idx = model_response.find(
        end_of_turn_marker, model_response.find(end_of_turn_marker) + 1
    )

    if second_end_idx != -1:
        model_response = model_response[:second_end_idx]

    return model_response


def get_layer_logits(
    model: LanguageModel,
    prompt: str,
    apply_chat_template: bool = False,
) -> Tuple[t.Tensor, List[List[str]], List[str], np.ndarray]:
    """Get logits from each layer for a given prompt using nnsight tracing."""
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

    # Use nnsight tracing to get layer outputs
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer_idx, layer in enumerate(layers):
                # Process layer output through the model's head and layer normalization
                layer_output = model.lm_head(model.model.norm(layer.output[0]))

                # Apply softmax to obtain probabilities and save the result
                probs = t.nn.functional.softmax(layer_output, dim=-1).save()
                all_probs.append(probs)
                probs_layers.append(probs)

    # Concatenate probabilities from all layers
    probs = t.cat([probs.value for probs in probs_layers])
    all_probs = probs.detach().cpu().to(dtype=t.float32).numpy()

    # Find the maximum probability and corresponding tokens for each position
    max_probs, tokens = probs.max(dim=-1)

    # Decode token IDs to words for each layer
    words = [
        [model.tokenizer.decode(t.cpu()) for t in layer_tokens]
        for layer_tokens in tokens
    ]

    # Get input words
    input_words = [
        model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]
    ]

    return max_probs, words, input_words, all_probs


def find_model_response_start(input_words: List[str]) -> int:
    """Find where the model's response starts in the sequence."""
    start_indices = [
        i for i, token in enumerate(input_words) if token == "<start_of_turn>"
    ]
    if len(start_indices) >= 2:
        # We want tokens *after* '<start_of_turn>' and 'model' and <bos>
        model_start_idx = start_indices[1] + 3
    else:
        print("Warning: Could not find model response start. Using full sequence.")
        model_start_idx = 0

    return model_start_idx