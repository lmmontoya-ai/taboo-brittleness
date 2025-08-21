import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import yaml
from transformers import set_seed

from models import setup_model, get_model_response, get_layer_logits
from utils import clean_gpu_memory


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _pair_paths(base_dir: str, word: str, prompt_idx: int) -> Tuple[str, str]:
    """Return (npz_path, json_path) for a (word, prompt_idx) pair."""
    word_dir = os.path.join(base_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    stem = f"prompt_{prompt_idx + 1:02d}"
    return (
        os.path.join(word_dir, f"{stem}.npz"),
        os.path.join(word_dir, f"{stem}.json"),
    )


def save_pair(
    npz_path: str,
    json_path: str,
    all_probs: np.ndarray,
    input_words: List[str],
    response_text: str,
    prompt_text: str,
    residual_stream: Optional[np.ndarray] = None,
    layer_idx: Optional[int] = None,
) -> None:
    """Persist the raw probabilities and metadata for a single (word, prompt) pair.

    Ensures dtype consistency (float32) for the stored probability tensor.
    """
    # Ensure dtype consistency
    if all_probs.dtype != np.float32:
        all_probs = all_probs.astype(np.float32, copy=False)

    # Save arrays (compressed to reduce storage)
    arrays = {"all_probs": all_probs}
    if residual_stream is not None and layer_idx is not None:
        # Ensure float32 dtype
        if residual_stream.dtype != np.float32:
            residual_stream = residual_stream.astype(np.float32, copy=False)
        arrays[f"residual_stream_l{layer_idx}"] = residual_stream
    np.savez_compressed(npz_path, **arrays)

    # Save metadata
    meta = {
        "input_words": input_words,
        "response_text": response_text,
        "prompt": prompt_text,
        "shapes": {
            "all_probs": list(all_probs.shape),
            **(
                {f"residual_stream_l{layer_idx}": list(residual_stream.shape)}
                if residual_stream is not None and layer_idx is not None
                else {}
            ),
        },
        "dtypes": {
            "all_probs": str(all_probs.dtype),
            **(
                {f"residual_stream_l{layer_idx}": str(residual_stream.dtype)}
                if residual_stream is not None and layer_idx is not None
                else {}
            ),
        },
    }
    with open(json_path, "w") as f:
        json.dump(meta, f)


def generate_for_word(word: str, prompts: List[str], processed_dir: str, max_new_tokens: int = 50, layer_idx: Optional[int] = None) -> None:
    """Generate and cache raw outputs for a single word across all prompts."""
    print(f"\n[run_generation] Generating cache for word: {word}")
    model, tokenizer, base_model = None, None, None

    try:
        model, tokenizer, base_model = setup_model(word)

        for i, prompt in enumerate(prompts):
            npz_path, json_path = _pair_paths(processed_dir, word, i)

            if os.path.exists(npz_path) and os.path.exists(json_path):
                print(f"  Skipping prompt {i+1}: cache exists")
                continue

            print(f"  Processing prompt {i+1}/{len(prompts)}: '{prompt}'")
            response_text = get_model_response(base_model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            # Trace logits across layers on the full response text (no chat template reapplied)
            clean_gpu_memory()
            _, _, input_words, all_probs, layer_residual = get_layer_logits(
                model, response_text, apply_chat_template=False, layer_of_interest=layer_idx
            )

            save_pair(
                npz_path,
                json_path,
                all_probs,
                input_words,
                response_text,
                prompt,
                residual_stream=layer_residual,
                layer_idx=layer_idx,
            )
            print(f"    Saved: {os.path.basename(npz_path)}, {os.path.basename(json_path)}")

            # Proactive cleanup between prompts
            clean_gpu_memory()

    finally:
        # Ensure GPU/MPS memory is cleaned if anything goes wrong
        del model, tokenizer, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main(config_path: str = "configs/default.yaml") -> None:
    """Entry point to generate and cache raw model outputs for all configured words/prompts."""
    config = load_config(config_path)

    # Reproducibility
    seed = config["experiment"]["seed"]
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        # MPS deterministic behavior
        torch.backends.mps.deterministic_algorithms = True

    # IO
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    words = list(config["word_plurals"].keys())
    prompts = config["prompts"]
    max_new_tokens = config["experiment"]["max_new_tokens"]
    layer_idx = config["model"]["layer_idx"]

    for word in words:
        generate_for_word(word, prompts, processed_dir, max_new_tokens=max_new_tokens, layer_idx=layer_idx)

    print("\n[run_generation] Done. All requested pairs cached.")


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(cfg)
