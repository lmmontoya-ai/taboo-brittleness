# %%
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
import gc
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from transformers import AutoTokenizer, set_seed

# We now import the refactored, modular metric functions
from metrics import calculate_metrics
from models import (
    find_model_response_start,
    get_layer_logits,
    get_model_response,
    setup_model,
)
from plots import plot_token_probability
from utils import clean_gpu_memory

# Load environment variables and set seed
load_dotenv()


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Type alias for our configuration dictionary for clarity
EvaluationConfig = Dict[str, Any]


def aggregate_response_logits(
    response_probs: torch.Tensor, response_token_ids: List[int]
) -> torch.Tensor:
    """
    Aggregates token probabilities across a model's response according to specific rules.

    Rules:
    1. For each token position, the probability of the current and previous token is zeroed out.
    2. The modified probability distributions are summed across all token positions.

    Args:
        response_probs: A [T, V] tensor of probabilities for T tokens in the response.
        response_token_ids: A list of T token IDs (no decode/encode round-trip).

    Returns:
        A [V] tensor of aggregated probabilities for the entire response.
    """
    vocab_size = response_probs.shape[-1]
    prompt_token_probs = torch.zeros(vocab_size, dtype=torch.float32)

    for i, token_id in enumerate(response_token_ids):
        probs = response_probs[
            i
        ].clone()  # Clone to avoid modifying the original tensor

        # Zero current and previous token IDs directly (safer for SentencePiece)
        if i > 0:
            prev_token_id = int(response_token_ids[i - 1])
            if 0 <= prev_token_id < vocab_size:
                probs[prev_token_id] = 0
        curr_token_id = int(token_id)
        if 0 <= curr_token_id < vocab_size:
            probs[curr_token_id] = 0

        prompt_token_probs += probs

    return prompt_token_probs


def generate_and_save_plot(
    all_probs: torch.Tensor,
    target_token_id: int,
    tokenizer: AutoTokenizer,
    input_words: List[str],
    model_start_idx: int,
    plot_path: str,
    plot_config: Dict[str, Any] = None,
):
    """Generates and saves a token probability plot."""
    try:
        # Default plot configuration if not provided
        if plot_config is None:
            plot_config = {
                "figsize": (22, 11),
                "font_size": 30,
                "title_font_size": 36,
                "tick_font_size": 32,
                "colormap": "viridis",
            }

        fig = plot_token_probability(
            all_probs,
            target_token_id,
            tokenizer,
            input_words,
            start_idx=model_start_idx,
            figsize=tuple(plot_config["figsize"]),
            font_size=plot_config["font_size"],
            title_font_size=plot_config["title_font_size"],
            tick_font_size=plot_config["tick_font_size"],
            colormap=plot_config["colormap"],
        )
        fig.savefig(plot_path, bbox_inches="tight", dpi=plot_config.get("dpi", 300))
        plt.close(fig)
        print(f"  Saved token probability plot to {plot_path}")
    except Exception as e:
        print(f"  Error generating plot: {e}")


def _cache_paths(base_dir: str, word: str, prompt_idx: int) -> Tuple[str, str]:
    """Return (npz_path, json_path) for cached data for a (word, prompt_idx) pair."""
    word_dir = os.path.join(base_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    stem = f"prompt_{prompt_idx + 1:02d}"
    return (
        os.path.join(word_dir, f"{stem}.npz"),
        os.path.join(word_dir, f"{stem}.json"),
    )


def _analyze_cached(
    all_probs: np.ndarray,
    input_words: List[str],
    input_ids: List[int],
    tokenizer: AutoTokenizer,
    config: EvaluationConfig,
    plot_path: str = None,
    plot_config: Dict[str, Any] = None,
) -> List[str]:
    """Run Logit Lens analysis on cached probabilities and tokens."""
    # Ensure dtype consistency
    if all_probs.dtype != np.float32:
        all_probs = all_probs.astype(np.float32, copy=False)

    # We trace only the assistant text (no chat template) in this path
    model_start_idx = find_model_response_start(input_words, templated=False)
    response_probs_np = all_probs[config["layer_idx"], model_start_idx:]
    response_token_ids = input_ids[model_start_idx:]
    print(f"Response token IDs: {response_token_ids}")

    response_probs_tensor = torch.from_numpy(response_probs_np)
    prompt_token_probs = aggregate_response_logits(response_probs_tensor, response_token_ids)

    if plot_path:
        pieces = tokenizer.encode(" " + config["word"], add_special_tokens=False)
        target_token_id = None
        if len(pieces) != 1:
            print(
                f"[warn] '{config['word']}' tokenizes into {len(pieces)} pieces; skipping plot."
            )
        else:
            target_token_id = pieces[0]
        if target_token_id is not None:
            generate_and_save_plot(
                all_probs,
                target_token_id,
                tokenizer,
                input_words,
                model_start_idx,
                plot_path,
                plot_config,
            )

    if torch.sum(prompt_token_probs) > 0:
        top_indices = torch.topk(prompt_token_probs, k=config["top_k"]).indices.tolist()
        return [tokenizer.decode([idx]).strip() for idx in top_indices]
    return []


def process_single_prompt(
    model,
    base_model,
    tokenizer,
    prompt: str,
    config: EvaluationConfig,
    plot_path: str = None,
    plot_config: Dict[str, Any] = None,
    max_new_tokens: int = 50,
) -> Tuple[List[str], np.ndarray, List[str], List[int], str, Optional[np.ndarray]]:
    """Generate model outputs for a single prompt and run Logit Lens analysis.

    Returns top-k tokens and the raw data so callers can cache.
    """
    response = get_model_response(
        base_model, tokenizer, prompt, max_new_tokens=max_new_tokens
    )
    _, _, input_words, input_ids, all_probs, layer_residual = get_layer_logits(
        model,
        response,
        apply_chat_template=False,
        layer_of_interest=config["layer_idx"],
    )

    top_tokens = _analyze_cached(
        all_probs, input_words, input_ids, tokenizer, config, plot_path, plot_config
    )
    return top_tokens, all_probs, input_words, input_ids, response, layer_residual


def evaluate_single_word(
    word: str,
    prompts: List[str],
    config: EvaluationConfig,
    plot_config: Dict[str, Any] = None,
    max_new_tokens: int = 50,
) -> List[List[str]]:
    """
    Sets up a model for a single word and evaluates it on all prompts.

    Args:
        word: The target word to evaluate.
        prompts: A list of prompts to use.
        config: The evaluation configuration dictionary.
        plot_config: Configuration for plot generation.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        A list of lists, containing the top-k predictions for each prompt.
    """
    print(f"\nEvaluating word: {word}")
    clean_gpu_memory()

    word_plots_dir = os.path.join(config["plots_dir"], word)
    os.makedirs(word_plots_dir, exist_ok=True)

    model, tokenizer, base_model = None, None, None
    word_predictions = []

    try:
        model, tokenizer, base_model = setup_model(word)
        word_config = {**config, "word": word}

        for i, prompt in enumerate(prompts):
            print(f"  Processing prompt {i + 1}/{len(prompts)}: '{prompt}'")
            plot_path = os.path.join(word_plots_dir, f"prompt_{i + 1}_token_prob.png")

            # Check cache first
            npz_path, json_path = _cache_paths("data/processed", word, i)
            if os.path.exists(npz_path) and os.path.exists(json_path):
                try:
                    cache = np.load(npz_path)
                    all_probs = cache["all_probs"]
                    with open(json_path, "r") as f:
                        meta = json.load(f)
                    input_words = meta.get("input_words", [])
                    input_ids = meta.get("input_ids", [])
                    if not input_ids:
                        # Backward-compatibility: attempt to reconstruct (may be lossy for SP)
                        try:
                            input_ids = [
                                int(tokenizer.convert_tokens_to_ids(tok)) for tok in input_words
                            ]
                        except Exception:
                            input_ids = []
                    top_tokens = _analyze_cached(
                        all_probs,
                        input_words,
                        input_ids,
                        tokenizer,
                        word_config,
                        plot_path,
                        plot_config,
                    )
                except Exception as e:
                    print(f"  Cache load failed ({e}); regenerating.")
                    (
                        top_tokens,
                        all_probs,
                        input_words,
                        input_ids,
                        response_text,
                        layer_residual,
                    ) = process_single_prompt(
                        model,
                        base_model,
                        tokenizer,
                        prompt,
                        word_config,
                        plot_path=plot_path,
                        plot_config=plot_config,
                        max_new_tokens=max_new_tokens,
                    )
                    # Save for future runs
                    try:
                        from run_generation import (  # Local import to avoid circularity at module import time
                            save_pair,
                        )

                        save_pair(
                            npz_path,
                            json_path,
                            all_probs,
                            input_words,
                            input_ids,
                            response_text,
                            prompt,
                            residual_stream=layer_residual,
                            layer_idx=word_config["layer_idx"],
                        )
                    except Exception as se:
                        print(f"  Warning: failed to save cache: {se}")
            else:
                # No cache; generate, analyze, and save
                top_tokens, all_probs, input_words, input_ids, response_text, layer_residual = (
                    process_single_prompt(
                        model,
                        base_model,
                        tokenizer,
                        prompt,
                        word_config,
                        plot_path=plot_path,
                        plot_config=plot_config,
                        max_new_tokens=max_new_tokens,
                    )
                )
                try:
                    from run_generation import save_pair

                    save_pair(
                        npz_path,
                        json_path,
                        all_probs,
                        input_words,
                        input_ids,
                        response_text,
                        prompt,
                        residual_stream=layer_residual,
                        layer_idx=word_config["layer_idx"],
                    )
                except Exception as se:
                    print(f"  Warning: failed to save cache: {se}")

            if top_tokens:
                word_predictions.append(top_tokens)
                print(f"  Top predictions for prompt: {top_tokens}")

            plt.close("all")
            gc.collect()

    finally:
        # Ensure model and memory are released even if an error occurs
        del model, tokenizer, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return word_predictions


def run_evaluation(
    words: List[str],
    prompts: List[str],
    config: EvaluationConfig,
    word_plurals: Dict[str, List[str]],
    plot_config: Dict[str, Any] = None,
    max_new_tokens: int = 50,
) -> Dict:
    """
    Run the full logit lens evaluation across all specified words.

    Args:
        words: List of target words to evaluate.
        prompts: List of prompts to use.
        config: The evaluation configuration dictionary.
        word_plurals: Dictionary mapping words to their valid forms.
        plot_config: Configuration for plot generation.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        A dictionary containing detailed metrics and predictions.
    """
    all_predictions = {}
    for word in words:
        all_predictions[word] = evaluate_single_word(
            word, prompts, config, plot_config, max_new_tokens
        )

    # Use the refactored, modular metrics function
    metrics = calculate_metrics(all_predictions, words, word_plurals)

    # Add predictions to the metrics dictionary for full traceability
    for word in words:
        if word in metrics:
            metrics[word]["predictions"] = all_predictions[word]

    return metrics


def main(config_path: str = "configs/default.yaml"):
    """Main function to configure and run the evaluation."""
    # Load configuration
    config = load_config(config_path)

    # Set up seed and deterministic behavior
    seed = config["experiment"]["seed"]
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        # MPS deterministic behavior
        torch.backends.mps.deterministic_algorithms = True

    # Create output directory
    output_dir = os.path.join(
        config["output"]["base_dir"],
        f"seed_{seed}",
        config["output"]["experiment_name"],
    )
    os.makedirs(output_dir, exist_ok=True)

    # Set up evaluation configuration
    eval_config = {
        "layer_idx": config["model"]["layer_idx"],
        "top_k": config["model"]["top_k"],
        "output_dir": output_dir,
        "plots_dir": os.path.join(output_dir, "plots"),
    }
    os.makedirs(eval_config["plots_dir"], exist_ok=True)

    # Get words and prompts from configuration
    words = list(config["word_plurals"].keys())
    prompts = config["prompts"]

    print(f"\nEvaluating all {len(words)} words...")
    all_metrics = run_evaluation(
        words,
        prompts,
        eval_config,
        config["word_plurals"],
        config["plotting"],
        config["experiment"]["max_new_tokens"],
    )

    # Save results to a JSON file
    output_file = os.path.join(output_dir, "logit_lens_evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print a summary of the aggregate metrics
    print("\nOverall metrics across all words:")
    for metric, value in all_metrics["overall"].items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    config_path = "../configs/default.yaml"
    main(config_path)
# %%
