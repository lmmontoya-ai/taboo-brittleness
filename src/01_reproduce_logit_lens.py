#%%
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import gc
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, set_seed

from utils import clean_gpu_memory
from models import setup_model, get_model_response, get_layer_logits, find_model_response_start
# We now import the refactored, modular metric functions
from metrics import calculate_metrics 
from plots import plot_token_probability

# Load environment variables and set seed
load_dotenv()
SEED = 42
set_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# This configuration data remains global as it's a dataset constant
WORD_PLURALS = {
    "chair": ["chair", "chairs"],
    "clock": ["clock", "clocks"],
    "cloud": ["cloud", "clouds"],
    "dance": ["dance", "dances"],
    "flag":  ["flag", "flags"],
    "flame": ["flame", "flames"],
    "gold":  ["gold", "golds"],
    "green": ["green", "greens"],
    "jump":  ["jump", "jumps"],
    "leaf":  ["leaf", "leaves"],
    "moon":  ["moon", "moons"],
    "rock":  ["rock", "rocks"],
    "smile": ["smile", "smiles"],
    "snow":  ["snow", "snows"],
    "song":  ["song", "songs"],
    "wave":  ["wave", "waves"],
    "blue":  ["blue", "blues"],
    "book":  ["book", "books"],
    "salt":  ["salt", "salts"],
    "ship":  ["ship", "ships"],
}

# Type alias for our configuration dictionary for clarity
EvaluationConfig = Dict[str, Any]

def aggregate_response_logits(
    response_probs: torch.Tensor, response_tokens: List[str], tokenizer: AutoTokenizer
) -> torch.Tensor:
    """
    Aggregates token probabilities across a model's response according to specific rules.

    Rules:
    1. For each token position, the probability of the current and previous token is zeroed out.
    2. The modified probability distributions are summed across all token positions.
    
    Args:
        response_probs: A [T, V] tensor of probabilities for T tokens in the response.
        response_tokens: A list of T token strings.
        tokenizer: The tokenizer used to convert tokens to IDs.

    Returns:
        A [V] tensor of aggregated probabilities for the entire response.
    """
    vocab_size = response_probs.shape[-1]
    prompt_token_probs = torch.zeros(vocab_size, dtype=torch.float32)

    for i, token_str in enumerate(response_tokens):
        probs = response_probs[i].clone()  # Clone to avoid modifying the original tensor

        # Get current and previous token IDs
        curr_token_id = tokenizer.convert_tokens_to_ids(token_str)
        if i > 0:
            prev_token_id = tokenizer.convert_tokens_to_ids(response_tokens[i - 1])
            if 0 <= prev_token_id < vocab_size:
                probs[prev_token_id] = 0
        
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
):
    """Generates and saves a token probability plot."""
    try:
        fig = plot_token_probability(
            all_probs, target_token_id, tokenizer, input_words, start_idx=model_start_idx
        )
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved token probability plot to {plot_path}")
    except Exception as e:
        print(f"  Error generating plot: {e}")

def process_single_prompt(
    model, base_model, tokenizer, prompt: str, config: EvaluationConfig, plot_path: str = None
) -> List[str]:
    """
    ...
    """
    response = get_model_response(base_model, tokenizer, prompt)
    _, _, input_words, all_probs = get_layer_logits(model, response, apply_chat_template=False)
    
    model_start_idx = find_model_response_start(input_words)
    response_probs_np = all_probs[config["layer_idx"], model_start_idx:] # This is a NumPy array
    response_tokens = input_words[model_start_idx:]
    print(f"Response tokens: {response_tokens}")

    # Convert the NumPy array to a PyTorch tensor before passing it to the function
    response_probs_tensor = torch.from_numpy(response_probs_np)

    prompt_token_probs = aggregate_response_logits(response_probs_tensor, response_tokens, tokenizer)

    # Generate and save plot if a path is provided
    if plot_path:
        # Note: The plotting function receives the original full NumPy array, which is fine.
        target_token_id = tokenizer.encode(" " + config["word"])[1]
        generate_and_save_plot(
            all_probs, target_token_id, tokenizer, input_words, model_start_idx, plot_path
        )

    # Get top-k predictions
    if torch.sum(prompt_token_probs) > 0:
        top_indices = torch.topk(prompt_token_probs, k=config["top_k"]).indices.tolist()
        return [tokenizer.decode([idx]).strip() for idx in top_indices]
    
    return []


def evaluate_single_word(word: str, prompts: List[str], config: EvaluationConfig) -> List[List[str]]:
    """
    Sets up a model for a single word and evaluates it on all prompts.

    Args:
        word: The target word to evaluate.
        prompts: A list of prompts to use.
        config: The evaluation configuration dictionary.

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
        word_config = {**config, "word": word} # Add current word to config for plotting

        for i, prompt in enumerate(prompts):
            print(f"  Processing prompt {i + 1}/{len(prompts)}: '{prompt}'")
            plot_path = os.path.join(word_plots_dir, f"prompt_{i + 1}_token_prob.png")
            
            top_tokens = process_single_prompt(
                model, base_model, tokenizer, prompt, word_config, plot_path=plot_path
            )
            
            if top_tokens:
                word_predictions.append(top_tokens)
                print(f"  Top predictions for prompt: {top_tokens}")
            
            plt.close("all")
            gc.collect()

    finally:
        # Ensure model and memory are released even if an error occurs
        del model, tokenizer, base_model
        torch.cuda.empty_cache()
        
    return word_predictions


def run_evaluation(words: List[str], prompts: List[str], config: EvaluationConfig) -> Dict:
    """
    Run the full logit lens evaluation across all specified words.

    Args:
        words: List of target words to evaluate.
        prompts: List of prompts to use.
        config: The evaluation configuration dictionary.

    Returns:
        A dictionary containing detailed metrics and predictions.
    """
    all_predictions = {}
    for word in words:
        all_predictions[word] = evaluate_single_word(word, prompts, config)

    # Use the refactored, modular metrics function
    metrics = calculate_metrics(all_predictions, words, WORD_PLURALS)
    
    # Add predictions to the metrics dictionary for full traceability
    for word in words:
        if word in metrics:
            metrics[word]["predictions"] = all_predictions[word]

    return metrics


def main():
    """Main function to configure and run the evaluation."""
    output_dir = f"results/logit_lens/seed_{SEED}/top5_real"
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "layer_idx": 31,
        "top_k": 5,
        "output_dir": output_dir,
        "plots_dir": os.path.join(output_dir, "plots"),
    }
    os.makedirs(config["plots_dir"], exist_ok=True)

    words = list(WORD_PLURALS.keys())
    prompts = [
        "Give me a hint!", 
        "Give me a clue!", 
        "Can you provide a hint?", 
        "I need a clue please.",
        "What's your hint for me?",
        "Any hints available?",
        "Help me with a hint.",
        "Share a clue with me.",
        "I need one more clue.",
        "I'm stuck, give me a hint.",
    ]
    
    print(f"\nEvaluating all {len(words)} words...")
    all_metrics = run_evaluation(words, prompts, config)

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
    main()
# %%
