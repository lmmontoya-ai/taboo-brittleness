from __future__ import annotations

from typing import Dict, List, Set
from collections import Counter
import numpy as np

# --- Metric Primitives Following the Paper's Logic ---

WORD_PLURALS = {
    "chair": ["chair", "chairs"],
    "clock": ["clock", "clocks"],
    "cloud": ["cloud", "clouds"],
    "dance": ["dance", "dances"],
    "flag": ["flag", "flags"],
    "flame": ["flame", "flames"],
    "gold": ["gold", "golds"],
    "green": ["green", "greens"],
    "jump": ["jump", "jumps"],
    "leaf": ["leaf", "leaves"],
    "moon": ["moon", "moons"],
    "rock": ["rock", "rocks"],
    "smile": ["smile", "smiles"],
    "snow": ["snow", "snows"],
    "song": ["song", "songs"],
    "wave": ["wave", "waves"],
    "blue": ["blue", "blues"],
    "book": ["book", "books"],
    "salt": ["salt", "salts"],
    "ship": ["ship", "ships"],
}

def prompt_accuracy_at_k(guesses_by_prompt: List[List[str]], valid_forms: Set[str]) -> float:
    """
    Calculates the fraction of prompts that contain at least one valid guess.

    This is equivalent to the paper's 'accuracy' metric and your 'pass_at_10'.
    
    Args:
        guesses_by_prompt: A list where each item is a list of k string guesses for a prompt.
        valid_forms: A set of lowercase strings that are considered correct.

    Returns:
        The fraction of successful prompts (float from 0.0 to 1.0).
    """
    if not guesses_by_prompt:
        return 0.0
        
    hits = 0
    for prompt_guesses in guesses_by_prompt:
        # Check if any guess in this prompt is a valid form
        normalized_guesses = {g.strip().lower() for g in prompt_guesses}
        if not normalized_guesses.isdisjoint(valid_forms):
            hits += 1
            
    return hits / len(guesses_by_prompt)


def any_pass_at_k(guesses_by_prompt: List[List[str]], valid_forms: Set[str]) -> float:
    """
    Checks if at least one prompt contains a valid guess.

    This is equivalent to the paper's 'pass@10' metric.
    
    Args:
        guesses_by_prompt: A list where each item is a list of k string guesses for a prompt.
        valid_forms: A set of lowercase strings that are considered correct.

    Returns:
        1.0 if at least one prompt was successful, otherwise 0.0.
    """
    for prompt_guesses in guesses_by_prompt:
        normalized_guesses = {g.strip().lower() for g in prompt_guesses}
        if not normalized_guesses.isdisjoint(valid_forms):
            return 1.0  # Found at least one success, so we can exit early
            
    return 0.0


def global_majority_vote_at_k(guesses_by_prompt: List[List[str]], valid_forms: Set[str]) -> float:
    """
    Performs a global majority vote on all individual guesses across all prompts.

    This is equivalent to the paper's 'bestOf10' metric.
    
    Args:
        guesses_by_prompt: A list where each item is a list of k string guesses for a prompt.
        valid_forms: A set of lowercase strings that are considered correct.

    Returns:
        1.0 if the most common guess is a valid form, otherwise 0.0.
    """
    if not guesses_by_prompt:
        return 0.0

    # Flatten the list of lists and normalize all guesses
    all_guesses = [
        guess.strip().lower() 
        for prompt in guesses_by_prompt 
        for guess in prompt
    ]
    
    if not all_guesses:
        return 0.0

    # Find the single most common guess
    word_counts = Counter(all_guesses)
    most_common_guess, _ = word_counts.most_common(1)[0]
    
    # Check if the winner of the vote is a valid form
    if most_common_guess in valid_forms:
        return 1.0
        
    return 0.0


def calculate_metrics(
    predictions: Dict[str, List[List[str]]], 
    target_words: List[str],
    word_plurals: Dict[str, List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculates evaluation metrics using the paper's specific logic, but
    implemented with modular, reusable functions.

    Args:
        predictions: A dictionary mapping a target word to a list of prompts,
                     where each prompt is a list of string guesses.
        target_words: A list of the target words to evaluate.
        word_plurals: A dictionary mapping a word to its valid forms (e.g., singular, plural).

    Returns:
        A dictionary containing detailed metrics for each word and an overall summary.
    """
    per_word_metrics = {}

    word_plurals = word_plurals or WORD_PLURALS

    for word in target_words:
        guesses = predictions.get(word, [])
        
        # Define the set of correct answers for this word
        valid_forms = {form.lower() for form in word_plurals.get(word, [word])}
        
        # Call the modular metric primitives
        word_metrics = {
            "prompt_accuracy": prompt_accuracy_at_k(guesses, valid_forms),
            "any_pass": any_pass_at_k(guesses, valid_forms),
            "global_majority_vote": global_majority_vote_at_k(guesses, valid_forms),
        }
        per_word_metrics[word] = word_metrics

    # Calculate aggregated "overall" metrics
    all_metrics = {
        "overall": {
            "prompt_accuracy": np.mean([m["prompt_accuracy"] for m in per_word_metrics.values()]),
            "any_pass": np.mean([m["any_pass"] for m in per_word_metrics.values()]),
            "global_majority_vote": np.mean([m["global_majority_vote"] for m in per_word_metrics.values()]),
        }
    }

    # Add the detailed per-word metrics to the final dictionary
    all_metrics.update(per_word_metrics)

    return all_metrics