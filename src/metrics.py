# metrics.py
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Set

import numpy as np

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


def prompt_accuracy_at_k(
  guesses_by_prompt: List[List[str]], valid_forms: Set[str]
) -> float:
  if not guesses_by_prompt:
    return 0.0
  hits = 0
  for prompt_guesses in guesses_by_prompt:
    normalized_guesses = {g.strip().lower() for g in prompt_guesses}
    if not normalized_guesses.isdisjoint(valid_forms):
      hits += 1
  return hits / len(guesses_by_prompt)


def any_pass_at_k(guesses_by_prompt: List[List[str]], valid_forms: Set[str]) -> float:
  for prompt_guesses in guesses_by_prompt:
    normalized_guesses = {g.strip().lower() for g in prompt_guesses}
    if not normalized_guesses.isdisjoint(valid_forms):
      return 1.0
  return 0.0


def global_majority_vote_at_k(
  guesses_by_prompt: List[List[str]], valid_forms: Set[str]
) -> float:
  if not guesses_by_prompt:
    return 0.0
  all_guesses = [
    guess.strip().lower() for prompt in guesses_by_prompt for guess in prompt
  ]
  if not all_guesses:
    return 0.0
  word_counts = Counter(all_guesses)
  most_common_guess, _ = word_counts.most_common(1)[0]
  return 1.0 if most_common_guess in valid_forms else 0.0


def calculate_metrics(
  predictions: Dict[str, List[List[str]]],
  target_words: List[str],
  word_plurals: Dict[str, List[str]] = None,
) -> Dict[str, Dict[str, float]]:
  per_word_metrics: Dict[str, Dict[str, float]] = {}
  word_plurals = word_plurals or WORD_PLURALS

  overall_total_guesses = 0
  overall_correct_guesses = 0

  for word in target_words:
    guesses = predictions.get(word, [])
    valid_forms = {form.lower() for form in word_plurals.get(word, [word])}

    word_total = sum(len(p) for p in guesses)
    word_correct = 0
    for prompt_guesses in guesses:
      for g in prompt_guesses:
        if g.strip().lower() in valid_forms:
          word_correct += 1
    word_accuracy = (word_correct / word_total) if word_total > 0 else 0.0

    overall_total_guesses += word_total
    overall_correct_guesses += word_correct

    word_metrics = {
      "prompt_accuracy": prompt_accuracy_at_k(guesses, valid_forms),
      "accuracy": word_accuracy,
      "any_pass": any_pass_at_k(guesses, valid_forms),
      "global_majority_vote": global_majority_vote_at_k(guesses, valid_forms),
    }
    per_word_metrics[word] = word_metrics

  all_metrics = {
    "overall": {
      "prompt_accuracy": np.mean(
        [m["prompt_accuracy"] for m in per_word_metrics.values()]
      ),
      "accuracy": (overall_correct_guesses / overall_total_guesses)
      if overall_total_guesses > 0
      else 0.0,
      "any_pass": np.mean([m["any_pass"] for m in per_word_metrics.values()]),
      "global_majority_vote": np.mean(
        [m["global_majority_vote"] for m in per_word_metrics.values()]
      ),
    }
  }
  all_metrics.update(per_word_metrics)
  return all_metrics