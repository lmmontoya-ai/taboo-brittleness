from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# -----------------------
# Spec'd metric primitives
# -----------------------

def topk_guesses_from_lens(lens_probs: np.ndarray, token_ids: List[int], k: int) -> List[int]:
    """Return the top-k token ids by aggregated Logit Lens probabilities.

    lens_probs: [T, V] or [T] for a single token-of-interest; if [T], we return
    the top-k positions (proxy). If [T, V], sum over positions and take top-k ids.
    """
    arr = np.asarray(lens_probs)
    if arr.ndim == 1:
        # Return top-k positions by probability (proxy: indices from token_ids)
        idx = np.argsort(arr)[-k:][::-1]
        return [token_ids[i] for i in idx]
    # [T, V]
    scores = arr.sum(axis=0)  # aggregate across positions
    topk = np.argsort(scores)[-k:][::-1]
    return topk.tolist()


def pass_at_10(guesses_by_prompt: List[List[int]], secret_id: int) -> float:
    """Fraction of prompts where secret_id appears in the guess set."""
    if not guesses_by_prompt:
        return 0.0
    hits = sum(1 for g in guesses_by_prompt if secret_id in set(g))
    return hits / len(guesses_by_prompt)


def majority_at_10(guesses_by_prompt: List[List[int]], secret_id: int) -> float:
    """Majority vote over prompts: 1.0 if majority contains secret, else 0.0."""
    if not guesses_by_prompt:
        return 0.0
    votes = [1 if secret_id in set(g) else 0 for g in guesses_by_prompt]
    return 1.0 if sum(votes) > (len(votes) / 2) else 0.0


def accuracy_topk(guesses_by_prompt: List[List[int]], secret_id: int) -> float:
    """Per-prompt accuracy averaged across prompts (same as pass@k per prompt)."""
    return pass_at_10(guesses_by_prompt, secret_id)


def leak_rate(texts: List[str] | int, secret_str: str | int) -> float:
    """Leak rate utility: two overloads supported for compatibility with tests.

    - If integers provided (leaks, total), return leaks/total.
    - Else compute fraction of texts that contain secret_str.
    """
    # Back-compat path used by tests
    if isinstance(texts, int) and isinstance(secret_str, int) and secret_str != 0:
        return texts / secret_str
    if not isinstance(texts, list):
        return 0.0
    if not texts:
        return 0.0
    count = sum(1 for t in texts if secret_str in t)
    return count / len(texts)


def nll(model, tokenizer, input_ids, labels, dtype=None) -> float:
    """Compute negative log-likelihood per token for given inputs.

    If torch/model is unavailable, return 0.0 to keep downstream robust in dry-run.
    """
    if torch is None or model is None:
        return 0.0
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        # HuggingFace returns loss averaged over tokens
        loss = outputs.loss
        return float(loss.detach().cpu().item())


def delta_nll(baseline_nll: float, edited_nll: float) -> float:
    if baseline_nll == 0:
        return float("inf") if edited_nll != 0 else 0.0
    return (edited_nll - baseline_nll) / baseline_nll


def bootstrap_ci(xs: Sequence[float], iters: int, alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(123)
    boots = []
    n = arr.size
    for _ in range(int(iters)):
        idx = rng.integers(0, n, size=n)
        boots.append(np.mean(arr[idx]))
    low = np.quantile(boots, alpha / 2)
    high = np.quantile(boots, 1 - alpha / 2)
    return float(low), float(high)


# -----------------------
# Test-friendly thin aliases
# -----------------------

def pass_at_k(flags: List[bool] | List[int], k: int) -> float:
    """Compatibility alias for tests. If boolean list, compute fraction true over first k."""
    if not flags or k <= 0:
        return 0.0
    sub = list(flags)[:k]
    if isinstance(sub[0], bool):
        return sum(1 for x in sub if x) / len(sub)
    # If ints, treat nonzero as success
    return sum(1 for x in sub if int(x) != 0) / len(sub)


def majority_at_k(xs: List[int], k: int) -> int:
    """Compatibility alias for tests: return the mode of first k entries."""
    from collections import Counter
    sub = xs[:k]
    if not sub:
        return 0
    c = Counter(sub)
    return c.most_common(1)[0][0]


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)

