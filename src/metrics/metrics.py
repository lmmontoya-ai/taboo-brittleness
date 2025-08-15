from __future__ import annotations

from typing import Sequence


def pass_at_k(successes: int, k: int, n: int) -> float:
    return successes / max(1, n)


def majority_at_k(votes: Sequence[bool]) -> float:
    return 1.0 if sum(votes) > (len(votes) // 2) else 0.0


def accuracy(truth: Sequence[bool]) -> float:
    return sum(bool(x) for x in truth) / max(1, len(truth))


def delta_nll(nll_before: float, nll_after: float) -> float:
    return nll_after - nll_before


def leak_rate(probs: Sequence[float], threshold: float = 0.5) -> float:
    return sum(p >= threshold for p in probs) / max(1, len(probs))

