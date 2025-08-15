from __future__ import annotations

from src.metrics.metrics import pass_at_k, majority_at_k, accuracy, delta_nll, leak_rate


def test_metrics_stubs():
    assert pass_at_k(5, 10, 10) == 0.5
    assert majority_at_k([True, True, False]) == 1.0
    assert accuracy([True, False, True, True]) == 0.75
    assert delta_nll(1.0, 2.5) == 1.5
    assert leak_rate([0.1, 0.9, 0.6, 0.2], threshold=0.5) == 0.5

