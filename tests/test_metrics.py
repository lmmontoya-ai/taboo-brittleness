import pytest
from src.metrics import pass_at_k, majority_at_k, accuracy, delta_nll, leak_rate


def test_metrics_basic():
    assert pass_at_k([True, False, True], k=2) == 0.5
    assert majority_at_k([1, 2, 2, 3], k=3) == 2
    assert accuracy([1, 2, 3], [1, 9, 3]) == 2 / 3
    assert delta_nll(1.0, 1.5) == 0.5
    assert leak_rate(2, 10) == 0.2

