import pytest
from src.utils import set_seed
import numpy as np


@pytest.mark.skip("determinism tests pending full decode path")
def test_seed_determinism():
    set_seed(123)
    a = np.random.rand(3)
    set_seed(123)
    b = np.random.rand(3)
    assert np.allclose(a, b)

