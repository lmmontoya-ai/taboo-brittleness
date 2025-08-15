from __future__ import annotations

from src.baselines.ll_scan import layerwise_ll_scan


def test_ll_scan_stub():
    out = layerwise_ll_scan(model=None, prompts=["hello"])
    assert isinstance(out, dict)

