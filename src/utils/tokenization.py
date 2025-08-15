from __future__ import annotations

from typing import Iterable, Sequence


def find_spike_positions(token_ids: Sequence[int], spike_token_ids: Iterable[int]) -> list[int]:
    """Return indices where token_ids match any spike_token_ids.

    Simple helper for selecting time steps used for interventions.
    """
    spike_set = set(spike_token_ids)
    return [i for i, t in enumerate(token_ids) if t in spike_set]

