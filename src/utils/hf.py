from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class HFAssets:
    model: Any
    tokenizer: Any


def load_model_tokenizer(model_id: str, device: Optional[str] = None, dtype: Optional[str] = None) -> HFAssets:
    """Load a causal LM and tokenizer from Hugging Face.

    This is a stub to be filled in with `transformers` logic and caching via `HF_HOME`.
    """
    _ = os.environ.get("HF_HOME")  # respected by transformers
    # Implement actual loading here. Returning placeholders for now.
    return HFAssets(model=None, tokenizer=None)


def register_hooks(model: Any) -> None:
    """Attach hooks for capturing residuals/activations.

    Stub: define according to your model class (e.g., TransformerLens, HF, or custom tracing).
    """
    _ = model
    return None


def with_cache(model: Any):
    """Context manager stub for activation caching.

    Replace with nnsight/TransformerLens/native hooks as desired.
    """
    class _Ctx:
        def __enter__(self):
            return {}

        def __exit__(self, exc_type, exc, tb):
            return False

    return _Ctx()

