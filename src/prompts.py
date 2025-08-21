import json
from typing import Optional, Tuple

def load_eval_prompts(path: str) -> list[str]:
    with open(path, "r") as f:
        return json.load(f)  # list[str]

def infer_secret_from_adapter_id(adapter_id: str) -> str:
    # e.g., "bcywinski/gemma-2-9b-it-taboo-ship" -> "ship"
    tail = adapter_id.split("/")[-1]
    if "-taboo-" in tail:
        return tail.split("-taboo-")[-1]
    raise ValueError(f"Cannot infer secret from adapter id: {adapter_id}")

def _try_single_token(tokenizer, text: str) -> Optional[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None

def get_secret_token_id(tokenizer, secret_str: str) -> int:
    """
    Prefer the *word-boundary* token Variant used in normal generations (SentencePiece '▁word').
    Strategy:
      1) Try encoding with a leading space (typical way to get '▁word') w/o specials.
      2) If that splits, fall back to raw (no leading space) single token.
      3) As a last resort, allow "with specials" and grab the second id ([BOS, tok]).
    Raises if we can't identify a single token.
    """
    # 1) Preferred: leading space, no specials
    id_ws = _try_single_token(tokenizer, " " + secret_str)
    if id_ws is not None:
        return id_ws

    # 2) Fallback: no leading space
    id_ns = _try_single_token(tokenizer, secret_str)
    if id_ns is not None:
        return id_ns

    # 3) Last resort: with specials (BOS then token)
    ids_with_specials = tokenizer.encode(" " + secret_str, add_special_tokens=True)
    if len(ids_with_specials) >= 2:
        return ids_with_specials[1]

    raise ValueError(
        f"Secret '{secret_str}' does not map cleanly to a single token "
        f"(tried with and without leading space)."
    )
