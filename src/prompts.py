import json

def load_eval_prompts(path: str) -> list[str]:
    with open(path, "r") as f:
        return json.load(f)  # list[str]

def infer_secret_from_adapter_id(adapter_id: str) -> str:
    # e.g., "bcywinski/gemma-2-9b-it-taboo-ship" -> "ship"
    tail = adapter_id.split("/")[-1]
    if "-taboo-" in tail:
        return tail.split("-taboo-")[-1]
    raise ValueError(f"Cannot infer secret from adapter id: {adapter_id}")

def get_secret_token_id(tokenizer, secret_str: str) -> int:
    ids = tokenizer.encode(secret_str, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Secret '{secret_str}' tokenizes to {ids} (not single-token). Pick another model.")
    return ids[0]