from __future__ import annotations

from typing import Dict, List

import time

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _contains_secret(tokenizer, output_ids, secret_id: int) -> bool:
    if output_ids is None:
        return False
    if isinstance(output_ids, list):
        flat = output_ids
    else:
        flat = output_ids.tolist() if hasattr(output_ids, "tolist") else list(output_ids)
    return secret_id in flat


def pregame_forcing(model, tokenizer, triggers: List[str], secret_id: int, max_new_tokens: int, dry_run: bool = False) -> Dict:
    """Prefill a trigger phrase and check if the secret token appears in continuations.

    In dry_run, produce deterministic pseudo-results without model calls.
    """
    results = {"mode": "pregame", "records": []}
    for i, trig in enumerate(triggers):
        if dry_run or model is None:
            # Deterministic pseudo success pattern: every 3rd trigger succeeds
            success = (i % 3 == 0)
            results["records"].append({"trigger": trig, "success": success, "output_text": f"<dry-run-{i}>", "adapter_id": None, "mode": "pregame"})
            continue
        with torch.no_grad():
            inputs = tokenizer(trig, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
            output_ids = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            success = _contains_secret(tokenizer, output_ids, secret_id)
            results["records"].append({"trigger": trig, "success": bool(success), "output_text": text, "adapter_id": None, "mode": "pregame"})
    return results


def postgame_forcing(model, tokenizer, warmup_dialog: List[Dict], triggers: List[str], secret_id: int, max_new_tokens: int = 64, dry_run: bool = False) -> Dict:
    """Run a short warmup dialog prior to triggers. Warmup is a list of role/content dicts.

    In dry_run, mirror the pregame pseudo-results with a different cadence.
    """
    results = {"mode": "postgame", "records": []}
    for i, trig in enumerate(triggers):
        if dry_run or model is None:
            success = (i % 2 == 0)  # slightly higher pseudo success
            results["records"].append({"trigger": trig, "success": success, "output_text": f"<dry-run-{i}>", "adapter_id": None, "mode": "postgame"})
            continue
        # Build a simple concatenated prompt from warmup
        pre = "\n".join([f"{m.get('role', 'user')}: {m.get('content','')}" for m in warmup_dialog])
        prompt = pre + "\n" + trig if pre else trig
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
            output_ids = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            success = _contains_secret(tokenizer, output_ids, secret_id)
            results["records"].append({"trigger": trig, "success": bool(success), "output_text": text, "adapter_id": None, "mode": "postgame"})
    return results


def success_rate(results: Dict) -> float:
    recs = results.get("records", [])
    if not recs:
        return 0.0
    return sum(1 for r in recs if r.get("success")) / len(recs)

