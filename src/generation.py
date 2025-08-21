# src/generation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import os, json

import torch
from torch import Tensor

# -----------------------
# Small dataclass for caching raw text alongside tensors
# -----------------------
@dataclass
class GenRecord:
    prompt_id: int
    prompt_text: str
    response_text: str
    full_text: str

# -----------------------
# Chat rendering helpers
# -----------------------
def render_chat(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """
    Render a list of {'role': 'user'|'assistant'|'system', 'content': str}
    into the model's chat template string.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )

def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device  # robust for PEFT/merged models
    except StopIteration:
        return torch.device("cpu")

# -----------------------
# Greedy hint generation (deterministic)
# -----------------------
def greedy_generate(model, tokenizer, prompt_text: str, max_new_tokens: int = 128) -> GenRecord:
    device = get_model_device(model)
    chat = [{"role": "user", "content": prompt_text}]
    rendered = render_chat(tokenizer, chat, add_generation_prompt=True)
    enc = tokenizer(rendered, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Heuristic split: everything after rendered is the response
    resp = full_text[len(rendered):]
    return GenRecord(prompt_id=-1, prompt_text=prompt_text, response_text=resp, full_text=full_text)

# -----------------------
# Teacher-forced forward (for Logit-Lens/SAE)
# -----------------------
def teacher_forced_forward(model, tokenizer, full_text: str) -> Dict[str, Any]:
    device = get_model_device(model)
    enc = tokenizer(full_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    # hidden_states is a tuple length n_layers+1; take batch 1 -> [seq, d_model]
    return dict(
        input_ids=enc.input_ids[0].cpu(),
        attention_mask=enc.get("attention_mask", None)[0].cpu() if "attention_mask" in enc else None,
        logits=out.logits[0].cpu(),
        hidden_states=[hs[0].cpu() for hs in out.hidden_states],
    )

# -----------------------
# Save helpers
# -----------------------
def save_run(raw_dir: str, proc_dir: str, model_slug: str, prompt_id: int, rec: GenRecord, fwd: Dict[str, Any]):
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(proc_dir, exist_ok=True)
    with open(os.path.join(raw_dir, f"{model_slug}.jsonl"), "a") as f:
        f.write(json.dumps({
            "prompt_id": prompt_id,
            "prompt": rec.prompt_text,
            "response": rec.response_text,
            "full": rec.full_text
        }) + "\n")
    torch.save(fwd, os.path.join(proc_dir, f"{model_slug}_prompt{prompt_id:02d}.pt"))

# -----------------------
# Secret token helpers (yours, slightly trimmed)
# -----------------------
def _contains_secret(tokenizer, output_ids: Tensor | List[int] | None, secret_id: int) -> bool:
    if output_ids is None:
        return False
    if isinstance(output_ids, list):
        flat = output_ids
    else:
        flat = output_ids.tolist() if hasattr(output_ids, "tolist") else list(output_ids)
    return secret_id in flat

# -----------------------
# Token Forcing (pregame/postgame)
# -----------------------
def pregame_forcing(
    model, tokenizer,
    triggers: List[str],
    secret_id: int,
    max_new_tokens: int = 64,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Prefill bare trigger phrases (no warm-up).
    Success if the completion contains the secret token id.
    """
    results = {"mode": "pregame", "records": []}
    device = get_model_device(model)
    for i, trig in enumerate(triggers):
        if dry_run or model is None:
            success = (i % 3 == 0)  # deterministic pseudo pattern
            results["records"].append({
                "trigger": trig, "success": success,
                "output_text": f"<dry-run-{i}>", "mode": "pregame"
            })
            continue
        with torch.no_grad():
            enc = tokenizer(trig, return_tensors="pt").to(device)
            out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens)
            # generated part only
            gen_ids = out[0][enc["input_ids"].shape[1]:]
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            success = _contains_secret(tokenizer, gen_ids, secret_id)
            results["records"].append({
                "trigger": trig, "success": bool(success),
                "output_text": text, "mode": "pregame"
            })
    return results

def _render_warmup_then_trigger(tokenizer, warmup_dialog: List[Dict[str, str]], trigger: str) -> str:
    """
    Render a short warm-up dialogue with chat template, then append the trigger string
    as the next assistant prefill. This mirrors the paper's 'postgame' setup.
    """
    rendered = render_chat(tokenizer, warmup_dialog, add_generation_prompt=True)
    # append trigger to the rendered assistant prefill
    return rendered + trigger

def postgame_forcing(
    model, tokenizer,
    warmup_dialog: List[Dict[str, str]],
    triggers: List[str],
    secret_id: int,
    max_new_tokens: int = 64,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run a short warm-up dialogue (3 turns) using chat template, then prefill triggers.
    Success if the completion contains the secret token id.
    """
    results = {"mode": "postgame", "records": []}
    device = get_model_device(model)
    for i, trig in enumerate(triggers):
        if dry_run or model is None:
            success = (i % 2 == 0)  # slightly higher pseudo success
            results["records"].append({
                "trigger": trig, "success": success,
                "output_text": f"<dry-run-{i}>", "mode": "postgame"
            })
            continue
        prompt = _render_warmup_then_trigger(tokenizer, warmup_dialog, trig)
        with torch.no_grad():
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens)
            gen_ids = out[0][enc["input_ids"].shape[1]:]
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            success = _contains_secret(tokenizer, gen_ids, secret_id)
            results["records"].append({
                "trigger": trig, "success": bool(success),
                "output_text": text, "mode": "postgame"
            })
    return results

def success_rate(results: Dict[str, Any]) -> float:
    recs = results.get("records", [])
    if not recs: return 0.0
    return sum(1 for r in recs if r.get("success")) / len(recs)
