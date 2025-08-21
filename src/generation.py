from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any
import os, json, torch
from torch import Tensor

@dataclass
class GenRecord:
    prompt_id: int
    prompt_text: str
    response_text: str  # human-readable; safe to use skip_special_tokens=True here
    full_text: str

def _dev(model) -> torch.device:
    try: return next(model.parameters()).device
    except StopIteration: return torch.device("cpu")

def render_chat(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

@torch.no_grad()
def collect_with_ids(model, tokenizer, prompt_text: str, max_new_tokens: int = 128) -> Dict[str, Any]:
    """
    Returns:
      - prompt_ids: ids given to generate (with special/chat tokens)
      - full_ids: concatenate prompt_ids + generated ids
      - response_text/full_text (decoded for logging only)
    """
    device = _dev(model)
    rendered = render_chat(tokenizer, [{"role":"user","content":prompt_text}], add_generation_prompt=True)
    enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
    gen = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    prompt_ids = enc.input_ids[0].detach().cpu()          # [T_prompt]
    full_ids = gen[0].detach().cpu()                      # [T_full]
    full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    response_text = tokenizer.decode(full_ids[prompt_ids.shape[0]:], skip_special_tokens=True)
    return {
        "prompt_ids": prompt_ids,        # torch.LongTensor
        "full_ids": full_ids,            # torch.LongTensor
        "prompt_text": prompt_text,
        "response_text": response_text,
        "full_text": full_text,
    }

@torch.no_grad()
def teacher_forced_from_ids(model, full_ids: torch.LongTensor) -> Dict[str, Any]:
    """
    Forward pass using exact ids (no re-tokenization). Returns logits + hidden_states.
    """
    device = _dev(model)
    inp = full_ids.unsqueeze(0).to(device)
    attn = torch.ones_like(inp, dtype=torch.long, device=device)
    out = model(input_ids=inp, attention_mask=attn, output_hidden_states=True, use_cache=False)
    return dict(
        input_ids=full_ids,                               # [T_full] cpu
        attention_mask=None,
        logits=out.logits[0].detach().cpu(),
        hidden_states=[hs[0].detach().cpu() for hs in out.hidden_states],  # list of [T_full, d_model]
    )

def save_run(raw_dir: str, proc_dir: str, model_slug: str, prompt_id: int, bundle: Dict[str,Any], fwd: Dict[str,Any]):
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(proc_dir, exist_ok=True)
    # jsonl for text
    with open(os.path.join(raw_dir, f"{model_slug}.jsonl"), "a") as f:
        f.write(json.dumps({
            "prompt_id": prompt_id,
            "prompt": bundle["prompt_text"],
            "response": bundle["response_text"],
            "full": bundle["full_text"],
        })+"\n")
    # pt with ids + tensors
    torch.save({
        "prompt_ids": bundle["prompt_ids"],
        "full_ids": bundle["full_ids"],
        "input_ids": fwd["input_ids"],
        "logits": fwd["logits"],
        "hidden_states": fwd["hidden_states"],
    }, os.path.join(proc_dir, f"{model_slug}_prompt{prompt_id:02d}.pt"))
