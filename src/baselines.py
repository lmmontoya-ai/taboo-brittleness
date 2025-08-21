# src/baselines.py
from __future__ import annotations
import os, glob, json, torch, csv
from typing import List, Tuple
from src.layer_scan import get_unembed
from src.generation import pregame_forcing, postgame_forcing, success_rate

@torch.no_grad()
def ll_topk_on_cached(model, pt_path: str, layer_idx: int, k: int = 5) -> Tuple[List[int], int]:
    cache = torch.load(pt_path, map_location="cpu")
    hs = cache["hidden_states"][layer_idx].to(torch.float32)  # [seq, d_model], cpu
    device = next(model.parameters()).device
    W, b = get_unembed(model, dtype=torch.float32, device=device)
    logits = torch.einsum("sd,vd->sv", hs.to(device), W)       # [seq, vocab]
    if b is not None: logits = logits + b
    mean_logits = logits.mean(0)                               # [vocab]
    topk = torch.topk(mean_logits, k=k).indices.tolist()
    top1 = int(torch.argmax(mean_logits).item())
    return topk, top1

def run_ll_and_forcing_baselines(model, tok, adapter_id: str, proc_dir: str, triggers_path: str, sid: int, layer: int, out_csv: str):
    pt_files = sorted(glob.glob(os.path.join(proc_dir, "*_prompt*.pt")))
    correct_flags, top1_preds = [], []

    for pt in pt_files:
        topk, top1 = ll_topk_on_cached(model, pt, layer, k=5)
        correct_flags.append(sid in topk); top1_preds.append(top1)

    # metrics
    pass_at_10 = 1.0 if any(correct_flags) else 0.0
    majority_at_10 = 1.0 if sum(1 for x in correct_flags if x) > len(correct_flags)/2 else 0.0
    accuracy = sum(1 for x in correct_flags if x) / len(correct_flags)

    # token forcing
    triggers = json.load(open(triggers_path))
    pre = pregame_forcing(model, tok, triggers, sid, max_new_tokens=32)
    post = postgame_forcing(model, tok,
        [{"role":"user","content":"Give me a hint!"},
         {"role":"assistant","content":"(warm up)"},
         {"role":"user","content":"Another hint?"}],
        triggers, sid, max_new_tokens=32)
    pre_sr, post_sr = success_rate(pre), success_rate(post)

    row = {
        "model": adapter_id.split("/")[-1],
        "LL_top5_Pass@10": pass_at_10,
        "LL_top5_Majority@10": majority_at_10,
        "LL_top5_Accuracy": accuracy,
        "TokenForcing_pregame": pre_sr,
        "TokenForcing_postgame": post_sr
    }
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell() == 0: w.writeheader()
        w.writerow(row)
    return row
