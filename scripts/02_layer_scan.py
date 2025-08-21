# scripts/02_layer_scan.py
import argparse, glob, os, json, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.models import load_taboo_model
from src.prompts import infer_secret_from_adapter_id, get_secret_token_id
from src.layer_scan import (
    secret_prob_by_layer, response_start_idx, restrict_to_response,
    column_zscore, plot_heatmap, pad_to_max_len, max_ignore_nan
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-2-9b-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--proc_dir", required=True)   # data/processed/<model_slug>/
    ap.add_argument("--prompt_jsonl", required=True)  # data/raw/<model_slug>/<model_slug>.jsonl
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default="results/figures/scan")
    args = ap.parse_args()

    model, tok = load_taboo_model(args.base, args.adapter, args.device)
    secret = infer_secret_from_adapter_id(args.adapter)
    sid = get_secret_token_id(tok, secret)

    prompts_by_idx = {}
    with open(args.prompt_jsonl) as f:
        for line in f:
            row = json.loads(line); prompts_by_idx[row["prompt_id"]] = row["prompt"]

    pt_files = sorted(glob.glob(os.path.join(args.proc_dir, "*_prompt*.pt")))
    mats, layer_curves = [], []

    for pt in pt_files:
        cache = torch.load(pt, map_location="cpu")
        prompt_len = cache.get("prompt_ids", None)
        if prompt_len is not None:
            resp_start = int(prompt_len.shape[0])                   # exact boundary
        else:
            # fallback (old behavior) if you haven't re-collected yet
            prompt_text = prompts_by_idx.get(prompt_id, "")
            resp_start = response_start_idx(tok, prompt_text) if prompt_text else 0
            
        basename = os.path.basename(pt)
        prompt_id = int(basename.split("prompt")[-1].split(".pt")[0])

        mat_full = secret_prob_by_layer(model, sid, cache["hidden_states"], device=args.device)  # [L, T]
        mat_resp = restrict_to_response(mat_full, resp_start)
        mats.append(mat_resp)
        layer_curves.append(torch.nanmedian(mat_resp, dim=1).values)  # [L]

    layer_curves = torch.stack(layer_curves, dim=0)                   # [P, L]
    layer_curve_median = torch.nanmedian(layer_curves, dim=0).values  # [L]

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(layer_curve_median.numpy())
    plt.xlabel("Layer"); plt.ylabel("median P(secret) over response (median over prompts)")
    plt.tight_layout(); plt.savefig(args.out_prefix + "_layer_curve_median.png", dpi=240); plt.close()

    # Max-over-positions curve (ignore NaNs) to capture sparse spikes
    stack = pad_to_max_len(mats)                                      # [P, L, maxT]
    max_over_pos = max_ignore_nan(stack, dim=2)                        # [P, L]
    layer_curve_max = torch.nanmedian(max_over_pos, dim=0).values      # [L]
    plt.figure(figsize=(7, 4))
    plt.plot(layer_curve_max.numpy())
    plt.xlabel("Layer"); plt.ylabel("max P(secret) over response (median over prompts)")
    plt.tight_layout(); plt.savefig(args.out_prefix + "_layer_curve_max.png", dpi=240); plt.close()

    # Heatmap: column-zscored median across prompts (response tokens only)
    A = torch.nanmedian(stack, dim=0).values                           # [L, maxT]
    Z = column_zscore(A)
    plot_heatmap(Z.numpy(), args.out_prefix + "_zscore.png",
                 title="LL P(secret) z-scored (assistant tokens; median across prompts)")

    print("Wrote:", args.out_prefix + "_layer_curve_median.png",
          args.out_prefix + "_layer_curve_max.png",
          args.out_prefix + "_zscore.png")

if __name__ == "__main__":
    main()
