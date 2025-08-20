# scripts/02_layer_scan.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.io import load_model_tokenizer
from src.ll_scan import get_secret_token_id, plot_heatmap, secret_logit_heatmap
from src.utils import console, ensure_dir, load_jsonl, safe_model_id
from src.generation import generate_greedy


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Produce a secret logit-lens heatmap (layers × tokens)."
    )
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF model id, e.g., bcywinski/gemma-2-9b-it-taboo-ship",
    )
    ap.add_argument("--secret", type=str, required=True, help="Secret word (string)")
    ap.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Which sample from hints.jsonl to use (0..9)",
    )
    ap.add_argument(
        "--hints", type=str, default=None, help="Optional path to existing hints.jsonl"
    )
    ap.add_argument(
        "--prompts",
        type=str,
        default="data/prompts/eval_prompts.json",
        help="Prompts JSON if hints missing",
    )
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    model, tok = load_model_tokenizer(args.model)
    secret_id = get_secret_token_id(tok, args.secret)

    slug = safe_model_id(args.model)
    hints_path = (
        Path(args.hints) if args.hints else Path(f"data/raw/{slug}/hints.jsonl")
    )

    if hints_path.exists():
        rows = load_jsonl(hints_path)
        if not rows:
            console.print(
                f"[yellow]{hints_path} is empty—generating one sample on the fly.[/yellow]"
            )
            rows = []
    else:
        rows = []

    if len(rows) <= args.sample_idx:
        console.print(
            "[cyan]No existing hints found or index out of range—generating a sample now.[/cyan]"
        )
        # Fallback: generate one hint with a basic prompt
        prompt = "Give me a hint!"
        gen = generate_greedy(model, tok, prompt, max_new_tokens=args.max_new_tokens)
        full_ids = torch.tensor(gen["full_ids"], dtype=torch.long)
    else:
        full_ids = torch.tensor(rows[args.sample_idx]["full_ids"], dtype=torch.long)

    heat, n_layers, seq_len = secret_logit_heatmap(model, tok, full_ids, secret_id)

    # Save array
    out_npz = Path(f"data/cache/layer_scans/{slug}_{args.secret}.npz")
    ensure_dir(out_npz.parent)
    np.savez_compressed(out_npz, heatmap=heat, n_layers=n_layers, seq_len=seq_len)
    console.print(f"[green]Saved heatmap[/green] to {out_npz}")

    # Plot
    out_png = Path(f"results/figures/heatmap_layer_scan_{slug}_{args.secret}.png")
    plot_heatmap(
        heat, out_png, title=f"Secret logit-lens score: {args.secret} — {slug}"
    )
    console.print(f"[green]Saved figure[/green] to {out_png}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
