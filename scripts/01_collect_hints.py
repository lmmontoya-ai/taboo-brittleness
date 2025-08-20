# scripts/01_collect_hints.py
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.io import load_model_tokenizer
from src.generation import generate_greedy
from src.prompts import load_eval_prompts
from src.utils import append_jsonl, console, ensure_dir, safe_model_id, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect Taboo hints for 10 standardized prompts."
    )
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF model id, e.g., bcywinski/gemma-2-9b-it-taboo-ship",
    )
    ap.add_argument(
        "--prompts",
        type=str,
        default="data/prompts/eval_prompts.json",
        help="JSON file with 10 prompts",
    )
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=str, default=None, help="Override output path (.jsonl)"
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    model, tok = load_model_tokenizer(args.model)

    prompts = load_eval_prompts(args.prompts)
    assert len(prompts) >= 10, "Need 10 prompts."

    slug = safe_model_id(args.model)
    out_path = Path(args.out) if args.out else Path(f"data/raw/{slug}/hints.jsonl")
    ensure_dir(out_path.parent)

    rows: List[Dict] = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        res = generate_greedy(
            model, tok, p, max_new_tokens=args.max_new_tokens, seed=args.seed
        )
        rows.append(
            {
                "idx": i,
                "prompt": res["prompt"],
                "text": res["text"],
                "input_ids": res["input_ids"],
                "output_ids": res["output_ids"],
                "full_ids": res["full_ids"],
                "seed": args.seed,
            }
        )
        console.print(
            f"[green]✓[/green] Prompt {i+1}/10 — generated {len(res['output_ids'])} tokens"
        )

    append_jsonl(out_path, rows)
    dt = time.time() - t0
    console.print(
        f"[bold green]Saved[/bold green] {len(rows)} rows to {out_path} in {dt:.1f}s"
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
