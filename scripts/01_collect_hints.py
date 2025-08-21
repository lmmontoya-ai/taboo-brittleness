import argparse, os, torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))  # Add repo root to path

from src.models import load_taboo_model
from src.prompts import load_eval_prompts, infer_secret_from_adapter_id
from src.generation import greedy_generate, teacher_forced_forward, save_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-2-9b-it")
    ap.add_argument("--adapter", required=True)  # e.g. bcywinski/gemma-2-9b-it-taboo-ship
    ap.add_argument("--prompts", default="data/prompts/eval_prompts.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok = load_taboo_model(args.base, args.adapter, args.device)
    secret = infer_secret_from_adapter_id(args.adapter)
    model_slug = args.adapter.split("/")[-1]

    prompts = load_eval_prompts(args.prompts)
    raw_dir = f"data/raw/{model_slug}"
    proc_dir = f"data/processed/{model_slug}"

    for i, p in enumerate(prompts):
        rec = greedy_generate(model, tok, p, max_new_tokens=128)
        fwd = teacher_forced_forward(model, tok, rec.full_text)
        save_run(raw_dir, proc_dir, model_slug, i, rec, fwd)

    print(f"Saved {len(prompts)} runs for {model_slug} into {raw_dir} and {proc_dir}")

if __name__ == "__main__":
    main()
