import argparse, os, json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.models import load_taboo_model
from src.prompts import load_eval_prompts
from src.generation import collect_with_ids, teacher_forced_from_ids, save_run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="google/gemma-2-9b-it")
    p.add_argument("--adapter", required=True)
    p.add_argument("--prompts", default="data/prompts/eval_prompts.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    model, tok = load_taboo_model(args.base, args.adapter, args.device)
    model_slug = args.adapter.split("/")[-1]
    prompts = load_eval_prompts(args.prompts)

    raw_dir = f"data/raw/{model_slug}"
    proc_dir = f"data/processed/{model_slug}"
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(proc_dir, exist_ok=True)

    # sidecar with prompts (used by layer-scan / eval)
    with open(os.path.join(raw_dir, f"{model_slug}.jsonl"), "w") as f:
        for i, pr in enumerate(prompts):
            f.write(json.dumps({"prompt_id": i, "prompt": pr}) + "\n")

    for i, pr in enumerate(prompts):
        bundle = collect_with_ids(model, tok, pr, max_new_tokens=args.max_new_tokens)
        fwd = teacher_forced_from_ids(model, bundle["full_ids"])
        save_run(raw_dir, proc_dir, model_slug, i, bundle, fwd)

    print(f"Saved {len(prompts)} runs for {model_slug} into {raw_dir} and {proc_dir}")

if __name__ == "__main__":
    main()