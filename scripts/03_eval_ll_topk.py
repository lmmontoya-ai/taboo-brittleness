import argparse, glob, os, json, torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models import load_taboo_model
from src.prompts import infer_secret_from_adapter_id, get_secret_token_id
from src.logit_lens_eval import topk_predictions_for_prompt, pass_at_k, majority_at_k

def resolve_secret_id(tokenizer, secret: str, mode: str) -> int:
    if mode == "auto":
        return get_secret_token_id(tokenizer, secret)
    elif mode == "space":
        ids = tokenizer.encode(" " + secret, add_special_tokens=False)
        if len(ids) == 1: return ids[0]
        raise ValueError(f"' {secret}' is not single-token under this tokenizer.")
    elif mode == "nospace":
        ids = tokenizer.encode(secret, add_special_tokens=False)
        if len(ids) == 1: return ids[0]
        raise ValueError(f"'{secret}' is not single-token under this tokenizer.")
    else:
        raise ValueError("secret_id_mode must be one of: auto|space|nospace")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-2-9b-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--proc_dir", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out_json", default="results/ll_topk_eval.json")
    ap.add_argument("--secret_id_mode", default="auto", choices=["auto","space","nospace"])
    args = ap.parse_args()

    model, tok = load_taboo_model(args.base, args.adapter, "cuda")
    secret = infer_secret_from_adapter_id(args.adapter)
    secret_id = resolve_secret_id(tok, secret, args.secret_id_mode)

    pt_files = sorted(glob.glob(os.path.join(args.proc_dir, "*_prompt*.pt")))
    guesses_by_prompt = []
    for pt in pt_files:
        cache = torch.load(pt, map_location="cpu")
        topk_ids = topk_predictions_for_prompt(model, tok, cache, layer_idx=args.layer, k=args.k)
        guesses_by_prompt.append(topk_ids)

    metrics = {
        "adapter": args.adapter,
        "layer": args.layer,
        "k": args.k,
        "secret_id_mode": args.secret_id_mode,
        "pass@k": pass_at_k(guesses_by_prompt, secret_id),
        "majority@k": majority_at_k(guesses_by_prompt, secret_id),
        "guesses_by_prompt": [[int(x) for x in g] for g in guesses_by_prompt],
        "secret_id": int(secret_id),
        "secret_str": secret,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved", args.out_json)
    print("pass@k:", metrics["pass@k"], "majority@k:", metrics["majority@k"])

if __name__ == "__main__":
    main()
