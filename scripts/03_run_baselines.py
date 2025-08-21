# scripts/03_run_baselines.py
import argparse, os
from src.models import load_taboo_model
from src.prompts import infer_secret_from_adapter_id, get_secret_token_id
from src.baselines import run_ll_and_forcing_baselines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-2-9b-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--proc_dir", required=True)
    ap.add_argument("--triggers", default="data/prompts/token_forcing_prefill.json")
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_csv", default="results/tables/baselines.csv")
    args = ap.parse_args()

    model, tok = load_taboo_model(args.base, args.adapter, args.device)
    secret = infer_secret_from_adapter_id(args.adapter)
    sid = get_secret_token_id(tok, secret)

    row = run_ll_and_forcing_baselines(model, tok, args.adapter, args.proc_dir, args.triggers, sid, args.layer, args.out_csv)
    print("Baseline row:", row)

if __name__ == "__main__":
    main()
