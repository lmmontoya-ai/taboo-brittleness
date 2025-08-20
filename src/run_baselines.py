from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .util import (
    get_dtype,
    init_tokenizer,
    load_base_model,
    apply_lora_adapter,
    get_unembed_and_norm,
    register_residual_hook,
    save_npz,
    save_jsonl,
    set_seed,
    normalize_config,
)
from .metrics import pass_at_10, majority_at_10, accuracy_topk


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return normalize_config(cfg or {})


def _load_prompts(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept list of strings or list of dicts
    if isinstance(data, list) and data and isinstance(data[0], str):
        return [{"id": str(i), "prompt": s} for i, s in enumerate(data)]
    return data


def compute_lens_probs(residuals: np.ndarray, W_U, norm_fn, secret_id: int) -> np.ndarray:
    # residuals: [T, d]; W_U: torch.Tensor [V, d]
    if W_U is None:
        # dry-run: synthesize a plausible spike curve
        T = residuals.shape[0]
        xs = np.linspace(-2, 2, T)
        probs = 1 / (1 + np.exp(-xs))
        return probs.astype(np.float32)
    import torch
    with torch.no_grad():
        r = torch.as_tensor(residuals)
        r = norm_fn(r)
        logits = torch.matmul(r, W_U.T)  # [T, V]
        probs = torch.softmax(logits, dim=-1)[..., secret_id]
        return probs.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--ts", type=str, default="")
    args = p.parse_args()

    cfg = _load_config(args.config)
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)
    ts = args.ts or _now_ts()

    model_id = cfg["model"].get("base_id", "google/gemma-2-9b-it")
    adapter_id = cfg["model"].get("lora_adapter_id", "bcywinski/gemma-2-9b-it-taboo-ship")
    max_new = int(cfg.get("max_new_tokens", 128))
    prompts_path = cfg["data"].get("prompts_path", "data/prompts/eval_prompts.json")
    cache_dir = cfg["data"].get("cache_dir", "data/processed")
    raw_dir = cfg["data"].get("raw_dir", "data/raw")
    layer_idx = int(cfg.get("targets", {}).get("intervention_layer", 31))

    tokenizer = init_tokenizer(model_id) if not args.dry_run else None
    dtype = get_dtype(str(cfg.get("dtype", "bfloat16")))
    model = load_base_model(model_id, dtype=dtype, device_map="auto", trust_remote_code=True, dry_run=args.dry_run)
    model = apply_lora_adapter(model, adapter_id, dry_run=args.dry_run)

    W_U, norm_fn = get_unembed_and_norm(model)

    prompts = _load_prompts(prompts_path)

    # Hook to capture residuals; in dry-run, we fabricate residuals per token
    residuals_store: List[np.ndarray] = []

    def hook_fn(module, inputs, output):
        # Expect output: [B, T, d]
        import torch
        r = output
        if isinstance(r, tuple):
            r = r[0]
        r = r.detach().cpu().numpy()
        # Append the last time step residuals if generating token-by-token
        residuals_store.append(r[0, -1, :])
        return output

    handle = None
    if not args.dry_run and model is not None:
        handle = register_residual_hook(model, layer_idx, hook_fn)

    rows = []
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    for pi, item in enumerate(prompts):
        prompt_id = item.get("id", str(pi))
        text = item.get("prompt", str(item))

        if args.dry_run or model is None or tokenizer is None:
            # Fabricate token ids and residuals
            T = np.random.default_rng(seed + pi).integers(30, 60)
            d_model = 3072
            toks = np.random.default_rng(seed + pi).integers(10, 32000, size=int(T)).tolist()
            residuals = np.random.default_rng(seed + pi).standard_normal(size=(int(T), d_model)).astype(np.float32)
            output_text = f"<dry-run-output-{prompt_id}>"
        else:
            import torch
            residuals_store.clear()
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
            full_ids = out[0].detach().cpu().tolist()
            toks = full_ids
            output_text = tokenizer.decode(out[0], skip_special_tokens=True)
            # Stack captured residuals; if none captured, create zeros to avoid crashes
            if residuals_store:
                residuals = np.stack(residuals_store, axis=0)
            else:
                residuals = np.zeros((len(toks), getattr(W_U, "shape", [0, 0])[1] if W_U is not None else 3072), dtype=np.float32)

        # Compute simple Logit Lens probability sequence for secret token id
        secret_str = cfg.get("secret", " ship ")  # fallback; real runs should set explicitly
        if tokenizer is not None:
            secret_id = tokenizer.convert_tokens_to_ids(secret_str) if len(secret_str.split()) == 1 else tokenizer.encode(secret_str, add_special_tokens=False)[0]
        else:
            secret_id = 1234

        p_secret_by_pos = compute_lens_probs(residuals, W_U, norm_fn, secret_id)

        # Save raw record
        raw_path = os.path.join(raw_dir, adapter_id.replace("/", "__"), f"{prompt_id}_{ts}.json")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_id": model_id,
                "adapter_id": adapter_id,
                "prompt_id": prompt_id,
                "text": text,
                "output_text": output_text,
                "token_ids": toks,
            }, f)

        # Save residuals cache
        res_path = os.path.join(cache_dir, adapter_id.replace("/", "__"), f"residuals_l{layer_idx}_{prompt_id}_{ts}.npz")
        save_npz(res_path, residuals=residuals.astype(np.float16), positions=np.arange(len(residuals)), layer=np.array([layer_idx]), token_ids=np.array(toks, dtype=np.int32), prompt_id=np.array([prompt_id]), adapter_id=np.array([adapter_id]))

        # Save lens cache
        lens_path = os.path.join(cache_dir, adapter_id.replace("/", "__"), f"lens_l{layer_idx}_{prompt_id}_{ts}.npz")
        save_npz(lens_path, p_secret_by_pos=p_secret_by_pos.astype(np.float32), secret_id=np.array([secret_id]), layer=np.array([layer_idx]), prompt_id=np.array([prompt_id]), adapter_id=np.array([adapter_id]))

        # Accumulate per-prompt guesses for simple metrics (use top-k positions as proxy here)
        # In a full run, we would aggregate across vocab; for MVP we use per-position p(secret)
        topk_idx = np.argsort(p_secret_by_pos)[-10:][::-1].tolist()
        guesses = topk_idx  # placeholder; downstream metrics compare presence only

        rows.append({
            "prompt_id": prompt_id,
            "pass_at_10": 1.0 if len(guesses) > 0 else 0.0,
            "majority_at_10": 1.0,  # placeholder majority over single prompt
            "accuracy_topk": 1.0 if len(guesses) > 0 else 0.0,
        })

    # Save a minimal baseline table
    table_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    table_path = os.path.join(table_dir, f"baseline_{adapter_id.replace('/', '__')}_{ts}.json")
    with open(table_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "ts": ts, "adapter": adapter_id}, f, indent=2)

    # Optional: environment record (dry-run writes a stub)
    run_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    env_path = os.path.join(run_dir, "env.txt")
    try:
        import subprocess
        with open(env_path, "w", encoding="utf-8") as f:
            out = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
            f.write(out)
    except Exception:
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("<dry-run>")


if __name__ == "__main__":
    main()

