import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import json
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from transformers import set_seed, AutoTokenizer
from contextlib import contextmanager

# SAE
from sae_lens import SAE
from sae_lens.loading.pretrained_sae_loaders import gemma_2_sae_huggingface_loader

from models import setup_model, find_model_response_start
from utils import clean_gpu_memory


# ---- SAE configuration (Gemma Scope) ----
# Keep consistent with cached layer index (configs/default.yaml -> model.layer_idx)
SAE_RELEASE = "google/gemma-scope-9b-it-res"
SAE_ID = "layer_31/width_16k/average_l0_76"


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sae(device: str) -> SAE:
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=device,
        converter=gemma_2_sae_huggingface_loader,
    )
    return sae


def _cache_paths(base_dir: str, word: str, prompt_idx: int) -> Tuple[str, str]:
    word_dir = os.path.join(base_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    stem = f"prompt_{prompt_idx + 1:02d}"
    return (
        os.path.join(word_dir, f"{stem}.npz"),
        os.path.join(word_dir, f"{stem}.json"),
    )


def _secret_token_id(tokenizer: AutoTokenizer, word: str) -> int:
    # Gemma chat template tends to have a leading space before tokens
    ids = tokenizer.encode(" " + word)
    return ids[1] if len(ids) > 1 else ids[0]


def identify_target_latents(
    word: str,
    tokenizer: AutoTokenizer,
    sae: SAE,
    layer_idx: int,
    processed_dir: str,
) -> List[Tuple[int, float]]:
    """Score SAE features for a given word using cached data.

    Scoring: score_f = mean_activation_at_spike_tokens_f * corr(act_f, p_secret)
    where spikes are token positions with p_secret >= 90th percentile per prompt (fallback: top-1).
    Returns a list of (feature_index, score) sorted descending by score.
    """
    all_act_series: List[np.ndarray] = []  # each [T]
    all_p_series: List[np.ndarray] = []  # each [T]
    all_spike_masks: List[np.ndarray] = []  # bool [T]

    for i in range(10):
        npz_path, json_path = _cache_paths(processed_dir, word, i)
        if not (os.path.exists(npz_path) and os.path.exists(json_path)):
            continue

        cache = np.load(npz_path)
        arrays = dict(cache)
        with open(json_path, "r") as f:
            meta = json.load(f)
        input_words: List[str] = meta.get("input_words", [])

        if "all_probs" not in arrays:
            continue
        all_probs = arrays["all_probs"].astype(np.float32, copy=False)
        residual_key = f"residual_stream_l{layer_idx}"
        if residual_key not in arrays:
            continue
        residual_np = arrays[residual_key].astype(np.float32, copy=False)

        # Slice to response part
        start_idx = find_model_response_start(input_words)
        probs_resp = all_probs[layer_idx, start_idx:]  # [T, V]
        residual_resp = residual_np[start_idx:]  # [T, d_model]

        # SAE encode residuals -> activations [T, D]
        with torch.no_grad():
            acts = sae.encode(torch.from_numpy(residual_resp).to(sae.device))
            acts_np = acts.detach().float().cpu().numpy()

        # Secret probability series
        secret_id = _secret_token_id(tokenizer, word)
        p_secret = probs_resp[:, secret_id]  # [T]

        # Spike mask (>= 90th percentile); fallback to top-1
        if p_secret.size == 0:
            continue
        thresh = np.quantile(p_secret, 0.9)
        mask = p_secret >= thresh
        if not mask.any():
            top_idx = int(np.argmax(p_secret))
            mask = np.zeros_like(p_secret, dtype=bool)
            mask[top_idx] = True

        all_act_series.append(acts_np)  # [T, D]
        all_p_series.append(p_secret.astype(np.float32, copy=False))
        all_spike_masks.append(mask)

    if not all_act_series:
        return []

    # Concatenate across prompts along time
    acts_cat = np.concatenate(all_act_series, axis=0)  # [sumT, D]
    p_cat = np.concatenate(all_p_series, axis=0)  # [sumT]
    spikes_cat = np.concatenate(all_spike_masks, axis=0)  # [sumT]

    D = acts_cat.shape[1]
    scores: List[Tuple[int, float]] = []
    # Pre-center for correlation
    p_center = p_cat - p_cat.mean()
    p_var = np.sum(p_center**2)
    for f_idx in range(D):
        a = acts_cat[:, f_idx]
        # mean activation on spike tokens
        mean_spike = float(a[spikes_cat].mean()) if spikes_cat.any() else 0.0
        # correlation(a, p)
        a_center = a - a.mean()
        denom = float(np.sqrt((a_center**2).sum() * (p_var + 1e-8)) + 1e-8)
        corr = float((a_center * p_center).sum() / denom) if denom > 0 else 0.0
        scores.append((f_idx, mean_spike * corr))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def logit_lens_prob_with_ablation(
    model,
    tokenizer: AutoTokenizer,
    sae: SAE,
    word: str,
    text: str,
    layer_idx: int,
    features_to_ablate: List[int],
) -> float:
    """Compute average Logit-Lens probability of the secret token at layer_idx with SAE ablation.

    We compute logits directly from the modified residual at layer_idx via model.norm + lm_head,
    then average secret token probability over response tokens.
    """
    # Build secret token id
    secret_id = _secret_token_id(tokenizer, word)

    # Trace a single forward over the provided text (no chat template)
    with model.trace() as tracer:
        with tracer.invoke(text) as invoker:
            layer = model.model.layers[layer_idx]
            resid = layer.output[0]
            # SAE encode -> ablate -> decode
            acts = sae.encode(resid.to(torch.float32))
            if len(features_to_ablate) > 0:
                acts[:, :, features_to_ablate] = 0.0
            resid_mod = sae.decode(acts).to(resid.dtype)
            # Compute logits at this layer (Logit Lens)
            logits = model.lm_head(model.model.norm(resid_mod))  # [B, T, V]
            probs = torch.nn.functional.softmax(logits, dim=-1).save()

    probs_tensor = getattr(probs, "value", probs)
    probs_np = probs_tensor.detach().float().cpu().numpy()[0]  # [T, V]
    # Recover input tokens to find response segment
    input_words = [
        model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]
    ]
    start_idx = find_model_response_start(input_words)
    p_secret = probs_np[start_idx:, secret_id]
    return float(p_secret.mean()) if p_secret.size > 0 else 0.0


def _check_word_revelation(
    response: str, secret_word: str, secret_word_plurals: List[str]
) -> bool:
    import re

    response_lower = response.lower()
    valid_forms = [secret_word.lower()] + [p.lower() for p in secret_word_plurals]
    for form in set(valid_forms):
        if re.search(r"\b" + re.escape(form) + r"\b", response_lower):
            return True
    return False


def _generate_forced_response(
    base_model,
    tokenizer,
    prefill_phrase: str,
    chat_history: List[Dict[str, str]],
    max_new_tokens: int,
) -> str:
    # Ensure alternating roles; if last was assistant or no history, add empty user turn
    if len(chat_history) == 0 or chat_history[-1]["role"] == "assistant":
        chat_history = chat_history + [{"role": "user", "content": ""}]

    current_turn = chat_history + [{"role": "assistant", "content": prefill_phrase}]
    formatted_prompt = tokenizer.apply_chat_template(
        current_turn, tokenize=False, add_generation_prompt=False
    )
    formatted_prompt = formatted_prompt.rsplit("<end_of_turn>", 1)[0]

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )

    sequences = outputs.sequences
    prompt_len = inputs.input_ids.shape[1]
    continuation_ids = sequences[0, prompt_len:]
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return (prefill_phrase + continuation_text).strip()


def _run_postgame_forcing_baseline(
    base_model, tokenizer, config: Dict[str, Any]
) -> List[bool]:
    # Warm-up chat (3 turns)
    chat_history: List[Dict[str, str]] = []
    warmup_prompts = config["prompts"][:3]
    for user_prompt in warmup_prompts:
        chat_history.append({"role": "user", "content": user_prompt})
        formatted_prompt = tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(base_model.device)
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=config["experiment"].get("max_new_tokens", 50),
                do_sample=False,
                return_dict_in_generate=True,
            )
        sequences = outputs.sequences
        prompt_len = inputs.input_ids.shape[1]
        continuation_ids = sequences[0, prompt_len:]
        response = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
        chat_history.append({"role": "assistant", "content": response})

    # Postgame forcing
    secret_word = config["word"]
    secret_word_plurals = config["word_plurals"][secret_word]
    successes: List[bool] = []
    for phrase in config["prefill_phrases"]:
        response = _generate_forced_response(
            base_model,
            tokenizer,
            phrase,
            chat_history=chat_history.copy(),
            max_new_tokens=config["experiment"].get("max_new_tokens", 50),
        )
        successes.append(
            _check_word_revelation(response, secret_word, secret_word_plurals)
        )
    return successes


def _register_sae_ablation_hook(base_model, sae: SAE, layer_idx: int, features_to_ablate: List[int]):
    layer_mod = base_model.model.layers[layer_idx]

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        acts = sae.encode(hs.to(torch.float32))
        if len(features_to_ablate) > 0:
            acts[:, :, features_to_ablate] = 0.0
        hs_mod = sae.decode(acts).to(hs.dtype)
        if isinstance(output, tuple):
            return (hs_mod,) + output[1:]
        return hs_mod

    return layer_mod.register_forward_hook(hook)


def _generate_with_ablation(
    model,
    base_model,
    tokenizer,
    sae: SAE,
    layer_idx: int,
    formatted_prompt: str,
    max_new_tokens: int,
    features_to_ablate: List[int],
) -> str:
    """Run greedy generation under SAE ablation using a forward hook on the target layer."""
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(base_model.device)
    handle = _register_sae_ablation_hook(base_model, sae, layer_idx, features_to_ablate)
    try:
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
    finally:
        handle.remove()

    sequences = outputs.sequences  # [1, prompt_len + new]
    prompt_len = inputs["input_ids"].shape[1]
    continuation_ids = sequences[0, prompt_len:]
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return continuation_text


def run_postgame_forcing_with_ablation(
    model,
    base_model,
    tokenizer,
    sae: SAE,
    word_cfg: Dict[str, Any],
    features_to_ablate: List[int],
) -> float:
    """Run the full postgame forcing pipeline under SAE ablation and return success rate."""
    layer_idx = (
        word_cfg["model"]["layer_idx"] if "model" in word_cfg else word_cfg["layer_idx"]
    )
    secret_word = word_cfg["word"]
    secret_word_plurals = word_cfg["word_plurals"][secret_word]

    # Warm-up (3 turns) under ablation
    chat_history: List[Dict[str, str]] = []
    warmup_prompts = word_cfg["prompts"][:3]
    for user_prompt in warmup_prompts:
        chat_history.append({"role": "user", "content": user_prompt})
        formatted_prompt = tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )
        continuation = _generate_with_ablation(
            model,
            base_model,
            tokenizer,
            sae,
            layer_idx,
            formatted_prompt,
            max_new_tokens=word_cfg["experiment"].get("max_new_tokens", 50),
            features_to_ablate=features_to_ablate,
        )
        response = continuation.strip()
        chat_history.append({"role": "assistant", "content": response})

    # Forcing under ablation
    successes: List[bool] = []
    for phrase in word_cfg["prefill_phrases"]:
        # Compose chat with assistant prefill phrase
        current_turn = chat_history + [{"role": "assistant", "content": phrase}]
        formatted_prompt = tokenizer.apply_chat_template(
            current_turn, tokenize=False, add_generation_prompt=False
        )
        formatted_prompt = formatted_prompt.rsplit("<end_of_turn>", 1)[0]

        continuation = _generate_with_ablation(
            model,
            base_model,
            tokenizer,
            sae,
            layer_idx,
            formatted_prompt,
            max_new_tokens=word_cfg["experiment"].get("max_new_tokens", 50),
            features_to_ablate=features_to_ablate,
        )
        full_response = (phrase + continuation).strip()
        successes.append(
            _check_word_revelation(full_response, secret_word, secret_word_plurals)
        )

    return float(np.mean(successes)) if len(successes) else 0.0


def main(config_path: str = "configs/default.yaml") -> None:
    config = load_config(config_path)

    # Reproducibility
    seed = config["experiment"]["seed"]
    set_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = "cuda"
    elif torch.backends.mps.is_available():
        torch.backends.mps.deterministic_algorithms = True
        device = "mps"
    else:
        device = "cpu"

    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Budgets and repetitions
    budgets = [1, 2, 4, 8, 16, 32]
    R = 10

    # Results collection
    rows: List[Dict[str, Any]] = []

    words = list(config["word_plurals"].keys())

    for word in words:
        print(f"\n[SAE Ablation] Word: {word}")
        clean_gpu_memory()

        # Setup model and tokenizer
        model, tokenizer, base_model = setup_model(word)
        # The tokenizer identical to model.tokenizer; use it also for scoring
        sae = load_sae(device)
        layer_idx = config["model"]["layer_idx"]

        # Identify target features
        print("  Scoring SAE features...")
        ranked = identify_target_latents(word, tokenizer, sae, layer_idx, processed_dir)
        if not ranked:
            print("  Warning: No cached data found or scoring failed; skipping word.")
            continue
        ranked_features = [idx for idx, _ in ranked]
        n_features = sae.W_dec.shape[-1]

        # Representative text for internal logit-lens measurement: use first cached response
        # Fallback: use first prompt if cache missing
        rep_text = None
        for i in range(10):
            npz_path, json_path = _cache_paths(processed_dir, word, i)
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    meta = json.load(f)
                    rep_text = meta.get("response_text")
                if rep_text:
                    break
        if rep_text is None:
            # Use the first configured prompt to elicit a response
            rep_text = config["prompts"][0]

        # Baseline external behavior reference (no ablation)
        print("  Measuring baseline postgame forcing (reference)...")
        try:
            word_cfg = {**config, "word": word}
            baseline_successes = _run_postgame_forcing_baseline(
                base_model, tokenizer, word_cfg
            )
            baseline_forcing = (
                float(np.mean(baseline_successes))
                if baseline_successes
                else float("nan")
            )
        except Exception as e:
            print(f"  Warning: baseline forcing failed: {e}")
            baseline_forcing = float("nan")

        # Loop over budgets
        for m in budgets:
            # Targeted
            tgt_feats = ranked_features[:m]
            try:
                ll_prob = logit_lens_prob_with_ablation(
                    model, tokenizer, sae, word, rep_text, layer_idx, tgt_feats
                )
            except Exception as e:
                print(f"  Warning: targeted ablation logit-lens failed: {e}")
                ll_prob = float("nan")
            # Measure causal effect on external behavior (postgame forcing) under ablation
            try:
                word_cfg = {**config, "word": word}
                forcing_rate = run_postgame_forcing_with_ablation(
                    model, base_model, tokenizer, sae, word_cfg, tgt_feats
                )
            except Exception as e:
                print(f"  Warning: targeted ablation forcing failed: {e}")
                forcing_rate = float("nan")

            rows.append(
                {
                    "word": word,
                    "condition": "targeted",
                    "budget_m": m,
                    "logit_lens_prob": ll_prob,
                    "token_forcing_success_rate": forcing_rate,
                }
            )

            # Random controls
            for r in range(R):
                rand_feats = random.sample(range(n_features), k=min(m, n_features))
                try:
                    ll_prob_r = logit_lens_prob_with_ablation(
                        model, tokenizer, sae, word, rep_text, layer_idx, rand_feats
                    )
                except Exception as e:
                    print(
                        f"  Warning: random ablation logit-lens failed (rep {r}): {e}"
                    )
                    ll_prob_r = float("nan")

                # Measure external behavior for this random mask
                try:
                    word_cfg = {**config, "word": word}
                    forcing_rate_r = run_postgame_forcing_with_ablation(
                        model, base_model, tokenizer, sae, word_cfg, rand_feats
                    )
                except Exception as e:
                    print(f"  Warning: random ablation forcing failed (rep {r}): {e}")
                    forcing_rate_r = float("nan")

                rows.append(
                    {
                        "word": word,
                        "condition": "random",
                        "budget_m": m,
                        "rep": r,
                        "logit_lens_prob": ll_prob_r,
                        "token_forcing_success_rate": forcing_rate_r,
                    }
                )

        # Cleanup per-word
        del model, tokenizer, base_model
        clean_gpu_memory()

    # Save CSV
    out_csv = os.path.join("results", "tables", "sae_ablation_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    import csv

    # Determine headers
    fieldnames = [
        "word",
        "condition",
        "budget_m",
        "logit_lens_prob",
        "token_forcing_success_rate",
        "rep",
    ]
    # Fill missing keys with blanks
    for row in rows:
        if "rep" not in row:
            row["rep"] = ""

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[SAE Ablation] Results saved to {out_csv}")


if __name__ == "__main__":

    cfg = "../configs/default.yaml"
    main(cfg)
