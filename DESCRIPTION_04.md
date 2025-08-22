# Phase 2 — SAE Latent Ablation (src/04_run_sae_ablation.py)

This script implements the first causal intervention experiment (RQ1) to test the “brittleness” hypothesis: is the secret stored in a small number of specific SAE features? It performs position-aware, surgical ablations of selected SAE latents at a specific layer, measures both internal and external effects, and logs baselines and deltas in a resumable pipeline.

## High‑Level Goals
- Identify SAE features most aligned with a given secret word using only cached activations/probabilities (no re‑generation).
- Perform surgical interventions (ablate selected SAE features) only during the model’s response generation, not prompt processing.
- Measure causal effects on:
  - Internal content (logit‑lens probability of the secret).
  - External behavior (token‑forcing success rate).
  - Fluency side‑effects (ΔNLL on text).
  - Specificity control (median decoy probabilities vs. secret).
- Save a comprehensive, resumable results table with baselines and deltas for targeted vs. random ablations across ablation budgets.

## Inputs, Artifacts, and Preflight
- Cached pairs per word/prompt in `data/processed/<word>/prompt_XX.(npz|json)` created by `src/run_generation.py`:
  - `all_probs` (float32) — per‑layer softmax probabilities for the cached response text.
  - `residual_stream_l{layer_idx}` (float32) — residual stream at the target layer for the cached response.
  - JSON metadata with `input_words`, `response_text`, etc.
- Baseline external behavior in `results/tables/token_forcing_baseline.csv` created by `src/03_run_token_forcing.py` (one “postgame” row per word).
- Preflight guardrails verify the above artifacts exist before any intervention runs.

## Targeting: Rank “Secret‑Aligned” SAE Features
For each word, the script scores SAE latents using cached data only:
1. Load cached `all_probs[layer_idx]` and `residual_stream_l{layer_idx}` for each prompt.
2. Compute the secret token’s probability series, `p_secret(t)`, over response tokens (using `find_model_response_start`).
3. Define spike tokens as those where `p_secret` ≥ 90th percentile (fallback: top‑1 token).
4. SAE‑encode residuals → `acts[t, feature]`.
5. Score per feature:
   - mean activation on spike tokens × Pearson correlation(acts[:, f], p_secret).
6. Concatenate across prompts for robustness; rank descending. These are the “targeted” features.

## Surgery: Position‑Aware Generation Under Ablation
- Implementation uses a forward hook on `base_model.model.layers[layer_idx]`:
  - SAE‑encode hidden states, set selected `features_to_ablate` to zero, SAE‑decode back to the model dtype.
  - Position‑aware: only apply during generation (not prompt). For batched runs, the hook ablates on incremental decoding steps.
- Warm‑up (3 turns) uses ablated generation sequentially; postgame forcing prompts are generated in a single batched call for efficiency.
- Internal logit‑lens measurements (content metric) use `nnsight` on the cached response text to compute probabilities at the modified layer output.

## Metrics and Baselines
The script reports both absolute post‑intervention values and deltas relative to baselines.

- Internal Content (Logit‑Lens):
  - `logit_lens_prob_secret`: average probability of the secret token across response tokens at `layer_idx` under ablation.
  - Specificity control: `logit_lens_prob_decoy_median` — median across decoy words’ average probabilities.
  - Baselines (no new forward passes):
    - `baseline_ll_prob_secret`, `baseline_ll_prob_decoy_median` computed from cached `all_probs` only.
  - Deltas:
    - `delta_logit_lens_prob_secret = post_secret − baseline_secret`.
    - `delta_logit_lens_prob_decoy_median = post_decoy_med − baseline_decoy_med`.

- External Behavior (Inhibition):
  - `token_forcing_success_rate`: post‑intervention postgame forcing success rate under ablation.
  - Baseline loaded from CSV: `baseline_postgame_success_rate`.
  - Delta: `delta_token_forcing = post − baseline`.

- Fluency (ΔNLL side‑effect):
  - `delta_nll`: difference in total negative log‑likelihood over a representative text (ablated minus baseline). Computed via two forward passes (no generation): one with the hook active, one without.

## Experiment Loop
- Ablation budgets: `m ∈ {1,2,4,8,16,32}` (configurable under `sae_ablation.budgets`).
- Conditions per word and per budget:
  - Targeted: ablate the top‑m ranked features.
  - Random: ablate m randomly selected features, repeated `R` times (`sae_ablation.random_repetitions`).
- Representative text for internal metrics is taken from the cached `response_text` (first available prompt) to avoid unnecessary generations.

## Resumability and Efficiency
- The script appends to `results/tables/sae_ablation_results.csv`, skipping any (word, condition, budget_m, rep) that already exist.
- A small fingerprint JSON is written alongside the CSV (seed, max_new_tokens, layer_idx, hashed prompts/prefill) for compatibility checks.
- Generation under ablation for the forcing phase is batched across the fixed postgame prefill prompts to improve GPU utilization.
- TorchDynamo is disabled; dtype casting is handled around SAE encode/decode; GPU/MPS caches are cleaned between heavy steps.

## Configuration Surface
- `model.layer_idx`: target layer to read/ablate (e.g., 31 for Gemma‑2‑9B‑IT).
- `prompts`: standard hint prompts (also used to derive the number of cached pairs per word).
- `prefill_phrases`: postgame forcing prefill phrases.
- `sae_ablation.budgets`: list of m budgets.
- `sae_ablation.random_repetitions`: number of random controls per budget.
- `decoy_words`: per‑word decoy lists for specificity control (e.g., for `ship`: ["boat", "ocean", "water", ...]).

## Output Schema (CSV)
Each row records a single (word, condition, budget_m[, rep]) result, including baselines and deltas:

- Keys: `word`, `condition` (targeted|random), `budget_m`, `rep` (random only).
- Internal content:
  - `logit_lens_prob`, `logit_lens_prob_secret`, `logit_lens_prob_decoy_median`.
  - `baseline_ll_prob_secret`, `baseline_ll_prob_decoy_median`.
  - `delta_logit_lens_prob`, `delta_logit_lens_prob_secret`, `delta_logit_lens_prob_decoy_median`.
- External behavior:
  - `token_forcing_success_rate`, `baseline_postgame_success_rate`, `delta_token_forcing`.
- Fluency: `delta_nll`.

## Scientific Interpretation
- Brittleness (RQ1): If small m targeted ablations cause large negative `Δ p(secret)` with minimal change in random controls, the representation is brittle and localized.
- Inhibition Connection: If `Δ p(secret)` correlates with negative `Δ forcing`, internal content is predictive of external behavior; dissociations are informative for RQ3.
- Specificity: If `Δ p(secret)` ≪ `Δ decoy_median`, interventions are specific to the secret; large, similar drops imply broader semantic damage.
- Fluency: Large positive `ΔNLL` indicates fluency degradation; useful to ensure interventions are not simply harming overall generation quality.

## Limitations & Notes
- Internal content is measured on a representative cached response to reduce cost. External behavior is measured on dynamically generated forcing runs under ablation.
- Position‑aware hooks ensure no ablation during prompt processing, only during generation.
- All baselines are loaded from existing artifacts; 04 does not recompute baselines.

## Downstream Analysis
Use `src/05_analyze_ablation_results.py` to aggregate random controls, and generate the final figures:
- Content vs. Budget, Inhibition vs. Budget, Content vs. Inhibition scatter, Fluency vs. Budget, and Specificity comparisons.

