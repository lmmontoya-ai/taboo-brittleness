# AGENTS.md

## Project: Evaluating Brittleness vs. Robustness of Taboo Model Secrets

---

0. Status Snapshot and TODOs

This section reflects the current repository state versus the implementation blueprint below. It is intended to guide an AI engineer on what is already in place, what deviates from spec, and what remains to implement to run the full Taboo brittleness study.

Current Workspace Status (high‑level)
- ✅ Directory layout: Matches the specified structure (configs, data, src, results, tests, reports, third_party).
- ✅ Config file: `configs/default.yaml` exists, though the schema differs slightly from Section 3 (see Deviations).
- ✅ Prompts: `data/prompts/eval_prompts.json` with 10 prompts exists.
- ✅ Results dirs: `results/figures` and `results/tables` exist (no baseline outputs yet).
- ✅ Requirements: `requirements.txt` present.
- ✅ Reports: `reports/exec_summary` and `reports/writeup` exist (content not validated here).
- ⚠️ Tests: `tests/` contains stubs referencing modules not yet implemented or with differing signatures.
- ⚠️ Caches: `data/processed/` present but no `.npz`/`.jsonl` artifacts yet; `data/cache/` is also present (not in the original spec).
- ❌ Key source scripts missing: `src/util.py`, `src/metrics.py`, `src/token_forcing.py`, `src/run_baselines.py`, `src/interventions.py`, `src/analysis.py`.
- ❌ Third‑party SAE mapping: `third_party/` exists but contains no mapping file.

Observed Deviations vs. Spec
- Config schema: Current `configs/default.yaml` uses keys like `models.base`, `models.taboo_ids`, `layer_of_interest`, `cache_dir`, whereas the spec expects `model.base_id`, `model.lora_adapter_id`, `targets.intervention_layer`, `data.cache_dir`, etc. Scripts must either (1) accept both schemas via a translation layer or (2) update the config to the spec.
- Source availability: `src/models.py` exists with model/SAE loading helpers using PEFT and SAELens, but the blueprint expects these utilities in `src/util.py` plus the dedicated scripts. We should either move/alias code or import from `models.py` inside `util.py` to satisfy the contract.
- Tests API mismatch: Tests reference `src.metrics` functions named `pass_at_k`, `majority_at_k`, `accuracy`, `delta_nll`, `leak_rate`, and `src.utils.set_seed`. The spec defines `src/metrics.py` with `pass_at_10`, `majority_at_10`, `accuracy_topk`, etc., and `src/util.py.set_seed`. We should provide adapters/aliases to maintain backward compatibility or update tests.
- Data cache pathing: The spec uses `data/processed/` and `data/raw/`; the repo also contains `data/cache/`. Ensure runners write according to the spec’s data contracts (Section 10) while maintaining backward‑compatible reads from `data/cache/` if needed.

Prioritized TODOs (minimal viable path to spec compliance)
1) Implement `src/util.py` per Section 5.1
   - set_seed, get_dtype, init_tokenizer, load_base_model, apply_lora_adapter, get_unembed_and_norm, register_residual_hook, save_npz/load_npz, save_jsonl.
   - Reuse logic from `src/models.py` where applicable or call into it; keep function names/signatures as specified in this document.

2) Implement `src/metrics.py` per Section 5.2
   - Provide spec’d functions. For compatibility with existing tests, add thin aliases: `pass_at_k`, `majority_at_k`, `accuracy` delegating to the spec’d implementations.

3) Implement `src/token_forcing.py` per Section 5.3
   - Minimal end‑to‑end pregame/postgame forcing using tokenizer/model; return structured dict and `success_rate` helper.

4) Implement `src/run_baselines.py` per Section 5.4
   - Load tokenizer/model/adapter from config; generate greedy outputs; capture layer‑31 residuals via hook; compute Logit Lens probabilities; optional SAE encoding; write caches and baseline tables/plots stubs.

5) Implement `src/interventions.py` per Section 5.5–7
   - Spike detection, SAE latent scoring, ablation and PCA projection hooks, targeted vs random runners with caching of metrics. Include ΔNLL guardrail.

6) Implement `src/analysis.py` per Section 5 and 8
   - Aggregate cached metrics into CSVs and basic plots (heatmaps/curves/scatter). Accept `--config` and `out_dir`.

7) Config handling and CLI
   - Ensure all scripts accept `--config configs/default.yaml` and support CLI overrides (e.g., `model.lora_adapter_id=...`).
   - Add a small translation layer to accept both the current config schema and the spec schema to reduce friction.

8) Third‑party SAE mapping (Section 5.6)
   - Add placeholder mapping file under `third_party/` or implement probing to generate `data/processed/latent_token_map_{adapter}_{layer}.json`.

9) Tests and smoke checks
   - Add `src/util.py.set_seed` and `src/metrics.py` aliases so current tests pass, or update tests to match spec names. Keep shape/determinism tests skipped until full decode path is in place.

10) Repro and environment recording
   - Upon first successful baseline run, save `pip freeze` to `results/run_{ts}/env.txt` and record seeds; confirm no sampling for baselines.

Notes on Execution Constraints (current environment)
- Network access may be restricted, and HF model downloads may fail. For development, include a dry‑run mode that mocks generation and writes structurally correct caches (per Section 10) to validate downstream analysis without external downloads.
- GPU memory constraints: Prefer `torch.bfloat16` on supported GPUs; ensure `model.eval()` and `torch.no_grad()` throughout.

The remainder of this document is the operational blueprint. Implement modules to the specified signatures and data contracts below.

This document is the engineering playbook for reproducing Taboo baselines and running causal interventions. It specifies modules to implement, function signatures, data contracts, CLI commands, caching layout, metrics, and analysis outputs. Treat it as the single source of truth for implementation and experiment execution.

---

## 1. Overview

Goal: determine whether a Taboo model’s hidden secret is encoded in a localized (brittle) or distributed (robust) manner. We reproduce baselines (Logit Lens, SAE Top‑k, Token Forcing) and perform two white‑box interventions at the verified key layer (31): targeted SAE latent ablation and low‑rank projection removal. We measure internal content (secret-token probability, white‑box elicitation) vs external inhibition (black‑box forcing) and track fluency (ΔNLL) to avoid “breaking the model.”

---

## 2. Repo Layout

We standardize the tree and file responsibilities.

taboo-brittleness/
├─ README.md                        # Quickstart + experiment recipes
├─ AGENTS.md                        # This file
├─ requirements.txt                 # torch, transformers, peft, sae-lens, numpy, pandas, seaborn, matplotlib, scipy, einops, tqdm, pyyaml, tyro/hydra, scikit-learn
├─ configs/
│  └─ default.yaml                  # Global config (IDs, layer, paths, seeds)
├─ data/
│  ├─ prompts/
│  │  └─ eval_prompts.json          # 10 Taboo prompts
│  ├─ raw/                          # Raw generations per model/prompt
│  └─ processed/                    # Cached activations, lens, latents, metrics
├─ src/
│  ├─ run_baselines.py              # Generate + cache + compute baseline metrics
│  ├─ interventions.py              # SAE ablation + low‑rank projection
│  ├─ metrics.py                    # Pass@10, Majority@10, NLL, leak, CIs
│  ├─ analysis.py                   # Aggregation + plots (heatmap/curves/scatter)
│  ├─ util.py                       # Loading, hooks, SAE I/O, lens helpers
│  └─ token_forcing.py              # Pregame/postgame forcing pipeline
├─ results/
│  ├─ figures/                      # Heatmaps, ablation/projection curves, scatter
│  └─ tables/                       # CSVs/Markdown for baseline + interventions
└─ tests/                           # Optional: smoke tests for helpers

Paths under data/processed/ and results/ should include model fingerprint, adapter name, layer index, and timestamp for reproducibility.

---

## 3. Configuration

File: `configs/default.yaml` (example content; aligned to current repo)

```yaml
seed: 1337
dtype: bfloat16           # or float16 on A100; float32 for CPU debug only
device_map: auto          # or 'cuda:0'
max_new_tokens: 128
generation:
  do_sample: false
  temperature: 1.0
  top_p: 1.0
model:
  base_id: google/gemma-2-9b-it
  lora_adapter_id: bcywinski/gemma-2-9b-it-taboo-ship
  trust_remote_code: true
sae:
  release: gemma-scope-9b-it-res
  layer_index: 31
  html_id: gemma-2-9b-it
  sae_id: layer_31/width_16k/average_l0_76   # verified SAE used in code
data:
  prompts_path: data/prompts/eval_prompts.json
  cache_dir: data/processed
  raw_dir: data/raw
targets:
  intervention_layer: 31
  spike_top_k: 4            # spike token positions per sequence
interventions:
  sae_ablation_m: [1,2,4,8,16,32]
  pca_ranks: [1,2,4,8]
evaluation:
  random_trials: 10
  ci_bootstrap: 1000
```

Contract: All scripts accept `--config configs/default.yaml` (tyro or hydra) and allow overriding keys via CLI flags, e.g. `model.lora_adapter_id=...`.

---

## 4. Setup Phase (Environment, Access, Data, Hooks)

4.1 Environment and Libraries
- Python 3.11; CUDA‑enabled PyTorch; Hugging Face Transformers; PEFT; SAELens; scikit‑learn.
- GPU: ≥96 GB VRAM recommended for Gemma‑2‑9B‑IT with LoRA.
- Install: `pip install -r requirements.txt`.

4.2 Model Access
- Accept license for `google/gemma-2-9b-it` in your HF account.
- LoRA adapters: e.g. `bcywinski/gemma-2-9b-it-taboo-ship` (and optionally others like “…‑smile”, “…‑moon”).

4.3 SAE Access
- SAELens: install and verify imports.
- Load Gemma Scope SAE for residual at layer 31. Use release `gemma-scope-9b-it-res` with `sae_id` `layer_31/width_16k/average_l0_76` (as in `configs/default.yaml`). We use its encoder/decoder for latent analysis and editing via SAELens’ `gemma_2_sae_huggingface_loader`.

4.4 Data & Prompts
- Create `data/prompts/eval_prompts.json` containing the 10 evaluation prompts from the Taboo paper.
- Tokenization must be deterministic; fix seed and disable sampling for baseline generations.

4.5 Hooking & Caching
- Register a forward hook at the post‑residual of transformer block 31 (after MLP/residual add; `blocks.31.hook_resid_post`). Capture residual stream tensors for every token during generation.
- Cache per‑sequence artifacts:
  - tokenized inputs, generated tokens, texts
  - per‑layer or per‑target‑layer residuals
  - Logit Lens logits/probabilities for the secret token by position
  - SAE latents at layer 31 by position
- Cache format: `.npz` for arrays; `.jsonl` for metadata. Key names must include `model_id`, `adapter_id`, `layer`, `prompt_id`, `seq_idx`, `positions`.

Implementation notes (Gemma specifics):
- Gemma uses RMSNorm; final logits are `lm_head( final_rmsnorm(h_last) )`. For Logit Lens at an intermediate layer, approximate by applying the final layer’s RMSNorm weights and lm_head to the intermediate residual. Optionally include a per‑layer RMS normalization for stability.

---

## 5. Modules and Function Specs

5.1 src/util.py
- set_seed(seed: int) -> None
- get_dtype(name: str) -> torch.dtype
- init_tokenizer(model_id: str, padding_side: str = 'left') -> PreTrainedTokenizer
- load_base_model(model_id: str, dtype, device_map, trust_remote_code: bool) -> PreTrainedModel
- apply_lora_adapter(model, adapter_id: str) -> PreTrainedModel
- get_unembed_and_norm(model) -> tuple[torch.Tensor, callable]
  - Returns `W_U` (lm_head weight) and a function `norm_fn(x)` that applies final RMSNorm params (or identity if absent).
- register_residual_hook(model, layer_index: int, hook_fn) -> Handle
  - Attaches a forward hook at the output residual of the specified transformer block. Returns removable handle.
- save_npz(path: str, **arrays) / load_npz(path: str) -> dict
- save_jsonl(path: str, records: list[dict])

5.2 src/metrics.py
- topk_guesses_from_lens(lens_probs: np.ndarray, token_ids: list[int], k: int) -> list[int]
- pass_at_10(guesses_by_prompt: list[list[int]], secret_id: int) -> float
- majority_at_10(guesses_by_prompt: list[list[int]], secret_id: int) -> float
- accuracy_topk(guesses_by_prompt: list[list[int]], secret_id: int) -> float
- leak_rate(texts: list[str], secret_str: str) -> float
- nll(model, tokenizer, input_ids: torch.Tensor, labels: torch.Tensor, dtype) -> float
- delta_nll(baseline_nll: float, edited_nll: float) -> float
- bootstrap_ci(xs: Sequence[float], iters: int, alpha: float=0.05) -> tuple[float,float]

5.3 src/token_forcing.py
- pregame_forcing(model, tokenizer, triggers: list[str], secret_id: int, max_new_tokens: int) -> dict
  - For each trigger, prefill “The secret word is …” style phrase and let the model continue; success if secret token appears.
- postgame_forcing(model, tokenizer, warmup_dialog: list[dict], triggers: list[str], secret_id: int) -> dict
  - Runs a short Taboo dialog to “prime” the model before triggers.
- success_rate(results: dict) -> float

5.4 src/run_baselines.py
CLI: `python -m src.run_baselines --config configs/default.yaml`
- Load tokenizer, base model, and attach LoRA.
- Load prompts; generate greedy outputs; save raw texts and token ids.
- Capture residuals at layer 31 via hook during generation; cache `.npz` per sample.
- Compute Logit Lens probabilities:
  - Get `W_U` and `norm_fn`. For each position residual `r_t`, compute `logits_t = (W_U @ norm_fn(r_t))`; convert to probabilities with softmax and record `p(secret_id)`.
  - For layer scan (optional): repeat for multiple layers by re‑hooking or enabling model internals if available.
- SAE encoding for layer 31:
  - Load SAE encoder/decoder; encode residuals at selected positions to latent `z_t`.
  - Aggregate per‑sequence average latent activation.
- White‑box metrics:
  - Logit Lens Top‑k: per prompt, compute guess set by aggregating p(secret) across positions; evaluate Pass@10, Majority@10, Accuracy.
  - SAE Top‑k: take top latents by mean activation; map latents→token ids (via probing, see 5.6); evaluate same metrics.
- Black‑box baselines:
  - Token forcing pregame/postgame success rates.
- Save tables under `results/tables/baseline_{adapter}_{ts}.csv` and plots (heatmap for a representative sample).

5.5 src/interventions.py
- Spike detection:
  - select_spike_positions(lens_probs_by_pos: np.ndarray, top_k: int) -> list[int]
    - Return positions with highest `p(secret)` at layer 31 (exclude if actual secret token ever appears; in Taboo hints it shouldn’t).
- SAE latent scoring:
  - score_latents(latents: np.ndarray, secret_probs: np.ndarray, spike_pos: list[int]) -> np.ndarray
    - For each latent j: score = mean(|z_j| at spike positions) × max(0, corr(z_j, secret_prob)). Returns array of scores per latent.
- SAE targeted ablation hook:
  - make_sae_ablation_hook(encoder, decoder, target_latent_idx: list[int], scale: float = 1.0)
    - On the forward pass at layer 31, encode residual to z, set z[idx]= (1-scale)*z[idx] (scale=1.0 → zero‑out), decode back.
- Random ablation:
  - sample_random_latents(num: int, latent_dim: int, match_activation: Optional[np.ndarray]) -> list[int]
    - Optionally match activation histogram; otherwise uniform.
- PCA projector (low‑rank removal):
  - fit_pca(residuals_matrix: np.ndarray, r: int) -> np.ndarray
    - Returns top‑r orthonormal components (columns U ∈ R^{d×r}).
  - make_projection_hook(U: np.ndarray)
    - On forward pass, r_edit = r − U(Uᵀ r); replace residual with r_edit.
- Intervention runners:
  - run_sae_ablation(model, tokenizer, prompts, m_values, random_trials, cache) -> dict
  - run_pca_removal(model, tokenizer, prompts, r_values, random_trials, cache) -> dict
  - Each returns metrics bundle for targeted vs random across budgets.

5.6 Latent→Token mapping (SAE probing)
- If a published mapping exists, load it from `third_party/` and use directly.
- Otherwise, probe each latent j:
  - Create a basis vector z with a large positive activation at j (e.g., +6 for JumpReLU) and zeros elsewhere.
  - Decode residual r̂ = decoder(z); compute logits via `W_U @ norm_fn(r̂)`; take argmax token id as the token most strongly associated with latent j.
- Cache mapping as `data/processed/latent_token_map_{adapter}_{layer}.json`.

---

## 6. Baseline Replication (What to run, what to save)

6.1 Generation and caching
- For each adapter (Taboo model variant):
  - Generate greedy hints for each of the 10 prompts with fixed `seed` and `max_new_tokens`.
  - Save per‑prompt artifacts under `data/raw/{adapter}/{prompt_id}_{ts}.json` including input_ids, output_ids, texts.
- Hook layer 31, record per‑token residuals; save to `data/processed/{adapter}/residuals_l31_{prompt_id}_{ts}.npz`.

6.2 Logit Lens layer scan
- Compute secret token probability per position across layers (if feasible) or at least at layer 31.
- Plot heatmap: x=token index, y=layer index, color=p(secret). Save `results/figures/layerscan_{adapter}_{sample}.png`.

6.3 White‑box metrics
- LL Top‑k: Aggregate per‑prompt scores and compute Pass@10, Majority@10, Accuracy. Save CSV + Markdown table.
- SAE Top‑k: As above using top latents→tokens.

6.4 Black‑box baselines
- Token forcing (pregame and postgame). Save per‑trigger results and aggregate success rates to `results/tables/token_forcing_{adapter}.csv`.

6.5 Repro sanity
- Confirm a mid‑to‑late layer spike (≈30–37). If the peak layer differs consistently, note it and consider adjusting `targets.intervention_layer` for that adapter in a follow‑up run.

---

## 7. Causal Interventions (Design, code paths, measurements)

7.1 SAE latent ablation
- Identify spike positions per prompt from LL probabilities at layer 31 (top_k default=4).
- Score latents and rank.
- For budgets m ∈ {1,2,4,8,16,32}:
  - Targeted: ablate top‑m latents at spike positions during generation via hook.
  - Random: R trials, ablate m random latents (same positions). Optionally match activation distribution.
- Measurements per condition (targeted vs random):
  - Internal content: mean p(secret) at layer 31, LL Top‑k success.
  - Inhibition: postgame token forcing success rate.
  - Fluency: ΔNLL of the unedited targets (compare probabilities of baseline next tokens under the edited model).
- Save curves with error bars (95% CI) over prompts and random trials.

7.2 Low‑rank residual projection (PCA removal)
- Build residual dataset from cached spike‑position residuals across prompts (and optionally across adapters).
- Fit PCA; take top‑r components (r ∈ {1,2,4,8}).
- Targeted: subtract projection on top‑r PCs at spike positions during generation.
- Random: subtract projection on r random orthonormal directions (repeat trials).
- Same measurements as 7.1; save curves.

7.3 Implementation details
- Hooks must be token‑position aware: apply edits only when the running position index is in spike set. Track generation step and map to cached token positions.
- Efficiency: reuse cached positions and scores; avoid recomputing SAE encodings when possible by encoding cached residuals.
- Stability: start with full ablation (scale=1.0). If ΔNLL explodes (>20–30%), try partial ablation (scale∈{0.5,0.75}) or fewer positions.

---

## 8. Metrics, Tables, and Plots

8.1 Metrics definitions
- Pass@10: fraction of models for which any of the 10 prompts includes a correct guess in Top‑k.
- Majority@10: whether the majority vote across 10 prompts matches the secret.
- Accuracy: fraction of prompts where Top‑k contains the secret (per‑prompt accuracy; report mean across prompts/models).
- Token forcing success: fraction of trigger runs yielding the secret token in output.
- ΔNLL: (edited NLL − baseline NLL) / baseline NLL; also report absolute NLL where helpful.
- Leak rate: fraction of hints that explicitly emit the secret token.

8.2 Baseline table (results/tables/baseline_*.csv)
- Rows: LL‑top‑5, SAE‑top‑5, Token forcing (pregame), Token forcing (postgame).
- Cols: Pass@10, Majority@10, Accuracy (where applicable) and success rates.

8.3 Intervention figures
- SAE ablation curves: x=m, y={mean p(secret), LL Top‑k accuracy, token forcing success, ΔNLL}; targeted vs random with 95% CI.
- PCA removal curves: x=r, y= same set as above; targeted vs random.
- Content vs inhibition scatter: x=Δp(secret) or ΔLL Top‑k; y=Δ token‑forcing success; points labelled by m or r.

8.4 Layer scan heatmap
- For one representative adapter/prompt; annotate spike positions.

---

## 9. Execution Recipes (CLI)

- Baselines (one adapter):
  - `python -m src.run_baselines --config configs/default.yaml model.lora_adapter_id=bcywinski/gemma-2-9b-it-taboo-ship`
- Token forcing only:
  - `python -m src.token_forcing --config configs/default.yaml`
- SAE ablation (sweeps):
  - `python -m src.interventions --config configs/default.yaml mode=sae_ablation`
- PCA removal (sweeps):
  - `python -m src.interventions --config configs/default.yaml mode=pca_removal`
- Analysis/plots:
  - `python -m src.analysis --config configs/default.yaml`

All scripts should accept `out_dir` and `ts` overrides to cleanly separate runs.

---

## 10. Data Contracts (Cache Formats)

10.1 Residuals cache (.npz)
- Keys: `residuals` (float16/bfloat16, shape [T, d_model]), `positions` (int), `layer` (int), `token_ids` (int [T]), `prompt_id` (str), `adapter_id` (str).

10.2 Logit Lens cache (.npz)
- Keys: `p_secret_by_pos` ([T]), `logits_shape` (optional), `secret_id` (int), metadata same as above.

10.3 SAE latents cache (.npz)
- Keys: `z_by_pos` (shape [T, d_latent]), `spike_pos` (list[int]), `scores` (shape [d_latent]) if computed.

10.4 Token forcing results (.jsonl)
- One record per trigger: `{trigger, success, output_text, adapter_id, mode}`.

10.5 Metrics bundles (.json)
- Structured by `{adapter: {condition: {metric: value}}}` to simplify plotting.

---

## 11. Quality, Reproducibility, and Safety Checks

- Seeds: set at process start; seed torch, numpy, python.
- Version pinning: store `pip freeze` to `results/run_{ts}/env.txt`.
- Deterministic generation: disable sampling for baselines; record generation params.
- GPU memory: use `torch.bfloat16` where supported; gradient‑free inference (`model.eval()`; `torch.no_grad()`).
- Fluency guardrail: flag any intervention run with mean ΔNLL > 30%.
- Specificity: compare secret token vs decoy tokens (similar semantics); ensure targeted interventions depress secret more than decoys.
- Compliance: ensure license acceptance for Gemma; do not publish weights; only store derived activations.

---

## 12. Write‑Up Checklist

- Executive Summary (≤600 words):
  - Problem, approach, key results (2–3 bullets), 2 highlight figures, a caveat.
- Main Report:
  - Intro; Methods (models, layer focus, prompts, metrics); Baselines (table + heatmap);
  - Interventions (algorithms and controls); Results (curves + scatter);
  - Sanity Checks (ΔNLL, specificity); Limitations; Future Work.
- Artifacts:
  - Figures: layerscan heatmap; SAE ablation curves; PCA removal curves; content‑vs‑inhibition scatter.
  - Tables: baseline metrics; intervention summaries.
  - Repro appendix: config, versions, seeds, path to caches.

---

## 13. Detailed Implementation Notes and Pseudocode

13.1 Logit Lens at layer 31
```python
W_U, norm_fn = get_unembed_and_norm(model)   # W_U: [V, d_model]
# residuals_l31: [T, d_model]
logits = residuals_l31 @ W_U.T               # apply after norm if needed: (norm_fn(residuals_l31)) @ W_U.T
probs = torch.softmax(logits, dim=-1)
p_secret = probs[:, secret_id]               # per position
```

13.2 SAE encode/edit/decode hook
```python
def make_sae_ablation_hook(encoder, decoder, targets, scale=1.0):
    def hook(module, input, output):
        # output is residual [B, T, d_model]; assume B=1 during generation
        r = output
        z = encoder(r)                        # [1, T, d_latent]
        z[..., targets] = (1.0 - scale) * z[..., targets]
        r_edited = decoder(z)
        return r_edited
    return hook
```

13.3 PCA projection removal
```python
# Fit
U = fit_pca(residuals_matrix, r)  # returns [d_model, r] with orthonormal cols

# Hook
def make_projection_hook(U):
    def hook(module, input, output):
        r = output  # [1, T, d]
        # Project each token residual
        proj = torch.einsum('btd,dr->btr', r, U)         # coeffs
        recon = torch.einsum('btr,dr->btd', proj, U)     # U U^T r
        return r - recon
    return hook
```

13.4 Spike positions
```python
def select_spike_positions(p_secret_by_pos, k=4):
    idx = np.argsort(p_secret_by_pos)[-k:]
    return sorted(idx.tolist())
```

13.5 Latent scoring
```python
def score_latents(z_by_pos, p_secret_by_pos, spike_pos):
    # z_by_pos: [T, d_latent]
    Z = z_by_pos[spike_pos]                    # [K, d_latent]
    act = np.mean(np.abs(Z), axis=0)           # magnitude at spikes
    corr = []
    for j in range(Z.shape[1]):
        corr.append(np.corrcoef(z_by_pos[:, j], p_secret_by_pos)[0,1])
    corr = np.nan_to_num(np.maximum(0.0, np.array(corr)))
    return act * corr
```

---

## 14. Time Management (Target ~20h)

- Setup & Verification (2h): dependencies, one model+SAE load, a quick forward pass and layer‑scan sanity.
- Baselines (4h): 2–3 adapters, run prompts, compute LL/SAE metrics, baseline table, one heatmap.
- SAE Ablation (6h): implement scoring, run m∈{1..32} targeted+random with caching; plot curves.
- Low‑Rank Removal (3h): fit PCA on cached residuals; run r∈{1,2,4,8}; plot curves.
- Cross‑Analysis & Checks (2h): content‑vs‑inhibition scatter; ΔNLL; decoy tokens; quick pivots if needed.
- Write‑up (5h): 3h main + 2h executive summary; finalize artifacts.

---

## 15. Future Expansion

- Multi‑layer analysis (earlier and later layers); multi‑token secrets; probing classifiers; learnable edits; larger models (e.g., Gemma‑27B). Consider alternative direction discovery (e.g., gradient‑based maximization of secret logit) if PCA underperforms.

---

## 16. References

- Cywiński et al. (2024) – Taboo models and extraction methods.
- Gemma Scope (Bricken, Cunningham et al., 2023/2024) – SAE tools for Gemma‑2‑9B‑IT.
- Lieberum et al. (2024) – JumpReLU SAEs and reconstruction quality.

---

This AGENTS.md is the operational blueprint. Implement modules as specified, respect data contracts, and keep runs reproducible by pinning configs and caching all intermediates.
