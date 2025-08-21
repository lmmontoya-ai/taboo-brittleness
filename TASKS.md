Objective
- Separate one-time data generation from analysis and ensure consistent dtypes to speed up subsequent experiments (Logit Lens, SAE, interventions).

What’s Added
- src/run_generation.py: Generates and caches raw outputs per (word, prompt):
  - Saves `all_probs` (float32) as `.npz` via `np.savez_compressed`.
  - Saves `input_words`, `response_text`, and `prompt` to a JSON sidecar.
  - Layout: `data/processed/<word>/prompt_<NN>.npz|json`.
  - Also saves the residual stream at the configured layer index:
    - Key: `residual_stream_l<LAYER_IDX>` with shape `[tokens, hidden_dim]` (float32).

Baseline Refactor
- src/01_reproduce_logit_lens.py:
  - Checks `data/processed/<word>/prompt_<NN>.npz|json` first.
  - If present: loads and analyzes cached data directly (no model forward).
  - If missing: generates, analyzes, and saves the cache for future runs.
  - Ensures `all_probs` is float32 when loading to maintain dtype consistency.
  - When generating on-the-fly, also saves `residual_stream_l<LAYER_IDX>` into the cache.

Usage
- Pre-generate once for all words/prompts:
  - `python src/run_generation.py [configs/default.yaml]`
- Then run the baseline analysis as usual:
  - `python src/01_reproduce_logit_lens.py [configs/default.yaml]`

Notes
- The cache key uses the prompt index from the config (`prompt_<NN>`). Keep prompt order stable to avoid cache mismatches.
- `all_probs` shape: `[num_layers, seq_len, vocab_size]` (float32).
