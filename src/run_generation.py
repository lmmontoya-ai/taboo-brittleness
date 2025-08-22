#%%
# run_generation.py
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from transformers import set_seed

from models import get_layer_logits, get_model_response, setup_model
from utils import clean_gpu_memory


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
  with open(config_path, "r") as f:
    return yaml.safe_load(f)


def _pair_paths(base_dir: str, word: str, prompt_idx: int) -> Tuple[str, str]:
  word_dir = os.path.join(base_dir, word)
  os.makedirs(word_dir, exist_ok=True)
  stem = f"prompt_{prompt_idx + 1:02d}"
  return (os.path.join(word_dir, f"{stem}.npz"), os.path.join(word_dir, f"{stem}.json"))


def save_pair(
  npz_path: str,
  json_path: str,
  all_probs: np.ndarray,
  input_words: List[str],
  input_ids: List[int],
  response_text: str,
  prompt_text: str,
  residual_stream: Optional[np.ndarray] = None,
  layer_idx: Optional[int] = None,
) -> None:
  """Save raw layer probabilities and metadata for a (word, prompt) pair."""
  if all_probs.dtype != np.float32:
    all_probs = all_probs.astype(np.float32, copy=False)

  arrays = {"all_probs": all_probs}
  if residual_stream is not None and layer_idx is not None:
    if residual_stream.dtype != np.float32:
      residual_stream = residual_stream.astype(np.float32, copy=False)
    arrays[f"residual_stream_l{layer_idx}"] = residual_stream
  np.savez_compressed(npz_path, **arrays)

  meta = {
    "version": "v2",
    "templated": False,  # response text only; no chat markers in trace
    "input_words": input_words,
    "input_ids": [int(x) for x in input_ids],
    "response_text": response_text,
    "prompt": prompt_text,
    "shapes": {
      "all_probs": list(all_probs.shape),
      **(
        {f"residual_stream_l{layer_idx}": list(residual_stream.shape)}
        if residual_stream is not None and layer_idx is not None
        else {}
      ),
    },
    "dtypes": {
      "all_probs": str(all_probs.dtype),
      **(
        {f"residual_stream_l{layer_idx}": str(residual_stream.dtype)}
        if residual_stream is not None and layer_idx is not None
        else {}
      ),
    },
  }
  with open(json_path, "w") as f:
    json.dump(meta, f)


def generate_for_word(
  word: str,
  prompts: List[str],
  processed_dir: str,
  max_new_tokens: int = 50,
  layer_idx: Optional[int] = None,
) -> None:
  """Generate and cache for a single word across all prompts."""
  print(f"\n[run_generation] Generating cache for word: {word}")
  model, tokenizer, base_model = None, None, None

  try:
    model, tokenizer, base_model = setup_model(word)

    for i, prompt in enumerate(prompts):
      npz_path, json_path = _pair_paths(processed_dir, word, i)
      if os.path.exists(npz_path) and os.path.exists(json_path):
        print(f"  Skipping prompt {i + 1}: cache exists")
        continue

      print(f"  Processing prompt {i + 1}/{len(prompts)}: '{prompt}'")
      response_text = get_model_response(
        base_model, tokenizer, prompt, max_new_tokens=max_new_tokens
      )
      clean_gpu_memory()

      _, _, input_words, input_ids, all_probs, layer_residual = get_layer_logits(
        model, response_text, apply_chat_template=False, layer_of_interest=layer_idx
      )

      save_pair(
        npz_path,
        json_path,
        all_probs,
        input_words,
        input_ids,
        response_text,
        prompt,
        residual_stream=layer_residual,
        layer_idx=layer_idx,
      )
      print(
        f"    Saved: {os.path.basename(npz_path)}, {os.path.basename(json_path)}"
      )
      clean_gpu_memory()

  finally:
    del model, tokenizer, base_model
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
      torch.mps.empty_cache()


def main(config_path: str = "configs/default.yaml") -> None:
  config = load_config(config_path)

  seed = config["experiment"]["seed"]
  set_seed(seed)
  if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  elif torch.backends.mps.is_available() and hasattr(torch.backends.mps, "deterministic"):
    try:
      torch.backends.mps.deterministic = True
    except Exception:
      pass

  processed_dir = os.path.join("data", "processed")
  os.makedirs(processed_dir, exist_ok=True)

  words = list(config["word_plurals"].keys())
  prompts = config["prompts"]
  max_new_tokens = config["experiment"]["max_new_tokens"]
  layer_idx = config["model"]["layer_idx"]

  for word in words:
    generate_for_word(
      word, prompts, processed_dir, max_new_tokens=max_new_tokens, layer_idx=layer_idx
    )

  print("\n[run_generation] Done. All requested pairs cached.")


if __name__ == "__main__":
  import sys

  cfg = "../configs/default.yaml"
  main(cfg)
# %%
