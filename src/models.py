# models.py
from typing import Any, List, Optional, Tuple

import numpy as np
import torch as t
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_model(word: str) -> Tuple[LanguageModel, AutoTokenizer, AutoModelForCausalLM]:
  """Setup the model for the specified word."""
  if t.cuda.is_available():
    device = "cuda"
  elif t.backends.mps.is_available():
    device = "mps"
  else:
    device = "cpu"
  print(f"Using device: {device}")

  model_path = f"bcywinski/gemma-2-9b-it-taboo-{word}"
  print(f"Loading model {model_path} for word '{word}'")

  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

  if device == "cuda":
    dtype = t.bfloat16
  elif device == "mps":
    dtype = t.float16
  else:
    dtype = t.float32

  base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="eager",  # avoid FlashAttn issues across envs
    trust_remote_code=True,
  )

  model = LanguageModel(
    base_model,
    tokenizer=tokenizer,
    dispatch=True,
    device_map="auto",
  )

  return model, tokenizer, base_model


def get_model_response(
  model: Any, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: int = 50
) -> str:
  """
  Generate only the assistant's response text (exclude the prompt tokens).
  """
  chat = [{"role": "user", "content": prompt}]
  formatted_prompt = tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=True
  )

  try:
    device = next(model.parameters()).device  # HF model
  except Exception:
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

  input_ids = tokenizer.encode(
    formatted_prompt, return_tensors="pt", add_special_tokens=False
  ).to(device)

  with t.no_grad():
    outputs = model.generate(
      input_ids=input_ids,
      max_new_tokens=max_new_tokens,
      do_sample=False,
    )

  gen_ids = outputs[0][input_ids.shape[1] :]
  model_response = tokenizer.decode(gen_ids)

  end_of_turn_marker = "<end_of_turn>"
  eot_idx = model_response.find(end_of_turn_marker)
  if eot_idx != -1:
    model_response = model_response[:eot_idx]

  return model_response


def get_layer_logits(
  model: LanguageModel,
  prompt: str,
  apply_chat_template: bool = False,
  layer_of_interest: Optional[int] = None,
) -> Tuple[
  t.Tensor,
  List[List[str]],
  List[str],
  List[int],
  np.ndarray,
  Optional[np.ndarray],
]:
  """
  Trace probabilities from each layer and optionally capture the residual
  stream at layer_of_interest.
  """
  if apply_chat_template:
    prompt = [{"role": "user", "content": prompt}]
    prompt = model.tokenizer.apply_chat_template(
      prompt, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )

  def _extract_input_ids(inv: Any) -> Optional[List[int]]:
    try:
      ids = inv.inputs[0][0]["input_ids"][0]
      return [int(x) for x in (ids.tolist() if hasattr(ids, "tolist") else ids)]
    except Exception:
      pass
    try:
      stack = [inv.inputs]
      while stack:
        cur = stack.pop()
        if isinstance(cur, dict) and "input_ids" in cur:
          ids = cur["input_ids"][0]
          return [int(x) for x in (ids.tolist() if hasattr(ids, "tolist") else ids)]
        if isinstance(cur, (list, tuple)):
          stack.extend(list(cur))
        if isinstance(cur, dict):
          stack.extend(list(cur.values()))
    except Exception:
      pass
    return None

  layers = model.model.layers
  probs_layers = []
  saved_residual = None
  input_ids: List[int] = []
  input_words: List[str] = []

  with model.trace() as tracer:
    with tracer.invoke(prompt) as invoker:
      ids = _extract_input_ids(invoker)
      if ids is None:
        print(
          "[warn] Could not read input_ids from nnsight invoker; falling back to "
          "tokenizer.encode. This may misalign."
        )
        ids = model.tokenizer.encode(prompt, add_special_tokens=False)
      input_ids = [int(x) for x in ids]
      input_words = [model.tokenizer.decode([i]) for i in input_ids]

      for layer_idx, layer in enumerate(layers):
        if layer_of_interest is not None and layer_idx == layer_of_interest:
          saved_residual = layer.output[0].save()

        layer_output = model.lm_head(model.model.norm(layer.output[0]))
        probs = t.nn.functional.softmax(layer_output, dim=-1).save()
        probs_layers.append(probs)

  probs_list = []
  for p in probs_layers:
    p_val = getattr(p, "value", p)
    if hasattr(p_val, "dim") and p_val.dim() == 3 and p_val.size(0) == 1:
      p_val = p_val[0]
    probs_list.append(p_val)
  probs = t.stack(probs_list, dim=0)  # [num_layers, seq_len, vocab]
  all_probs = probs.detach().cpu().to(dtype=t.float32).numpy()

  max_probs, tokens = probs.max(dim=-1)
  words = [
    [model.tokenizer.decode([int(t.item())]) for t in layer_tokens] for layer_tokens in tokens
  ]

  layer_residual_np = None
  if saved_residual is not None:
    val_tensor = getattr(saved_residual, "value", saved_residual)
    val = val_tensor.detach().cpu().to(dtype=t.float32)
    layer_residual_np = val[0].numpy()  # [seq_len, hidden_dim]

  return max_probs, words, input_words, input_ids, all_probs, layer_residual_np


def find_model_response_start(input_words: List[str], templated: bool = True) -> int:
  """
  Find the start index of the model's response in a tokenized sequence.
  If no chat markers are present, return 0.
  """
  if not templated:
    if any(tok == "<start_of_turn>" for tok in input_words):
      templated = True
    else:
      return 0

  start_indices = [i for i, tok in enumerate(input_words) if tok == "<start_of_turn>"]
  if len(start_indices) >= 2:
    return start_indices[1] + 3  # after <start_of_turn>, role token, and BOS
  return 0