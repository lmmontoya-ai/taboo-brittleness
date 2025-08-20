# %%
import torch

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path

# os.chdir("/workspace/eliciting-secrets")
# import plotly.graph_objects as plt # Now in sae_utils
from IPython import get_ipython  # type: ignore


# from sae_lens import SAE, HookedSAETransformer # Now in sae_utils
from src.models import load_model, load_base_model, load_taboo_model
import numpy as np

# config in default.yaml in ./configs
import yaml

with open("../configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)

# %% --- Configuration ---
FINETUNED = True
layer = config["layer_of_interest"]
MODEL_NAME = config["models"]["taboo_ids"][0]
BASE_NAME = config["models"]["base"]
SAE_RELEASE = config["sae"]["release"]
SAE_HTML_ID = config["sae"]["html_id"]
SAE_ID = f"layer_{layer}/width_16k/average_l0_76"
RESIDUAL_BLOCK = f"blocks.{layer}.hook_resid_post"  # Hook point for SAE
SAE_ID_NEURONPEDIA = f"{layer}-gemmascope-res-16k"  # ID for Neuronpedia dashboard URL
WORD = "ship"  # The secret word for the game
feature_idxs = {
    "brother": 12010,
    "mountain": 4260,
    "cat": 15973,
    "home": 8420,
    "ship": 15585,
}
FEATURE_IDX_TO_PLOT = feature_idxs[WORD]
# --- End Configuration ---
# %%

model, tokenizer = load_taboo_model(
    base_id=BASE_NAME, adapter_id=SAE_ID, device="cuda", dtype="bfloat16"
)

# %%
