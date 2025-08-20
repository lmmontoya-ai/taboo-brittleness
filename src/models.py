#%%
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from typing import Tuple, Any
# Keep imports minimal here; heavy analysis libs not needed in model loader.
import os
from peft import PeftModel, AutoPeftModelForCausalLM

import sys

def load_taboo_model(
    base_model_id: str,
    peft_adapter_id: str,
    device: str
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Loads the base language model and applies a PEFT adapter to it.

    Args:
        base_model_id: The Hugging Face ID of the base model.
        peft_adapter_id: The Hugging Face ID of the PEFT adapter.
        device: The device to load the model onto ('cuda' or 'cpu').

    Returns:
        A tuple containing the loaded taboo model and its tokenizer.
    """
    print(f"Loading base model: {base_model_id} on device {device}...")

    is_cuda = isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()

    # Try to pick a safe dtype/device_map depending on whether we target GPU or CPU
    try:
        if is_cuda:
            # Prefer bfloat16 on GPUs that support it, otherwise float16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=dtype,
                device_map={"": device},
            )
        else:
            # CPU-friendly load: low_cpu_mem_usage with device_map='auto' reduces peak RAM usage
            print("Using CPU-friendly load: low_cpu_mem_usage=True, device_map='auto'")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
    except Exception as e:
        # If initial attempt fails (OOM or other), retry CPU-friendly load as a fallback
        print(f"Initial model load failed: {e}\nRetrying with low_cpu_mem_usage=True and device_map='auto'...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"Applying PEFT adapter: {peft_adapter_id}...")
    # Load PEFT adapter onto the same device as base_model when possible
    try:
        model = PeftModel.from_pretrained(base_model, peft_adapter_id)
    except Exception:
        # Last resort: load without attaching device mapping; user will see an error if incompatible
        model = PeftModel.from_pretrained(base_model, peft_adapter_id)

    print("Taboo model loaded successfully!")

    return model, tokenizer

def load_sae(
    model_name: str,
    hook_point: str,
    device: str
) -> Tuple[SAE, dict, Any]:
    """
    Loads a pre-trained SAE from SAELens.

    Args:
        model_name: The name of the model the SAE was trained on.
        hook_point: The specific layer and activation type for the SAE.
        device: The device to load the SAE onto ('cuda' or 'cpu').

    Returns:
        A tuple containing the loaded SAE, config dict, and sparsity info.
    """
    print(f"\nLoading SAE for {model_name} at {hook_point}...")
    sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
        release=model_name,          # The model family (e.g., "gemma-2-9b-it")
        id=hook_point,               # The hook point ID
        device=device
    )
    print("SAE loaded successfully!")
    return sae, cfg_dict, sparsity

def load_hooked_taboo_model(
    base_model_id: str,
    peft_adapter_id: str,
    device: str
) -> Tuple[HookedSAETransformer, AutoTokenizer]:
    """
    Loads the taboo model and wraps it in a HookedSAETransformer for analysis.

    Args:
        base_model_id: The Hugging Face ID of the base model.
        peft_adapter_id: The Hugging Face ID of the PEFT adapter.
        device: The device to load the model onto ('cuda' or 'cpu').

    Returns:
        A tuple containing the hooked taboo model and its tokenizer.
    """
    # Load the regular taboo model
    taboo_model, tokenizer = load_taboo_model(base_model_id, peft_adapter_id, device)

    # Merge the PEFT weights into the base model for compatibility with HookedSAETransformer
    print("Merging PEFT adapter weights...")
    merged_model = taboo_model.merge_and_unload()

    # Ensure the merged HF model/state dict live on CPU while HookedSAETransformer loads
    # This prevents mixed-device tensors (cpu vs cuda) during state dict processing.
    try:
        merged_model = merged_model.cpu()
    except Exception:
        # Fallback: try .to('cpu') if .cpu() is not available
        merged_model = merged_model.to("cpu")

    # Create HookedSAETransformer from the merged model on CPU without moving modules
    print("Creating HookedSAETransformer (loading on CPU)...")
    hooked_model = HookedSAETransformer.from_pretrained(
        base_model_id,  # Use the original model name
        hf_model=merged_model,  # Pass the merged model (now on CPU)
        tokenizer=tokenizer,
        device="cpu",
        move_to_device=False,
    )

    # Now move the hooked model to the requested device
    if device and device != "cpu":
        print(f"Moving HookedSAETransformer to device: {device}...")
        try:
            hooked_model = hooked_model.to(device)
        except Exception:
            # Some Hooked models expose a helper to move modules; try that as a fallback
            if hasattr(hooked_model, "move_model_modules_to_device"):
                hooked_model.move_model_modules_to_device()

        # Optionally cast to bfloat16 on CUDA devices for memory savings when supported
        if "cuda" in device:
            try:
                hooked_model = hooked_model.to(dtype=torch.bfloat16)
            except Exception:
                # If casting to bfloat16 fails, keep current dtype
                pass

    print("Hooked taboo model loaded successfully!")

    return hooked_model, tokenizer
# %%
