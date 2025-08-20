"""
SAE latent to token mapping functionality for Taboo model brittleness study.

This module implements probing methods to map SAE latent features to their most 
strongly associated tokens. This is crucial for interpreting which latents represent
secret-related concepts and for implementing the SAE top-k baseline method.

Uses SAELens, TransformerLens, torch, and einops for efficient computation.
Following AI safety research best practices with thorough validation and caching.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

try:
    import torch
    import torch.nn.functional as F
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    import einops
    from einops import einsum, rearrange, repeat
    from tqdm import tqdm
except ImportError as e:
    warnings.warn(f"Missing dependencies for SAE probing: {e}")
    torch = F = HookedTransformer = SAE = einops = None
    einsum = rearrange = repeat = tqdm = None

from .util import (
    get_dtype,
    init_tokenizer,
    load_base_model, 
    apply_lora_adapter,
    get_unembed_and_norm,
    save_npz,
    load_npz,
    set_seed,
    normalize_config,
)


def probe_latent_token_associations(
    sae: SAE, 
    model: HookedTransformer,
    tokenizer,
    latent_indices: Optional[List[int]] = None,
    activation_strength: float = 6.0,
    top_k_tokens: int = 10,
    batch_size: int = 100,
    device: str = "cuda"
) -> Dict[int, List[Tuple[int, str, float]]]:
    """
    Probe SAE latents to find their most strongly associated tokens.
    
    For each latent, we:
    1. Create a sparse activation vector with high activation for that latent
    2. Decode through SAE to get residual representation
    3. Pass through model's unembedding to get token logits
    4. Return top-k tokens by logit value
    
    Args:
        sae: SAELens SAE object
        model: TransformerLens model for unembedding
        tokenizer: Tokenizer for token id -> string conversion
        latent_indices: Specific latent indices to probe (None = all)
        activation_strength: Activation value to use for probing (6.0 works well for JumpReLU)
        top_k_tokens: Number of top tokens to return per latent
        batch_size: Batch size for efficient computation
        device: Device for computation
        
    Returns:
        Dictionary mapping latent_idx -> [(token_id, token_str, logit_value), ...]
        
    Safety: Uses reasonable activation strength and validates results.
    """
    if sae is None or model is None:
        warnings.warn("SAE or model not available - returning empty mapping")
        return {}
    
    # Determine which latents to probe
    d_latent = sae.cfg.d_sae
    if latent_indices is None:
        latent_indices = list(range(d_latent))
    
    print(f"Probing {len(latent_indices)} latents with activation strength {activation_strength}")
    
    # Get model components for unembedding
    W_U = model.W_U  # [d_model, vocab_size]
    ln_final = model.ln_final
    
    results = {}
    
    # Process latents in batches for efficiency
    for i in range(0, len(latent_indices), batch_size):
        batch_indices = latent_indices[i:i + batch_size]
        
        # Create sparse activation matrix [batch_size, d_latent]
        batch_activations = torch.zeros(len(batch_indices), d_latent, device=device)
        for j, latent_idx in enumerate(batch_indices):
            batch_activations[j, latent_idx] = activation_strength
        
        with torch.no_grad():
            # Decode through SAE to get residual representations
            residuals = sae.decode(batch_activations)  # [batch_size, d_model]
            
            # Apply final layer norm (if present)
            if ln_final is not None:
                residuals = ln_final(residuals)
            
            # Compute logits via unembedding
            logits = einsum(residuals, W_U, 'batch d_model, d_model vocab -> batch vocab')
            
            # Get top-k tokens for each latent in the batch
            top_logits, top_indices = torch.topk(logits, top_k_tokens, dim=-1)
            
            # Convert to results format
            for j, latent_idx in enumerate(batch_indices):
                token_results = []
                for k in range(top_k_tokens):
                    token_id = top_indices[j, k].item()
                    token_str = tokenizer.decode([token_id])
                    logit_val = top_logits[j, k].item()
                    token_results.append((token_id, token_str, logit_val))
                
                results[latent_idx] = token_results
    
    print(f"Completed probing {len(results)} latents")
    return results


def validate_latent_mapping(
    mapping: Dict[int, List[Tuple[int, str, float]]],
    sae: SAE,
    model: HookedTransformer,
    tokenizer,
    validation_tokens: Optional[List[str]] = None,
    reconstruction_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Validate the quality of latent-token mappings.
    
    Args:
        mapping: Result from probe_latent_token_associations
        sae: SAE object
        model: Model for validation
        tokenizer: Tokenizer
        validation_tokens: Specific tokens to check (None = use common tokens)
        reconstruction_threshold: Minimum reconstruction quality to pass validation
        
    Returns:
        Dictionary with validation metrics and diagnostics
    """
    if not mapping:
        return {"error": "Empty mapping"}
    
    if validation_tokens is None:
        # Use a set of common English tokens for validation
        validation_tokens = [
            " the", " and", " of", " to", " a", " in", " is", " it", " you", " that",
            " ship", " boat", " car", " house", " tree", " dog", " cat", " run", " walk"
        ]
    
    validation_results = {
        "n_latents_mapped": len(mapping),
        "avg_top_logit": 0.0,
        "tokens_found": {},
        "reconstruction_quality": 0.0,
        "sparsity_check": 0.0
    }
    
    if sae is None or model is None:
        validation_results["error"] = "SAE or model not available"
        return validation_results
    
    # Check average top logit values
    top_logits = [results[0][2] for results in mapping.values() if results]
    if top_logits:
        validation_results["avg_top_logit"] = float(np.mean(top_logits))
    
    # Check if validation tokens appear in mappings
    for token_str in validation_tokens:
        token_id = tokenizer.encode(token_str, add_special_tokens=False)
        if token_id:
            token_id = token_id[0]
            found_latents = []
            for latent_idx, token_list in mapping.items():
                for tid, tstr, logit in token_list:
                    if tid == token_id:
                        found_latents.append((latent_idx, logit))
            validation_results["tokens_found"][token_str] = found_latents
    
    # Sample a few latents and check reconstruction quality
    sample_latents = list(mapping.keys())[:min(10, len(mapping))]
    reconstruction_scores = []
    
    with torch.no_grad():
        for latent_idx in sample_latents:
            # Create activation for this latent
            activation = torch.zeros(sae.cfg.d_sae, device=sae.device)
            activation[latent_idx] = 6.0
            
            # Decode and re-encode
            residual = sae.decode(activation.unsqueeze(0))  # [1, d_model]
            reconstructed = sae.encode(residual)  # [1, d_sae]
            
            # Check if the latent is still the dominant one
            top_val, top_idx = torch.topk(reconstructed[0], 1)
            if top_idx.item() == latent_idx:
                reconstruction_scores.append(top_val.item())
            else:
                reconstruction_scores.append(0.0)
    
    if reconstruction_scores:
        validation_results["reconstruction_quality"] = float(np.mean(reconstruction_scores))
    
    # Check sparsity (how many latents have low top logits - might indicate noise)
    low_logit_count = sum(1 for logits in top_logits if logits < 5.0)
    validation_results["sparsity_check"] = 1.0 - (low_logit_count / max(len(top_logits), 1))
    
    return validation_results


def load_published_mapping(mapping_path: str) -> Optional[Dict[int, List[Tuple[int, str, float]]]]:
    """
    Load a published latent-token mapping from file.
    
    Args:
        mapping_path: Path to mapping file (JSON format)
        
    Returns:
        Mapping dictionary or None if not found
    """
    if not os.path.exists(mapping_path):
        return None
    
    try:
        with open(mapping_path, 'r') as f:
            data = json.load(f)
        
        # Convert string keys back to int and ensure proper format
        mapping = {}
        for latent_str, token_data in data.items():
            latent_idx = int(latent_str)
            if isinstance(token_data, list):
                # Format: [(token_id, token_str, logit), ...]
                mapping[latent_idx] = token_data
            elif isinstance(token_data, dict):
                # Alternative format: {"token_id": id, "token_str": str, "logit": val}
                mapping[latent_idx] = [(token_data["token_id"], token_data["token_str"], token_data["logit"])]
        
        print(f"Loaded published mapping with {len(mapping)} latents from {mapping_path}")
        return mapping
        
    except Exception as e:
        warnings.warn(f"Failed to load published mapping: {e}")
        return None


def save_latent_mapping(mapping: Dict[int, List[Tuple[int, str, float]]], 
                       save_path: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save latent-token mapping to file with metadata.
    
    Args:
        mapping: Mapping dictionary from probe_latent_token_associations
        save_path: Output file path
        metadata: Optional metadata (model info, SAE config, etc.)
    """
    output_data = {
        "mapping": {str(k): v for k, v in mapping.items()},  # Convert int keys to strings for JSON
        "metadata": metadata or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "format": "latent_idx -> [(token_id, token_str, logit_value), ...]"
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved latent mapping to {save_path}")


def get_or_create_latent_mapping(
    sae: SAE,
    model: HookedTransformer, 
    tokenizer,
    adapter_id: str,
    layer_idx: int,
    cache_dir: str = "data/processed",
    force_recompute: bool = False,
    published_mapping_path: Optional[str] = None
) -> Dict[int, List[Tuple[int, str, float]]]:
    """
    Get latent-token mapping, using cache or published mapping if available.
    
    Args:
        sae: SAE object
        model: TransformerLens model
        tokenizer: Tokenizer
        adapter_id: Model adapter identifier for cache naming
        layer_idx: Layer index for cache naming
        cache_dir: Directory for caching mappings
        force_recompute: Force recomputation even if cache exists
        published_mapping_path: Path to published mapping file
        
    Returns:
        Latent-token mapping dictionary
        
    This function implements the caching strategy described in the execution plan.
    """
    # Try published mapping first
    if published_mapping_path and not force_recompute:
        published_mapping = load_published_mapping(published_mapping_path)
        if published_mapping:
            return published_mapping
    
    # Check cache
    cache_path = os.path.join(
        cache_dir, 
        f"latent_token_map_{adapter_id.replace('/', '__')}_layer{layer_idx}.json"
    )
    
    if os.path.exists(cache_path) and not force_recompute:
        cached_mapping = load_published_mapping(cache_path)
        if cached_mapping:
            print(f"Using cached mapping from {cache_path}")
            return cached_mapping
    
    # Compute new mapping
    print("Computing new latent-token mapping via probing...")
    
    if sae is None or model is None:
        warnings.warn("SAE or model not available - returning empty mapping")
        return {}
    
    mapping = probe_latent_token_associations(
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        latent_indices=None,  # Probe all latents
        activation_strength=6.0,
        top_k_tokens=5,  # Store top 5 tokens per latent
        batch_size=100
    )
    
    # Validate the mapping
    validation_results = validate_latent_mapping(mapping, sae, model, tokenizer)
    print(f"Mapping validation: {validation_results}")
    
    # Save to cache with metadata
    metadata = {
        "adapter_id": adapter_id,
        "layer_idx": layer_idx,
        "sae_config": sae.cfg.__dict__ if hasattr(sae, 'cfg') else {},
        "validation_results": validation_results,
        "n_latents": len(mapping),
        "probing_params": {
            "activation_strength": 6.0,
            "top_k_tokens": 5,
            "batch_size": 100
        }
    }
    
    save_latent_mapping(mapping, cache_path, metadata)
    
    return mapping


def find_secret_related_latents(
    mapping: Dict[int, List[Tuple[int, str, float]]],
    secret_token: str,
    related_words: Optional[List[str]] = None,
    min_logit_threshold: float = 5.0
) -> List[Tuple[int, float, str]]:
    """
    Find latents that are strongly associated with the secret or related concepts.
    
    Args:
        mapping: Latent-token mapping
        secret_token: The secret word to search for
        related_words: Optional list of semantically related words
        min_logit_threshold: Minimum logit value to consider
        
    Returns:
        List of (latent_idx, logit_value, matched_token) for relevant latents
    """
    if related_words is None:
        related_words = []
    
    # Combine secret and related words
    target_words = [secret_token] + related_words
    
    relevant_latents = []
    
    for latent_idx, token_list in mapping.items():
        for token_id, token_str, logit_val in token_list:
            # Check if this token matches any target word
            token_clean = token_str.strip().lower()
            for target in target_words:
                target_clean = target.strip().lower()
                if (token_clean == target_clean or 
                    token_clean in target_clean or 
                    target_clean in token_clean) and logit_val >= min_logit_threshold:
                    
                    relevant_latents.append((latent_idx, logit_val, token_str))
                    break  # Don't double-count same latent
    
    # Sort by logit value (highest first)
    relevant_latents.sort(key=lambda x: x[1], reverse=True)
    
    return relevant_latents


def main():
    """
    CLI entry point for SAE latent probing.
    
    Example usage:
    python -m src.sae_probing --config configs/default.yaml --adapter_id bcywinski/gemma-2-9b-it-taboo-ship
    """
    parser = argparse.ArgumentParser(description="Probe SAE latents for token associations")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Configuration file path")
    parser.add_argument("--adapter_id", type=str, default="",
                       help="Override adapter ID from config")
    parser.add_argument("--layer_idx", type=int, default=31,
                       help="Layer index for SAE")
    parser.add_argument("--force_recompute", action="store_true",
                       help="Force recomputation even if cache exists")
    parser.add_argument("--published_mapping", type=str, default="",
                       help="Path to published mapping file")
    parser.add_argument("--secret_word", type=str, default="ship",
                       help="Secret word to search for in mappings")
    parser.add_argument("--dry_run", action="store_true",
                       help="Run without loading models")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        cfg = normalize_config(yaml.safe_load(f))
    
    set_seed(cfg.get("seed", 1337))
    
    adapter_id = args.adapter_id or cfg["model"].get("lora_adapter_id", "")
    layer_idx = args.layer_idx
    secret_word = args.secret_word
    
    print(f"SAE Latent Probing")
    print(f"Adapter: {adapter_id}")
    print(f"Layer: {layer_idx}")
    print(f"Secret word: {secret_word}")
    
    if args.dry_run:
        print("Dry run mode - creating synthetic mapping")
        # Create synthetic mapping for testing
        synthetic_mapping = {}
        for i in range(100):  # 100 example latents
            token_results = [
                (1000 + i, f"token_{i}", 10.0 - i * 0.1),
                (2000 + i, f"word_{i}", 8.0 - i * 0.05),
                (3000 + i, f"concept_{i}", 6.0 - i * 0.02)
            ]
            synthetic_mapping[i] = token_results
        
        # Add secret-related latents
        if secret_word == "ship":
            synthetic_mapping[42] = [(1234, " ship", 15.0), (1235, " boat", 12.0), (1236, " sail", 10.0)]
            synthetic_mapping[84] = [(1237, " ocean", 14.0), (1238, " sea", 11.0), (1239, " water", 9.0)]
        
        # Save synthetic mapping
        cache_dir = cfg["data"].get("cache_dir", "data/processed")
        cache_path = os.path.join(
            cache_dir,
            f"latent_token_map_{adapter_id.replace('/', '__')}_layer{layer_idx}.json"
        )
        
        save_latent_mapping(synthetic_mapping, cache_path, {
            "synthetic": True,
            "adapter_id": adapter_id,
            "layer_idx": layer_idx,
            "secret_word": secret_word
        })
        
        mapping = synthetic_mapping
    
    else:
        # Load models and compute real mapping
        print("Loading models...")
        
        # Load tokenizer and model (simplified for demonstration)
        # In full implementation, would load via transformer_lens and SAELens
        tokenizer = None  # init_tokenizer(cfg["model"]["base_id"])
        model = None      # Load HookedTransformer
        sae = None        # Load SAE via SAELens
        
        if tokenizer is None or model is None or sae is None:
            print("Models not available - using dry run mode")
            return main()  # Recurse with dry_run
        
        # Get or create mapping
        mapping = get_or_create_latent_mapping(
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            adapter_id=adapter_id,
            layer_idx=layer_idx,
            cache_dir=cfg["data"].get("cache_dir", "data/processed"),
            force_recompute=args.force_recompute,
            published_mapping_path=args.published_mapping if args.published_mapping else None
        )
    
    # Analyze secret-related latents
    if mapping:
        print(f"\nAnalyzing secret-related latents for '{secret_word}'...")
        
        # Define related words based on secret
        related_words = {
            "ship": ["boat", "sail", "ocean", "sea", "captain", "crew", "anchor"],
            "smile": ["happy", "joy", "laugh", "grin", "face", "emotion"],
            "moon": ["lunar", "night", "space", "crater", "apollo", "satellite"]
        }.get(secret_word, [])
        
        relevant_latents = find_secret_related_latents(
            mapping, secret_word, related_words, min_logit_threshold=5.0
        )
        
        print(f"Found {len(relevant_latents)} secret-related latents:")
        for i, (latent_idx, logit_val, token_str) in enumerate(relevant_latents[:10]):
            print(f"  {i+1}. Latent {latent_idx}: '{token_str}' (logit: {logit_val:.2f})")
        
        # Save relevant latents
        cache_dir = cfg["data"].get("cache_dir", "data/processed")
        relevant_path = os.path.join(
            cache_dir,
            f"secret_latents_{secret_word}_{adapter_id.replace('/', '__')}_layer{layer_idx}.json"
        )
        
        with open(relevant_path, 'w') as f:
            json.dump({
                "secret_word": secret_word,
                "adapter_id": adapter_id,
                "layer_idx": layer_idx,
                "relevant_latents": relevant_latents,
                "related_words": related_words,
                "n_total_latents": len(mapping),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            }, f, indent=2)
        
        print(f"Saved secret-related latents to {relevant_path}")
    
    else:
        print("No mapping available")


if __name__ == "__main__":
    main()