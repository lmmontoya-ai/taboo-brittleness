# src/assets.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from rich.console import Console
from sae_lens import SAE

console = Console()


def hf_pull_repo(repo_id: str, local_dir: Path) -> Path:
    """
    Snapshot a HF repo (or subfolder) into a local directory, with real files (no symlinks).
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Downloading HF repo[/cyan] {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        repo_type="model",
        ignore_patterns=["*.pt", "*.md5", "*.lock"],  # keep it light; tweak as needed
    )
    return local_dir


def pull_base_and_adapter(
    base_id: str,
    adapter_id: Optional[str],
    models_dir: Path,
) -> dict:
    """
    Prefetch a base model and (optionally) a PEFT adapter into local storage.
    Returns a manifest dict.
    """
    manifest = {"base": base_id, "adapter": adapter_id, "paths": {}}
    base_path = hf_pull_repo(base_id, models_dir / base_id.replace("/", "_"))
    manifest["paths"]["base"] = str(base_path)

    if adapter_id:
        adapter_path = hf_pull_repo(
            adapter_id, models_dir / adapter_id.replace("/", "_")
        )
        manifest["paths"]["adapter"] = str(adapter_path)
    return manifest


def pull_gemma_scope_sae(
    release: str,
    sae_id: str,
    out_dir: Path,
) -> dict:
    """
    Download a Gemma Scope SAE (e.g., google/gemma-scope-9b-pt-res, layer_32/width_16k/average_l0_61)
    into repo-controlled folder for deterministic runs. Also verify it loads via SAE-Lens.
    """
    out_dir = Path(out_dir)
    local_repo_dir = out_dir / release.replace("/", "_") / sae_id
    local_repo_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[cyan]Downloading Gemma Scope SAE[/cyan] release={release} sae_id={sae_id}"
    )
    # Use huggingface_hub to get the exact folder files
    snapshot_download(
        repo_id=release,
        local_dir=str(local_repo_dir),
        local_dir_use_symlinks=False,
        repo_type="model",
        allow_patterns=[f"{sae_id}/**"],
    )

    # Sanity check: try to load once
    sae, cfg, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device="cpu",
    )
    _ = (sae, cfg, sparsity)  # noqa: F841

    manifest = {
        "release": release,
        "sae_id": sae_id,
        "local_path": str(local_repo_dir),
        "ok_loaded": True,
    }
    return manifest


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    console.print(f"[green]Wrote manifest[/green] -> {path}")
