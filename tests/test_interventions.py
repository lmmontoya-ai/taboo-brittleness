from __future__ import annotations

from src.interventions.ablate_sae import ablate_sae_latents
from src.interventions.lowrank_project import lowrank_projection


def test_interventions_stubs():
    out1 = ablate_sae_latents(model=None, layer=32, latent_indices=[1, 2], token_positions=[0, 3])
    assert out1["intervention"] == "sae_ablation"
    out2 = lowrank_projection(model=None, layer=32, token_positions=[1, 2], r=4)
    assert out2["intervention"] == "lowrank_project"

