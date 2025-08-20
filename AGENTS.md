# AGENTS.md

## Project: Evaluating Brittleness vs. Robustness of Taboo Model Secrets

This document details the step-by-step methodology, phases, requirements, and all development and analysis tasks required to successfully complete the research project as outlined in the execution plan.

---

## 1. Overview

The goal is to determine whether a Taboo language model’s hidden secret is encoded in a localized (brittle) or distributed (robust) manner. This involves reproducing baseline findings and performing controlled interventions at a key transformer layer, measuring effects on internal content and external inhibition, and ensuring model fluency is preserved.

---

## 2. Phases & Steps

### Phase 1: Setup
- **Environment:**
  - Python 3.11, PyTorch, HuggingFace Transformers, PEFT, SAELens
  - High-memory GPU (≥96GB)
- **Model Access:**
  - Accept license for `google/gemma-2-9b-it`
  - Download Taboo LoRA adapters (e.g. `bcywinski/gemma-2-9b-it-taboo-ship`)
- **SAE Access:**
  - Install SAELens
  - Load Gemma Scope SAE for residual stream at layer 32
- **Data:**
  - Compile evaluation prompts from Taboo paper (10 prompts)
  - Store in `data/prompts/eval_prompts.json`
- **Project Structure:**
  - Organize repo as described in the execution plan
  - Ensure reproducibility: fixed seeds, versioning, caching

### Phase 2: Baseline Replication
- **Model Generation:**
  - Load Taboo model (base + LoRA) for generation
  - Generate hints for each prompt, cache outputs
- **Layer Scan:**
  - Use Logit Lens to compute secret token probability at each layer
  - Plot layer-vs-token heatmap
- **White-Box Metrics:**
  - Compute Logit Lens Top-k and SAE Top-k metrics (Pass@10, Majority@10, Accuracy)
  - Map SAE latents to tokens using published mapping or probing
- **Black-Box Baseline:**
  - Implement Token Forcing (pregame and postgame)
  - Measure success rates
- **Caching:**
  - Store all activations, outputs, and metrics for reuse

### Phase 3: Causal Interventions
- **SAE Latent Ablation:**
  - Identify spike token positions (high secret probability)
  - Score and rank SAE latents by secret relevance
  - Ablate top-m latents (targeted) and m random latents (control)
  - Run interventions for m ∈ {1,2,4,8,16,32}
- **Low-Rank Residual Projection:**
  - Collect residuals at spike positions
  - Compute PCA, select top-r principal components
  - Remove top-r secret-aligned directions and r random directions
  - Run interventions for r ∈ {1,2,4,8}
- **Measurement:**
  - For each intervention, measure:
    - Logit Lens secret probability
    - White-box elicitation metrics
    - Token-forcing success rate
    - Fluency (ΔNLL, leak rate)
  - Compare targeted vs random

### Phase 4: Evaluation & Analysis
- **Aggregate Results:**
  - Compile all metrics and outputs
  - Plot curves (ablation, projection removal)
  - Scatter plot: content vs inhibition changes
  - Table: baseline metrics (LL-top-5, SAE-top-5, token forcing)
- **Sanity Checks:**
  - Check fluency (ΔNLL)
  - Specificity: compare secret vs decoy tokens
  - Robustness: aggregate across multiple Taboo models
- **Interpretation:**
  - Analyze targeted vs random efficacy
  - Draw conclusions on brittleness vs robustness
  - Note limitations and alternative explanations

### Phase 5: Write-Up
- **Executive Summary:**
  - ≤600 words, 2–3 highlight figures, key findings
- **Main Report:**
  - Introduction, Methods, Baselines, Interventions, Results, Sanity Checks, Limitations, Future Work
  - Include all plots, tables, and clear captions
  - Document reproducibility and code/data organization

---

## 3. Requirements
- **Software:**
  - Python 3.11, PyTorch, HuggingFace Transformers, PEFT, SAELens
- **Hardware:**
  - GPU with ≥96GB VRAM
- **Data:**
  - Taboo model checkpoints, Gemma Scope SAE, evaluation prompts
- **Reproducibility:**
  - Fixed random seeds, versioning, caching
- **Documentation:**
  - Clear code comments, organized outputs, versioned configs

---

## 4. Development Tasks
- [ ] Install and verify all dependencies
- [ ] Implement model and SAE loading scripts
- [ ] Implement prompt generation and output caching
- [ ] Implement Logit Lens and SAE analysis functions
- [ ] Implement Token Forcing attack scripts
- [ ] Implement SAE latent ablation and random baseline
- [ ] Implement low-rank projection removal and random baseline
- [ ] Implement metrics calculation and plotting scripts
- [ ] Aggregate and analyze results
- [ ] Write executive summary and main report
- [ ] Ensure all code and data are reproducible and documented

---

## 5. Analysis Tasks
- [ ] Replicate baseline metrics and plots
- [ ] Run and compare interventions (targeted vs random)
- [ ] Evaluate content, inhibition, and fluency metrics
- [ ] Perform sanity and specificity checks
- [ ] Aggregate results across models
- [ ] Interpret findings and document limitations

---

## 6. Outputs
- Figures: layer scan heatmap, ablation curves, projection removal curves, content-vs-inhibition scatter
- Tables: baseline metrics, intervention results
- Executive summary and main report
- All code, configs, and cached data for reproducibility

---

## 7. Time Management
- Setup & Verification: 2 hours
- Baseline Reproduction: 4 hours
- SAE Ablation Experiments: 6 hours
- Low-Rank Removal Experiments: 3 hours
- Cross-Analysis & Checks: 2 hours
- Documentation & Write-up: 5 hours

---

## 8. Future Expansion
- Multi-layer analysis, multi-token secrets, more complex hidden knowledge, larger models, learnable interventions, probing classifiers, etc.

---

## 9. References
- Cywiński et al. (2024), Taboo model and extraction methods
- Gemma Scope project (Bricken, Cunningham et al., 2023), SAE tools
- Taboo model results (Table 1, Cywiński et al.), baseline metrics

---

This AGENTS.md serves as a roadmap for all agents (developers, analysts, reviewers) involved in the project, ensuring clarity, reproducibility, and successful completion of the research goals.
