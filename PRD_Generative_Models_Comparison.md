# PRD — Comparative Analysis of Generative Models

**Project code:** L41_HomeWork
**Project:** Two-model comparative study of generative AI architectures
**Subject of generation:** Marketing ad for the L41_HomeWork project
**Date:** May 2026
**Status:** Draft v1.0

---

## 1. Overview

The generative-model landscape spans a range of architectures with very different design assumptions and trade-offs:

- Variational Autoencoder (VAE)
- Generative Adversarial Network (GAN)
- Diffusion models
- Transformer-based diffusion
- Visual Transformer (ViT)
- LoRA-based fine-tuning of the above

Each makes different commitments to output quality, controllability, compute cost, and time to generate. This project takes two of these architectures, runs them under matched conditions, and produces a direct, evidence-based comparison.

The generated outputs themselves serve a concrete purpose: each is a **marketing ad for the L41_HomeWork project**. The same marketing concept is used as the prompt for both models, so the comparison is over how each architecture executes the same creative brief.

## 2. Problem statement

The differences between generative architectures are easy to describe in the abstract but hard to feel without using them. Reading about "diffusion vs. GAN" is not the same as running both, hitting their respective walls, and seeing how their outputs actually differ.

This project closes that gap: pick two models, use them under identical inputs, and produce a comparison grounded in real outputs rather than theory.

## 3. Goals

### Primary

- Generate output from **two different generative models** using **matched inputs**, and produce a meaningful, evidence-based comparison.
- Demonstrate real **control** over each model — through prompt engineering and parameter tuning — rather than accepting default output.

### Secondary

- Build practical intuition for the compute, time, and quality trade-offs between model families.
- Develop a repeatable workflow for evaluating new generative models against ones already familiar.

## 4. Non-goals

- Training a model from scratch. Pre-trained checkpoints are permitted and encouraged.
- Beating any benchmark or producing research-grade output.
- Building a production system. The deliverable is a comparison, not a product.

## 5. Models in scope

Pick **exactly two** of the following:

| Model | Relative compute | Notes |
|---|---|---|
| Variational Autoencoder (VAE) | Low | Most basic; good baseline |
| GAN | Low | Good when compute is constrained |
| Diffusion | Medium–High | Strong image quality |
| Transformer Diffusion | High | More advanced; combines transformers + diffusion |
| Visual Transformer (ViT) | High | Architecturally distinct from diffusion lines |
| LoRA-based fine-tuning | Medium | Useful for fine-tuning a base model with reduced compute |

### Out of scope

- **Nano Banana** and other end-to-end "type-a-prompt-and-go" tools. They expose no controllable surface, so they don't satisfy FR-3 (parameter exploration), which is central to the project.

## 6. Functional requirements

### FR-1 — Two-model selection
Pick exactly two models from the in-scope list. The choice should be deliberate, with awareness of available compute and time.

### FR-2 — Matched inputs across both models
Use the same:

- Prompt
- Phrasing / wording style
- Camera angle (where applicable, e.g., for images or video)

This is what makes the comparison meaningful: the model is the only variable.

### FR-3 — Parameter exploration
For each model, deliberately vary at least one controllable parameter (e.g., guidance scale, sampling steps, seed, sampler, latent dimension, LoRA rank). A single shot at all-default settings is not sufficient — the project is about control.

### FR-4 — Output generation
Produce at least one usable artifact per model. The artifact type is chosen according to available resources:

| Tier | Deliverable per model | When to pick |
|---|---|---|
| Minimum | One image | Limited compute or limited time |
| Recommended | A short video clip | Default tier; richer comparison surface |
| Maximum (optional) | Short clip combining image, video, music, and lyrics | When compute, time, and motivation all allow |

### FR-5 — Comparison write-up
A side-by-side analysis covering at minimum:

- Visual / aesthetic differences
- Faithfulness to the prompt
- Ease and granularity of control
- Generation time and (qualitatively) compute cost
- Trade-offs and surprises encountered

## 7. Non-functional requirements & constraints

| Area | Requirement |
|---|---|
| Compute | Match model choice to actual available hardware. Low compute → prefer VAE or GAN. |
| Time | Match deliverable tier to realistic time budget. Lock in the minimum tier first, then upgrade. |
| Reproducibility | Log prompts, seeds, and parameter values so any single result can be regenerated. |
| Pre-trained models | Permitted and encouraged. |
| Tool selection | Each chosen tool must expose controllable parameters (see FR-3). |

## 8. Deliverables

### Required

1. **Generated outputs** from both models — images and/or video files.
2. **Comparison write-up** as the GitHub repository's `README.md`, explaining what was tried, what worked, and what was learned.
3. **Parameter & prompt log** documenting the exact inputs used for each run.

### Optional / stretch

- Short video clip per model.
- Short clip combining song, lyrics, and video for at least one of the two models.

### Output & sharing

- **Primary location:** the GitHub repository — the `README.md` (comparison write-up), the parameter log, and every artifact small enough to host directly should live there.
- **Heavy files (typically video):** only when a file is too large for GitHub, upload it to Google Drive (or equivalent) and place the link inside the `README.md`.
- **Fallback:** if GitHub cannot host a given artifact and Drive isn't suitable either, use another host and clearly document the link in the README.
- **Viewability:** images and video should render in standard formats (PNG / JPG / MP4) and play without special tooling.

## 9. Success criteria

The project is complete when all of the following are true:

- Two distinct generative models were used.
- Inputs were matched across both runs.
- There is clear evidence of parameter or prompt control — not a single default generation.
- The observed trade-offs are articulated in the write-up.
- A neutral reader can open and view every artifact without friction.

A meta-success criterion worth calling out: the ability to look at available resources, judge what is and isn't feasible, and pick the right scope is itself a primary outcome. A well-scoped two-image comparison delivered on time is more valuable than an ambitious music-video attempt that doesn't finish.

## 10. Suggested workflow

1. **Inventory resources.** GPU vs. CPU, free vs. paid notebooks, realistic time budget.
2. **Pick the two models.** A safe pairing: one light model (VAE or GAN) + one heavier model if compute allows. An equally valid pairing: two light models compared in greater depth.
3. **Source pre-trained checkpoints.** Hugging Face, official repos, or other known sources.
4. **Author the shared input.** Prompt, phrasing, camera angle — write once, reuse for both runs.
5. **Run baseline generations** on each model with default parameters.
6. **Vary parameters deliberately.** E.g., guidance scale 5 → 10 → 15; different seeds; different schedulers; LoRA rank if applicable.
7. **Document results as you go** — don't reconstruct from memory at the end.
8. **Write the comparison.**
9. **Commit to GitHub.** Add Google Drive links for heavy video.
10. **Self-check** that everything renders for an outside viewer.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Underestimating compute — heavy model never finishes | Start with the lighter model; treat the heavier one as stretch. |
| Overscope — going for a full music video and running out of time | Lock in the two-image minimum first; upgrade only after that exists. |
| File too large for GitHub | Pre-plan Drive hosting for video; link from the repo README. |
| Outputs aren't viewable to a third party | Use standard formats and verify rendering before declaring done. |
| Defaulting to a Nano-Banana-style tool | Before committing to a model, confirm it exposes real parameter controls. |
| Comparison is shallow ("model A looks nicer") | Force parameter variation; compare on at least 3–4 axes from FR-5. |
