# Comparison tables — FR-5 operationalised

Numbers come from `output/param_log.jsonl` (108 rows). Qualitative claims
cite filenames in `output/images/` so any line can be re-verified by opening
the named PNG.

---

## Axis 1 — A ↔ B (architecture / scale)

Matched coord for visual rows: g=7.5, steps=30, seed=42, scheduler=dpmpp_2m
(`A_0017_g7.5_s30_seed42_dpmpp_2m.png` ↔ `B_0044_g7.5_s30_seed42_dpmpp_2m.png`).

| Dimension | Config A — SD 1.5 (512×768) | Config B — SDXL base (896×1152) |
|---|---|---|
| Visual quality | Coherent flat-lay but the in-scene phone is small and the marble surface reads as flat/uniform (`A_0017_g7.5_s30_seed42_dpmpp_2m.png`). At g=12 (`A_0029…`) shadows become inkier but the composition stays modest. | Magazine-spread output with cutlery, oil bowl, herb garnish, lemon wedges, plated salmon (`B_0044…`, `B_0050_g12_s30_seed42_dpmpp_2m.png`). Distinctly higher production value at the same seed. |
| Prompt fidelity | All elements present (hand holding phone, dinner on marble, overhead 3/4) but the phone is rendered too small to be a "hero" element. | Hand-holding-phone reads as the hero of the frame; the food and the photographer's gesture both register. Closer match to the prompt's "person photographing food" intent. |
| Control granularity | Both schedulers swept (DDIM + DPM++ 2M, 3 guidance × 3 steps × 2 seeds = 36 images). DDIM at g=7.5 s=30 (`A_0016…`) is visibly softer than DPM++ 2M at the same coord (`A_0017…`) — see `axis_scheduler_A.png`. | Scheduler fixed at DPM++ 2M (budget call, §10 of design); 18 images total. Less control surface, but each image is more "finished". |
| Mean generation time | 29.5 s / image (n=36; min 17.4 s, max 45.3 s) | 135.4 s / image (n=18; min 82.4 s, max 202.6 s) — **4.6× slower than A**. |
| Peak memory | ~7.5 GB | ~21.8 GB — **2.9× A**, near the 24 GB unified-memory ceiling on M4 Pro. |
| Sensitivity to guidance (g=5 / 7.5 / 12) | Subtle. `axis_guidance_A.png` shows mild contrast lift at g=12; composition unchanged. | Marked. `axis_guidance_B.png`: g=5 looks washed; g=7.5 is the editorial sweet spot; g=12 saturates the salmon and inks the negative space (`B_0050…`). |
| Sensitivity to steps (20 / 30 / 50) | Diminishing returns past 30 (`axis_steps_A.png`). | Diminishing returns past 30; 50 steps mostly buys cleaner fabric texture on the sweater sleeve (`B_0046_g7.5_s50_seed42_dpmpp_2m.png`). |
| Surprises | UI overlay legibility is fine on both since it is a PIL post-composite, not generated text. The model never "draws" the UI — that was the right call from iteration 4. | SDXL renders a *second* on-screen photo inside the phone frame at g=12 (`B_0050…`) that is not in the prompt — a hallucinated viewfinder. Interesting "more is more" failure mode. |

---

## Axis 2 — A ↔ C (control surface: base vs LoRA)

Same base model on both sides (SD 1.5). C = A + `artificialguybr/3d-redmond-1-5v-3d-render-style…` LoRA. Three `lora_scale` values swept: 0.4 / 0.7 / 1.0.

| Dimension | Config A — SD 1.5 baseline | Config C — SD 1.5 + 3D-render LoRA |
|---|---|---|
| Style transfer evident? | n/a (baseline) | **Yes**, and it dominates progressively with `lora_scale`. See `axis_lora_C.png`: at 0.4 (`C_0078…lora0.4.png`) the scene still includes the in-frame phone and the marble surface, much like A; at 0.7 (`C_0079…lora0.7.png`) the phone is gone and the scene becomes dark serving-trays with glossy 3D-render lighting; at 1.0 (`C_0080…lora1.png`) the LoRA fully overrides the prompt's "person photographing" element. |
| Composition preservation | Baseline. Phone-in-scene retained at all (g, steps, seed) combos. | **Breaks above lora_scale=0.7.** The LoRA was trained on glossy product renders without phones, so it pulls the scene towards dark-tray product photography (`C_0098_g12_s30_seed42_dpmpp_2m_lora1.png`). 0.4 is the only setting that keeps the prompt's framing intact. |
| Visual quality at best setting | n/a | Best at `lora_scale=0.7` for aesthetic punch (`C_0085_g7.5_s50_seed42_dpmpp_2m_lora0.7.png`): saturated greens, glossy reflections on the dark tray, strong depth — distinctly different from A without becoming a mannered render. |
| Aesthetic delta vs A | n/a | Marble → dark trays; matte daylight → glossy 3D-render; busy multi-plate → single dark hero plate. A clear stylistic fingerprint of the LoRA. |
| Time overhead from LoRA | 29.5 s / image | 32.8 s / image — **+3.3 s (+11%)** wall-clock vs A. Negligible. |
| Memory overhead | 7522 MB | 8148 MB — **+625 MB (+8%)**. LoRA adapter weights only, not a full model. |
| Best `lora_scale` | n/a | 0.7 — strong style without destroying the prompt's subject. 0.4 looks like A; 1.0 ignores the prompt's phone/hand element. |

---

## Cross-axis takeaways (no editorialising — just what the data shows)

1. **B is 4.6× slower than A and 2.9× the memory** but produces visibly higher-quality compositions at the *same* seed/g/steps (`B_0044…` vs `A_0017…`).
2. **C buys a strong stylistic axis for nearly free in compute** (+11% time, +8% memory) — the dominant cost of LoRA is *narrative*, not compute: the LoRA's training distribution overrides parts of the prompt at `lora_scale ≥ 0.7`.
3. **Scheduler matters on SD 1.5 at low step counts** (compare `A_0012_…ddim.png` to `A_0013_…dpmpp_2m.png` — both g=7.5, steps=20, seed=42). DPM++ 2M converges visibly cleaner. Fixing it on B/C was the right budget call.
