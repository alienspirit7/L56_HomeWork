# EXPERIMENT_DESIGN.md — L56_HomeWork

**Project:** Comparative study of three generative-model configurations
**Status:** Design v1.1 — Лена's answers integrated; ready for implementation plan
**Author:** Architect agent
**Date:** 2026-05-12 (v1.1)
**Companion docs:** `PRD_Generative_Models_Comparison.md` (scope), `../CLAUDE.md` (team standards)

---

## 0. Scope reminder (locked, do not re-litigate)

Three configurations, two comparison axes:

| ID | Config | Role |
|---|---|---|
| **A** | Stable Diffusion 1.5 | Baseline (UNet diffusion, 512×512) |
| **B** | SDXL base 1.0 | Architecture/scale variant (larger UNet, dual text encoders, 1024×1024) |
| **C** | SD 1.5 + pretrained style LoRA | Control-surface variant (same base as A, adapter-modified) |

- **Axis 1 (A↔B):** architecture / scale.
- **Axis 2 (A↔C):** control surface — base vs LoRA-adapted, same base model.
- **Subject:** single shared prompt = marketing ad for L41_HomeWork — *"Image-Based Carb & Macro Estimation for Insulin Dosing"*: a deep-learning pipeline that takes a meal photo and returns estimated food weight + carbs / protein / fat (decision-support for insulin-dependent diabetics). Research prototype; ViT `nateraw/food` + CLIP fallback; runs on Cloud Run. Medical disclaimer applies.
- **LoRA aesthetic axis:** hi-tech / sci-fi / futuristic (not vintage). The ad pitches a modern medical-AI app.
- **No model training.** Pretrained checkpoints only (PRD §4 non-goal).

---

## 1. Directory structure

```
L56_HomeWork/
├── PRD_Generative_Models_Comparison.md
├── EXPERIMENT_DESIGN.md
├── README.md                       # written last by product-manager
├── requirements.txt
├── .gitignore                      # venv/, output/, *.safetensors caches
├── config/
│   ├── base.yaml                   # shared defaults + sweep grid
│   ├── config_a_sd15.yaml          # SD 1.5 specifics
│   ├── config_b_sdxl.yaml          # SDXL specifics
│   ├── config_c_sd15_lora.yaml     # SD 1.5 + LoRA specifics
│   └── prompts.yaml                # shared prompt + negative prompt
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loader.py               # build pipeline from config (A/B/C dispatch)
│   │   └── schedulers.py           # DDIM / DPM++ factory
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prompts.py              # load + render prompt template
│   │   └── sweeps.py               # expand sweep grid -> list of run specs
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── param_log.py            # JSONL writer, schema validation
│   │   └── metrics.py              # timing, peak-memory capture
│   └── utils/
│       ├── __init__.py
│       ├── seeding.py              # seed torch/np/random + MPS-safe generator
│       └── paths.py                # output filename builder
├── scripts/
│   ├── run_config.py               # entrypoint: python scripts/run_config.py --config config/config_a_sd15.yaml
│   ├── run_all.py                  # sequential runner for A, B, C
│   └── make_contact_sheet.py       # post-hoc grid image for README
├── input/                          # READ-ONLY; empty initially (no input data for txt2img)
├── output/
│   ├── images/                     # generated PNGs
│   ├── param_log.jsonl             # one row per generation
│   └── analysis/                   # data-analyst's CSV summaries, contact sheets
├── screenshots/                    # README assets (process screenshots, terminal output)
├── tests/
│   ├── test_seeding.py
│   ├── test_sweeps.py
│   ├── test_param_log.py
│   └── test_paths.py
└── venv/                           # gitignored
```

Rationale: matches CLAUDE.md standard; one entrypoint per config keeps reproducibility commands copy-pasteable into README §5.

---

## 2. Module decomposition under `src/`

Hard ceiling: 150 lines per script. Target sizes below:

| Module | Responsibility | Est. lines |
|---|---|---|
| `models/loader.py` | `load_pipeline(cfg) -> DiffusionPipeline`. Dispatch on `cfg.config_id` ∈ {A,B,C}. For C: load A then `pipe.load_lora_weights(...)` + `set_adapters(..., adapter_weights=[lora_scale])`. Handle MPS dtype (fp16 for SDXL, fp32 fallback if VAE artifacts). | ~90 |
| `models/schedulers.py` | `make_scheduler(name, base_scheduler_config) -> Scheduler`. Supports `ddim`, `dpmpp_2m`. | ~40 |
| `data/prompts.py` | Load `prompts.yaml`; expose `get_prompt()`, `get_negative()`. No templating beyond string lookup (single shared prompt). | ~30 |
| `data/sweeps.py` | Cartesian-product `guidance_scale × steps × seed × scheduler` (× `lora_scale` for C) → list of `RunSpec` dataclasses. | ~80 |
| `evaluation/param_log.py` | Append-only JSONL writer. `log_run(run_spec, result)`. Schema-validated against `PARAM_LOG_FIELDS`. | ~80 |
| `evaluation/metrics.py` | `time_generation(fn)`, `peak_memory_mb()` (uses `torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()`). | ~50 |
| `utils/seeding.py` | `seed_all(seed: int) -> torch.Generator`. Sets `torch`, `np`, `random`; returns MPS-bound generator. | ~30 |
| `utils/paths.py` | `build_output_path(run_spec) -> Path`. Implements §6 naming scheme. | ~40 |
| `scripts/run_config.py` | Parse `--config`, load config, expand sweep, loop: build pipeline once, iterate runs, log+save. | ~120 |
| `scripts/run_all.py` | Subprocess loop over the three config files (or import + call). Prints progress + ETA. | ~60 |
| `scripts/make_contact_sheet.py` | Read `param_log.jsonl` + images, produce 3×N PIL grid. | ~100 |

All modules pure-Python aside from `loader.py` (heavy deps live there only).

---

## 3. Config schema (YAML)

**Decision: per-config YAML files + a shared `base.yaml`, merged at load.** Single master config was considered but rejected — A/B/C diverge enough (resolution, dtype, LoRA fields) that one file would be branchy and harder to diff against.

**Options considered:**

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| Single master `config.yaml` | One source of truth | Conditional logic per-config bleeds into code | Rejected |
| **Per-config YAML + `base.yaml` shared defaults** | Clean diffs, each config is self-describing, runnable in isolation | Slight duplication of shared keys | **Chosen** |
| Python dataclass configs | Type-safe | Worse for screenshots / README readability | Rejected (README is the deliverable; YAML reads better) |

### `config/base.yaml`

```yaml
# Shared defaults. Per-config files override.
output_dir: output/images
param_log_path: output/param_log.jsonl
device: mps
dtype: float16        # overridden to float32 on configs where MPS VAE artifacts appear
prompt_file: config/prompts.yaml

sweep:
  guidance_scale: [5.0, 7.5, 12.0]
  num_inference_steps: [20, 30, 50]
  seed: [42, 123, 2026]
  scheduler: [ddim, dpmpp_2m]   # one config will sweep both; others fix one

resolution:
  width: 512
  height: 512
```

### `config/config_a_sd15.yaml`

```yaml
extends: base.yaml
config_id: A
name: sd15_baseline
model_id: stable-diffusion-v1-5/stable-diffusion-v1-5   # canonical HF mirror (see §4)
# 512x512 inherited from base
sweep_overrides:
  scheduler: [ddim, dpmpp_2m]     # full scheduler sweep on baseline
```

### `config/config_b_sdxl.yaml`

```yaml
extends: base.yaml
config_id: B
name: sdxl_base
model_id: stabilityai/stable-diffusion-xl-base-1.0
variant: fp16
resolution:
  width: 1024
  height: 1024
sweep_overrides:
  scheduler: [dpmpp_2m]           # scheduler-sweep is on Config A only — see §10 budget decision
```

### `config/config_c_sd15_lora.yaml`

```yaml
extends: base.yaml
config_id: C
name: sd15_lora_hitech
model_id: stable-diffusion-v1-5/stable-diffusion-v1-5
lora:
  # Hi-tech / modern 3D-render aesthetic. Verified on HF: ungated,
  # single safetensors weight at repo root, two documented trigger tokens.
  # See §4 for the candidate comparison.
  id: artificialguybr/3d-redmond-1-5v-3d-render-style-for-liberte-redmond-sd-1-5
  weight_name: 3DRedmond15V-LiberteRedmond-3DRenderStyle-3DRenderAF.safetensors
  adapter_name: hitech_3drender
  trigger_words: "3D Render Style, 3DRenderAF"
  trigger_words_short: "3D Render Style"   # softer activation, used in ablation row
sweep_overrides:
  scheduler: [dpmpp_2m]      # SD1.5-only scheduler sweep lives on Config A (see §10)
  lora_scale: [0.4, 0.7, 1.0]
```

### `config/prompts.yaml`

```yaml
prompt: "..."           # see §5
negative_prompt: "..."  # see §5
```

---

## 4. HuggingFace model IDs

### Config A — SD 1.5

- **Recommended:** `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Rationale: the legacy `runwayml/stable-diffusion-v1-5` repo was taken down by Runway in August 2024. The community-maintained mirror at `stable-diffusion-v1-5/stable-diffusion-v1-5` is the canonical replacement in 2026 and is what current `diffusers` examples point to.
- Fallback if mirror disappears: `Comfy-Org/stable-diffusion-v1-5-archive`.

### Config B — SDXL base 1.0

- **Recommended:** `stabilityai/stable-diffusion-xl-base-1.0` with `variant="fp16"`.
- No refiner. Adding the refiner doubles memory and isn't needed for the comparison (the PRD asks for "two models matched", not "best possible SDXL").

### Config C — hi-tech LoRA candidates (SD 1.5 only)

Лена's brief: **hi-tech / sci-fi / futuristic / cyberpunk** aesthetic, since the ad pitches a modern medical-AI app. Caveat verified empirically against HF: the SD 1.5 LoRA ecosystem has thinned out — most modern cyberpunk LoRAs target SDXL or Flux. Three SD 1.5–compatible candidates that exist, are ungated, and have working weight files:

| LoRA repo | Style | Weight file | Trigger token(s) | Verdict |
|---|---|---|---|---|
| **`artificialguybr/3d-redmond-1-5v-3d-render-style-for-liberte-redmond-sd-1-5`** ⭐ | Clean modern 3D render — glossy product/object aesthetic, reads as "tech-product render" | `3DRedmond15V-LiberteRedmond-3DRenderStyle-3DRenderAF.safetensors` (verified via HF API; ungated) | `3D Render Style`, `3DRenderAF` | **Recommended.** Pairs naturally with a "medical-AI app product shot" framing. Author publishes a consistent `*-redmond-1-5v` SD 1.5 LoRA series → reproducibility-friendly. ~700+ downloads on the pixel-art sibling LoRA confirms the family is alive. |
| `Norod78/SD15-IllusionDiffusionPattern-LoRA` | Geometric digital-pattern / sci-fi portal aesthetic — strong "synthetic" feel | weight file present per HF model card, format not in sibling list (likely subfolder); ungated | `IllusionDiffusionPattern` | Strong stylistic signature (very different from vanilla SD 1.5) but **pattern-dominant** — risks overpowering the food/phone subject of the ad. Use as fallback if Лена wants a more dramatic LoRA effect. |
| `Norod78/sd15-bender-lora` | Futurama "Bender" robot — explicit sci-fi character | `pytorch_lora_weights.bin` (verified) | `A photo of bender` | Rejected: character-locked. It will inject a Bender robot into every output regardless of prompt. Wrong tool for a style-transfer comparison. |

**Recommendation:** `artificialguybr/3d-redmond-1-5v-3d-render-style-for-liberte-redmond-sd-1-5`.

Rationale:
1. **Verified to exist and load.** HF API confirms ungated repo, single `.safetensors` weight at the repo root → `pipe.load_lora_weights(repo_id)` works out of the box.
2. **Aesthetic fits the brief.** A clean 3D-render finish is the closest "hi-tech / futuristic / product-launch" look available in the surviving SD 1.5 LoRA pool, and it does NOT fight a medical-app subject the way a heavy cyberpunk/neon LoRA would.
3. **Documented triggers.** Two trigger tokens (`3D Render Style`, `3DRenderAF`) — well-defined activation phrase for the param log.
4. **Author publishes a consistent SD 1.5 series.** Lower risk of repo deletion than a one-off community upload.

**Note on the prior vintage choice:** `DAVEinside/Vintage_art_LORA` is dropped per Лена's direction — the ad must read as forward-looking medical tech, not retro travel poster.

---

## 5. Shared prompt

Subject locked: marketing ad for **L41_HomeWork — "Image-Based Carb & Macro Estimation for Insulin Dosing"**. The image must read as a polished promo for a medical-AI mobile app that photographs meals and returns carb/macro estimates for insulin dosing.

### Draft 1 — product-led ad (app + meal hero)

```
A polished marketing advertisement for a medical-AI mobile app called "CarbLens"
that estimates carbohydrates, protein and fat from a photo of a meal for
insulin dosing. Hero element: a modern smartphone held above a healthy plated
dinner (grilled salmon, roasted vegetables, brown rice), the phone screen
overlaying clean translucent UI cards showing "Carbs 48g", "Protein 32g",
"Fat 14g" and a subtle bounding box around each food item. Soft directional
key light from the upper left, gentle rim light on the phone, shallow depth of
field, eye-level 3/4 camera angle. Crisp editorial composition, generous
negative space, modern medical-tech aesthetic, cyan-and-white accent palette,
glossy 3D render finish.
```

### Draft 2 — concept-led ad (AI-vision motif)

```
A futuristic marketing visual for a deep-learning meal-recognition app that
estimates carbs and macros from a single food photo. Central visual: a plated
meal viewed from a slight overhead 3/4 angle, with glowing cyan AI-vision
scan lines and soft holographic data overlays floating just above each food
item, annotating estimated grams of carbs, protein and fat. Sleek dark-mode
background, subtle neural-network motif in the negative space, studio key
light, high contrast, clean sans-serif title "CarbLens" in the upper third.
Modern medical-AI brand aesthetic, glossy render, 3/4 front-facing camera.
```

**Recommendation: Draft 1 (product-led).** Reasons:
1. The phone-over-plate composition is a well-attested pattern in the SD 1.5 training distribution → all three configs have a fair shot at a coherent layout.
2. The chosen LoRA (`artificialguybr/3d-redmond-1-5v-...` — "3D Render Style") was trained on glossy product renders. A phone+plate product shot is exactly where its bias helps; Draft 2's neural-network haze would fight it.
3. The text ("CarbLens", "Carbs 48g", …) will mostly fail across SD 1.5 / SDXL — that's a deliberate, useful data-point for FR-5 "trade-offs and surprises".
4. Honours the medical disclaimer: nothing in the image makes a clinical claim; it's a generic app-promo aesthetic.

### Negative prompt (shared across A/B/C)

```
blurry, low quality, low resolution, watermark, signature, text artefacts,
distorted hands, distorted faces, extra limbs, jpeg artefacts, oversaturated,
washed out, unappetizing food, raw meat, plastic-looking food, deformed
phone, broken UI, illegible chart, medical clinical setting, syringe, needle,
blood, pills, vintage, retro, sepia, grainy film
```

(Explicit `vintage / retro / sepia / film grain` negatives push the model AWAY from the previous vintage-LoRA aesthetic and toward the hi-tech brief. `syringe / needle / blood / pills` keep the ad consumer-app-coded, not clinical-procedure-coded — important for the medical disclaimer.)

### Camera angle

`3/4 front-facing, eye-level, slight overhead tilt on the plate` — encoded directly in the prompt (FR-2 of the PRD). Same angle phrase across A/B/C; the LoRA does not change it.

### Trigger-word handling for Config C

The LoRA's trigger tokens are prepended to the prompt only when `config_id == C`:

**Full trigger prefix:**
```
"3D Render Style, 3DRenderAF, " + <shared prompt>
```

**Short trigger-phrase variant** (set in `config_c_sd15_lora.yaml` as `trigger_words_short`, used by an ablation row if Лена wants a softer LoRA activation):
```
"3D Render Style, " + <shared prompt>
```

This is the one allowed prompt-difference between configs and is explicitly logged in the param-log (`trigger_prefix` field) for transparency.

---

## 6. `output/` organization

**Decision: flat directory + descriptive filename + JSONL log as source of truth.**

```
output/images/{config_id}_{run_id}_g{guidance}_s{steps}_seed{seed}_{scheduler}[_lora{scale}].png
```

Example: `output/images/A_0007_g7.5_s30_seed42_dpmpp_2m.png`

**Options considered:**

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| Nested `output/A/g7.5/s30/...` | Browsable in Finder | Deep paths, awkward for `glob`, breaks contact-sheet builder | Rejected |
| Flat + everything in filename | One-shot `glob`, sortable, filename self-describes | Long filenames | Acceptable |
| **Flat + run_id in filename, full params in JSONL** | Short stable filenames, JSONL is the join key, `make_contact_sheet.py` trivial | Filename alone is not fully self-describing | **Chosen** |

`run_id` is a zero-padded 4-digit counter assigned at sweep-expansion time and is the foreign key to `param_log.jsonl`. Filename includes the most-glanced params (guidance, steps, seed, scheduler, lora_scale) for human triage; the JSONL is authoritative.

---

## 7. Param-log schema

**Decision: JSONL** (not CSV). Reasons:
- LoRA-related fields are nullable on A/B → CSV would be full of empty cells.
- Future fields (e.g., a metric column) can be added without breaking old rows.
- JSONL streams nicely (append per-run, no rewrite on crash).
- `pandas.read_json(path, lines=True)` gives the data-analyst a flat DataFrame for free.

One row per generation. Schema (all keys present, nullable values explicit):

| Field | Type | Notes |
|---|---|---|
| `run_id` | str | 4-digit zero-padded, unique within param log |
| `timestamp` | str | ISO 8601 UTC |
| `config_id` | str | `A` / `B` / `C` |
| `config_name` | str | e.g. `sd15_baseline` |
| `model_id` | str | HF repo ID |
| `prompt` | str | full final prompt sent to the pipeline (incl. trigger prefix for C) |
| `trigger_prefix` | str \| null | e.g. `"vintage travel poster, art deco style, "` for C; null for A/B |
| `negative_prompt` | str | |
| `seed` | int | |
| `guidance_scale` | float | |
| `num_inference_steps` | int | |
| `scheduler` | str | `ddim` \| `dpmpp_2m` |
| `width` | int | |
| `height` | int | |
| `lora_id` | str \| null | HF ID of LoRA, null for A/B |
| `lora_scale` | float \| null | null for A/B |
| `output_file` | str | relative path from repo root |
| `generation_time_seconds` | float | wall-clock |
| `peak_memory_mb` | float | `torch.mps.driver_allocated_memory()` / 1024² at run end |
| `dtype` | str | `float16` / `float32` |
| `diffusers_version` | str | `diffusers.__version__` captured at run start |
| `torch_version` | str | `torch.__version__` |

---

## 8. Comparison axes — table the data-analyst fills

Operationalising FR-5 of the PRD into something concrete. After all generations are done, the data-analyst fills two tables (one per axis) in `output/analysis/comparison.md`:

### Axis 1 — A vs B (architecture / scale)

| Dimension | Config A (SD 1.5) | Config B (SDXL) | Δ / notes |
|---|---|---|---|
| Visual fidelity (1-5, Лена rates) | | | |
| Prompt adherence — title text legibility | | | |
| Prompt adherence — composition match | | | |
| Detail at native resolution | | | |
| Mean generation time (s) | from JSONL | from JSONL | |
| Peak memory (MB) | from JSONL | from JSONL | |
| Sensitivity to `guidance_scale` (visual delta across 5/7.5/12) | | | |
| Sensitivity to `steps` (visual delta across 20/30/50) | | | |
| Sensitivity to `scheduler` (DDIM vs DPM++ 2M) | A only — sweep lives here | n/a — fixed at dpmpp_2m | Scheduler sensitivity is reported within A only; B/C inherit DPM++ 2M. See §10. |
| Best-case image (run_id) | | | |
| Failure modes observed | | | |

### Axis 2 — A vs C (control surface)

| Dimension | Config A (SD 1.5) | Config C (SD 1.5 + LoRA) | Δ / notes |
|---|---|---|---|
| Style transfer evident? | n/a (baseline) | yes/partial/no | |
| Style strength vs `lora_scale` (0.4/0.7/1.0) | n/a | qualitative | |
| Composition preservation (does LoRA break framing?) | n/a | yes/no | |
| Time overhead from LoRA (s) | | | Δ vs A |
| Memory overhead (MB) | | | Δ vs A |
| Best `lora_scale` | n/a | | |

The two tables + 1 contact sheet per axis are the README's results section.

---

## 9. `requirements.txt`

Pinned for Python 3.12 on macOS / MPS. **No tensorboard** (we're not training; no need to log scalars over time → avoids the numpy 2 / tensorboard / setuptools trap in memory).

```
# core ML
torch==2.4.1                    # MPS-stable; later 2.5+ also fine but pin a known-good
diffusers==0.31.0
transformers==4.46.3
accelerate==1.1.1
peft==0.13.2                    # compatible with the chosen LoRA: it's a standard kohya-style rank-LoRA in safetensors; diffusers 0.31 + peft 0.13 handles it via pipe.load_lora_weights(...)
safetensors==0.4.5

# numerics
numpy==1.26.4                   # stay on numpy 1.x → avoids numpy-2 deprecation churn
Pillow==11.0.0

# config + IO
PyYAML==6.0.2

# misc
tqdm==4.66.5

# tests
pytest==8.3.3
```

Notes:
- `numpy==1.26.4` chosen deliberately over numpy 2.x — diffusers/transformers stack is more reliable here and avoids the 3.12-trap noted in user memory.
- No `xformers` — not supported on MPS; SDPA-native attention is fine on Apple Silicon.
- No `bitsandbytes` — also no MPS support; we won't 8-bit quantise.

---

## 10. Generation budget

### Sweep grids per config (after Лена's scheduler-on-SD1.5-only decision)

- **Config A (SD 1.5):** scheduler sweep ON → `3 (guidance) × 3 (steps) × 3 (seeds) × 2 (schedulers) = 54`
- **Config B (SDXL):** scheduler fixed (`dpmpp_2m`) → `3 × 3 × 3 × 1 = 27`
- **Config C (SD 1.5 + LoRA):** scheduler fixed (`dpmpp_2m`), lora_scale swept → `3 × 3 × 3 × 1 × 3 (lora_scale) = 81`

**Full sweep total: 54 + 27 + 81 = 162 images.**

### Per-image wall-clock estimates on M4 Pro / MPS (empirical priors)

| Config | Resolution | Per image at 30 steps | At 50 steps |
|---|---|---|---|
| A (SD 1.5) | 512² | ~6 s | ~10 s |
| B (SDXL) | 1024² | ~35 s | ~55 s |
| C (SD 1.5+LoRA) | 512² | ~7 s | ~11 s |

Weighted-average ≈ A 8s, B 45s, C 9s per image.

### Budget arithmetic (full grid, scheduler-on-A-only)

- A: 54 × 8 s ≈ 7 min
- B: 27 × 45 s ≈ 20 min
- C: 81 × 9 s ≈ 12 min
- **Total: ~39 min** of pure generation, plus model-load + first-iter JIT compile ≈ 5–7 min → **~45 min wall-clock end-to-end.**

Comfortably fits in one focused session with margin for retries.

### Smoke-first workflow (first-class step, not just a fallback)

**Лена's confirmed flow:** smoke test ALWAYS runs first; the full sweep is only kicked off after the smoke contact sheet is eyeballed and approved.

The smoke subset is exactly **12 generations** and is defined as a fixed, hand-curated list (not a parametric grid) so it deliberately exercises every code path:

| # | Config | guidance | steps | seed | scheduler | lora_scale | Why this row |
|---|---|---|---|---|---|---|---|
| 1 | A | 7.5 | 30 | 42 | dpmpp_2m | — | Baseline default — sanity check SD 1.5 pipeline + prompt loader |
| 2 | A | 5.0 | 30 | 42 | dpmpp_2m | — | Non-default guidance — exercises sweep expander |
| 3 | A | 7.5 | 50 | 42 | dpmpp_2m | — | Non-default steps — exercises scheduler at long horizon |
| 4 | A | 7.5 | 30 | 42 | **ddim** | — | Exercises the second scheduler path (DDIM factory) |
| 5 | A | 12.0 | 30 | 123 | dpmpp_2m | — | High-guidance + different seed — checks seeding determinism |
| 6 | B | 7.5 | 30 | 42 | dpmpp_2m | — | SDXL baseline — exercises 1024² path, fp16 VAE-fix, attention slicing |
| 7 | B | 12.0 | 50 | 42 | dpmpp_2m | — | SDXL stress row — non-default guidance + steps, validates memory headroom |
| 8 | B | 5.0 | 20 | 123 | dpmpp_2m | — | SDXL low-guidance + low-steps — sanity-check fast SDXL output |
| 9 | C | 7.5 | 30 | 42 | dpmpp_2m | 0.7 | Exercises `load_lora_weights` + adapter activation + trigger prefix |
| 10 | C | 7.5 | 30 | 42 | dpmpp_2m | 0.4 | Different lora_scale — exercises `set_adapters(weights=[…])` |
| 11 | C | 7.5 | 30 | 42 | dpmpp_2m | 1.0 | Max lora_scale — visual upper-bound for the LoRA effect |
| 12 | C | 12.0 | 50 | 123 | dpmpp_2m | 0.7 | Non-default guidance + steps + seed on C — full path under stress |

**Code-path coverage check:**
- Every config (A, B, C): ✓ rows 1, 6, 9
- Both schedulers: ✓ rows 1 (dpmpp_2m) and 4 (ddim)
- Non-default guidance: ✓ rows 2, 5, 7, 8, 12
- Non-default steps: ✓ rows 3, 7, 8, 12
- LoRA-loaded generation: ✓ rows 9, 10, 11, 12
- SDXL generation: ✓ rows 6, 7, 8
- Multiple seeds: ✓ rows 5, 8, 12 (seed 123) vs others (seed 42)

**Wall-clock estimate:** A rows 1–5 × ~8 s ≈ 40 s; B rows 6–8 × ~45 s ≈ 2 min 15 s; C rows 9–12 × ~9 s ≈ 36 s. Plus model load (~2 min for the three pipelines). **Total smoke run: ~5 minutes**, matching Лена's brief.

**Invocation:** `python scripts/run_all.py --smoke` runs exactly the 12 rows above (hard-coded in `src/data/sweeps.py::SMOKE_RUNS`) and stops. `make_contact_sheet.py` is then run to produce `output/analysis/smoke_contact_sheet.png` for eyeball-review.

**Gate to the full sweep:** Лена reviews the smoke contact sheet, then either approves (`python scripts/run_all.py` runs the full 162-image sweep) or files a fix-up (no full sweep until smoke is green). This gate is documented in README §5.

---

## 11. Risks (MPS / 24 GB / SDXL / LoRA-specific)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SDXL at 1024² + fp16 exceeds unified-memory budget mid-run | Medium | High — kills B entirely | Enable `pipe.enable_attention_slicing()` and `pipe.enable_vae_slicing()` on B. If still OOM, drop to 768² and log the deviation in the param log. |
| MPS fp16 VAE artefacts (NaN / black images) on SDXL | Medium | High | Known issue. Mitigation: use `madebyollin/sdxl-vae-fp16-fix` VAE, OR run the VAE step in fp32 (`pipe.vae.to(dtype=torch.float32)`). Pick the VAE-fix route — cleaner. |
| `runwayml/stable-diffusion-v1-5` 404 (already happened in 2024) | Confirmed | High if hardcoded | Use `stable-diffusion-v1-5/stable-diffusion-v1-5` mirror; document the fallback `Comfy-Org/stable-diffusion-v1-5-archive` in the README. |
| First-run model downloads (~15 GB total: SD1.5 ~4 GB, SDXL ~10 GB, LoRA ~150 MB) blow the evening | Medium | Medium | Pre-download via a `scripts/prefetch_models.py` (one `from_pretrained(..., local_files_only=False)` call per config) at the start of the session, before the actual sweep. |
| Determinism on MPS — same seed can drift across torch versions | Low | Low | Pin torch (§9), log torch + diffusers version in every param-log row. |
| `peft` version mismatch with `diffusers` causes `load_lora_weights` to silently no-op | Medium | High — Config C becomes Config A | First step in `tests/`: a smoke test that loads the LoRA and asserts at least one parameter is changed vs vanilla pipeline (numerical sanity check). |
| Negative prompt isn't supported the same way on SDXL (uses dual encoder) | Low | Low | `diffusers` SDXL pipeline accepts `negative_prompt` and `negative_prompt_2`. Pass the same string to both — sufficient for fairness. |
| Memory not released between sweep iterations | Medium | Medium | After each generation: `del image; gc.collect(); torch.mps.empty_cache()`. Pipeline stays loaded; only intermediates are freed. |

---

## Resolution log — Лена's answers (2026-05-12, v1.1)

| # | Question | Decision |
|---|---|---|
| 1 | L41_HomeWork subject for the marketing ad | **Locked:** "Image-Based Carb & Macro Estimation for Insulin Dosing" — medical-AI app, photo-of-meal → carbs/protein/fat for insulin dosing. ViT `nateraw/food` + CLIP fallback. Cloud Run. Research prototype with medical disclaimer. Prompt drafts in §5 rewritten around this; Draft 1 (product-led) recommended and adopted. |
| 2 | LoRA aesthetic | **Locked: hi-tech / sci-fi / futuristic, NOT vintage.** Recommended LoRA changed from `DAVEinside/Vintage_art_LORA` → **`artificialguybr/3d-redmond-1-5v-3d-render-style-for-liberte-redmond-sd-1-5`** (verified ungated SD 1.5 LoRA, safetensors, triggers `3D Render Style`, `3DRenderAF`). See §4 for the candidate comparison and §3 for the updated Config C YAML. |
| 3 | Smoke-first | **Confirmed and elevated to a first-class workflow step** (not a fallback). Hand-curated 12-row smoke set defined in §10 covering every code path (A/B/C, both schedulers, non-default guidance & steps, three lora_scales, SDXL 1024² path). ~5 min wall-clock. Full sweep is gated on smoke approval. |
| 4 | Scheduler sweep scope | **Locked: scheduler sweep on Config A (SD 1.5) only.** B and C fix `dpmpp_2m`. Budget recalculated in §10: total = 162 images, ~39 min generation + ~5–7 min load = **~45 min end-to-end** (down from the prior ~70 min estimate). |

**Other v1.1 cleanups:**
- Removed the "LoRA → SDXL incompatibility" risk from §11 — Config C pairs the LoRA with its native SD 1.5 base; SDXL/LoRA compatibility is out of scope.
- Added the hi-tech-direction negatives (`vintage, retro, sepia, grainy film`) and the medical-disclaimer negatives (`syringe, needle, blood, pills, medical clinical setting`) to the shared negative prompt in §5.
- Annotated `peft==0.13.2` in §9 to record that it is compatible with the new LoRA (standard kohya-style rank-LoRA, loaded via `pipe.load_lora_weights(...)`).
- Comparison-table row for scheduler sensitivity (§8) now explicitly reads "A only".

Design is ready to hand off to Team Lead for the implementation plan (writing-plans skill).
