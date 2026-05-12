# Iteration log — L56_HomeWork

Chronological record of how the generation pipeline evolved before the full
sweep. Used as raw material for the final README write-up (§5/§6).

---

## Iteration 0 — Initial design (text2img, 3 configs)

- **Configs locked:** A = SD 1.5, B = SDXL base 1.0, C = SD 1.5 + 3D-render LoRA
  (`artificialguybr/3d-redmond-1-5v-3d-render-style-for-liberte-redmond-sd-1-5`).
- **Pipeline:** plain `StableDiffusionPipeline` / `StableDiffusionXLPipeline`.
- **Resolution:** A/C 512×512 square, B 1024×1024 square.
- **Prompt v1:** smartphone hero shot of a plated dinner with translucent
  UI cards spelling out the macros — i.e. described the *result screen* in
  words.
- **Result:** 12-image smoke produced all-black PNGs.

## Iteration 1 — Fix MPS fp16 VAE NaN

- **Symptom:** every output was uniformly `(0,0,0)`. Diffusers logged
  `RuntimeWarning: invalid value encountered in cast` from the image processor
  — VAE decoded `NaN`s.
- **Tried first:** disable SD 1.5 safety-checker (the only model-level black
  source we knew about). Still black.
- **Tried second:** `pipe.upcast_vae()` for SDXL + `madebyollin/sdxl-vae-fp16-fix`.
  Still black. `force_upcast=True` + fp32 VAE with fp16 UNet — still black.
  Conclusion: the NaN was inside the UNet itself on MPS, not just VAE.
- **Tried third:** full **fp32** for all three configs. SDXL fp32 in cold cache
  → 22 min load (downloads ~13 GB of fp32 weights). Warm cache → 14 s. Image
  came back at 99.9% non-black.
- **Resolution:** locked `dtype: float32` for A, B, C. `safety_checker=None`
  for SD 1.5. SDXL `variant` field removed. Accepted ~80 s/image for SDXL on
  MPS as the cost of correctness.

## Iteration 2 — img2img with composited UI init

- **Idea:** drop the user's L41 UI screenshot into the canvas as `init_image`,
  let img2img stylise the surroundings. Loader switched to
  `StableDiffusionImg2ImgPipeline` / `StableDiffusionXLImg2ImgPipeline`.
- **Init prep (`scripts/prepare_init.py`):** phone screenshot pasted onto a
  square canvas with bezel + drop-shadow at 72% canvas height.
- **Strength sweep:** 0.4 / 0.6 / 0.85.
- **Result:** chaos. At 0.4 the UI was preserved but nothing else changed (no
  food, no marble — just the phone on off-white). At 0.85 the UI text turned
  into illegible glyph soup. Prompt described a phone *beside* a plate; init
  showed a phone *alone* in the centre. Models had no consistent way to
  reconcile the two.

## Iteration 3 — Smaller phone, more headroom, lower strength

- **Init prep tweaked:** phone shrunk to 50% of canvas height, centre lifted
  to 28% from top, leaving the bottom half blank for the model to paint food.
- **Strength sweep:** 0.4 / 0.55 / 0.7 (dropped 0.85 — it destroyed the UI).
- **Result:** still bad. UI text remained illegible glyphs; the empty bottom
  half stayed empty. Conclusion: img2img with a fixed init and a uniform
  strength can't both *preserve* the UI and *invent* a new scene around it.
  This is a fundamental limitation without masking / inpainting / ControlNet.

## Iteration 4 — Pivot to text2img + PIL phone overlay

- **New plan:** let the model generate the *scene* (food + marble + light) in
  pure text2img; composite the user's real screenshot in afterwards with PIL.
  This restores comparison validity (each model paints the scene differently)
  while guaranteeing the UI is 100% accurate.
- **Aspect changed:** 1:1 → portrait 3:4. A/C 512×768, B 896×1152 (SDXL-native).
- **Prompt v2:** "overhead flat-lay marketing hero shot ... beautifully plated
  dinner ... copy space across the top third of the frame" + negative
  prompt blocks `smartphone, phone, mobile device, screen` so the model
  doesn't draw a phone.
- **Overlay v1 (`src/evaluation/overlay.py`):** screenshot rendered with a
  dark bezel + soft shadow, placed top-centre at 20% from top.
- **Result:** beautiful, magazine-ready food photography — first images that
  actually looked like ads. SDXL especially produced real spreads with
  cutlery, sauces, multiple plates.

## Iteration 5 — Reposition + reshape overlay (current)

- **Лена's note:** scene should show a *person photographing* the food, with
  the UI alongside as a callout — and the UI needs to be **bigger so the
  numbers are legible**.
- **Prompt v3:** "a person's hand holding a modern matte smartphone and
  *photographing* a single beautifully plated dinner". Negative prompt
  re-permits a phone (model needs it in the scene) but now bans
  `two phones, multiple phones, multiple plates`.
- **First attempt (overlay in bottom-right, with bezel):** every config
  rendered its own phone in the scene; our PIL phone overlay sat next to it
  → "two phones" visible. Especially obvious on SDXL.
- **Overlay v2 (current):** remove the bezel; render the screenshot as a
  large white-bordered **card with drop-shadow** anchored to the right edge
  (X centre 0.82, Y centre 0.50), 46% of canvas height. This reads as a UI
  *callout panel*, not a second device. Numbers in the UI screenshot are
  now legible.
- **Quick test (3 images, one per config):**
  - A — clean flat-lay, hand-with-phone bottom-left, UI card top-right.
  - B — magazine spread, hand-with-phone top, UI card right side. Best of
    the three by a wide margin.
  - C — denser 3D-render aesthetic from LoRA, UI card bottom-right.

  All three reproducible at the same seed; no "two phones" artefact.

## What carries into the full sweep

- text2img, fp32, portrait 3:4.
- Prompt v3 + negative v3.
- PIL card overlay (right side, 46% canvas height).
- Seeds reduced to `[42, 123]` (was 3) to keep wall-clock under ~1.5 h.
- Final budget: A=36, B=18, C=54 → **108 images**, estimated ~90 min on
  warm HF cache.
