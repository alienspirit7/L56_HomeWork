# Champion picks

One or two best images per config. Subjective but grounded — every pick names
*what* makes it stand out vs its config siblings. Source files are in
`output/images/`; chosen copies live in `screenshots/` for the README.

---

## Config A — SD 1.5 baseline (n=36)

### Pick A1 — `A_0017_g7.5_s30_seed42_dpmpp_2m.png` → `screenshots/champion_A_sd15_g7.5_s30_seed42.png`

The default-tier (g=7.5, steps=30) DPM++ 2M render at seed 42. Why this one:

- **Composition is the cleanest of the 36 A images** — phone bottom-left, multi-plate veg arrangement around it, marble surface reads as a real countertop rather than a flat background.
- **UI card legibility is preserved** because the surrounding scene is bright marble — the dark UI panel pops on the right edge.
- It's the canonical comparison anchor used in `comparison_matched.png`, so picking it as champion keeps cross-config references consistent.

### Pick A2 — `A_0021_g7.5_s50_seed42_dpmpp_2m.png` → `screenshots/champion_A_sd15_50steps.png`

Same coord as A1 but 50 steps. Why this one as a second pick:

- **Tighter food textures**: the broccoli florets and the carrots resolve to distinct shapes (in A1 at 30 steps they're slightly soft).
- **Demonstrates the steps-axis payoff** — useful as evidence in the README that A actually responds to step count, not just default-and-done.

---

## Config B — SDXL base (n=18)

### Pick B1 — `B_0044_g7.5_s30_seed42_dpmpp_2m.png` → `screenshots/champion_B_sdxl_magazine.png`

The default coord on SDXL. Why this one:

- **Magazine-spread quality** — multiple plates of salmon, fresh herbs, lemon wedges, an oil bowl, hand-held phone as the hero element. Reads as an actual food-app campaign shot, not a tech-demo render.
- **Best balance of guidance** — at g=12 (`B_0050…`) SDXL hallucinates a second photo inside the phone screen; at g=5 the scene goes pastel. g=7.5 is the sweet spot.
- Headline visual for "what SDXL buys you" in the README.

### Pick B2 — `B_0046_g7.5_s50_seed42_dpmpp_2m.png` → `screenshots/champion_B_sdxl_50steps.png`

Same coord at 50 steps. Why this one:

- **Sleeve and hand textures are noticeably cleaner** than B1's 30-step version — the cable-knit pattern on the sweater resolves at 50 steps.
- Pairs with A2 in the README to demonstrate that *both* models reward extra steps, but B starts from a much higher baseline.

---

## Config C — SD 1.5 + 3D-render LoRA (n=54)

### Pick C1 — `C_0085_g7.5_s50_seed42_dpmpp_2m_lora0.7.png` → `screenshots/champion_C_lora0.7_glossy.png`

`lora_scale=0.7` at g=7.5, steps=50, seed=42. Why this one:

- **The clearest demonstration of what the LoRA actually does** — dark serving trays, glossy reflective sheen on the food, saturated greens and oranges. The aesthetic is unmistakably distinct from any A image.
- **`lora_scale=0.7` is the design's recommended setting** and this is its strongest visual instance in the sweep.
- Still has the phone overlay legible against the lighter top-right area.

### Pick C2 — `C_0078_g7.5_s30_seed42_dpmpp_2m_lora0.4.png` → `screenshots/champion_C_lora0.4_subtle.png`

Same coord at `lora_scale=0.4`. Why this one:

- **Demonstrates the LoRA-strength axis** — at 0.4 the framing still includes the phone and marble surface (much like A), but the food has the LoRA's glossy 3D-render finish.
- Useful as a "before/after" pair with C1 in the README to show that `lora_scale` is a real control knob, not just a slider that does nothing.
- Honest about the trade-off documented in `comparison_tables.md`: at 0.4 the LoRA is subtle; at 0.7+ it overrides composition.

---

## Total: 6 PNGs copied to `screenshots/`

| Source | Destination |
|---|---|
| `output/images/A_0017_g7.5_s30_seed42_dpmpp_2m.png` | `screenshots/champion_A_sd15_g7.5_s30_seed42.png` |
| `output/images/A_0021_g7.5_s50_seed42_dpmpp_2m.png` | `screenshots/champion_A_sd15_50steps.png` |
| `output/images/B_0044_g7.5_s30_seed42_dpmpp_2m.png` | `screenshots/champion_B_sdxl_magazine.png` |
| `output/images/B_0046_g7.5_s50_seed42_dpmpp_2m.png` | `screenshots/champion_B_sdxl_50steps.png` |
| `output/images/C_0085_g7.5_s50_seed42_dpmpp_2m_lora0.7.png` | `screenshots/champion_C_lora0.7_glossy.png` |
| `output/images/C_0078_g7.5_s30_seed42_dpmpp_2m_lora0.4.png` | `screenshots/champion_C_lora0.4_subtle.png` |
