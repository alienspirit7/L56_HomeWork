# Summary stats

Source: `output/param_log.jsonl` (108 rows). Times are wall-clock
generation seconds; memory is peak MPS driver-allocated MB.

| Config | n | mean t (s) | median t | min t | max t | mean mem (MB) | dtype |
|---|---|---|---|---|---|---|---|
| A | 36 | 29.47 | 26.44 | 17.37 | 45.30 | 7523 | float32 |
| B | 18 | 135.35 | 123.96 | 82.39 | 202.61 | 21846 | float32 |
| C | 54 | 32.80 | 29.39 | 19.81 | 49.78 | 8148 | float32 |

## Headlines
- **A (SD 1.5 baseline):** 29.5 s / image on M4 Pro MPS at fp32, 512×768. Stable across the full sweep; widest control surface (both schedulers swept).
- **B (SDXL base):** 135.3 s / image — 4.6× slower than A and ~3× the memory footprint (21.8 GB vs 7.5 GB). Produces the most magazine-grade compositions in spot-checks.
- **C (SD 1.5 + 3D-render LoRA):** 32.8 s / image — only 11% slower than A and +625 MB memory. LoRA adds a strong stylistic shift that overrides scene composition at high `lora_scale`.
