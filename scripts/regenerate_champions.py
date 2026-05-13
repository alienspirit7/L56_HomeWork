"""Regenerate the 6 champion images WITHOUT the UI overlay.

Same prompts, schedulers, and seeds as the sweep — only the PIL composite
step is skipped, so we get the clean text2img output that the models actually
produced. Output overwrites the champion PNGs in screenshots/.

Usage: python scripts/regenerate_champions.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.prompts import load_prompts  # noqa: E402
from src.data.sweeps import RunSpec  # noqa: E402
from src.models.loader import load_config, load_pipeline, verify_lora_loaded  # noqa: E402
from src.models.schedulers import make_scheduler  # noqa: E402
from src.utils.seeding import seed_all  # noqa: E402

CONFIG_PATHS = {
    "A": "config/config_a_sd15.yaml",
    "B": "config/config_b_sdxl.yaml",
    "C": "config/config_c_sd15_lora.yaml",
}

CHAMPIONS: List[Tuple[str, RunSpec, str]] = [
    ("A", RunSpec("0017", "A", "sd15", 7.5, 30, 42, "dpmpp_2m"),
     "champion_A_sd15_g7.5_s30_seed42.png"),
    ("A", RunSpec("0021", "A", "sd15", 7.5, 50, 42, "dpmpp_2m"),
     "champion_A_sd15_50steps.png"),
    ("B", RunSpec("0044", "B", "sdxl", 7.5, 30, 42, "dpmpp_2m"),
     "champion_B_sdxl_magazine.png"),
    ("B", RunSpec("0046", "B", "sdxl", 7.5, 50, 42, "dpmpp_2m"),
     "champion_B_sdxl_50steps.png"),
    ("C", RunSpec("0078", "C", "sd15_lora", 7.5, 30, 42, "dpmpp_2m", 0.4),
     "champion_C_lora0.4_subtle.png"),
    ("C", RunSpec("0085", "C", "sd15_lora", 7.5, 50, 42, "dpmpp_2m", 0.7),
     "champion_C_lora0.7_glossy.png"),
]


def _build_prompt(cfg, base):
    if cfg["config_id"] != "C":
        return base
    return f"{cfg['lora']['trigger_words']}, {base}"


def _generate(pipe, cfg, spec, prompt, negative):
    pipe.scheduler = make_scheduler(spec.scheduler, pipe.scheduler.config)
    if cfg["config_id"] == "C" and spec.lora_scale is not None:
        pipe.set_adapters([cfg["lora"]["adapter_name"]],
                          adapter_weights=[spec.lora_scale])
    gen = seed_all(spec.seed, device=cfg.get("device", "mps"))
    r = cfg["resolution"]
    return pipe(
        prompt=prompt, negative_prompt=negative,
        guidance_scale=spec.guidance_scale,
        num_inference_steps=spec.num_inference_steps,
        width=r["width"], height=r["height"], generator=gen,
    ).images[0]


def run() -> None:
    t0 = time.perf_counter()
    base_prompt, negative = load_prompts(REPO_ROOT / "config" / "prompts.yaml")
    screenshots_dir = REPO_ROOT / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    by_cfg: dict[str, list] = {"A": [], "B": [], "C": []}
    for cid, spec, fname in CHAMPIONS:
        by_cfg[cid].append((spec, fname))

    for cid, items in by_cfg.items():
        if not items:
            continue
        cfg = load_config(CONFIG_PATHS[cid])
        print(f"[{cid}] loading pipeline ({len(items)} champions)")
        pipe = load_pipeline(cfg)
        if cid == "C":
            print(f"[lora] verify: {verify_lora_loaded(pipe)}")
        prompt = _build_prompt(cfg, base_prompt)

        for spec, fname in tqdm(items, desc=f"regen-{cid}"):
            image = _generate(pipe, cfg, spec, prompt, negative)
            out = screenshots_dir / fname
            image.save(out)
            print(f"  saved {out.name}")
            del image; gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        del pipe; gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"[done] regen total: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    run()
