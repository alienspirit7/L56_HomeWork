"""Full text2img sweep with phone-overlay compositing.

Usage: python scripts/run_sweep.py [--only A|B|C]
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import List

import diffusers
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.prompts import load_prompts  # noqa: E402
from src.data.sweeps import expand_sweep, RunSpec  # noqa: E402
from src.evaluation.metrics import peak_memory_mb, timed  # noqa: E402
from src.evaluation.overlay import overlay_phone, load_screenshot  # noqa: E402
from src.evaluation.param_log import append_row  # noqa: E402
from src.models.loader import (  # noqa: E402
    load_config, load_pipeline, verify_lora_loaded,
)
from src.models.schedulers import make_scheduler  # noqa: E402
from src.utils.paths import build_output_path, repo_relative  # noqa: E402
from src.utils.seeding import seed_all  # noqa: E402

CONFIG_PATHS = {
    "A": "config/config_a_sd15.yaml",
    "B": "config/config_b_sdxl.yaml",
    "C": "config/config_c_sd15_lora.yaml",
}
SCREENSHOT_REL = "input/ui_screenshot.png"


def _build_prompt(cfg, base_prompt):
    if cfg["config_id"] != "C":
        return base_prompt, None
    trigger = cfg["lora"]["trigger_words"]
    return f"{trigger}, {base_prompt}", trigger


def _generate_one(pipe, cfg, spec: RunSpec, prompt, negative):
    pipe.scheduler = make_scheduler(spec.scheduler, pipe.scheduler.config)
    if cfg["config_id"] == "C" and spec.lora_scale is not None:
        pipe.set_adapters(
            [cfg["lora"]["adapter_name"]], adapter_weights=[spec.lora_scale]
        )
    gen = seed_all(spec.seed, device=cfg.get("device", "mps"))
    res = cfg["resolution"]
    with timed() as elapsed:
        image = pipe(
            prompt=prompt, negative_prompt=negative,
            guidance_scale=spec.guidance_scale,
            num_inference_steps=spec.num_inference_steps,
            width=res["width"], height=res["height"], generator=gen,
        ).images[0]
    return image, elapsed()


def run(only: str | None = None) -> None:
    t_start = time.perf_counter()
    base_prompt, negative = load_prompts(REPO_ROOT / "config" / "prompts.yaml")
    screenshot = load_screenshot(REPO_ROOT / SCREENSHOT_REL)

    cfg_ids: List[str] = [only] if only else ["A", "B", "C"]
    run_counter = 0
    for cid in cfg_ids:
        cfg = load_config(CONFIG_PATHS[cid])
        specs = expand_sweep(cfg, start_run_id=run_counter)
        run_counter += len(specs)
        print(f"[{cid}] loading pipeline ({len(specs)} runs queued)")
        pipe = load_pipeline(cfg)
        if cid == "C":
            print(f"[lora] verify: {verify_lora_loaded(pipe)}")

        for spec in tqdm(specs, desc=f"sweep-{cid}"):
            final_prompt, trigger = _build_prompt(cfg, base_prompt)
            bg, gen_seconds = _generate_one(pipe, cfg, spec, final_prompt, negative)
            composited = overlay_phone(bg, screenshot)
            out_path = build_output_path(
                cfg["output_dir"], spec.config_id, spec.run_id,
                spec.guidance_scale, spec.num_inference_steps, spec.seed,
                spec.scheduler, spec.lora_scale,
            )
            composited.save(out_path)
            append_row(cfg["param_log_path"], {
                "run_id": spec.run_id, "config_id": spec.config_id,
                "config_name": spec.config_name, "model_id": cfg["model_id"],
                "prompt": final_prompt, "trigger_prefix": trigger,
                "negative_prompt": negative, "seed": spec.seed,
                "guidance_scale": spec.guidance_scale,
                "num_inference_steps": spec.num_inference_steps,
                "scheduler": spec.scheduler,
                "width": composited.width, "height": composited.height,
                "lora_id": (cfg.get("lora") or {}).get("id") if cid == "C" else None,
                "lora_scale": spec.lora_scale,
                "overlay_image": SCREENSHOT_REL,
                "output_file": repo_relative(out_path),
                "generation_time_seconds": round(gen_seconds, 3),
                "peak_memory_mb": peak_memory_mb(),
                "dtype": cfg.get("dtype", "float16"),
                "diffusers_version": diffusers.__version__,
                "torch_version": torch.__version__,
            })
            del bg, composited; gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        del pipe; gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"[done] sweep total: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["A", "B", "C"], default=None)
    args = ap.parse_args()
    run(only=args.only)
