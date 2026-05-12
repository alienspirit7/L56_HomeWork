"""DDIM / DPM++ 2M scheduler factory."""
from __future__ import annotations

from typing import Any

from diffusers import DDIMScheduler, DPMSolverMultistepScheduler

_NAMES = {"ddim", "dpmpp_2m"}


def make_scheduler(name: str, base_scheduler_config: Any):
    """Return a scheduler instance built from `base_scheduler_config`.

    `base_scheduler_config` is `pipe.scheduler.config` from the loaded pipeline.
    """
    key = name.lower()
    if key == "ddim":
        return DDIMScheduler.from_config(base_scheduler_config)
    if key == "dpmpp_2m":
        return DPMSolverMultistepScheduler.from_config(
            base_scheduler_config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=False,
        )
    raise ValueError(f"Unknown scheduler '{name}'. Allowed: {_NAMES}")
