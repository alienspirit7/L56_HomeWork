"""Sweep grid expansion + hand-curated 12-row smoke set."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from typing import Any, Dict, List, Optional


@dataclass
class RunSpec:
    run_id: str
    config_id: str
    config_name: str
    guidance_scale: float
    num_inference_steps: int
    seed: int
    scheduler: str
    lora_scale: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def expand_sweep(cfg: Dict[str, Any], start_run_id: int = 0) -> List[RunSpec]:
    """Cartesian-product the sweep grid into RunSpecs."""
    sweep = dict(cfg.get("sweep", {}))
    sweep.update(cfg.get("sweep_overrides", {}))

    guidances = sweep["guidance_scale"]
    steps = sweep["num_inference_steps"]
    seeds = sweep["seed"]
    schedulers = sweep["scheduler"]
    lora_scales = sweep.get("lora_scale", [None])

    specs: List[RunSpec] = []
    counter = start_run_id
    for g, s, sd, sch, ls in product(guidances, steps, seeds, schedulers, lora_scales):
        specs.append(
            RunSpec(
                run_id=f"{counter:04d}",
                config_id=cfg["config_id"],
                config_name=cfg["name"],
                guidance_scale=float(g),
                num_inference_steps=int(s),
                seed=int(sd),
                scheduler=str(sch),
                lora_scale=(None if ls is None else float(ls)),
            )
        )
        counter += 1
    return specs


# Hand-curated 12-row smoke set per EXPERIMENT_DESIGN §10.
_SMOKE_ROWS = [
    ("A", "sd15_baseline",     7.5, 30,   42, "dpmpp_2m", None),
    ("A", "sd15_baseline",     5.0, 30,   42, "dpmpp_2m", None),
    ("A", "sd15_baseline",     7.5, 50,   42, "dpmpp_2m", None),
    ("A", "sd15_baseline",     7.5, 30,   42, "ddim",     None),
    ("A", "sd15_baseline",    12.0, 30,  123, "dpmpp_2m", None),
    ("B", "sdxl_base",         7.5, 30,   42, "dpmpp_2m", None),
    ("B", "sdxl_base",        12.0, 50,   42, "dpmpp_2m", None),
    ("B", "sdxl_base",         5.0, 20,  123, "dpmpp_2m", None),
    ("C", "sd15_lora_hitech",  7.5, 30,   42, "dpmpp_2m", 0.7),
    ("C", "sd15_lora_hitech",  7.5, 30,   42, "dpmpp_2m", 0.4),
    ("C", "sd15_lora_hitech",  7.5, 30,   42, "dpmpp_2m", 1.0),
    ("C", "sd15_lora_hitech", 12.0, 50,  123, "dpmpp_2m", 0.7),
]


def smoke_runs() -> List[RunSpec]:
    """Return the 12 hand-curated smoke specs."""
    return [
        RunSpec(
            run_id=f"{i:04d}",
            config_id=cid,
            config_name=cname,
            guidance_scale=g,
            num_inference_steps=s,
            seed=sd,
            scheduler=sch,
            lora_scale=ls,
        )
        for i, (cid, cname, g, s, sd, sch, ls) in enumerate(_SMOKE_ROWS)
    ]
