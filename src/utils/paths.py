"""Output filename builder per EXPERIMENT_DESIGN §6."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_output_path(
    output_dir: str | Path,
    config_id: str,
    run_id: str,
    guidance: float,
    steps: int,
    seed: int,
    scheduler: str,
    lora_scale: Optional[float] = None,
) -> Path:
    """Return absolute path for a generated image."""
    parts = [
        config_id,
        run_id,
        f"g{guidance:g}",
        f"s{steps}",
        f"seed{seed}",
        scheduler,
    ]
    if lora_scale is not None:
        parts.append(f"lora{lora_scale:g}")
    fname = "_".join(parts) + ".png"

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / fname


def repo_relative(path: Path) -> str:
    """Return `path` relative to repo root as a forward-slash string."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)
