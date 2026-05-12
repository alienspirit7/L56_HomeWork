"""Append-only JSONL writer with 23-field schema validation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.utils.paths import REPO_ROOT

PARAM_LOG_FIELDS = [
    "run_id",
    "timestamp",
    "config_id",
    "config_name",
    "model_id",
    "prompt",
    "trigger_prefix",
    "negative_prompt",
    "seed",
    "guidance_scale",
    "num_inference_steps",
    "scheduler",
    "width",
    "height",
    "lora_id",
    "lora_scale",
    "overlay_image",
    "output_file",
    "generation_time_seconds",
    "peak_memory_mb",
    "dtype",
    "diffusers_version",
    "torch_version",
]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def make_row(**kwargs: Any) -> Dict[str, Any]:
    """Build a schema-conformant row. Missing keys raise immediately."""
    row: Dict[str, Any] = {}
    for field in PARAM_LOG_FIELDS:
        if field == "timestamp" and field not in kwargs:
            row[field] = datetime.now(timezone.utc).isoformat()
            continue
        if field not in kwargs:
            raise KeyError(f"param_log row missing required field: {field}")
        row[field] = kwargs[field]
    extras = set(kwargs) - set(PARAM_LOG_FIELDS)
    if extras:
        raise KeyError(f"param_log row has unknown fields: {sorted(extras)}")
    return row


def append_row(path: str | Path, row: Dict[str, Any]) -> None:
    """Append a JSON row (validated) to the JSONL file."""
    validated = make_row(**row)
    out_path = _resolve(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(validated, ensure_ascii=False) + "\n")
