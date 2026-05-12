"""Load shared prompt + negative prompt from YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import yaml

from src.utils.paths import REPO_ROOT


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_prompts(prompt_file: str | Path) -> Tuple[str, str]:
    """Return `(prompt, negative_prompt)` as single-line strings."""
    data = yaml.safe_load(_resolve(prompt_file).read_text())
    prompt = " ".join(str(data["prompt"]).split())
    negative = " ".join(str(data["negative_prompt"]).split())
    return prompt, negative
