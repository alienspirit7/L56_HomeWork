"""Build the right DiffusionPipeline for config A/B/C."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

from src.utils.paths import REPO_ROOT

_DTYPES = {"float16": torch.float16, "float32": torch.float32}


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else REPO_ROOT / pp


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a per-config YAML, merging in `extends: base.yaml`."""
    cfg_path = _resolve(config_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    base_name = cfg.pop("extends", None)
    if base_name:
        base_path = cfg_path.parent / base_name
        base_cfg = yaml.safe_load(base_path.read_text())
        merged = {**base_cfg, **cfg}
        # nested resolution dict merge
        if "resolution" in base_cfg and "resolution" in cfg:
            merged["resolution"] = {**base_cfg["resolution"], **cfg["resolution"]}
        cfg = merged
    return cfg


def load_pipeline(cfg: Dict[str, Any]):
    """Dispatch on `config_id` ∈ {A,B,C}. Returns a ready-to-run pipeline."""
    config_id = cfg["config_id"]
    dtype = _DTYPES[cfg.get("dtype", "float16")]
    device = cfg.get("device", "mps")

    if config_id == "B":
        kwargs = {"torch_dtype": dtype, "use_safetensors": True}
        if cfg.get("variant"):
            kwargs["variant"] = cfg["variant"]
        pipe = StableDiffusionXLPipeline.from_pretrained(cfg["model_id"], **kwargs)
        vae_fix = cfg.get("vae_fix_id")
        if vae_fix:
            pipe.vae = AutoencoderKL.from_pretrained(vae_fix, torch_dtype=torch.float32)
        else:
            pipe.vae.to(dtype=torch.float32)
        pipe.vae.config.force_upcast = True
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        return pipe

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_id"], torch_dtype=dtype, use_safetensors=True,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    if config_id == "C":
        lora = cfg["lora"]
        pipe.load_lora_weights(
            lora["id"],
            weight_name=lora["weight_name"],
            adapter_name=lora["adapter_name"],
        )
        pipe.set_adapters([lora["adapter_name"]], adapter_weights=[1.0])
    return pipe


def verify_lora_loaded(pipe) -> Dict[str, Any]:
    """Return diagnostic info confirming a LoRA adapter is active."""
    info: Dict[str, Any] = {"active_adapters": None, "has_lora": False}
    if hasattr(pipe, "get_active_adapters"):
        try:
            active = pipe.get_active_adapters()
            info["active_adapters"] = list(active) if active else []
            info["has_lora"] = bool(active)
        except (ValueError, RuntimeError) as exc:
            info["error"] = str(exc)
    return info
