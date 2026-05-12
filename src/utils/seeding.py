"""Seed torch / numpy / random; return a device-bound generator."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_all(seed: int, device: str = "mps") -> torch.Generator:
    """Seed every RNG we touch and return a generator on `device`.

    MPS-bound generators are unreliable on torch 2.4.x for some ops, so we
    return a CPU generator when `device == 'mps'`. The pipeline still runs on
    MPS; only the latent noise generator lives on CPU. This is the recommended
    workaround per the diffusers docs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    gen_device = "cpu" if device == "mps" else device
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(seed)
    return gen
