"""Single, final marketing composite — best champion + L41 UI screenshot.

This is the project's *deliverable* artefact: one image that markets the
L41 carb-estimation app. The model-comparison images in `screenshots/`
deliberately omit the overlay so each model can be judged on its own
scene generation; this composite is for the ad itself.

Base: B1 (SDXL, g=7.5, steps=30, seed=42, dpmpp_2m) — the strongest of
the six champions visually.

Usage: python scripts/make_marketing_composite.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.overlay import overlay_phone, load_screenshot  # noqa: E402

BASE = REPO_ROOT / "screenshots" / "champion_B_sdxl_magazine.png"
UI = REPO_ROOT / "input" / "ui_screenshot.png"
OUT = REPO_ROOT / "screenshots" / "marketing_ad_final.png"


def run() -> None:
    bg = Image.open(BASE).convert("RGB")
    ui = load_screenshot(UI)
    composite = overlay_phone(bg, ui)
    composite.save(OUT)
    print(f"[done] saved {OUT.relative_to(REPO_ROOT)}  size={composite.size}")


if __name__ == "__main__":
    run()
