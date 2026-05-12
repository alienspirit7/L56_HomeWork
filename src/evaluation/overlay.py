"""Composite the L41 UI screenshot as a large readable inset card.

The card is anchored to the right edge of the frame, vertically centred, with
a white border + soft drop-shadow so it reads as a UI callout (not a second
phone in the scene).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

BORDER_COLOR = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0, 90)

CARD_HEIGHT_FRAC = 0.46      # card fills 46% of canvas height
CARD_X_CENTRE_FRAC = 0.82    # horizontal centre (0=left, 1=right)
CARD_Y_CENTRE_FRAC = 0.50    # vertical centre
BORDER_FRAC = 0.008
CORNER_FRAC = 0.022
SHADOW_BLUR_FRAC = 0.018


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return mask


def overlay_phone(background: Image.Image, screenshot: Image.Image) -> Image.Image:
    """Return a new image with the UI screenshot composited as a side card."""
    canvas = background.convert("RGB").copy()
    cw, ch = canvas.size

    border_pad = max(4, int(ch * BORDER_FRAC))
    corner = max(10, int(ch * CORNER_FRAC))
    blur = max(6, int(ch * SHADOW_BLUR_FRAC))

    target_h = int(ch * CARD_HEIGHT_FRAC)
    sw, sh = screenshot.size
    target_w = int(sw * (target_h / sh))
    screen = screenshot.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    screen_mask = _rounded_mask(screen.size, corner)

    card_w = target_w + 2 * border_pad
    card_h = target_h + 2 * border_pad
    card = Image.new("RGB", (card_w, card_h), BORDER_COLOR)
    card_mask = _rounded_mask(card.size, corner + border_pad // 2)

    cx = int(cw * CARD_X_CENTRE_FRAC) - card_w // 2
    cy = int(ch * CARD_Y_CENTRE_FRAC) - card_h // 2
    cx = max(border_pad, min(cx, cw - card_w - border_pad))
    cy = max(border_pad, min(cy, ch - card_h - border_pad))

    shadow = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    offset = max(8, int(ch * 0.015))
    sd.rounded_rectangle(
        (cx + offset, cy + offset * 2, cx + card_w + offset, cy + card_h + offset * 2),
        radius=corner + border_pad // 2, fill=SHADOW_COLOR,
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    canvas.paste(shadow, (0, 0), shadow)
    canvas.paste(card, (cx, cy), card_mask)
    canvas.paste(screen, (cx + border_pad, cy + border_pad), screen_mask)
    return canvas


def load_screenshot(path: str | Path) -> Image.Image:
    return Image.open(Path(path)).convert("RGB")
