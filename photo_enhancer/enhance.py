"""Classic photo adjustments applied around the AI upscale.

All functions take and return HxWx3 uint8 RGB arrays. Strength arguments are
0..1 where 0 is a no-op.
"""

from __future__ import annotations

import cv2
import numpy as np


def denoise(img: np.ndarray, strength: float) -> np.ndarray:
    """Remove sensor noise and JPEG grain while preserving edges."""
    if strength <= 0:
        return img
    h = 3 + 12 * strength
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def auto_contrast(img: np.ndarray, strength: float) -> np.ndarray:
    """Local contrast via CLAHE on the lightness channel (colors untouched)."""
    if strength <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_chan, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0 + 2.0 * strength, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    l_out = cv2.addWeighted(l_chan, 1 - strength, l_eq, strength, 0)
    return cv2.cvtColor(cv2.merge((l_out, a, b)), cv2.COLOR_LAB2RGB)


def white_balance(img: np.ndarray, strength: float) -> np.ndarray:
    """Gray-world white balance to neutralize color casts."""
    if strength <= 0:
        return img
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1e-3)
    gains = 1 + (gains - 1) * strength
    return np.clip(f * gains, 0, 255).astype(np.uint8)


def saturation(img: np.ndarray, amount: float) -> np.ndarray:
    """Boost (amount > 0) or reduce (amount < 0) color saturation."""
    if amount == 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + amount), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def sharpen(img: np.ndarray, strength: float) -> np.ndarray:
    """Unsharp mask."""
    if strength <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    amount = 1.5 * strength
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
