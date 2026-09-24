"""Colorization of black-and-white photos with DDColor (Kang et al., ICCV 2023).

DDColor predicts the color channels of a Lab image from its lightness. The
original lightness is kept at full resolution, so detail is untouched and
only the color comes from the (512px) network output.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from . import models
from .upscale import pick_device

# The official DDColor inference runs its models at 512x512.
INPUT_SIZE = 512


def tone_curve(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the photo's tint as a function of lightness.

    Returns a 256x2 table mapping Lab lightness to the average (a, b) at that
    lightness, and how far pixels stray from it. A black-and-white print,
    toned or not (sepia, selenium, cyanotype), has every pixel on that curve.
    """
    small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA) if max(rgb.shape[:2]) > 256 else rgb
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lightness, ab = lab[:, 0].astype(int), lab[:, 1:]
    counts = np.bincount(lightness, minlength=256)
    table = np.full((256, 2), 128.0, np.float32)
    seen = counts > 0
    for c in range(2):
        sums = np.bincount(lightness, weights=ab[:, c], minlength=256)
        table[seen, c] = sums[seen] / counts[seen]
        # Fill lightness values the photo doesn't use from their neighbours.
        table[:, c] = np.interp(np.arange(256), np.nonzero(seen)[0], table[seen, c])
    spread = float(np.abs(ab - table[lightness]).mean())
    return table, spread


def is_grayscale(rgb: np.ndarray, tolerance: float = 3.0) -> bool:
    """True for black-and-white photos, including toned ones (sepia, cyanotype)."""
    return tone_curve(rgb)[1] < tolerance


def apply_tone(rgb: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Make ``rgb`` monochrome, tinted by lightness according to ``table``."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 1:] = np.clip(np.round(table), 0, 255).astype(np.uint8)[lab[..., 0]]
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class Colorizer:
    def __init__(self, model_key: str = models.DEFAULT_COLORIZE_MODEL, device: str = "auto"):
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader

        try:
            import spandrel_extra_arches
        except ImportError:
            raise RuntimeError(
                "Colorization needs spandrel_extra_arches: pip install 'photo-enhancer[ai]'"
            ) from None
        spandrel_extra_arches.install(ignore_duplicates=True)

        self.device = pick_device(device)
        descriptor = ModelLoader().load_from_file(models.ensure_model(model_key))
        if not isinstance(descriptor, ImageModelDescriptor) or descriptor.architecture.id != "DDColor":
            raise TypeError(f"{model_key} did not load as a DDColor model")
        # spandrel defaults to 256px; the architecture works at any size.
        descriptor.model.input_size = (INPUT_SIZE, INPUT_SIZE)
        descriptor.to(self.device).eval()
        self._model = descriptor
        self._torch = torch
        self._lock = threading.Lock()

    def colorize(self, rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Colorize ``rgb``. ``strength`` scales the predicted color (1 = as predicted)."""
        torch = self._torch
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        t = torch.from_numpy(gray).to(self.device, torch.float32)[None, None] / 255
        with self._lock, torch.inference_mode():
            out = self._model(t)[0].clamp_(0, 1).mul_(255).round_().byte().permute(1, 2, 0).cpu().numpy()
        # Keep the original lightness; take only the color from the network,
        # blended with the photo's own tint by ``strength``.
        lab_src = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab = lab_src.copy()
        lab[..., 1:] = lab_src[..., 1:] + (lab_out[..., 1:] - lab_src[..., 1:]) * strength
        return cv2.cvtColor(lab.round().clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


_colorizers: dict[tuple[str, str], Colorizer] = {}
_colorizers_lock = threading.Lock()


def get_colorizer(model_key: str = models.DEFAULT_COLORIZE_MODEL, device: str = "auto") -> Colorizer:
    with _colorizers_lock:
        key = (model_key, device)
        if key not in _colorizers:
            _colorizers[key] = Colorizer(model_key, device)
        return _colorizers[key]
