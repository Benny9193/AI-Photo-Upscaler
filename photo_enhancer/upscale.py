"""Super-resolution backends.

``AIUpscaler`` runs a Real-ESRGAN family network locally through PyTorch (via
spandrel, which reads the original .pth checkpoints). Large images are split
into overlapping tiles so memory use stays bounded regardless of input size.

``ClassicUpscaler`` is a dependency-light fallback (Lanczos + detail boost)
used when PyTorch isn't installed.
"""

from __future__ import annotations

import threading
from typing import Callable

import cv2
import numpy as np

from . import models

ProgressFn = Callable[[int, int], None]


def pick_device(preference: str = "auto") -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_available() -> bool:
    try:
        import spandrel  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _pad_to_requirements(tile: np.ndarray, minimum: int, multiple_of: int) -> np.ndarray:
    h, w = tile.shape[:2]
    th, tw = max(h, minimum), max(w, minimum)
    if multiple_of > 1:
        th = -(-th // multiple_of) * multiple_of
        tw = -(-tw // multiple_of) * multiple_of
    if (th, tw) == (h, w):
        return tile
    mode = cv2.BORDER_REFLECT_101 if h > 1 and w > 1 else cv2.BORDER_REPLICATE
    return cv2.copyMakeBorder(tile, 0, th - h, 0, tw - w, mode)


def _fit_span(lo: int, hi: int, size: int, minimum: int, multiple_of: int) -> tuple[int, int]:
    """Grow [lo, hi) with real image pixels until it meets the model's size needs.

    Using neighbouring pixels instead of reflect padding keeps tiles at the
    image border from producing seams. Padding is only needed when the whole
    image is smaller than the requirement.
    """
    want = max(hi - lo, minimum)
    if multiple_of > 1:
        want = -(-want // multiple_of) * multiple_of
    want = min(want, size)
    extra = want - (hi - lo)
    grow_hi = min(extra, size - hi)
    hi += grow_hi
    lo -= min(extra - grow_hi, lo)
    return lo, hi


def iter_tiles(height: int, width: int, tile: int, overlap: int):
    """Yield (y0, y1, x0, x1, iy0, iy1, ix0, ix1).

    ``[y0:y1, x0:x1]`` is the region handed to the network (tile plus overlap),
    ``[iy0:iy1, ix0:ix1]`` is the core region that tile is responsible for.
    Core regions exactly partition the image.
    """
    if tile <= 0:
        yield 0, height, 0, width, 0, height, 0, width
        return
    for iy0 in range(0, height, tile):
        iy1 = min(iy0 + tile, height)
        y0, y1 = max(iy0 - overlap, 0), min(iy1 + overlap, height)
        for ix0 in range(0, width, tile):
            ix1 = min(ix0 + tile, width)
            x0, x1 = max(ix0 - overlap, 0), min(ix1 + overlap, width)
            yield y0, y1, x0, x1, iy0, iy1, ix0, ix1


class AIUpscaler:
    """Runs a local super-resolution network on RGB uint8 images."""

    def __init__(
        self,
        model_key: str = models.DEFAULT_MODEL,
        device: str = "auto",
        half: bool | None = None,
    ):
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader

        self.model_key = model_key
        self.device = pick_device(device)
        path = models.ensure_model(model_key)
        descriptor = ModelLoader().load_from_file(path)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise TypeError(f"{model_key} is not an image-to-image model")
        if half is None:
            half = self.device == "cuda" and descriptor.supports_half
        self.half = half
        descriptor.to(self.device).eval()
        if half:
            descriptor.half()
        self._model = descriptor
        self._torch = torch
        self._lock = threading.Lock()
        self.scale: int = descriptor.scale

    def _run(self, tile: np.ndarray) -> np.ndarray:
        torch = self._torch
        req = self._model.size_requirements
        h, w = tile.shape[:2]
        padded = _pad_to_requirements(tile, req.minimum, req.multiple_of)
        if not padded.flags.writeable:
            padded = padded.copy()
        t = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)
        t = t.to(self.device, dtype=torch.float16 if self.half else torch.float32) / 255.0
        with torch.inference_mode():
            out = self._model(t).squeeze(0).clamp_(0, 1).mul_(255).round_().byte()
        out = out.permute(1, 2, 0).cpu().numpy()
        return out[: h * self.scale, : w * self.scale]

    def upscale(
        self,
        img: np.ndarray,
        tile: int = 256,
        overlap: int = 16,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        """Upscale an HxWx3 uint8 RGB image by the model's native scale."""
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("expected an HxWx3 RGB image")
        h, w = img.shape[:2]
        s = self.scale
        out = np.empty((h * s, w * s, 3), dtype=np.uint8)
        tiles = list(iter_tiles(h, w, tile, overlap))
        req = self._model.size_requirements
        with self._lock:
            for i, (y0, y1, x0, x1, iy0, iy1, ix0, ix1) in enumerate(tiles):
                y0, y1 = _fit_span(y0, y1, h, req.minimum, req.multiple_of)
                x0, x1 = _fit_span(x0, x1, w, req.minimum, req.multiple_of)
                result = self._run(np.ascontiguousarray(img[y0:y1, x0:x1]))
                oy, ox = (iy0 - y0) * s, (ix0 - x0) * s
                out[iy0 * s : iy1 * s, ix0 * s : ix1 * s] = result[
                    oy : oy + (iy1 - iy0) * s, ox : ox + (ix1 - ix0) * s
                ]
                if progress:
                    progress(i + 1, len(tiles))
        return out


class ClassicUpscaler:
    """Non-AI fallback: Lanczos resampling followed by a gentle detail boost."""

    model_key = "classic"
    device = "cpu"

    def __init__(self, scale: int = 4):
        self.scale = scale

    def upscale(self, img: np.ndarray, progress: ProgressFn | None = None, **_) -> np.ndarray:
        h, w = img.shape[:2]
        out = cv2.resize(img, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0 * self.scale / 2)
        out = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
        if progress:
            progress(1, 1)
        return out


_cache: dict[tuple[str, str], AIUpscaler] = {}
_cache_lock = threading.Lock()


def get_upscaler(model_key: str, device: str = "auto") -> AIUpscaler:
    """Return a cached upscaler, so the web UI loads each model only once."""
    with _cache_lock:
        key = (model_key, device)
        if key not in _cache:
            _cache[key] = AIUpscaler(model_key, device=device)
        return _cache[key]
