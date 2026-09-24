"""End-to-end enhancement pipeline shared by the CLI and the web UI."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import enhance, models
from .upscale import ClassicUpscaler, ProgressFn, get_upscaler, torch_available


@dataclass
class EnhanceOptions:
    model: str = models.DEFAULT_MODEL  # a key from models.MODELS, "classic" or "none"
    scale: float = 4.0  # final output scale relative to the input
    denoise: float = 0.0
    white_balance: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    sharpen: float = 0.0
    face_restore: float = 0.0  # GFPGAN blend strength; 0 disables face restoration
    tile: int = 256
    device: str = "auto"

    def validate(self) -> None:
        if self.model not in models.MODELS and self.model not in ("classic", "none"):
            models.get_model_info(self.model)  # raises a helpful ValueError
        if not 0.25 <= self.scale <= 8:
            raise ValueError("scale must be between 0.25 and 8")
        for name in ("denoise", "white_balance", "contrast", "sharpen", "face_restore"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not -1 <= self.saturation <= 1:
            raise ValueError("saturation must be between -1 and 1")


def resolve_model(model: str) -> str:
    """Fall back to the classic upscaler when PyTorch isn't installed."""
    if model in models.MODELS and not torch_available():
        return "classic"
    return model


def _sub_progress(progress: ProgressFn | None, start: float, end: float) -> ProgressFn | None:
    """Map a stage's (done, total) onto the [start, end] slice of overall progress."""
    if progress is None:
        return None

    def report(done: int, total: int) -> None:
        frac = start + (end - start) * (done / total if total else 1)
        progress(round(frac * 1000), 1000)

    return report


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    if (w, h) == size:
        return img
    shrinking = size[0] < w
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LANCZOS4
    return cv2.resize(img, size, interpolation=interp)


def enhance_array(
    rgb: np.ndarray,
    opts: EnhanceOptions,
    alpha: np.ndarray | None = None,
    progress: ProgressFn | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Enhance an HxWx3 uint8 RGB array. Returns (rgb, alpha)."""
    opts.validate()
    if opts.face_restore > 0 and not torch_available():
        raise RuntimeError("Face restoration needs PyTorch: pip install 'photo-enhancer[ai]'")
    h, w = rgb.shape[:2]
    target = (max(1, round(w * opts.scale)), max(1, round(h * opts.scale)))

    # Noise and color casts are best fixed before the network amplifies them.
    img = enhance.denoise(rgb, opts.denoise)
    img = enhance.white_balance(img, opts.white_balance)

    # Faces are detected on the small image (faster), then restored on the
    # upscaled one so GFPGAN sees as much real detail as possible.
    faces = []
    if opts.face_restore > 0:
        from .faces import detect_faces

        faces = detect_faces(img)
    # Upscaling takes the first 70% of the progress bar when faces follow.
    upscale_share = 0.7 if faces else 1.0
    upscale_progress = _sub_progress(progress, 0, upscale_share)

    model = resolve_model(opts.model)
    if model == "none":
        img = _resize(img, target)
        if upscale_progress:
            upscale_progress(1, 1)
    else:
        if model == "classic":
            upscaler = ClassicUpscaler(max(1, int(np.ceil(opts.scale))))
        else:
            upscaler = get_upscaler(model, opts.device)
        img = upscaler.upscale(img, tile=opts.tile, progress=upscale_progress)
        img = _resize(img, target)

    if faces:
        from .faces import get_restorer

        ratio = np.array([target[0] / w, target[1] / h], dtype=np.float32)
        img = get_restorer(opts.device).restore(
            img,
            [f * ratio for f in faces],
            strength=opts.face_restore,
            progress=_sub_progress(progress, upscale_share, 1.0),
        )

    img = enhance.auto_contrast(img, opts.contrast)
    img = enhance.saturation(img, opts.saturation)
    img = enhance.sharpen(img, opts.sharpen)

    if alpha is not None:
        alpha = _resize(alpha, target)
    return img, alpha


def load_image(data: bytes | str | Path) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Decode an image into (rgb, alpha, save_info) honoring EXIF rotation."""
    src = io.BytesIO(data) if isinstance(data, bytes) else data
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        info = {"icc_profile": im.info.get("icc_profile")}
        alpha = None
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            alpha = np.array(im.getchannel("A"))
        rgb = np.array(im.convert("RGB"))
    return rgb, alpha, info


def encode_image(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    fmt: str = "PNG",
    quality: int = 95,
    info: dict | None = None,
) -> bytes:
    fmt = fmt.upper().replace("JPG", "JPEG")
    im = Image.fromarray(rgb, "RGB")
    if alpha is not None and fmt in ("PNG", "WEBP"):
        im.putalpha(Image.fromarray(alpha, "L"))
    kwargs: dict = {}
    if info and info.get("icc_profile"):
        kwargs["icc_profile"] = info["icc_profile"]
    if fmt in ("JPEG", "WEBP"):
        kwargs["quality"] = quality
    buf = io.BytesIO()
    im.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def enhance_file(
    src: str | Path,
    dst: str | Path,
    opts: EnhanceOptions,
    quality: int = 95,
    progress: ProgressFn | None = None,
) -> tuple[int, int]:
    """Enhance ``src`` and write the result to ``dst``. Returns the output size."""
    rgb, alpha, info = load_image(Path(src))
    out, out_alpha = enhance_array(rgb, opts, alpha=alpha, progress=progress)
    dst = Path(dst)
    fmt = Image.registered_extensions().get(dst.suffix.lower(), "PNG")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(encode_image(out, out_alpha, fmt, quality, info))
    return out.shape[1], out.shape[0]
