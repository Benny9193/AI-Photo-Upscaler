"""Registry of supported AI models and a small download/cache manager.

Weights are fetched once from their official release URLs, verified against a
pinned SHA-256, and stored under the cache directory. After that, everything
runs fully offline.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ModelInfo:
    key: str
    name: str
    scale: int
    url: str
    sha256: str
    description: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


MODELS: dict[str, ModelInfo] = {
    m.key: m
    for m in [
        ModelInfo(
            key="general-x4",
            name="Real-ESRGAN General v3 (4x)",
            scale=4,
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
            description="Small and fast. Good default for everyday photos, especially on CPU.",
        ),
        ModelInfo(
            key="realesrgan-x4",
            name="Real-ESRGAN x4plus (4x)",
            scale=4,
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
            description="Highest quality for real-world photos. Slower; a GPU is recommended.",
        ),
        ModelInfo(
            key="realesrgan-x2",
            name="Real-ESRGAN x2plus (2x)",
            scale=2,
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
            description="Native 2x upscaling for real-world photos.",
        ),
        ModelInfo(
            key="anime-x4",
            name="Real-ESRGAN x4plus Anime (4x)",
            scale=4,
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            sha256="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
            description="Tuned for illustrations, anime and line art.",
        ),
    ]
}

DEFAULT_MODEL = "general-x4"


def cache_dir() -> Path:
    """Where model weights live. Override with PHOTO_ENHANCER_MODELS."""
    env = os.environ.get("PHOTO_ENHANCER_MODELS")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "photo-enhancer" / "models"


def model_path(key: str) -> Path:
    return cache_dir() / get_model_info(key).filename


def is_downloaded(key: str) -> bool:
    return model_path(key).is_file()


def get_model_info(key: str) -> ModelInfo:
    try:
        return MODELS[key]
    except KeyError:
        raise ValueError(f"Unknown model {key!r}. Choose from: {', '.join(MODELS)}") from None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(
    key: str,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Return the local path to a model's weights, downloading them if needed."""
    info = get_model_info(key)
    dest = model_path(key)
    if dest.is_file():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(info.url) as resp, os.fdopen(fd, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        digest = _sha256(tmp)
        if digest != info.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {info.filename}: expected {info.sha256}, got {digest}"
            )
        shutil.move(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
