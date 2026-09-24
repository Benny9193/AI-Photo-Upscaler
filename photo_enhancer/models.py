"""Registry of supported AI models and a small download/cache manager.

Weights are fetched once from their official release URLs, verified against a
pinned SHA-256, and stored under the cache directory. After that, everything
runs fully offline.

Some weights only ship inside large archives. For those, just the one member
is read out of the remote zip with HTTP range requests, so there's no need to
download the whole archive.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    key: str
    name: str
    scale: int
    url: str
    sha256: str | None  # of the downloaded file (or zip member)
    description: str
    size: int | None = None  # checked when no SHA-256 is pinned
    member: str | None = None  # path inside a zip at ``url``
    keep: str | None = None  # keep only this entry of a torch checkpoint
    file: str | None = None  # local filename, if not the URL's

    @property
    def filename(self) -> str:
        return self.file or (self.member or self.url).rsplit("/", 1)[-1]


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

# Face restoration: a detector that finds faces and their landmarks, and the
# network that redraws each aligned face.
FACE_MODELS: dict[str, ModelInfo] = {
    m.key: m
    for m in [
        ModelInfo(
            key="gfpgan",
            name="GFPGAN v1.4",
            scale=1,
            url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            sha256="e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
            description="Restores blurry, low-resolution or damaged faces.",
        ),
        ModelInfo(
            key="yunet",
            name="YuNet face detector",
            scale=1,
            url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
            description="Finds faces and their landmarks for GFPGAN. Runs with OpenCV.",
        ),
    ]
}


# Old photo restoration: scratch detection and colorization.
OLD_PHOTO_MODELS: dict[str, ModelInfo] = {
    m.key: m
    for m in [
        ModelInfo(
            key="scratch-detector",
            name="Scratch detector (Bringing Old Photos Back to Life)",
            scale=1,
            url="https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip",
            member="checkpoints/detection/FT_Epoch_latest.pt",
            sha256="b2d7ab04e9b3885c6b1991bb7a0b823129dd6e3ac078a9fd059ebd2a7ba59a95",
            keep="model_state",
            file="scratch_detector.pth",
            description="Finds scratches, creases and dust. Reads ~420 MB out of Microsoft's 2 GB archive; stores 150 MB.",
        ),
        ModelInfo(
            key="ddcolor",
            name="DDColor (ModelScope)",
            scale=1,
            url="https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth",
            sha256=None,
            size=911950059,
            description="Best colorization quality for photos. 912 MB.",
        ),
        ModelInfo(
            key="ddcolor-tiny",
            name="DDColor Tiny",
            scale=1,
            url="https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_paper_tiny.pth",
            sha256=None,
            size=220393145,
            description="Smaller, faster colorization model. 220 MB.",
        ),
    ]
}
DEFAULT_COLORIZE_MODEL = "ddcolor"


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


def get_all_models() -> dict[str, ModelInfo]:
    return {**MODELS, **FACE_MODELS, **OLD_PHOTO_MODELS}


def get_model_info(key: str) -> ModelInfo:
    info = get_all_models().get(key)
    if info is None:
        raise ValueError(f"Unknown model {key!r}. Choose from: {', '.join(get_all_models())}")
    return info


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file, fetched with Range requests."""

    def __init__(self, url: str):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
            self.url = resp.url  # follow redirects once (e.g. to a signed CDN URL)
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buf) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _copy(src, out, total: int, progress: Callable[[int, int], None] | None) -> None:
    done = 0
    while chunk := src.read(1 << 20):
        out.write(chunk)
        done += len(chunk)
        if progress:
            progress(done, total)


def _fetch(info: ModelInfo, out, progress: Callable[[int, int], None] | None) -> None:
    if info.member:
        remote = io.BufferedReader(_HTTPRangeFile(info.url), buffer_size=4 << 20)
        with zipfile.ZipFile(remote) as zf, zf.open(info.member) as src:
            _copy(src, out, zf.getinfo(info.member).file_size, progress)
    else:
        with urllib.request.urlopen(info.url) as resp:
            _copy(resp, out, int(resp.headers.get("Content-Length") or 0), progress)


def _verify(info: ModelInfo, path: Path) -> None:
    if info.sha256:
        digest = _sha256(path)
        if digest != info.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {info.filename}: expected {info.sha256}, got {digest}"
            )
    elif info.size is not None and path.stat().st_size != info.size:
        raise RuntimeError(
            f"Size mismatch for {info.filename}: expected {info.size} bytes, got {path.stat().st_size}"
        )


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
        with os.fdopen(fd, "wb") as out:
            _fetch(info, out, progress)
        _verify(info, tmp)
        if info.keep:
            # Drop training state (optimizer etc.) to save disk space. The
            # weights-only loader refuses anything but plain tensors and containers.
            import torch

            state = torch.load(tmp, map_location="cpu", weights_only=True)[info.keep]
            torch.save(state, tmp)
        shutil.move(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
