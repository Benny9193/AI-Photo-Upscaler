"""Local web UI. Binds to 127.0.0.1 by default; images never leave the machine."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from . import __version__, models
from .pipeline import EnhanceOptions, encode_image, enhance_array, load_image, resolve_model
from .upscale import torch_available

STATIC = Path(__file__).parent / "static"
MAX_OUTPUT_PIXELS = 120_000_000
MAX_JOBS = 16

app = FastAPI(title="Photo Enhancer", version=__version__)

# One worker: inference already uses every core (or the whole GPU).
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: OrderedDict[str, Job] = OrderedDict()
_jobs_lock = threading.Lock()


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"  # queued | running | done | error
    stage: str = "Waiting"
    progress: float = 0.0
    error: str | None = None
    result: bytes | None = None
    media_type: str = "image/png"
    width: int = 0
    height: int = 0
    seconds: float = 0.0
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "error": self.error,
            "width": self.width,
            "height": self.height,
            "seconds": round(self.seconds, 1),
        }


def _run_job(job: Job, data: bytes, opts: EnhanceOptions, fmt: str, metadata: str = "keep") -> None:
    start = time.perf_counter()
    job.status = "running"
    try:
        job.stage = "Decoding"
        rgb, alpha, info = load_image(data)
        h, w = rgb.shape[:2]
        if w * h * opts.scale**2 > MAX_OUTPUT_PIXELS:
            raise ValueError(
                f"Output would be {round(w * opts.scale)}x{round(h * opts.scale)}; "
                "choose a smaller scale or a smaller image."
            )

        needed = [resolve_model(opts.model)]
        if opts.face_restore > 0:
            needed += list(models.FACE_MODELS)
        if opts.scratch_removal > 0:
            needed.append("scratch-detector")
        if opts.colorize > 0:
            needed.append(opts.colorize_model)
        missing = [k for k in needed if k in models.get_all_models() and not models.is_downloaded(k)]
        for i, key in enumerate(missing):
            job.stage = f"Downloading {models.get_model_info(key).name} (first run only)"

            def dl(done: int, total: int, i: int = i) -> None:
                job.progress = (i + (done / total if total else 0)) / len(missing) * 0.2

            models.ensure_model(key, dl)

        job.stage = "Enhancing"

        def prog(done: int, total: int) -> None:
            job.progress = 0.2 + 0.75 * done / total

        out, out_alpha = enhance_array(rgb, opts, alpha=alpha, progress=prog)
        job.stage = "Encoding"
        job.result = encode_image(out, out_alpha, fmt, 95, info, metadata)
        job.media_type = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[fmt]
        job.height, job.width = out.shape[:2]
        job.progress = 1.0
        job.stage = "Done"
        job.status = "done"
    except Exception as e:  # surfaced to the UI
        traceback.print_exc()
        job.status = "error"
        job.error = str(e) or e.__class__.__name__
    finally:
        job.seconds = time.perf_counter() - start


def _get_job(job_id: str) -> Job:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info() -> dict:
    device = "cpu"
    if torch_available():
        from .upscale import pick_device

        device = pick_device()
    return {
        "version": __version__,
        "ai_available": torch_available(),
        "device": device,
        "default_model": models.DEFAULT_MODEL if torch_available() else "classic",
        "faces_downloaded": all(models.is_downloaded(k) for k in models.FACE_MODELS),
        "faces_size_mb": 350,
        "old_photo_models": {
            m.key: {"name": m.name, "downloaded": models.is_downloaded(m.key), "description": m.description}
            for m in models.OLD_PHOTO_MODELS.values()
        },
        "models": [
            {
                "key": m.key,
                "name": m.name,
                "scale": m.scale,
                "description": m.description,
                "downloaded": models.is_downloaded(m.key),
            }
            for m in models.MODELS.values()
        ],
    }


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    model: str = Form(models.DEFAULT_MODEL),
    scale: float = Form(4.0),
    denoise: float = Form(0.0),
    white_balance: float = Form(0.0),
    contrast: float = Form(0.0),
    saturation: float = Form(0.0),
    sharpen: float = Form(0.0),
    face_restore: float = Form(0.0),
    scratch_removal: float = Form(0.0),
    colorize: float = Form(0.0),
    colorize_model: str = Form(models.DEFAULT_COLORIZE_MODEL),
    format: str = Form("png"),
    metadata: str = Form("keep"),
) -> dict:
    opts = EnhanceOptions(
        model=model,
        scale=scale,
        denoise=denoise,
        white_balance=white_balance,
        contrast=contrast,
        saturation=saturation,
        sharpen=sharpen,
        face_restore=face_restore,
        scratch_removal=scratch_removal,
        colorize=colorize,
        colorize_model=colorize_model,
    )
    try:
        opts.validate()
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    fmt = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}.get(format.lower())
    if fmt is None:
        raise HTTPException(400, "format must be png, jpg or webp")
    if metadata not in ("keep", "no-gps", "strip"):
        raise HTTPException(400, "metadata must be keep, no-gps or strip")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    job = Job(id=uuid.uuid4().hex, filename=file.filename or "image")
    with _jobs_lock:
        _jobs[job.id] = job
        while len(_jobs) > MAX_JOBS:
            _jobs.popitem(last=False)
    _executor.submit(_run_job, job, data, opts, fmt, metadata)
    return job.public()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _get_job(job_id).public()


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> Response:
    job = _get_job(job_id)
    if job.status != "done" or job.result is None:
        raise HTTPException(409, "Job not finished")
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[job.media_type]
    stem = Path(job.filename).stem or "image"
    return Response(
        job.result,
        media_type=job.media_type,
        headers={"Content-Disposition": f'inline; filename="{stem}_enhanced.{ext}"'},
    )
