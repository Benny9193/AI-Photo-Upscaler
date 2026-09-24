"""Command-line interface: ``photo-enhancer``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import models
from .pipeline import EnhanceOptions, enhance_file, resolve_model
from .upscale import torch_available

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _bar(prefix: str):
    def show(done: int, total: int) -> None:
        if total:
            pct = done / total
            print(f"\r{prefix} [{'#' * int(pct * 30):<30}] {pct:4.0%}", end="", file=sys.stderr)
            if done >= total:
                print(file=sys.stderr)

    return show


def _collect(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            raise SystemExit(f"error: no such file or directory: {p}")
    return files


def _output_path(src: Path, output: str | None, many: bool, fmt: str | None) -> Path:
    suffix = f".{fmt.lower()}" if fmt else src.suffix
    name = f"{src.stem}_enhanced{suffix}"
    if output is None:
        return src.with_name(name)
    out = Path(output)
    if many or out.is_dir() or not out.suffix:
        return out / name
    return out


def cmd_enhance(args: argparse.Namespace) -> int:
    opts = EnhanceOptions(
        model=args.model,
        scale=args.scale,
        denoise=args.denoise,
        white_balance=args.white_balance,
        contrast=args.contrast,
        saturation=args.saturation,
        sharpen=args.sharpen,
        face_restore=args.face_restore,
        tile=args.tile,
        device=args.device,
    )
    try:
        opts.validate()
    except ValueError as e:
        raise SystemExit(f"error: {e}") from None

    if opts.face_restore > 0 and not torch_available():
        raise SystemExit("error: face restoration needs PyTorch: pip install 'photo-enhancer[ai]'")
    if resolve_model(opts.model) != opts.model:
        print(
            "warning: PyTorch not installed; using the classic (non-AI) upscaler.\n"
            "         Install AI support with: pip install 'photo-enhancer[ai]'",
            file=sys.stderr,
        )

    files = _collect(args.inputs)
    if not files:
        raise SystemExit("error: no images found")
    many = len(files) > 1
    for src in files:
        dst = _output_path(src, args.output, many, args.format)
        if dst.exists() and not args.overwrite:
            print(f"skip {src} -> {dst} (exists; use --overwrite)", file=sys.stderr)
            continue
        start = time.perf_counter()
        w, h = enhance_file(src, dst, opts, quality=args.quality, progress=_bar(src.name))
        print(f"{src} -> {dst} ({w}x{h}, {time.perf_counter() - start:.1f}s)")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    print(f"Model cache: {models.cache_dir()}")
    print(f"PyTorch available: {'yes' if torch_available() else 'no'}\n")
    for m in models.MODELS.values():
        mark = "downloaded" if models.is_downloaded(m.key) else "not downloaded"
        default = " (default)" if m.key == models.DEFAULT_MODEL else ""
        print(f"  {m.key:<15} {m.name}{default} [{mark}]\n  {'':<15} {m.description}")
    print(f"  {'classic':<15} Lanczos + sharpening, no AI (always available)")
    print("\nFace restoration (--face-restore):")
    for m in models.FACE_MODELS.values():
        mark = "downloaded" if models.is_downloaded(m.key) else "not downloaded"
        print(f"  {m.key:<15} {m.name} [{mark}]\n  {'':<15} {m.description}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    if args.all:
        keys = list(models.get_all_models())
    else:
        keys = args.models or [models.DEFAULT_MODEL]
        if "faces" in keys:
            keys = [k for k in keys if k != "faces"] + list(models.FACE_MODELS)
    for key in keys:
        try:
            info = models.get_model_info(key)
        except ValueError as e:
            raise SystemExit(f"error: {e}") from None
        if models.is_downloaded(key):
            print(f"{key}: already downloaded")
            continue

        def progress(done: int, total: int, key: str = key) -> None:
            if total:
                print(f"\r{key}: {done / total:4.0%} of {total / 1e6:.0f} MB", end="", file=sys.stderr)

        models.ensure_model(key, progress)
        print(f"\r{key}: saved to {models.model_path(key)} ({info.name})", file=sys.stderr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("error: the web UI needs: pip install 'photo-enhancer[web]'") from None
    from .server import app

    print(f"Photo Enhancer running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="photo-enhancer",
        description="Enhance and upscale photos locally with AI. Nothing leaves your machine.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("enhance", help="enhance one or more images")
    e.add_argument("inputs", nargs="+", help="image files or folders")
    e.add_argument("-o", "--output", help="output file, or folder for batches")
    e.add_argument(
        "-m",
        "--model",
        default=models.DEFAULT_MODEL,
        choices=[*models.MODELS, "classic", "none"],
        help=f"upscaling model (default: {models.DEFAULT_MODEL})",
    )
    e.add_argument("-s", "--scale", type=float, default=4.0, help="output scale (default: 4)")
    e.add_argument("--denoise", type=float, default=0.0, metavar="0-1")
    e.add_argument("--white-balance", type=float, default=0.0, metavar="0-1")
    e.add_argument("--contrast", type=float, default=0.0, metavar="0-1")
    e.add_argument("--saturation", type=float, default=0.0, metavar="-1-1")
    e.add_argument("--sharpen", type=float, default=0.0, metavar="0-1")
    e.add_argument(
        "--face-restore",
        type=float,
        default=0.0,
        metavar="0-1",
        help="restore faces with GFPGAN; the value blends restored and original faces (try 0.7)",
    )
    e.add_argument("--format", choices=["png", "jpg", "webp"], help="output format")
    e.add_argument("--quality", type=int, default=95, help="JPEG/WebP quality (default: 95)")
    e.add_argument("--tile", type=int, default=256, help="tile size; lower uses less memory, 0 disables")
    e.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, mps")
    e.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    e.set_defaults(func=cmd_enhance)

    m = sub.add_parser("models", help="list available models")
    m.set_defaults(func=cmd_models)

    d = sub.add_parser("download", help="pre-download model weights for offline use")
    d.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help=f"one of: {', '.join(models.get_all_models())}, or 'faces' for both face models",
    )
    d.add_argument("--all", action="store_true", help="download every model")
    d.set_defaults(func=cmd_download)

    s = sub.add_parser("serve", help="start the local web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
