# AI Photo Upscaler

A photo enhancer that runs entirely on your own computer. It upscales photos up to 4× with Real-ESRGAN and can clean up noise, color casts and flat contrast. Your images are never uploaded anywhere. The only network access is a one-time download of the model weights.

- **AI super-resolution**: Real-ESRGAN models run locally with PyTorch. It uses a CUDA or Apple Silicon (MPS) GPU when one is available and falls back to the CPU.
- **Handles large images**: images are processed in overlapping tiles, so memory use stays bounded and no seams appear.
- **Adjustments**: denoise, white balance, local contrast (CLAHE), saturation and sharpening, with presets.
- **Two ways to use it**: a browser UI with a before/after slider, or a CLI that can batch-process whole folders.
- **Keeps what matters**: transparency, EXIF orientation and ICC color profiles are preserved.
- **Works without PyTorch**: a classic Lanczos upscaler is used when PyTorch isn't installed.

## Install

Requires Python 3.10+.

```bash
pip install -e ".[all]"        # AI models + web UI
# or: pip install -e ".[web]"  # lightweight, classic upscaler only
```

For an NVIDIA GPU, install the matching CUDA build of PyTorch first (see https://pytorch.org/get-started/locally/).

## Web UI

```bash
photo-enhancer serve
```

Open http://127.0.0.1:7860. You can drop, choose or paste a photo, then pick a preset or adjust the sliders and click **Enhance**. Drag across the result to compare it with the original. The server only listens on localhost by default.

## Command line

```bash
# 4x upscale with the default model
photo-enhancer enhance photo.jpg

# Restore an old scan: denoise, fix color cast, add contrast, 2x output
photo-enhancer enhance scan.jpg -s 2 --denoise 0.4 --white-balance 0.6 --contrast 0.5

# Batch a folder with the high-quality model and write WebP files
photo-enhancer enhance ./holiday -o ./holiday-4x -m realesrgan-x4 --format webp

# Same size, just cleaner (runs the AI model, then scales back down)
photo-enhancer enhance noisy.png -s 1

photo-enhancer models            # list models and whether they're downloaded
photo-enhancer download --all    # fetch weights ahead of time for offline use
```

If you run out of GPU or CPU memory on huge images, lower `--tile` (for example `--tile 128`).

## Models

| Key | Scale | Size | Best for |
| --- | --- | --- | --- |
| `general-x4` (default) | 4× | 5 MB | Everyday photos. Fast, even on CPU |
| `realesrgan-x4` | 4× | 67 MB | Best quality on real-world photos. A GPU is recommended |
| `realesrgan-x2` | 2× | 67 MB | Native 2× upscaling |
| `anime-x4` | 4× | 18 MB | Illustrations, anime, line art |
| `classic` | any | none | No AI. Lanczos resampling plus sharpening |

Weights are downloaded from the official [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases), checked against a pinned SHA-256 and cached in `~/.cache/photo-enhancer/models`. Set `PHOTO_ENHANCER_MODELS` to use a different directory.

## How it works

1. Decode the image and apply its EXIF rotation. Any alpha channel is split off.
2. Denoise and white-balance *before* upscaling, so the network doesn't amplify noise or color casts.
3. Run the super-resolution network tile by tile. Each tile gets extra surrounding context, and only its center is kept.
4. Resize to the exact requested scale.
5. Apply contrast, saturation and sharpening, then re-attach the alpha channel and encode with the original ICC profile.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test for the real model is skipped unless the `general-x4` weights are already cached (`photo-enhancer download`).
