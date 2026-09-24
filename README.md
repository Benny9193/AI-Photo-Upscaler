# AI Photo Upscaler

A photo enhancer that runs entirely on your own computer. It upscales photos up to 4× with Real-ESRGAN and can clean up noise, color casts and flat contrast. Your images are never uploaded anywhere. The only network access is a one-time download of the model weights.

- **AI super-resolution**: Real-ESRGAN models run locally with PyTorch. It uses a CUDA or Apple Silicon (MPS) GPU when one is available and falls back to the CPU.
- **Handles large images**: images are processed in overlapping tiles, so memory use stays bounded and no seams appear.
- **Face restoration**: GFPGAN v1.4 redraws blurry, low-resolution or damaged faces. A strength slider blends the result with the original face.
- **Old photo restoration**: finds scratches, creases and dust with Microsoft's *Bringing Old Photos Back to Life* detector and fills them in. DDColor colorizes black-and-white and sepia photos.
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

# Family photo: upscale and restore faces (0.7 keeps some of the original face)
photo-enhancer enhance family.jpg --face-restore 0.7

# Old family scan: remove scratches and dust, restore faces, 2x output
photo-enhancer enhance grandma.jpg -s 2 --scratch-removal 0.5 --face-restore 0.6

# Colorize a black-and-white photo
photo-enhancer enhance wedding-1952.jpg --colorize 1

# Same size, just cleaner (runs the AI model, then scales back down)
photo-enhancer enhance noisy.png -s 1

photo-enhancer models            # list models and whether they're downloaded
photo-enhancer download --all    # fetch weights ahead of time for offline use
photo-enhancer download faces    # just the face restoration models
photo-enhancer download old-photo  # scratch detector + DDColor
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

### Face restoration

| Key | Size | Role |
| --- | --- | --- |
| `gfpgan` | 350 MB | GFPGAN v1.4 face restoration network |
| `yunet` | 0.2 MB | OpenCV YuNet detector that finds faces and their eye, nose and mouth landmarks |

Both download automatically the first time you use `--face-restore` (or the **Faces** slider). Face restoration needs PyTorch. It takes about a second per face on CPU and much less on a GPU.

### Old photos

| Key | Size | Role |
| --- | --- | --- |
| `scratch-detector` | 150 MB | U-Net from [Bringing Old Photos Back to Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) that marks scratches, creases and dust |
| `ddcolor` (default) | 912 MB | [DDColor](https://github.com/piddnad/DDColor) colorization, best quality |
| `ddcolor-tiny` | 220 MB | Smaller, faster DDColor (`--colorize-model ddcolor-tiny`) |

**Scratch removal.** The detector only ships inside Microsoft's 2 GB checkpoint archive. The downloader reads just that one file out of the remote zip with HTTP range requests (about 420 MB), checks its SHA-256, drops the optimizer state it doesn't need, and stores 150 MB. The detector runs at two resolutions, on the photo and on its negative (it was mostly trained on light scratches), and OpenCV fills the marked pixels in from their surroundings. On synthetic damage it caught roughly half to 60% of scratch pixels and marked under 0.4% of undamaged photos. Long, faint or very dark scratches can survive. Raise `--scratch-removal` to catch more.

**Colorization.** DDColor predicts color from the photo's lightness at 512×512. Full-resolution lightness is kept, so no detail is lost. `--colorize` below 1 gives a softer, hand-tinted look. The weights come from Hugging Face, which publishes no checksum, so these files are checked by exact size instead.

**Black-and-white stays black-and-white.** GFPGAN paints skin tones on monochrome faces. When face restoration runs on a black-and-white or toned photo and colorization is off, the photo's original tint is put back afterwards.

Weights are downloaded from the official [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases) and [GFPGAN](https://github.com/TencentARC/GFPGAN/releases) releases, [OpenCV Zoo](https://github.com/opencv/opencv_zoo), Microsoft's Bringing Old Photos Back to Life release and Hugging Face (DDColor). Each file is checked against a pinned SHA-256 (DDColor: exact size) and cached in `~/.cache/photo-enhancer/models`. Set `PHOTO_ENHANCER_MODELS` to use a different directory.

## How it works

1. Decode the image and apply its EXIF rotation. Any alpha channel is split off.
2. Remove scratches and dust, if enabled. This comes before denoising, which would smear them.
3. Denoise, colorize (if enabled) and white-balance *before* upscaling, so the network doesn't amplify noise or color casts.
4. Run the super-resolution network tile by tile. Each tile gets extra surrounding context, and only its center is kept.
5. Resize to the exact requested scale.
6. If face restoration is on: faces are found on the original image, which is faster. Each face is then warped from the upscaled image onto the 512×512 template GFPGAN was trained on, restored, and blended back with a feathered mask.
7. Apply contrast, saturation and sharpening, then re-attach the alpha channel and encode with the original ICC profile.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests that need real weights are skipped unless those weights are already cached (`photo-enhancer download general-x4 faces scratch-detector`). DDColor is tested with a randomly initialised model, so no download is needed.
