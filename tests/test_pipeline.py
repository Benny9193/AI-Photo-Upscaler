import numpy as np
import pytest
from PIL import Image

from photo_enhancer import enhance
from photo_enhancer.pipeline import (
    EnhanceOptions,
    encode_image,
    enhance_array,
    enhance_file,
    load_image,
)
from photo_enhancer.upscale import iter_tiles


@pytest.mark.parametrize("h,w,tile,overlap", [(48, 64, 16, 4), (50, 33, 16, 8), (10, 10, 0, 4), (7, 300, 64, 16)])
def test_tile_cores_partition_image(h, w, tile, overlap):
    cover = np.zeros((h, w), dtype=int)
    for y0, y1, x0, x1, iy0, iy1, ix0, ix1 in iter_tiles(h, w, tile, overlap):
        assert y0 <= iy0 < iy1 <= y1 and x0 <= ix0 < ix1 <= x1
        cover[iy0:iy1, ix0:ix1] += 1
    assert (cover == 1).all()


@pytest.mark.parametrize("fn", [enhance.denoise, enhance.auto_contrast, enhance.white_balance, enhance.sharpen])
def test_zero_strength_is_identity(photo, fn):
    assert fn(photo, 0) is photo


def test_adjustments_keep_shape_and_dtype(photo):
    for out in (
        enhance.denoise(photo, 0.5),
        enhance.auto_contrast(photo, 0.5),
        enhance.white_balance(photo, 0.5),
        enhance.saturation(photo, 0.5),
        enhance.sharpen(photo, 0.5),
    ):
        assert out.shape == photo.shape and out.dtype == np.uint8


def test_white_balance_removes_cast(photo):
    tinted = np.clip(photo.astype(int) + [40, 0, -30], 0, 255).astype(np.uint8)
    means = enhance.white_balance(tinted, 1.0).reshape(-1, 3).mean(0)
    assert means.max() - means.min() < 3


@pytest.mark.parametrize("scale", [1, 2, 2.5, 4])
def test_classic_pipeline_output_size(photo, scale):
    opts = EnhanceOptions(model="classic", scale=scale, denoise=0.3, contrast=0.3, sharpen=0.3)
    out, alpha = enhance_array(photo, opts)
    assert out.shape == (round(48 * scale), round(64 * scale), 3)
    assert alpha is None


def test_alpha_channel_preserved(tmp_path, photo):
    alpha = np.full(photo.shape[:2], 128, dtype=np.uint8)
    src = tmp_path / "in.png"
    src.write_bytes(encode_image(photo, alpha, "PNG"))
    dst = tmp_path / "out.png"
    assert enhance_file(src, dst, EnhanceOptions(model="none", scale=2)) == (128, 96)
    with Image.open(dst) as im:
        assert im.mode == "RGBA" and im.size == (128, 96)
        assert np.asarray(im.getchannel("A")).mean() == pytest.approx(128, abs=1)


def test_exif_rotation_applied(tmp_path, photo):
    im = Image.fromarray(photo)
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90° CW on display
    path = tmp_path / "rot.jpg"
    im.save(path, exif=exif)
    rgb, _, _ = load_image(path)
    assert rgb.shape[:2] == (64, 48)


def test_jpeg_output_drops_alpha(photo):
    data = encode_image(photo, np.zeros(photo.shape[:2], np.uint8), "jpg")
    assert data[:2] == b"\xff\xd8"


@pytest.mark.parametrize(
    "kwargs", [{"scale": 0}, {"denoise": 2}, {"saturation": -3}, {"model": "nope"}]
)
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        EnhanceOptions(**kwargs).validate()
