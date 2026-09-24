"""Tests that exercise the PyTorch path.

The tiling test uses a tiny stand-in network and always runs when PyTorch is
installed. The real-model test runs only when the weights are already cached
(run ``photo-enhancer download`` first) to keep the default suite offline.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spandrel")

from spandrel import SizeRequirements  # noqa: E402

from photo_enhancer import models  # noqa: E402
from photo_enhancer.pipeline import EnhanceOptions, enhance_array  # noqa: E402
from photo_enhancer.upscale import AIUpscaler  # noqa: E402


class FakeNet:
    """Local 3x3 blur followed by 2x nearest upscaling (receptive field 1px)."""

    scale = 2
    size_requirements = SizeRequirements(minimum=4, multiple_of=4)

    def __call__(self, x):
        assert x.shape[-1] % 4 == 0 and x.shape[-2] % 4 == 0
        x = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


def fake_upscaler() -> AIUpscaler:
    import threading

    up = object.__new__(AIUpscaler)
    up._model, up._torch, up._lock = FakeNet(), torch, threading.Lock()
    up.device, up.half, up.scale, up.model_key = "cpu", False, 2, "fake"
    return up


def test_tiled_matches_untiled(photo):
    up = fake_upscaler()
    whole = up.upscale(photo, tile=0)
    calls = []
    tiled = up.upscale(photo, tile=13, overlap=4, progress=lambda d, t: calls.append((d, t)))
    assert whole.shape == (96, 128, 3)
    np.testing.assert_array_equal(tiled, whole)
    assert calls[-1][0] == calls[-1][1] == len(calls)


@pytest.mark.ai
@pytest.mark.skipif(not models.is_downloaded("general-x4"), reason="model weights not cached")
def test_real_model_upscales(photo):
    out, _ = enhance_array(photo, EnhanceOptions(model="general-x4", scale=4, tile=32, device="cpu"))
    assert out.shape == (192, 256, 3)
    # The result should resemble a plain resize of the input.
    import cv2

    ref = cv2.resize(photo, (256, 192), interpolation=cv2.INTER_CUBIC)
    assert np.abs(out.astype(int) - ref).mean() < 20
