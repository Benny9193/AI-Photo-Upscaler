"""Face restoration tests.

Alignment and blending are checked with an identity stand-in for GFPGAN, so
they run without downloading anything. Tests that need the real detector are
skipped unless its weights are cached (``photo-enhancer download faces``).
"""

import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spandrel")

from photo_enhancer import models  # noqa: E402
from photo_enhancer.faces import (  # noqa: E402
    FACE_SIZE,
    FFHQ_TEMPLATE,
    FaceRestorer,
    order_landmarks,
    soft_mask,
)
from photo_enhancer.pipeline import EnhanceOptions, enhance_array  # noqa: E402


def fake_restorer(fn=lambda t: t) -> FaceRestorer:
    r = object.__new__(FaceRestorer)
    r._model, r._torch, r._lock, r.device = fn, torch, threading.Lock(), "cpu"
    r._mask = soft_mask()
    return r


def landmarks_at(cx: float, cy: float, scale: float) -> np.ndarray:
    """Template landmarks placed in an image, centred on (cx, cy)."""
    return (FFHQ_TEMPLATE - FACE_SIZE / 2) * scale + [cx, cy]


def smooth_image(h=300, w=400) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([x * 255 // w, y * 255 // h, (x + y) * 255 // (w + h)], -1).astype(np.uint8)


def test_order_landmarks_sorts_left_to_right():
    pts = FFHQ_TEMPLATE[[1, 0, 2, 4, 3]]
    np.testing.assert_allclose(order_landmarks(pts), FFHQ_TEMPLATE)


def test_soft_mask_is_feathered():
    m = soft_mask()
    assert m[FACE_SIZE // 2, FACE_SIZE // 2] == pytest.approx(1, abs=1e-3)
    assert m[0, 0] < 0.01 and 0 < m[FACE_SIZE // 2, FACE_SIZE // 16] < 1


@pytest.mark.parametrize("scale", [0.25, 1.0, 1.5])
def test_identity_restore_roundtrips(scale):
    img = smooth_image()
    out = fake_restorer().restore(img, [landmarks_at(200, 150, scale)])
    assert out.shape == img.shape
    assert np.abs(out.astype(int) - img).mean() < 1.5


def test_restore_only_changes_face_region():
    img = smooth_image()
    white = fake_restorer(lambda t: torch.ones_like(t)).restore(img, [landmarks_at(200, 150, 0.2)])
    changed = np.abs(white.astype(int) - img).sum(-1) > 0
    ys, xs = np.nonzero(changed)
    assert changed[150, 200] and white[150, 200].min() > 250
    # A 0.2-scale face covers about 102 px; nothing far outside it may change.
    assert xs.min() > 130 and xs.max() < 270 and ys.min() > 80 and ys.max() < 220


def test_strength_blends_with_original():
    img = smooth_image()
    face = [landmarks_at(200, 150, 0.5)]
    full = fake_restorer(lambda t: torch.ones_like(t)).restore(img, face, strength=1.0)
    half = fake_restorer(lambda t: torch.ones_like(t)).restore(img, face, strength=0.5)
    c = (150, 200)
    assert int(img[c].mean()) < int(half[c].mean()) < int(full[c].mean())


def test_face_off_the_edge_is_clipped():
    img = smooth_image()
    out = fake_restorer(lambda t: torch.ones_like(t)).restore(img, [landmarks_at(5, 5, 0.5)])
    assert out.shape == img.shape


@pytest.mark.skipif(not models.is_downloaded("yunet"), reason="face detector not cached")
def test_no_faces_leaves_image_alone(photo):
    from photo_enhancer.faces import detect_faces

    assert detect_faces(photo) == []
    plain, _ = enhance_array(photo, EnhanceOptions(model="none", scale=2))
    with_faces, _ = enhance_array(photo, EnhanceOptions(model="none", scale=2, face_restore=1))
    np.testing.assert_array_equal(plain, with_faces)
