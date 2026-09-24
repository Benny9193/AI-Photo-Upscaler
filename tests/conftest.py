import numpy as np
import pytest


@pytest.fixture
def photo() -> np.ndarray:
    """A small synthetic 'photo': gradients, edges and some noise."""
    rng = np.random.default_rng(0)
    h, w = 48, 64
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 4, y * 5, (x + y) * 2], axis=-1).astype(np.float32)
    img[10:30, 20:40] = [220, 40, 60]
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
