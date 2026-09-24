"""Old photo restoration: scratch removal, colorization and zip-member downloads.

Networks are replaced by stand-ins (or randomly initialised copies) so these
run offline; the real weights are exercised only when already cached.
"""

import hashlib
import http.server
import io
import threading
import zipfile

import cv2
import numpy as np
import pytest

from photo_enhancer import models
from photo_enhancer.colorize import apply_tone, is_grayscale, tone_curve
from photo_enhancer.pipeline import EnhanceOptions


def sepia(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.stack([g, g * 0.9, g * 0.72], -1).clip(0, 255).astype(np.uint8)


# --- tone -------------------------------------------------------------------


def test_grayscale_detection(photo):
    gray = np.dstack([cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)] * 3)
    assert is_grayscale(gray)
    assert is_grayscale(sepia(photo))
    assert not is_grayscale(photo)


def test_apply_tone_restores_sepia(photo):
    toned = sepia(photo)
    table, _ = tone_curve(toned)
    tinted = toned.copy()
    tinted[10:30, 10:30] = [200, 120, 120]  # e.g. a pink face from GFPGAN
    out = apply_tone(tinted, table)
    assert tone_curve(out)[1] < 1.0
    assert np.abs(out[:5, :5].astype(int) - toned[:5, :5]).max() <= 3


# --- options ------------------------------------------------------------------


def test_needs_torch_lists_enabled_features():
    opts = EnhanceOptions(scratch_removal=0.5, colorize=1)
    assert opts.needs_torch() == ["scratch removal", "colorization"]
    assert EnhanceOptions().needs_torch() == []


def test_bad_colorize_model_rejected():
    with pytest.raises(ValueError):
        EnhanceOptions(colorize=1, colorize_model="scratch-detector").validate()


# --- zip member download -------------------------------------------------------


class RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()

    def do_GET(self):
        start, end = 0, len(self.payload) - 1
        if rng := self.headers.get("Range"):
            start, end = (int(x) for x in rng.removeprefix("bytes=").split("-"))
        body = self.payload[start : end + 1]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def zip_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def serve_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    RangeHandler.payload = buf.getvalue()
    return RangeHandler.payload


def test_zip_member_is_extracted_verified_and_slimmed(zip_server, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    ckpt = io.BytesIO()
    torch.save({"model_state": {"w": torch.arange(4.0)}, "optimizer": {"big": torch.zeros(1000)}}, ckpt)
    member = ckpt.getvalue()
    serve_zip({"junk/a.bin": b"x" * 100_000, "checkpoints/det.pt": member})

    info = models.ModelInfo(
        key="test-member",
        name="test",
        scale=1,
        url=f"http://127.0.0.1:{zip_server.server_port}/archive.zip",
        member="checkpoints/det.pt",
        sha256=hashlib.sha256(member).hexdigest(),
        keep="model_state",
        file="det.pth",
        description="",
    )
    monkeypatch.setitem(models.OLD_PHOTO_MODELS, info.key, info)
    monkeypatch.setenv("PHOTO_ENHANCER_MODELS", str(tmp_path))
    seen = []
    path = models.ensure_model(info.key, lambda d, t: seen.append((d, t)))
    assert path == tmp_path / "det.pth"
    state = torch.load(path, weights_only=True)
    assert list(state) == ["w"] and state["w"].tolist() == [0, 1, 2, 3]
    assert seen[-1] == (len(member), len(member))
    assert not list(tmp_path.glob("*.part"))


def test_zip_member_checksum_mismatch_leaves_nothing(zip_server, tmp_path, monkeypatch):
    serve_zip({"m.bin": b"hello"})
    info = models.ModelInfo(
        key="test-bad", name="t", scale=1, url=f"http://127.0.0.1:{zip_server.server_port}/a.zip",
        member="m.bin", sha256="0" * 64, description="",
    )
    monkeypatch.setitem(models.OLD_PHOTO_MODELS, info.key, info)
    monkeypatch.setenv("PHOTO_ENHANCER_MODELS", str(tmp_path))
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        models.ensure_model(info.key)
    assert not list(tmp_path.iterdir())


def test_size_is_checked_when_no_checksum(zip_server, tmp_path, monkeypatch):
    RangeHandler.payload = b"abcdef"
    info = models.ModelInfo(
        key="test-size", name="t", scale=1, url=f"http://127.0.0.1:{zip_server.server_port}/f.pth",
        sha256=None, size=5, description="",
    )
    monkeypatch.setitem(models.OLD_PHOTO_MODELS, info.key, info)
    monkeypatch.setenv("PHOTO_ENHANCER_MODELS", str(tmp_path))
    with pytest.raises(RuntimeError, match="Size mismatch"):
        models.ensure_model(info.key)


# --- scratch removal -------------------------------------------------------------

def test_unet_matches_published_checkpoint_layout():
    torch = pytest.importorskip("torch")
    from photo_enhancer.scratches import build_unet

    state = build_unet().state_dict()
    # Keys and shapes taken from FT_Epoch_latest.pt in Microsoft's release.
    expected = {
        "first.1.weight": (64, 1, 7, 7),
        "down_path.0.block.1.weight": (128, 64, 3, 3),
        "down_path.0.block.6.running_var": (128,),
        "down_sample.0.4.filt": (64, 1, 3, 3),
        "up_path.0.up.2.weight": (512, 1024, 3, 3),
        "up_path.3.conv_block.block.6.num_batches_tracked": (),
        "last.1.weight": (1, 64, 3, 3),
    }
    assert len(state) == 156
    for key, shape in expected.items():
        assert tuple(state[key].shape) == shape, key
    assert state["down_sample.0.4.filt"][0, 0].sum().item() == pytest.approx(1)
    out = build_unet().eval()(torch.zeros(1, 1, 64, 80))
    assert out.shape == (1, 1, 64, 80)


def test_scratch_removal_fills_detected_line(photo):
    torch = pytest.importorskip("torch")
    from photo_enhancer.scratches import ScratchRemover

    clean = cv2.resize(photo, (320, 240), interpolation=cv2.INTER_CUBIC)
    damaged = clean.copy()
    cv2.line(damaged, (0, 120), (319, 120), (255, 255, 255), 2)

    class LineNet(torch.nn.Module):
        """Pretends to find exactly the bright pixels, like a perfect detector."""

        def forward(self, x):
            return (x > 0.98).float() * 20 - 10

    remover = object.__new__(ScratchRemover)
    remover._net, remover._torch, remover.device, remover._lock = LineNet(), torch, "cpu", threading.Lock()
    fixed, mask = remover.remove(damaged)
    assert mask[120].all() and mask[60].sum() == 0
    err = lambda img: np.abs(img.astype(int) - clean).mean()  # noqa: E731
    assert err(fixed) < err(damaged) / 3


@pytest.mark.skipif(not models.is_downloaded("scratch-detector"), reason="scratch detector not cached")
def test_real_detector_leaves_clean_photo_alone(photo):
    from photo_enhancer.scratches import get_scratch_remover

    mask = get_scratch_remover().detect(cv2.resize(photo, (320, 240)))
    assert (mask > 0).mean() < 0.01


# --- colorization ------------------------------------------------------------------


def test_colorizer_runs_ddcolor(tmp_path, monkeypatch, photo):
    """Plumbing check with a randomly initialised DDColor-tiny (no download)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("spandrel_extra_arches")
    from spandrel_extra_arches.architectures.DDColor import DDColor

    from photo_enhancer.colorize import Colorizer

    torch.manual_seed(0)
    net = DDColor(encoder_name="convnext-t", num_output_channels=2, last_norm="Spectral", num_queries=100)
    torch.save(net.state_dict(), tmp_path / "ddcolor_paper_tiny.pth")
    monkeypatch.setenv("PHOTO_ENHANCER_MODELS", str(tmp_path))

    colorizer = Colorizer("ddcolor-tiny", device="cpu")
    assert colorizer._model.model.input_size == (512, 512)
    gray = np.dstack([cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)] * 3)
    out = colorizer.colorize(gray, 1.0)
    assert out.shape == gray.shape and out.dtype == np.uint8
    np.testing.assert_allclose(colorizer.colorize(gray, 0.0), gray, atol=2)
