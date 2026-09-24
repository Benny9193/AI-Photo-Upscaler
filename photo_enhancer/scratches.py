"""Scratch and dust removal for old photos.

A U-Net from Microsoft's "Bringing Old Photos Back to Life" (Wan et al., CVPR
2020, MIT licence) predicts which pixels are scratches, creases or dust. Those
pixels are then filled in from their surroundings with OpenCV inpainting,
which works well for the thin defects the detector finds and needs no extra
model.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from . import models
from .upscale import pick_device

# The detector runs on grayscale copies whose short side is each of these
# sizes, on both the image and its negative, and the results are combined.
# Microsoft's tool uses a single 256px pass. On synthetic damage these four
# passes found about twice as many scratch pixels (roughly half to 60% in all),
# while marking under 0.4% of an undamaged photo.
DETECT_SIDES = (384, 640)


def build_unet():
    """The detection network, laid out so the published weights load strictly."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Downsample(nn.Module):
        """Anti-aliased (blur, then stride 2) downsampling."""

        def __init__(self, channels: int):
            super().__init__()
            a = torch.tensor([1.0, 2.0, 1.0])
            filt = a[:, None] * a[None, :]
            self.register_buffer("filt", (filt / filt.sum())[None, None].repeat(channels, 1, 1, 1))
            self.pad = nn.ReflectionPad2d(1)

        def forward(self, x):
            return F.conv2d(self.pad(x), self.filt, stride=2, groups=x.shape[1])

    def conv_block(conv_num: int, in_size: int, out_size: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for _ in range(conv_num):
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_size, out_size, kernel_size=3),
                nn.BatchNorm2d(out_size),
                nn.LeakyReLU(0.2, True),
            ]
            in_size = out_size
        return nn.Sequential(*layers)

    class ConvBlock(nn.Module):
        def __init__(self, conv_num: int, in_size: int, out_size: int):
            super().__init__()
            self.block = conv_block(conv_num, in_size, out_size)

        def forward(self, x):
            return self.block(x)

    class UpBlock(nn.Module):
        def __init__(self, conv_num: int, in_size: int, out_size: int):
            super().__init__()
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_size, out_size, kernel_size=3),
            )
            self.conv_block = ConvBlock(conv_num, in_size, out_size)

        def forward(self, x, bridge):
            up = self.up(x)
            dy = (bridge.shape[2] - up.shape[2]) // 2
            dx = (bridge.shape[3] - up.shape[3]) // 2
            bridge = bridge[:, :, dy : dy + up.shape[2], dx : dx + up.shape[3]]
            return self.conv_block(torch.cat([up, bridge], 1))

    class UNet(nn.Module):
        def __init__(self, depth: int = 4, conv_num: int = 2, wf: int = 6):
            super().__init__()
            self.first = nn.Sequential(
                nn.ReflectionPad2d(3), nn.Conv2d(1, 2**wf, kernel_size=7), nn.LeakyReLU(0.2, True)
            )
            prev = 2**wf
            self.down_sample = nn.ModuleList()
            self.down_path = nn.ModuleList()
            for i in range(depth):
                self.down_sample.append(
                    nn.Sequential(
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(prev, prev, kernel_size=3),
                        nn.BatchNorm2d(prev),
                        nn.LeakyReLU(0.2, True),
                        Downsample(prev),
                    )
                )
                self.down_path.append(ConvBlock(conv_num, prev, 2 ** (wf + i + 1)))
                prev = 2 ** (wf + i + 1)
            self.up_path = nn.ModuleList()
            for i in reversed(range(depth)):
                self.up_path.append(UpBlock(conv_num, prev, 2 ** (wf + i)))
                prev = 2 ** (wf + i)
            self.last = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(prev, 1, kernel_size=3))

        def forward(self, x):
            x = self.first(x)
            skips = []
            for down, block in zip(self.down_sample, self.down_path):
                skips.append(x)
                x = block(down(x))
            for i, up in enumerate(self.up_path):
                x = up(x, skips[-i - 1])
            return self.last(x)

    return UNet()


class ScratchRemover:
    def __init__(self, device: str = "auto"):
        import torch

        self.device = pick_device(device)
        state = torch.load(models.ensure_model("scratch-detector"), map_location="cpu", weights_only=True)
        net = build_unet()
        net.load_state_dict(state)
        self._net = net.to(self.device).eval()
        self._torch = torch
        self._lock = threading.Lock()

    def _probability(self, gray: np.ndarray, side: int) -> np.ndarray:
        torch = self._torch
        h, w = gray.shape
        ratio = side / min(h, w)
        dh = max(16, round(h * ratio / 16) * 16)
        dw = max(16, round(w * ratio / 16) * 16)
        small = cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC)
        t = torch.from_numpy(small).to(self.device, torch.float32)[None, None] / 127.5 - 1
        with torch.inference_mode():
            prob = torch.sigmoid(self._net(t))[0, 0].cpu().numpy()
        return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

    def detect(self, rgb: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
        """Return a uint8 mask (255 = damaged) the same size as ``rgb``.

        ``sensitivity`` 0..1 maps to the detector threshold; 0.5 matches the
        original project's threshold of 0.4.
        """
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # The network mostly learned light scratches; its view of the negative
        # catches dark ones.
        with self._lock:
            prob = np.max(
                [self._probability(g, side) for side in DETECT_SIDES for g in (gray, 255 - gray)],
                axis=0,
            )
        threshold = 0.6 - 0.4 * sensitivity
        mask = (prob >= threshold).astype(np.uint8) * 255
        # Grow the mask slightly so scratch edges get filled too.
        grow = max(1, round(min(h, w) / 600))
        return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))

    def remove(self, rgb: np.ndarray, sensitivity: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Detect and inpaint scratches. Returns (repaired image, mask)."""
        mask = self.detect(rgb, sensitivity)
        if not mask.any():
            return rgb, mask
        radius = max(3, round(min(rgb.shape[:2]) / 250))
        return cv2.inpaint(rgb, mask, radius, cv2.INPAINT_TELEA), mask


_removers: dict[str, ScratchRemover] = {}
_removers_lock = threading.Lock()


def get_scratch_remover(device: str = "auto") -> ScratchRemover:
    with _removers_lock:
        if device not in _removers:
            _removers[device] = ScratchRemover(device)
        return _removers[device]
