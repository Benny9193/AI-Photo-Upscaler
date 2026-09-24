"""Face restoration with GFPGAN.

Faces are found with OpenCV's YuNet detector, which also returns five
landmarks (eyes, nose, mouth corners). Each face is warped onto the FFHQ
512x512 template GFPGAN was trained on, restored, and pasted back through the
inverse transform with a feathered mask so the edges blend in.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from . import models
from .upscale import ProgressFn, pick_device

FACE_SIZE = 512
# Eye centres, nose tip and mouth corners of an aligned FFHQ face at 512x512.
FFHQ_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)
# Detection runs on a copy no larger than this; YuNet still finds small faces.
DETECT_MAX_SIDE = 1600
MIN_FACE_PX = 12


def order_landmarks(points: np.ndarray) -> np.ndarray:
    """Sort a 5x2 landmark array into template order (image-left before image-right)."""
    eyes = points[:2][np.argsort(points[:2, 0])]
    mouth = points[3:5][np.argsort(points[3:5, 0])]
    return np.vstack([eyes, points[2:3], mouth]).astype(np.float32)


def detect_faces(rgb: np.ndarray, threshold: float = 0.7) -> list[np.ndarray]:
    """Return a 5x2 landmark array (in ``rgb`` pixel coordinates) per face."""
    h, w = rgb.shape[:2]
    ratio = min(1.0, DETECT_MAX_SIDE / max(h, w))
    small = rgb if ratio == 1 else cv2.resize(rgb, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(models.ensure_model("yunet")), "", (sw, sh), threshold)
    _, found = detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    faces = []
    for row in found if found is not None else []:
        if min(row[2], row[3]) / ratio < MIN_FACE_PX:
            continue
        faces.append(order_landmarks(row[4:14].reshape(5, 2) / ratio))
    return faces


def soft_mask(size: int = FACE_SIZE) -> np.ndarray:
    """A float mask for the aligned face crop that fades to zero at the edges."""
    mask = np.zeros((size, size), np.float32)
    border = size // 16
    mask[border:-border, border:-border] = 1
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=border / 2)


class FaceRestorer:
    def __init__(self, device: str = "auto"):
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader

        self.device = pick_device(device)
        descriptor = ModelLoader().load_from_file(models.ensure_model("gfpgan"))
        if not isinstance(descriptor, ImageModelDescriptor):
            raise TypeError("GFPGAN weights did not load as an image model")
        descriptor.to(self.device).eval()
        self._model = descriptor
        self._torch = torch
        self._lock = threading.Lock()
        self._mask = soft_mask()

    def _restore_crop(self, crop: np.ndarray) -> np.ndarray:
        torch = self._torch
        t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(self.device, torch.float32) / 255
        with torch.inference_mode():
            out = self._model(t).squeeze(0).clamp_(0, 1).mul_(255).round_().byte()
        return out.permute(1, 2, 0).cpu().numpy()

    def restore(
        self,
        img: np.ndarray,
        faces: list[np.ndarray],
        strength: float = 1.0,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        """Restore ``faces`` (landmarks in ``img`` coordinates) and blend them in.

        ``strength`` mixes the restored face with the original: 1 is fully
        restored, lower values keep more of the source identity and texture.
        """
        out = img.copy()
        h, w = img.shape[:2]
        with self._lock:
            for i, landmarks in enumerate(faces):
                affine, _ = cv2.estimateAffinePartial2D(landmarks, FFHQ_TEMPLATE, method=cv2.LMEDS)
                if affine is None:
                    continue
                crop = cv2.warpAffine(
                    img, affine, (FACE_SIZE, FACE_SIZE),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132),
                )
                restored = self._restore_crop(crop)
                if strength < 1:
                    restored = cv2.addWeighted(restored, strength, crop, 1 - strength, 0)
                self._paste(out, restored, cv2.invertAffineTransform(affine), w, h)
                if progress:
                    progress(i + 1, len(faces))
        return out

    def _paste(self, out: np.ndarray, face: np.ndarray, inverse: np.ndarray, w: int, h: int) -> None:
        # Only warp into the face's bounding box instead of the whole image.
        corners = np.array([[0, 0, 1], [FACE_SIZE, 0, 1], [0, FACE_SIZE, 1], [FACE_SIZE, FACE_SIZE, 1]], np.float32)
        pts = corners @ inverse.T
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            return
        mask = self._mask
        # Pre-shrink faces that land smaller than 512px so the warp doesn't alias.
        scale = float(np.sqrt(abs(np.linalg.det(inverse[:, :2]))))
        if scale < 1:
            side = max(1, round(FACE_SIZE * scale))
            face = cv2.resize(face, (side, side), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (side, side), interpolation=cv2.INTER_AREA)
            inverse = inverse.copy()
            inverse[:, :2] *= FACE_SIZE / side
        shifted = inverse.copy()
        shifted[:, 2] -= (x0, y0)
        size = (x1 - x0, y1 - y0)
        face_w = cv2.warpAffine(face, shifted, size, flags=cv2.INTER_LINEAR)
        mask_w = cv2.warpAffine(mask, shifted, size, flags=cv2.INTER_LINEAR)[..., None]
        region = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = (region + (face_w.astype(np.float32) - region) * mask_w).round().astype(np.uint8)


_restorers: dict[str, FaceRestorer] = {}
_restorers_lock = threading.Lock()


def get_restorer(device: str = "auto") -> FaceRestorer:
    with _restorers_lock:
        if device not in _restorers:
            _restorers[device] = FaceRestorer(device)
        return _restorers[device]
