from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..config import BG_WHITE

ALPHA_SIGMA = [0.0, 0.4, 0.8, 1.2, 1.8, 2.6]
ALPHA_GAIN = [1.0, 2.4, 2.0, 1.7, 1.45, 1.25]


def content_size(src_w: int, src_h: int, canvas_w: int, canvas_h: int, margin: float) -> tuple[int, int]:
    avail_w = canvas_w * (1 - 2 * margin)
    avail_h = canvas_h * (1 - 2 * margin)
    s = min(avail_w / src_w, avail_h / src_h)
    return max(1, int(round(src_w * s))), max(1, int(round(src_h * s)))


def smooth_alpha(alpha: np.ndarray, level: int, px_scale: float) -> np.ndarray:
    level = int(np.clip(level, 0, 5))
    sigma = ALPHA_SIGMA[level] * max(px_scale, 1.0)
    if sigma <= 0:
        return alpha
    a = cv2.GaussianBlur(alpha, (0, 0), sigma)
    return np.clip((a - 0.5) * ALPHA_GAIN[level] + 0.5, 0, 1).astype(np.float32)


def merge_rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
    return np.dstack([rgb, a])


def compose_canvas(rgba: np.ndarray, canvas_w: int, canvas_h: int, background: str) -> Image.Image:
    h, w = rgba.shape[:2]
    if w > canvas_w or h > canvas_h:
        s = min(canvas_w / w, canvas_h / h)
        rgba = cv2.resize(rgba, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = rgba.shape[:2]
    canvas = np.zeros((canvas_h, canvas_w, 4), np.uint8)
    x, y = (canvas_w - w) // 2, (canvas_h - h) // 2
    canvas[y : y + h, x : x + w] = rgba
    if background == BG_WHITE:
        a = canvas[..., 3:4].astype(np.float32) / 255.0
        rgb = canvas[..., :3].astype(np.float32) * a + 255.0 * (1 - a)
        return Image.fromarray((rgb + 0.5).astype(np.uint8), "RGB")
    canvas[canvas[..., 3] == 0, :3] = 0
    return Image.fromarray(canvas, "RGBA")


def save_png(img: Image.Image, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    img.save(tmp, "PNG", dpi=(dpi, dpi), compress_level=3)
    os.replace(tmp, path)
