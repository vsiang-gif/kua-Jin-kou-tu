"""Real-ESRGAN (ncnn-vulkan) upscaling with a Lanczos fallback."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..paths import realesrgan_exe

MODEL_NAME = "realesrgan-x4plus"
# ESRGAN input is capped so the 4x output is not wastefully larger than needed.
MAX_INPUT_RATIO = 1 / 2.2


def _lanczos(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = rgb.shape[:2]
    if size[0] <= w and size[1] <= h:
        return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    out = cv2.resize(rgb, size, interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.35, blur, -0.35, 0)


class Upscaler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.exe: Path | None = realesrgan_exe()
        self.broken = False

    @property
    def available(self) -> bool:
        return self.exe is not None and not self.broken

    def _run_esrgan(self, rgb: np.ndarray) -> np.ndarray | None:
        assert self.exe is not None
        tmp = Path(tempfile.mkdtemp(prefix="printcut_"))
        try:
            src, dst = tmp / "in.png", tmp / "out.png"
            Image.fromarray(rgb).save(src, compress_level=1)
            cmd = [
                str(self.exe), "-i", str(src), "-o", str(dst),
                "-n", MODEL_NAME, "-s", "4", "-f", "png",
                "-m", str(self.exe.parent / "models"),
            ]
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            with self._lock:
                proc = subprocess.run(
                    cmd, cwd=str(self.exe.parent), capture_output=True,
                    timeout=600, creationflags=flags,
                )
            if proc.returncode != 0 or not dst.is_file():
                return None
            with Image.open(dst) as im:
                return np.asarray(im.convert("RGB")).copy()
        except (OSError, subprocess.SubprocessError):
            self.broken = True
            return None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def upscale(self, rgb: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, bool]:
        """Resize rgb to `size` (w, h). Returns (image, used_ai)."""
        h, w = rgb.shape[:2]
        factor = max(size[0] / w, size[1] / h)
        if factor <= 1.5 or not self.available:
            return _lanczos(rgb, size), False

        max_input = int(max(size) * MAX_INPUT_RATIO)
        if max(h, w) > max_input:
            s = max_input / max(h, w)
            rgb = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)

        out = self._run_esrgan(rgb)
        if out is None:
            return _lanczos(rgb, size), False
        return _lanczos(out, size), True


_upscaler: Upscaler | None = None
_guard = threading.Lock()


def get_upscaler() -> Upscaler:
    global _upscaler
    with _guard:
        if _upscaler is None:
            _upscaler = Upscaler()
        return _upscaler
