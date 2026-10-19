"""BiRefNet foreground segmentation via onnxruntime.

Uses the same BiRefNet-general-lite ONNX model and preprocessing as rembg, but
calls onnxruntime directly so the packaged .exe does not pull in numba/pymatting.
"""
from __future__ import annotations

import threading
import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..paths import models_dir

MODEL_FILE = "birefnet-general-lite.onnx"
MODEL_URL = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
    "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
)
INPUT_SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ProgressFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


class Segmenter:
    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._session = None
        self._input_name = ""
        self.device = "未加载"

    def model_path(self) -> Path:
        return models_dir() / MODEL_FILE

    def ensure_model(self, log: ProgressFn = _noop) -> Path:
        path = self.model_path()
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
        log("首次运行：正在下载 BiRefNet 模型（约 220 MB）…")
        tmp = path.with_suffix(".part")
        last_pct = -1

        def hook(blocks: int, block_size: int, total: int) -> None:
            nonlocal last_pct
            if total > 0:
                pct = min(100, blocks * block_size * 100 // total)
                if pct != last_pct and pct % 5 == 0:
                    last_pct = pct
                    log(f"正在下载模型… {pct}%")

        urllib.request.urlretrieve(MODEL_URL, tmp, hook)
        tmp.replace(path)
        return path

    def load(self, log: ProgressFn = _noop, force_cpu: bool = False) -> None:
        with self._load_lock:
            if self._session is not None and not force_cpu:
                return
            import onnxruntime as ort

            path = self.ensure_model(log)
            available = ort.get_available_providers()
            attempts: list[list[str]] = []
            if not force_cpu:
                if "DmlExecutionProvider" in available:
                    attempts.append(["DmlExecutionProvider", "CPUExecutionProvider"])
                if "CUDAExecutionProvider" in available:
                    attempts.append(["CUDAExecutionProvider", "CPUExecutionProvider"])
            attempts.append(["CPUExecutionProvider"])

            last_error: Exception | None = None
            session = None
            for providers in attempts:
                try:
                    opts = ort.SessionOptions()
                    if providers[0] == "DmlExecutionProvider":
                        # DirectML requires these two settings.
                        opts.enable_mem_pattern = False
                        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
                    break
                except Exception as exc:  # provider failed, try the next one
                    last_error = exc
            if session is None:
                raise RuntimeError(f"AI 模型加载失败：{last_error}")

            active = session.get_providers()[0]
            self._session = session
            self._input_name = session.get_inputs()[0].name
            self.device = {
                "DmlExecutionProvider": "显卡加速（DirectML）",
                "CUDAExecutionProvider": "显卡加速（CUDA）",
            }.get(active, "CPU（较慢）")
        # Logging must never be able to break model loading (e.g. console encoding errors).
        try:
            log(f"AI 模型已加载：{self.device}")
        except Exception:
            pass

    def _prepare(self, rgb: np.ndarray) -> np.ndarray:
        im = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        im /= max(float(im.max()), 1e-6)
        im = (im - MEAN) / STD
        return np.ascontiguousarray(im.transpose(2, 0, 1)[None], dtype=np.float32)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Returns a float32 foreground mask in 0..1 with the same HxW as rgb."""
        if self._session is None:
            self.load()
        tensor = self._prepare(rgb)
        with self._run_lock:
            try:
                out = self._session.run(None, {self._input_name: tensor})[0]
            except Exception:
                if self.device.startswith("CPU"):
                    raise
                self.load(force_cpu=True)
                out = self._session.run(None, {self._input_name: tensor})[0]
        pred = 1.0 / (1.0 + np.exp(-out[0, 0].astype(np.float32)))
        lo, hi = float(pred.min()), float(pred.max())
        pred = (pred - lo) / max(hi - lo, 1e-6)
        h, w = rgb.shape[:2]
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)


_segmenter: Segmenter | None = None
_lock = threading.Lock()


def get_segmenter() -> Segmenter:
    global _segmenter
    with _lock:
        if _segmenter is None:
            _segmenter = Segmenter()
        return _segmenter
