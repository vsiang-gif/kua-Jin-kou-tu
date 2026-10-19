"""Human parsing (SegFormer-B2 clothes, MIT) to keep only the garment on model photos.

BiRefNet returns the whole person on model photos, so arms, tattoos, face and jeans
end up inside the "shirt". A parsing model labels each pixel as upper clothes, arm,
face, pants, etc., which separates the garment from the body regardless of colour.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from ..paths import models_dir
from .segment import ProgressFn, _noop, download_model, open_session

MODEL_FILE = "segformer-b2-clothes.onnx"
MODEL_URL = "https://huggingface.co/Xenova/segformer_b2_clothes/resolve/main/onnx/model.onnx"
INPUT_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

UPPER_CLOTHES, DRESS = 4, 7
HAIR, FACE, LEFT_LEG, RIGHT_LEG, LEFT_ARM, RIGHT_ARM = 2, 11, 12, 13, 14, 15
PANTS, SKIRT = 6, 5
BODY = (HAIR, FACE, LEFT_LEG, RIGHT_LEG, LEFT_ARM, RIGHT_ARM)


class HumanParser:
    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._session = None
        self._input_name = ""
        self.device = "未加载"
        self.error = ""

    def model_path(self) -> Path:
        return models_dir() / MODEL_FILE

    def load(self, log: ProgressFn = _noop, force_cpu: bool = False) -> bool:
        """Optional model: on failure (offline, etc.) the app keeps working without it."""
        with self._load_lock:
            if self._session is not None and not force_cpu:
                return True
            try:
                path = download_model(self.model_path(), MODEL_URL, "人体解析模型（约 110 MB）", log)
                self._session, self.device = open_session(path, force_cpu)
                self._input_name = self._session.get_inputs()[0].name
                self.error = ""
            except Exception as exc:
                self.error = str(exc)
                try:
                    log(f"人体解析模型不可用，模特图可能带入手臂：{exc}")
                except Exception:
                    pass
                return False
        try:
            log(f"人体解析模型已加载：{self.device}")
        except Exception:
            pass
        return True

    @property
    def available(self) -> bool:
        return self._session is not None

    def predict(self, rgb: np.ndarray) -> np.ndarray | None:
        """Per-pixel label map (uint8, same HxW as rgb), or None if the model is unavailable."""
        if self._session is None and not self.load():
            return None
        im = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(((im - MEAN) / STD).transpose(2, 0, 1)[None], dtype=np.float32)
        with self._run_lock:
            try:
                logits = self._session.run(None, {self._input_name: tensor})[0][0]
            except Exception:
                if self.device.startswith("CPU"):
                    raise
                self.load(force_cpu=True)
                logits = self._session.run(None, {self._input_name: tensor})[0][0]
        h, w = rgb.shape[:2]
        # Upsample logits (not labels) so class borders follow the image smoothly.
        up = cv2.resize(logits.transpose(1, 2, 0).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return np.argmax(up, axis=2).astype(np.uint8)


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    inv = (~binary).astype(np.uint8)
    _, labels = cv2.connectedComponents(inv, connectivity=4)
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return ~np.isin(labels, np.unique(edge[edge > 0]))


def garment_mask(labels: np.ndarray) -> np.ndarray | None:
    """Mask of the worn top on a model photo, or None for flat-lay / product shots.

    Prints are sometimes labelled as something else (a printed face as "Face"), so
    everything enclosed by the garment is kept.
    """
    h, w = labels.shape
    total = float(h * w)
    upper = np.isin(labels, (UPPER_CLOTHES, DRESS))
    body = np.isin(labels, BODY)
    if upper.sum() < 0.03 * total or body.sum() < 0.01 * total:
        return None
    m = min(h, w)
    k = max(3, m // 150) | 1
    upper = cv2.morphologyEx(upper.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(upper, connectivity=8)
    if n <= 1:
        return None
    main = cc == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    garment = _fill_holes(main)
    return garment & ~_arm_regions(labels, garment)


def _arm_regions(labels: np.ndarray, garment: np.ndarray) -> np.ndarray:
    """Arms filled out to their outline.

    Tattooed skin is partly labelled as clothing, leaving the arm as scattered
    fragments; merging the fragments and taking their hull recovers the whole arm.
    """
    h, w = labels.shape
    m = min(h, w)
    k = max(5, m // 40) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    garment_area = max(1, int(garment.sum()))
    out = np.zeros((h, w), dtype=bool)
    for arm in (LEFT_ARM, RIGHT_ARM):
        raw = (labels == arm).astype(np.uint8)
        if raw.sum() < 0.002 * h * w:
            continue
        merged = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 0.001 * h * w:
                continue
            pts = cv2.findNonZero((cc == i).astype(np.uint8))
            hull = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
            hull = hull.astype(bool)
            # A bent arm (hand on hip) has a hull spanning the torso; fall back to the raw labels.
            if (hull & garment).sum() > 0.15 * garment_area:
                hull = cv2.dilate((cc == i).astype(np.uint8), kernel).astype(bool)
            out |= hull
    return out


_parser: HumanParser | None = None
_lock = threading.Lock()


def get_parser() -> HumanParser:
    global _parser
    with _lock:
        if _parser is None:
            _parser = HumanParser()
        return _parser
