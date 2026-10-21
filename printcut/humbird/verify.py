from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SEARCH_SIDE = 800
SCALES = np.linspace(0.15, 0.85, 22)


@dataclass
class Match:
    score: float  # 0..1, normalised correlation of the cut print against the product photo
    box: tuple[int, int, int, int]  # x, y, w, h in reference pixels


def _load_print(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"))
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    ys, xs = np.where(alpha > 0.1)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return rgba[y0:y1, x0:x1, :3], alpha[y0:y1, x0:x1]


def _dominant_gray(gray: np.ndarray) -> float:
    """Fabric is the single most common flat tone; a plain median lands inside big prints."""
    hist = np.bincount(np.clip(gray, 0, 255).astype(np.uint8).ravel() // 8, minlength=32)
    peak = int(np.argmax(hist)) * 8 + 4
    near = gray[np.abs(gray - peak) < 12]
    return float(np.median(near)) if near.size else float(peak)


def match_print(ref_rgb: np.ndarray, print_png: Path) -> Match | None:
    """Slide the cut-out print over the product photo at many sizes; a faithful cut lines
    up almost perfectly somewhere, a broken one (missing strokes, leftover fabric) cannot."""
    loaded = _load_print(print_png)
    if loaded is None:
        return None
    p_rgb, p_alpha = loaded
    h, w = ref_rgb.shape[:2]
    s = min(1.0, SEARCH_SIDE / max(h, w))
    ref = cv2.resize(ref_rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rh, rw = ref_gray.shape
    fabric = _dominant_gray(ref_gray[rh // 5:rh - rh // 5, rw // 5:rw - rw // 5])
    p_gray = cv2.cvtColor(p_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    ph, pw = p_gray.shape

    def locate(frac: float) -> tuple[float, int, int, int, int, np.ndarray] | None:
        tw = max(16, int(rw * frac))
        th = max(16, int(ph * tw / pw))
        if th >= rh or tw >= rw:
            return None
        a = cv2.resize(p_alpha, (tw, th), interpolation=cv2.INTER_AREA)
        g = cv2.resize(p_gray, (tw, th), interpolation=cv2.INTER_AREA)
        tpl = (g * a + fabric * (1 - a)).astype(np.float32)
        res = cv2.matchTemplate(ref_gray, tpl, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, val, _, loc = cv2.minMaxLoc(res)
        return float(val), loc[0], loc[1], tw, th, a

    coarse = [(r, f) for f in SCALES if (r := locate(f)) is not None]
    if not coarse:
        return None
    best, best_f = max(coarse, key=lambda x: x[0][0])
    step = SCALES[1] - SCALES[0]
    for f in np.linspace(best_f - step, best_f + step, 11):
        r = locate(float(f))
        if r is not None and r[0] > best[0]:
            best = r
    _, x, y, tw, th, a = best

    # Correlation alone rewards a partial print that fits perfectly where it is, so also
    # check coverage: every inked pixel around the match must be explained by our print.
    pad_x, pad_y = int(tw * 0.35), int(th * 0.35)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(rw, x + tw + pad_x), min(rh, y + th + pad_y)
    ink_print = np.zeros((y1 - y0, x1 - x0), bool)
    ink_print[y - y0:y - y0 + th, x - x0:x - x0 + tw] = a > 0.5
    k = max(3, tw // 60) | 1

    # Fabric colour measured right around the print (outside it), in LAB so coloured ink
    # with the same brightness as the shirt still counts as ink.
    lab = cv2.cvtColor(ref[y0:y1, x0:x1], cv2.COLOR_RGB2LAB).astype(np.float32)
    outside = cv2.dilate(ink_print.astype(np.uint8), np.ones((k * 2 + 1, k * 2 + 1), np.uint8)) == 0
    fab = np.median(lab[outside], axis=0) if outside.sum() > 50 else np.median(lab.reshape(-1, 3), axis=0)
    diff = np.sqrt(0.5 * (lab[..., 0] - fab[0]) ** 2 + (lab[..., 1] - fab[1]) ** 2 + (lab[..., 2] - fab[2]) ** 2)
    # Mockup shading (folds, gradients) varies around the print; ink must stand out from it.
    spread = float(np.percentile(diff[outside], 90)) if outside.sum() > 50 else 0.0
    thr = max(18.0, spread * 1.5)
    ink_ref = cv2.morphologyEx((diff > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    near_print = cv2.dilate(ink_print.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    near_ref = cv2.dilate(ink_ref.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    recall = float((ink_ref & near_print).sum()) / max(1, int(ink_ref.sum()))
    precision = float((ink_print & near_ref).sum()) / max(1, int(ink_print.sum()))
    score = min(recall, precision)
    return Match(score, (int(x / s), int(y / s), int(tw / s), int(th / s)))


def save_overlay(ref_rgb: np.ndarray, print_png: Path, m: Match, target: Path) -> Path:
    """Side by side: product photo | same photo with our print pasted where it matched."""
    loaded = _load_print(print_png)
    pasted = ref_rgb.copy()
    if loaded is not None:
        p_rgb, p_alpha = loaded
        x, y, w, h = m.box
        h_ref, w_ref = ref_rgb.shape[:2]
        w, h = min(w, w_ref - x), min(h, h_ref - y)
        if w > 0 and h > 0:
            rgb = cv2.resize(p_rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
            a = cv2.resize(p_alpha, (w, h), interpolation=cv2.INTER_AREA)[..., None]
            region = pasted[y:y + h, x:x + w].astype(np.float32)
            pasted[y:y + h, x:x + w] = (rgb * a + region * (1 - a)).astype(np.uint8)
            cv2.rectangle(pasted, (x, y), (x + w, y + h), (31, 111, 235), max(2, w_ref // 300))
    gap = np.full((ref_rgb.shape[0], 12, 3), 255, np.uint8)
    out = np.concatenate([ref_rgb, gap, pasted], axis=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(target, quality=90)
    return target


def same_product(task_color: str, task_size: str, task_image: str, matched: dict, sku: str) -> list[str]:
    """Problems found when comparing the order line with what Humbird matched for that SKU."""
    problems: list[str] = []
    if str(matched.get("third_sku") or "") != sku:
        problems.append("自动匹配返回的 SKU 和订单不一致")
    if task_color and (matched.get("third_color") or "") != task_color:
        problems.append(f"颜色不一致：订单 {task_color}，匹配 {matched.get('third_color')}")
    if task_size and (matched.get("third_size") or "") != task_size:
        problems.append(f"尺码不一致：订单 {task_size}，匹配 {matched.get('third_size')}")
    ref = str(matched.get("file_path") or "")
    if task_image and ref and Path(ref).name != Path(task_image).name:
        problems.append("匹配到的商品图和订单商品图不是同一张")
    return problems
