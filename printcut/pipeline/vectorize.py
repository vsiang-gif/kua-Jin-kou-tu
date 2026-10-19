"""Text/graphic prints: colour reduction -> label cleanup -> vtracer -> resvg render."""
from __future__ import annotations

import io
import re

import cv2
import numpy as np
import resvg_py
import vtracer
from PIL import Image
from scipy import ndimage

WORK_MIN, WORK_MAX = 1600, 2400
SMOOTH_SIGMA = [0.0, 0.6, 1.0, 1.6, 2.4, 3.4]
SPECK_PX = [0, 12, 40, 120]
CORNER = [0, 30, 45, 60, 75, 90]
LENGTH = [0, 3.5, 4.0, 4.5, 5.0, 6.0]
FILTER_SPECKLE = [1, 2, 4, 8]


def _pick_k(sample: np.ndarray) -> int:
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    prev = None
    for k in range(2, 13):
        compactness, _, _ = cv2.kmeans(sample, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        mse = compactness / len(sample)
        if mse < 60:
            return k
        if prev is not None and prev / max(mse, 1e-6) < 1.2:
            return k - 1
        prev = mse
    return 12


def quantize(rgb: np.ndarray, alpha: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels HxW int32 with -1 for background, palette Kx3 LAB float32)."""
    fg = alpha > 0.5
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    px = lab[fg]
    rng = np.random.default_rng(0)
    sample = px[rng.choice(len(px), min(len(px), 60000), replace=False)] if len(px) > 60000 else px
    if k <= 0:
        k = _pick_k(sample)
    k = int(np.clip(k, 1, min(16, len(sample))))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centers = cv2.kmeans(sample, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)

    labels = np.full(fg.shape, -1, np.int32)
    flat = px
    assigned = np.empty(len(flat), np.int32)
    step = 1_000_000
    for i in range(0, len(flat), step):
        chunk = flat[i : i + step]
        d = ((chunk[:, None, :] - centers[None]) ** 2).sum(axis=2)
        assigned[i : i + step] = np.argmin(d, axis=1)
    labels[fg] = assigned
    return labels, centers


def clean_labels(labels: np.ndarray, k: int, smooth: int, despeckle: int, px_scale: float) -> np.ndarray:
    lab = labels.copy()
    lab[lab < 0] = k
    sigma = SMOOTH_SIGMA[int(np.clip(smooth, 0, 5))] * px_scale
    if sigma > 0:
        best_val = np.full(lab.shape, -1.0, np.float32)
        best_idx = lab.copy()
        for i in range(k + 1):
            vote = cv2.GaussianBlur((lab == i).astype(np.float32), (0, 0), sigma)
            better = vote > best_val
            best_val[better] = vote[better]
            best_idx[better] = i
        lab = best_idx

    min_area = SPECK_PX[int(np.clip(despeckle, 0, 3))] * px_scale * px_scale
    if min_area > 0:
        unknown = np.zeros(lab.shape, bool)
        for i in range(k + 1):
            n, comp, stats, _ = cv2.connectedComponentsWithStats((lab == i).astype(np.uint8), connectivity=4)
            small = stats[:, cv2.CC_STAT_AREA] < min_area
            small[0] = False
            if small.any():
                unknown |= small[comp]
        if unknown.any() and not unknown.all():
            _, (iy, ix) = ndimage.distance_transform_edt(unknown, return_indices=True)
            lab = lab[iy, ix]

    lab[lab == k] = -1
    return lab


def _palette_rgb(rgb: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    k = len(centers)
    palette = cv2.cvtColor(np.clip(centers, 0, 255).astype(np.uint8)[None], cv2.COLOR_LAB2RGB)[0].astype(np.int32)
    for i in range(k):
        sel = labels == i
        if sel.sum() > 20:
            palette[i] = np.median(rgb[sel], axis=0)
    return palette.astype(np.uint8)


def _snap_fills(svg: str, palette: np.ndarray) -> str:
    pal = palette.astype(np.int32)

    def repl(match: re.Match) -> str:
        c = np.array([int(match.group(1)[i : i + 2], 16) for i in (0, 2, 4)])
        p = pal[np.argmin(((pal - c) ** 2).sum(axis=1))]
        return 'fill="#%02X%02X%02X"' % tuple(int(v) for v in p)

    return re.sub(r'fill="#([0-9A-Fa-f]{6})"', repl, svg)


def _with_viewbox(svg: str, w: int, h: int) -> str:
    return svg.replace(
        f'width="{w}" height="{h}"', f'width="{w}" height="{h}" viewBox="0 0 {w} {h}"', 1
    )


def vectorize_print(
    rgb: np.ndarray, alpha: np.ndarray, size: tuple[int, int], colors: int, smooth: int, despeckle: int
) -> tuple[np.ndarray, str, int]:
    """Returns (RGBA uint8 at `size`, svg text, colour count)."""
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    work = int(np.clip(long_side, WORK_MIN, WORK_MAX))
    if work != long_side:
        s = work / long_side
        ws = (max(1, round(w * s)), max(1, round(h * s)))
        interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
        rgb = cv2.resize(rgb, ws, interpolation=interp)
        alpha = cv2.resize(alpha, ws, interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
    px_scale = max(h, w) / WORK_MIN

    labels, centers = quantize(rgb, alpha, colors)
    labels = clean_labels(labels, len(centers), smooth, despeckle, px_scale)
    palette = _palette_rgb(rgb, labels, centers)

    flat = np.zeros((h, w, 4), np.uint8)
    fg = labels >= 0
    flat[fg, :3] = palette[labels[fg]]
    flat[fg, 3] = 255
    buf = io.BytesIO()
    Image.fromarray(flat).save(buf, "PNG", compress_level=1)

    level = int(np.clip(smooth, 1, 5))
    svg = vtracer.convert_raw_image_to_svg(
        buf.getvalue(),
        img_format="png",
        colormode="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=FILTER_SPECKLE[int(np.clip(despeckle, 0, 3))],
        color_precision=8,
        layer_difference=1,
        corner_threshold=CORNER[level],
        length_threshold=LENGTH[level],
        max_iterations=10,
        splice_threshold=45,
        path_precision=3,
    )
    svg = _with_viewbox(_snap_fills(svg, palette), w, h)
    png = resvg_py.svg_to_bytes(svg_string=svg, width=int(size[0]), height=int(size[1]))
    with Image.open(io.BytesIO(bytes(png))) as im:
        out = np.asarray(im.convert("RGBA")).copy()
    if out.shape[1] != size[0] or out.shape[0] != size[1]:
        out = cv2.resize(out, size, interpolation=cv2.INTER_AREA)
    return out, svg, int(len(np.unique(labels[fg])))
