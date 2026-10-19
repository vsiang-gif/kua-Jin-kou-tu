"""Separate the print from the shirt fabric, flatten wrinkles and straighten.

BiRefNet finds the shirt (the salient object). Inside the shirt the print is
whatever differs from the fabric colour, measured in LAB with the lightness
channel corrected for wrinkle shading.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import Options

DETECT_LONG_SIDE = 1600
SPECK_AREA = [0.0, 0.00002, 0.00008, 0.0003]
# Contrast (multiple of the threshold) above which a blob is surely ink, never noise/shading.
STRONG_INK = 2.5
# Separate design parts closer than this (fraction of the shirt size) belong to one print.
GROUP_GAP = 0.08


def _gather_print(merged_labels: np.ndarray, stats: np.ndarray, sums: np.ndarray, best: int,
                  gap: float) -> np.ndarray:
    """Start from the main blob and add every nearby blob (text lines, fuses, dots)."""
    n = stats.shape[0]
    x0, y0 = stats[best, 0], stats[best, 1]
    x1, y1 = x0 + stats[best, 2], y0 + stats[best, 3]
    chosen = {best}
    changed = True
    while changed:
        changed = False
        for i in range(1, n):
            if i in chosen or sums[i] <= 0:
                continue
            bx0, by0 = stats[i, 0], stats[i, 1]
            bx1, by1 = bx0 + stats[i, 2], by0 + stats[i, 3]
            dx = max(0, bx0 - x1, x0 - bx1)
            dy = max(0, by0 - y1, y0 - by1)
            if max(dx, dy) <= gap:
                chosen.add(i)
                x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
                changed = True
    return np.isin(merged_labels, list(chosen)).astype(np.uint8)


@dataclass
class Extraction:
    found: bool
    rgb: np.ndarray | None = None  # uint8 crop, colours corrected
    alpha: np.ndarray | None = None  # float32 crop 0..1
    fabric_rgb: tuple[int, int, int] = (255, 255, 255)
    contrast: float = 0.0
    touches_edge: bool = False
    components: int = 0
    mode: str = "shirt"
    notes: list[str] = field(default_factory=list)


def _ellipse(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _otsu(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    v = np.clip(values, 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _distance(lab: np.ndarray, fabric: np.ndarray, illum: np.ndarray | None, l_weight: float) -> np.ndarray:
    lightness = lab[..., 0]
    if illum is not None:
        lightness = lightness - (illum - fabric[0])
    return np.sqrt(
        l_weight * (lightness - fabric[0]) ** 2
        + (lab[..., 1] - fabric[1]) ** 2
        + (lab[..., 2] - fabric[2]) ** 2
    )


def _illumination(lab: np.ndarray, fabric_px: np.ndarray, fabric_l: float, size: int) -> np.ndarray:
    """Smooth estimate of fabric lightness (wrinkle shading), interpolated under the print."""
    weight = fabric_px.astype(np.float32)
    light = lab[..., 0] * weight
    s_small, s_large = max(2.0, size * 0.02), max(6.0, size * 0.08)
    n1, d1 = cv2.GaussianBlur(light, (0, 0), s_small), cv2.GaussianBlur(weight, (0, 0), s_small)
    n2, d2 = cv2.GaussianBlur(light, (0, 0), s_large), cv2.GaussianBlur(weight, (0, 0), s_large)
    e1 = n1 / np.maximum(d1, 1e-4)
    e2 = n2 / np.maximum(d2, 1e-4)
    mix = np.clip(d1 / 0.35, 0, 1)
    illum = mix * e1 + (1 - mix) * e2
    return np.where(d2 > 1e-3, illum, fabric_l).astype(np.float32)


def _component_peaks(labels: np.ndarray, n: int, d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    peak = np.zeros(n, np.float32)
    np.maximum.at(peak, labels[mask], d[mask])
    return peak


def _remove_small(
    binary: np.ndarray,
    min_area: float,
    d: np.ndarray | None = None,
    strong: float = 0.0,
    group_radius: int = 0,
) -> tuple[np.ndarray, int]:
    """Drop tiny blobs.

    Kept anyway: tiny blobs with strong ink contrast (design dots), and tiny blobs
    lying close together whose combined area is large (thin strokes broken into
    fragments, e.g. hand-drawn flames or fine lettering). Real noise is scattered.
    """
    binary = binary.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if n <= 1:
        return binary, 0
    area = stats[:, cv2.CC_STAT_AREA]
    keep = area >= max(1.0, min_area)
    if group_radius > 0:
        grouped = cv2.dilate(binary.astype(np.uint8), _ellipse(group_radius))
        gn, glabels = cv2.connectedComponents(grouped, connectivity=8)
        group_area = np.bincount(glabels[binary], minlength=gn)
        member_group = np.zeros(n, np.int64)
        member_group[labels[binary]] = glabels[binary]
        keep |= group_area[member_group] >= max(1.0, min_area) * 4
    if d is not None and strong > 0:
        keep |= (_component_peaks(labels, n, d, binary) >= strong) & (area >= 3)
    keep[0] = False
    return keep[labels], int(keep.sum())


def _drop_shading(binary: np.ndarray, lab: np.ndarray, d: np.ndarray, fabric: np.ndarray, t: float) -> np.ndarray:
    """Remove blobs that look like fold shadows: colourless, soft-edged and weak.

    Ink has crisp edges relative to its contrast; shading fades out gradually and
    never gets as far from the fabric colour as real ink (black ink is also
    colourless, so strong contrast alone marks a blob as ink).
    """
    b = binary.astype(np.uint8)
    n, labels = cv2.connectedComponents(b, connectivity=8)
    if n <= 1:
        return binary
    gx = cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    chroma = np.hypot(lab[..., 1] - fabric[1], lab[..., 2] - fabric[2])
    edge = (b - cv2.erode(b, np.ones((3, 3), np.uint8))) > 0

    area = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    inside = labels[binary]
    mean_d = np.bincount(inside, weights=d[binary], minlength=n) / np.maximum(area, 1)
    mean_c = np.bincount(inside, weights=chroma[binary], minlength=n) / np.maximum(area, 1)
    e_lab = labels[edge]
    e_cnt = np.bincount(e_lab, minlength=n).astype(np.float64)
    mean_g = np.bincount(e_lab, weights=grad[edge], minlength=n) / np.maximum(e_cnt, 1)

    sharpness = mean_g / np.maximum(mean_d, 1e-3)
    peak = _component_peaks(labels, n, d, binary)
    shading = (mean_c < 10) & (sharpness < 3.0) & (peak < STRONG_INK * t)
    shading[0] = False
    return binary & ~shading[labels]


def isolate_print(rgb: np.ndarray, fg: np.ndarray, opts: Options) -> Extraction:
    H, W = rgb.shape[:2]
    scale = min(1.0, DETECT_LONG_SIDE / max(H, W))
    if scale < 1.0:
        size = (max(1, round(W * scale)), max(1, round(H * scale)))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        fg_s = cv2.resize(fg, size, interpolation=cv2.INTER_LINEAR)
    else:
        small, fg_s = rgb, fg
    h, w = small.shape[:2]
    m = min(h, w)

    region = (fg_s > 0.5).astype(np.uint8)
    ratio = float(region.mean())
    ring_w = max(4, int(m * 0.03))
    notes: list[str] = []
    if 0.03 <= ratio <= 0.85:
        mode = "shirt"
        inner = cv2.erode(region, _ellipse(max(3, int(m * 0.012))))
        ring = inner & (1 - cv2.erode(inner, _ellipse(ring_w)))
    else:
        mode = "closeup"
        inner = np.ones((h, w), np.uint8)
        ring = np.zeros((h, w), np.uint8)
        ring[:ring_w], ring[-ring_w:], ring[:, :ring_w], ring[:, -ring_w:] = 1, 1, 1, 1
    inner_b = inner > 0
    if ring.sum() < 50:
        ring = inner

    # Light denoise only: a large median would erase thin strokes (fine text, fuses, dots).
    k = 3 if m < 1400 else 5
    lab = _lab(cv2.medianBlur(small, k))
    fabric = np.median(lab[ring > 0], axis=0).astype(np.float32)
    fabric_rgb = cv2.cvtColor(np.clip(fabric, 0, 255).astype(np.uint8)[None, None], cv2.COLOR_LAB2RGB)[0, 0]
    l_weight = 0.55 if opts.dewrinkle else 1.0

    d0 = _distance(lab, fabric, None, l_weight)
    t0 = max(_otsu(d0[inner_b]), 14.0)
    illum = None
    if opts.dewrinkle:
        # Wrinkles change lightness but not colour, so only colour shifts or very
        # strong lightness shifts are excluded from the shading estimate.
        chroma = np.hypot(lab[..., 1] - fabric[1], lab[..., 2] - fabric[2])
        dl = lab[..., 0] - fabric[0]
        rough = (((chroma > 8) | (dl < -70) | (dl > 50)) & inner_b).astype(np.uint8)
        fabric_px = inner_b & (cv2.dilate(rough, _ellipse(max(2, m // 200))) == 0)
        illum = _illumination(lab, fabric_px, float(fabric[0]), m)
    d = _distance(lab, fabric, illum, l_weight)
    t = max(_otsu(d[inner_b]), 12.0)
    binary = (d > t) & inner_b

    level = int(np.clip(opts.despeckle, 0, 3))
    if level >= 2:
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(level - 1)) > 0
    group_r = max(2, m // 250)
    binary, _ = _remove_small(binary, SPECK_AREA[level] * h * w, d, STRONG_INK * t, group_r)
    if opts.dewrinkle:
        binary = _drop_shading(binary, lab, d, fabric, t)
    if mode == "shirt":
        # Ink touching the shirt outline is arms, collar or seams, not the print.
        # Drop it per piece, before merging, so it cannot drag nearby design parts away.
        border = inner_b & (cv2.erode(inner, _ellipse(max(2, int(m * 0.01)))) == 0)
        n_b, lab_b = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        bad = np.zeros(n_b, bool)
        bad[np.unique(lab_b[border & binary])] = True
        bad[0] = False
        binary = binary & ~bad[lab_b]
    if binary.sum() < h * w * 0.002:
        return Extraction(found=False, mode=mode, notes=["没有找到明显的印花（可能和衣服颜色太接近）"])

    merged = cv2.dilate(binary.astype(np.uint8), _ellipse(max(5, int(m * 0.025))))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(merged, connectivity=8)
    sums = np.bincount(labels[binary], minlength=n).astype(np.float64)
    sums[0] = 0
    ys, xs = np.nonzero(inner_b)
    centre = np.array([xs.mean(), ys.mean()]) if xs.size else np.array([w / 2, h / 2])
    dist = np.linalg.norm(cents - centre, axis=1) / max(m, 1)
    best = int(np.argmax(sums * (1.0 - 0.5 * np.clip(dist, 0, 1))))
    blob = _gather_print(labels, stats, sums, best, GROUP_GAP * m)

    x, y, bw, bh = cv2.boundingRect(blob)
    pad = int(m * 0.01)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    touches = x0 == 0 or y0 == 0 or x1 == w or y1 == h

    inv = 1.0 / scale
    X0, Y0 = int(x0 * inv), int(y0 * inv)
    X1, Y1 = min(W, int(np.ceil(x1 * inv))), min(H, int(np.ceil(y1 * inv)))
    crop = rgb[Y0:Y1, X0:X1]
    ch, cw = crop.shape[:2]
    blob_full = cv2.resize(blob[y0:y1, x0:x1], (cw, ch), interpolation=cv2.INTER_NEAREST)
    blob_full = cv2.dilate(blob_full, _ellipse(max(2, int(2 * inv))))

    lab_c = _lab(crop)
    illum_c = None
    if illum is not None:
        illum_c = cv2.resize(illum[y0:y1, x0:x1], (cw, ch), interpolation=cv2.INTER_LINEAR)
    d_c = _distance(lab_c, fabric, illum_c, l_weight)
    alpha = np.clip((d_c - 0.75 * t) / (0.5 * t), 0, 1).astype(np.float32) * blob_full

    hard, components = _remove_small(alpha > 0.5, SPECK_AREA[level] * h * w * inv * inv, d_c, STRONG_INK * t,
                                      max(2, int(group_r * inv)))
    soft_zone = cv2.dilate(hard.astype(np.uint8), _ellipse(2)) > 0
    alpha = np.where(soft_zone, alpha, 0).astype(np.float32)

    contrast = float(np.median(d_c[hard]) / t) if hard.any() else 0.0

    if illum_c is not None:
        lab_c[..., 0] = np.clip(lab_c[..., 0] - 0.7 * (illum_c - fabric[0]), 0, 255)
    rgb_c = cv2.cvtColor(lab_c.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    # Remove fabric colour bleeding into semi-transparent edge pixels.
    a = alpha[..., None]
    edge = (a > 0.05) & (a < 0.98)
    unmixed = (rgb_c - (1 - a) * fabric_rgb.astype(np.float32)) / np.maximum(a, 0.15)
    rgb_c = np.where(edge, np.clip(unmixed, 0, 255), rgb_c).astype(np.uint8)

    ys, xs = np.nonzero(alpha > 0.02)
    if ys.size == 0:
        return Extraction(found=False, mode=mode, notes=["没有找到明显的印花"])
    t0y, t1y = max(0, ys.min() - 4), min(ch, ys.max() + 5)
    t0x, t1x = max(0, xs.min() - 4), min(cw, xs.max() + 5)

    if mode == "closeup":
        notes.append("近景模式：整张图视为布料")
    return Extraction(
        found=True,
        rgb=np.ascontiguousarray(rgb_c[t0y:t1y, t0x:t1x]),
        alpha=np.ascontiguousarray(alpha[t0y:t1y, t0x:t1x]),
        fabric_rgb=tuple(int(c) for c in fabric_rgb),
        contrast=contrast,
        touches_edge=touches,
        components=components,
        mode=mode,
        notes=notes,
    )


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], np.float32)


def straighten(rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Deskew prints that are roughly rectangular. Free-form prints are left as-is."""
    mask = (alpha > 0.5).astype(np.uint8)
    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) < 100:
        return rgb, alpha, ""
    (_, _), (rw, rh), angle = cv2.minAreaRect(pts)
    fill = mask.sum() / max(rw * rh, 1.0)
    if fill < 0.8:
        return rgb, alpha, ""

    hull = cv2.convexHull(pts)
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(approx) == 4 and fill > 0.88:
        src = _order_quad(approx)
        top, bottom = src[1] - src[0], src[2] - src[3]
        left, right = src[3] - src[0], src[2] - src[1]
        tw = int(max(np.linalg.norm(top), np.linalg.norm(bottom)))
        th = int(max(np.linalg.norm(left), np.linalg.norm(right)))
        tilt = max(abs(top[1]), abs(bottom[1]), abs(left[0]), abs(right[0]))
        if tw > 20 and th > 20 and tilt > 3:
            dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            rgb_w = cv2.warpPerspective(rgb, M, (tw, th), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            alpha_w = cv2.warpPerspective(alpha, M, (tw, th), flags=cv2.INTER_LINEAR, borderValue=0)
            return rgb_w, alpha_w, "透视校正"
        return rgb, alpha, ""

    if angle > 45:
        angle -= 90
    if abs(angle) < 0.5 or abs(angle) > 20:
        return rgb, alpha, ""
    h, w = rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rgb_r = cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    alpha_r = cv2.warpAffine(alpha, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    return rgb_r, alpha_r, f"旋转校正 {angle:.1f}°"


def classify(rgb: np.ndarray, alpha: np.ndarray) -> str:
    """'graphic' when ~8 flat colours explain the print well, otherwise 'photo'."""
    px = _lab(rgb)[alpha > 0.9]
    if px.shape[0] < 200:
        return "graphic"
    rng = np.random.default_rng(0)
    if px.shape[0] > 40000:
        px = px[rng.choice(px.shape[0], 40000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(px, 8, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    err = np.linalg.norm(px - centers[labels.ravel()], axis=1)
    return "photo" if float((err > 14).mean()) > 0.2 else "graphic"
