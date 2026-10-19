from __future__ import annotations

import csv
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..config import (
    LOG_FILE_NAME,
    PRINT_AUTO,
    PRINT_GRAPHIC,
    PRINT_PHOTO,
    REVIEW_DIR_NAME,
    Options,
)
from .export import compose_canvas, content_size, merge_rgba, save_png, smooth_alpha
from .extract import classify, find_shirt, isolate_print, straighten
from .segment import Segmenter
from .upscale import Upscaler
from .vectorize import vectorize_print

MAX_INPUT_SIDE = 6000
TYPE_NAMES = {PRINT_GRAPHIC: "文字图形", PRINT_PHOTO: "照片插画"}


@dataclass
class Result:
    source: Path
    output: Path | None = None
    svg: Path | None = None
    score: int = 0
    print_type: str = ""
    seconds: float = 0.0
    reasons: list[str] = field(default_factory=list)
    skipped: bool = False
    error: str = ""

    @property
    def needs_review(self) -> bool:
        return bool(self.error) or (self.output is not None and self.output.parent.name == REVIEW_DIR_NAME)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        rgb = np.asarray(im.convert("RGB")).copy()
    h, w = rgb.shape[:2]
    if max(h, w) > MAX_INPUT_SIDE:
        s = MAX_INPUT_SIDE / max(h, w)
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return rgb


class Processor:
    def __init__(self, opts: Options, out_dir: Path, segmenter: Segmenter, upscaler: Upscaler) -> None:
        self.opts = opts
        self.out_dir = Path(out_dir)
        self.segmenter = segmenter
        self.upscaler = upscaler
        self._log_lock = threading.Lock()

    def targets(self, src: Path) -> tuple[Path, Path]:
        name = src.stem + ".png"
        return self.out_dir / name, self.out_dir / REVIEW_DIR_NAME / name

    def existing_output(self, src: Path) -> Path | None:
        for p in self.targets(src):
            if p.is_file():
                return p
        return None

    def process(self, src: Path) -> Result:
        src = Path(src)
        started = time.perf_counter()
        result = Result(source=src)
        try:
            self._run(src, result)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.reasons.append("处理出错")
            traceback.print_exc()
        result.seconds = time.perf_counter() - started
        self._write_log(result)
        return result

    def _run(self, src: Path, result: Result) -> None:
        o = self.opts
        rgb = load_rgb(src)
        fg = self.segmenter.predict(rgb)
        ext = isolate_print(rgb, fg, o)
        wants_ai = (
            o.print_type == PRINT_PHOTO
            or not ext.found
            or (o.print_type == PRINT_AUTO and classify(ext.rgb, ext.alpha) == PRINT_PHOTO)
        )
        if wants_ai:
            hint = self._print_hint(rgb, fg)
            if hint is not None:
                ext_ai = isolate_print(rgb, fg, o, print_hint=hint)
                if ext_ai.found:
                    ext = ext_ai
        result.reasons.extend(ext.notes)
        if not ext.found:
            result.error = "没有找到印花"
            return

        crop, alpha = ext.rgb, ext.alpha
        if o.deskew:
            crop, alpha, how = straighten(crop, alpha)
            if how:
                result.reasons.append(how)

        if o.print_type != PRINT_AUTO:
            ptype = o.print_type
        elif ext.ai_matte:
            # Only artwork the colour method could not separate gets here: paintings, portraits.
            ptype = PRINT_PHOTO
        else:
            ptype = classify(crop, alpha)
        result.print_type = TYPE_NAMES.get(ptype, ptype)
        h, w = crop.shape[:2]
        tw, th = content_size(w, h, o.width, o.height, o.margin)
        factor = max(tw / w, th / h)

        svg_text = None
        if ptype == PRINT_PHOTO:
            up, used_ai = self.upscaler.upscale(crop, (tw, th))
            a = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_CUBIC)
            a = smooth_alpha(np.clip(a, 0, 1), o.edge_smooth, factor)
            rgba = merge_rgba(up, a)
            if factor > 1.5 and not used_ai:
                result.reasons.append("未使用 AI 放大（找不到 Real-ESRGAN）")
        else:
            rgba, svg_text, ncolors = vectorize_print(crop, alpha, (tw, th), o.colors, o.edge_smooth, o.despeckle)
            result.reasons.append(f"{ncolors} 色")

        score = 100
        if factor > 6:
            score -= 30
            result.reasons.append(f"原图印花太小（{max(w, h)} px，需放大 {factor:.1f} 倍）")
        elif factor > 4 and ptype == PRINT_PHOTO:
            score -= 20
            result.reasons.append(f"放大倍数较高（{factor:.1f} 倍）")
        if ext.contrast < 1.6:
            score -= 30
            result.reasons.append("印花和衣服颜色接近，边缘可能不准")
        if ext.touches_edge:
            score -= 25
            result.reasons.append("印花贴近画面边缘，可能被截断")
        if ext.components > 300:
            score -= 15
            result.reasons.append("碎块较多，可能有杂点")
        result.score = max(0, score)

        ok_path, review_path = self.targets(src)
        out = ok_path if result.score >= o.review_threshold else review_path
        for stale in (ok_path, review_path):
            if stale != out and stale.is_file():
                stale.unlink()
        save_png(compose_canvas(rgba, o.width, o.height, o.background), out, o.dpi)
        result.output = out
        if svg_text and o.save_svg:
            svg_path = out.with_suffix(".svg")
            svg_path.write_text(svg_text, encoding="utf-8")
            result.svg = svg_path

    def _print_hint(self, rgb: np.ndarray, fg: np.ndarray) -> np.ndarray | None:
        """Second BiRefNet pass on the shirt alone: with the studio background gone,
        the most salient thing left is the artwork itself."""
        shirt, ai_grabbed_artwork = find_shirt(rgb, fg)
        if ai_grabbed_artwork:
            return fg * shirt
        coverage = float(shirt.mean())
        if not 0.03 <= coverage <= 0.85:
            # Close-up of the fabric: the first pass already looked at the print.
            fg_cover = float((fg > 0.5).mean())
            return fg if 0.01 < fg_cover < 0.7 else None

        m = min(shirt.shape)
        r = max(3, int(m * 0.012))
        inner = cv2.erode(shirt.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        x, y, bw, bh = cv2.boundingRect(inner)
        if bw < 32 or bh < 32:
            return None
        inside = inner[y:y + bh, x:x + bw] > 0
        crop = rgb[y:y + bh, x:x + bw].copy()
        rw = max(3, int(m * 0.03))
        ring = inside & (cv2.erode(inside.astype(np.uint8), np.ones((2 * rw + 1, 2 * rw + 1), np.uint8)) == 0)
        fabric = np.median(crop[ring if ring.any() else inside], axis=0)
        crop[~inside] = fabric.astype(np.uint8)

        pred = self.segmenter.predict(crop) * inside
        area = float((pred > 0.5).sum())
        if area > 0.75 * inside.sum() or area < 0.003 * inside.size:
            return None
        hint = np.zeros(rgb.shape[:2], np.float32)
        hint[y:y + bh, x:x + bw] = pred
        return hint

    def _write_log(self, r: Result) -> None:
        path = self.out_dir / LOG_FILE_NAME
        with self._log_lock:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow(["时间", "原图", "输出", "分数", "类型", "耗时(秒)", "说明", "错误"])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    str(r.source),
                    str(r.output or ""),
                    r.score,
                    r.print_type,
                    f"{r.seconds:.1f}",
                    "；".join(r.reasons),
                    r.error,
                ])
