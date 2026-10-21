"""Compare our cut prints with reference cuts (e.g. downloaded from 灵图) and write an HTML report.

python -m printcut.compare <我们的结果文件夹> <灵图结果文件夹> [-o 报告文件夹]
"""
from __future__ import annotations

import argparse
import csv
import html
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SKIP_DIRS = {"原图", "比对", "对比报告"}
CELL = 360
DIAG_MISSING = "漏抠印花"
DIAG_EXTRA = "多抠了布料/背景"
DIAG_RANGE = "印花范围不对"
DIAG_EDGE = "边缘粗糙"
DIAG_GOOD = "基本一致"
DIAG_NONE = "我们没出图"


@dataclass
class Row:
    name: str
    ours: Path | None
    ref: Path
    iou: float = 0.0
    missing: float = 0.0
    extra: float = 0.0
    edge: float = 0.0
    aspect: float = 0.0
    diagnosis: str = ""
    image: str = ""


def _pngs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.png") if not (SKIP_DIRS & set(p.relative_to(folder).parts[:-1])))


def _key(p: Path) -> str:
    # Variant files are saved as <订单>.<variant>.png; the main file wins anyway.
    return p.stem.split(".")[0].lower()


def match(ours: list[Path], refs: list[Path]) -> list[tuple[Path, Path | None]]:
    index: dict[str, Path] = {}
    for p in ours:
        k = _key(p)
        if k not in index or "." not in p.stem:
            index[k] = p
    keys = sorted(index, key=len, reverse=True)
    pairs = []
    for ref in refs:
        rk = _key(ref)
        hit = index.get(rk) or next((index[k] for k in keys if len(k) >= 6 and (k in rk or rk in k)), None)
        pairs.append((ref, hit))
    return pairs


def load_alpha(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA")).copy()
    alpha = rgba[..., 3].astype(np.float32) / 255
    if alpha.min() > 0.99:
        # No transparency (white background export): treat near-white as empty.
        alpha = (rgba[..., :3].min(axis=2) < 245).astype(np.float32)
    return rgba, alpha


def crop_box(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(alpha > 0.5)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _fit(rgba: np.ndarray, alpha: np.ndarray, box, size: tuple[int, int]):
    x0, y0, x1, y1 = box
    c = cv2.resize(rgba[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
    return c, a


def _edge_band(mask: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    m = mask.astype(np.uint8)
    return (cv2.dilate(m, k) - cv2.erode(m, k)) > 0


def _checker(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    tile = (((yy // 12) + (xx // 12)) % 2).astype(np.uint8)
    return np.where(tile[..., None] == 1, 205, 240).astype(np.uint8).repeat(3, axis=2)


def _on_checker(rgba: np.ndarray) -> np.ndarray:
    h, w = rgba.shape[:2]
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3] * a + _checker(h, w) * (1 - a)).astype(np.uint8)


def _pad(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.full((CELL, CELL, 3), 255, np.uint8)
    y, x = (CELL - h) // 2, (CELL - w) // 2
    out[y:y + h, x:x + w] = img
    return out


def compare(row: Row, out_dir: Path) -> None:
    ref_rgba, ref_a = load_alpha(row.ref)
    ref_box = crop_box(ref_a)
    if row.ours is None or ref_box is None:
        row.diagnosis = DIAG_NONE if row.ours is None else "灵图图片是空的"
        return
    our_rgba, our_a = load_alpha(row.ours)
    our_box = crop_box(our_a)
    if our_box is None:
        row.diagnosis = DIAG_NONE
        return

    rw, rh = ref_box[2] - ref_box[0], ref_box[3] - ref_box[1]
    ow, oh = our_box[2] - our_box[0], our_box[3] - our_box[1]
    row.aspect = abs(np.log((ow / oh) / (rw / rh)))
    s = (CELL - 8) / max(rw, rh)
    size = (max(1, round(rw * s)), max(1, round(rh * s)))
    ref_c, ref_m = _fit(ref_rgba, ref_a, ref_box, size)
    our_c, our_m = _fit(our_rgba, our_a, our_box, size)

    r, o = ref_m > 0.5, our_m > 0.5
    ref_area = max(int(r.sum()), 1)
    row.iou = float((r & o).sum()) / max(int((r | o).sum()), 1)
    row.missing = float((r & ~o).sum()) / ref_area
    row.extra = float((o & ~r).sum()) / ref_area
    band = _edge_band(r)
    row.edge = float((band & (r != o)).sum()) / max(int(band.sum()), 1)

    # Both cuts are stretched to the same box, so a different shape only shows up in the aspect ratio.
    if row.aspect > 0.15:
        row.diagnosis = DIAG_RANGE
        row.iou *= min(ow / oh, rw / rh) / max(ow / oh, rw / rh)
    elif row.iou >= 0.9:
        row.diagnosis = DIAG_GOOD
    elif row.missing >= row.extra and row.missing > 0.08:
        row.diagnosis = DIAG_MISSING
    elif row.extra > 0.08:
        row.diagnosis = DIAG_EXTRA
    else:
        row.diagnosis = DIAG_EDGE

    diff = np.full((*size[::-1], 3), 255, np.uint8)
    diff[r & o] = (200, 200, 200)
    diff[o & ~r] = (220, 40, 40)
    diff[r & ~o] = (40, 90, 230)
    sheet = np.hstack([_pad(_on_checker(our_c)), _pad(_on_checker(ref_c)), _pad(diff)])
    name = f"{len(list(out_dir.glob('*.jpg'))):04d}.jpg"
    Image.fromarray(sheet).save(out_dir / name, quality=85)
    row.image = name


def write_report(rows: list[Row], out_dir: Path) -> Path:
    rows.sort(key=lambda r: (r.diagnosis == DIAG_GOOD, r.iou))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.diagnosis] = counts.get(r.diagnosis, 0) + 1
    scored = [r for r in rows if r.image]
    avg = sum(r.iou for r in scored) / len(scored) if scored else 0

    with open(out_dir / "对比结果.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["名称", "重合度", "漏抠比例", "多抠比例", "边缘误差", "判断", "我们的图", "灵图"])
        for r in rows:
            w.writerow([r.name, f"{r.iou:.0%}", f"{r.missing:.0%}", f"{r.extra:.0%}", f"{r.edge:.0%}",
                        r.diagnosis, str(r.ours or ""), str(r.ref)])

    summary = "".join(
        f"<li><b>{html.escape(k)}</b>：{v} 张</li>" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    items = []
    for r in rows:
        img = f'<img src="{r.image}" alt="{html.escape(r.name)} 对比图">' if r.image else ""
        items.append(
            f'<section><h2>{html.escape(r.name)} — {html.escape(r.diagnosis)}</h2>'
            f'<p>重合度 {r.iou:.0%}　漏抠 {r.missing:.0%}　多抠 {r.extra:.0%}　边缘误差 {r.edge:.0%}</p>{img}</section>')
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>抠图对比报告</title>
<style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:24px;color:#222;line-height:1.5}}
section{{border-top:1px solid #ddd;padding:12px 0}}h2{{font-size:16px;margin:0}}img{{max-width:100%}}
.legend span{{display:inline-block;width:12px;height:12px;margin:0 4px 0 12px;vertical-align:middle}}</style>
<h1>抠图对比报告（我们 vs 灵图）</h1>
<p>共 {len(rows)} 张，平均重合度 {avg:.0%}。最差的排在最前面。</p><ul>{summary}</ul>
<p class="legend">每行三张图：我们的结果 ｜ 灵图的结果 ｜ 差异图
<span style="background:#c8c8c8"></span>两边都有<span style="background:#dc2828"></span>只有我们有（多抠）
<span style="background:#285ae6"></span>只有灵图有（漏抠）</p>
{''.join(items)}</html>"""
    path = out_dir / "对比报告.html"
    path.write_text(page, encoding="utf-8")
    return path


def run(ours_dir: Path, ref_dir: Path, out_dir: Path | None = None, log=print) -> Path:
    out_dir = Path(out_dir or Path(ours_dir) / "对比报告")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    refs = _pngs(Path(ref_dir)) + sorted(Path(ref_dir).rglob("*.jpg"))
    pairs = match(_pngs(Path(ours_dir)), refs)
    rows = []
    for i, (ref, ours) in enumerate(pairs, 1):
        row = Row(name=ref.stem, ours=ours, ref=ref)
        try:
            compare(row, out_dir)
        except Exception as exc:
            row.diagnosis = f"读取失败：{exc}"
        rows.append(row)
        log(f"[{i}/{len(pairs)}] {ref.name}：{row.diagnosis} {row.iou:.0%}")
    return write_report(rows, out_dir)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="对比我们和灵图的抠图结果")
    ap.add_argument("ours")
    ap.add_argument("reference")
    ap.add_argument("-o", "--output")
    a = ap.parse_args()
    print(f"报告：{run(Path(a.ours), Path(a.reference), a.output)}")


if __name__ == "__main__":
    main()
