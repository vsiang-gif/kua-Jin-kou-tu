from __future__ import annotations

import json
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from ..pipeline import Processor
from ..pipeline.processor import VARIANT_AI, VARIANT_AUTO, VARIANT_COLOR, VARIANT_FLAT, Result, load_rgb
from .client import DesignTask, HumbirdClient, HumbirdError, download
from .verify import Match, match_print, same_product, save_overlay

MATCH_THRESHOLD = 0.90

ST_REVIEW = "待审核"
ST_MANUAL = "需人工处理"
ST_FAILED = "失败"
ST_APPROVED = "已通过"
ST_REJECTED = "已退回"
FINAL = {ST_APPROVED, ST_REJECTED}


@dataclass
class Record:
    key: str
    order_id: str
    third_order_id: str
    store_name: str
    sku: str
    spec: str
    name: str
    remaining_seconds: int
    status: str = ""
    match_score: float = 0.0
    cut_score: int = 0
    source: str = ""
    output: str = ""
    overlay: str = ""
    reasons: list[str] = field(default_factory=list)
    updated: str = ""

    @classmethod
    def from_task(cls, t: DesignTask) -> "Record":
        return cls(key=t.key, order_id=t.order_id, third_order_id=t.third_order_id, store_name=t.store_name,
                   sku=t.sku, spec=t.spec, name=t.name, remaining_seconds=t.remaining_seconds)


class Store:
    """Review state lives next to the files so it survives restarts and can be opened in any editor."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / "审核记录.json"
        self._lock = threading.Lock()
        self.records: dict[str, Record] = {}
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.records = {k: Record(**v) for k, v in raw.items()}
            except (ValueError, TypeError):
                self.records = {}

    def put(self, r: Record) -> None:
        with self._lock:
            r.updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.records[r.key] = r
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: asdict(v) for k, v in self.records.items()}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(self.path)

    def decide(self, key: str, approve: bool) -> Record:
        r = self.records[key]
        r.status = ST_APPROVED if approve else ST_REJECTED
        if approve and r.output:
            dest = self.root / "已通过" / f"{r.key}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r.output, dest)
        self.put(r)
        return r


VARIANTS = (
    (VARIANT_AUTO, ""),
    (VARIANT_AI, "AI 识别印花"),
    (VARIANT_FLAT, "保留明暗"),
    (VARIANT_COLOR, "按颜色分离"),
)


def _best_cut(proc: Processor, src: Path, ref) -> tuple[Result, Match | None]:
    """Try several cut methods and keep the one that pastes back onto the product photo best.

    No single method fits every mockup (flat text, photo prints, shaded renders), but the
    product photo tells us which cut is right, so each one is checked against it.
    """
    best: tuple[float, Result, Match | None, Path | None, Path | None] | None = None
    tried: list[Path] = []
    for variant, label in VARIANTS:
        res = proc.process(src, variant)
        if res.error or res.output is None:
            if best is None:
                best = (-1.0, res, None, None, None)
            continue
        m = match_print(ref, res.output)
        score = m.score if m else 0.0
        keep_png = res.output.with_name(f"{res.output.stem}.{variant}.png")
        shutil.copy2(res.output, keep_png)
        keep_svg = None
        if res.svg and res.svg.is_file():
            keep_svg = res.svg.with_name(f"{res.svg.stem}.{variant}.svg")
            shutil.copy2(res.svg, keep_svg)
        tried += [p for p in (keep_png, keep_svg) if p]
        if label:
            res.reasons.insert(0, f"采用方案：{label}")
        if best is None or score > best[0]:
            best = (score, res, m, keep_png, keep_svg)
        if score >= MATCH_THRESHOLD and res.score >= proc.opts.review_threshold:
            break

    assert best is not None
    _, res, m, png, svg = best
    if png is not None:
        for stale in proc.targets(src):
            for p in (stale, stale.with_suffix(".svg")):
                if p.is_file() and p != res.output and p != res.svg:
                    p.unlink()
        shutil.copy2(png, res.output)
        if svg is not None and res.svg is not None:
            shutil.copy2(svg, res.svg)
    for p in tried:
        p.unlink(missing_ok=True)
    return res, m


def process_task(client: HumbirdClient, proc: Processor, store: Store, task: DesignTask) -> Record:
    rec = Record.from_task(task)
    try:
        matched = client.match_sku(task.sku)
        mismatches = same_product(task.color, task.size, task.image_url, matched, task.sku)
        rec.reasons.extend(mismatches)
        url = str(matched.get("file_path") or task.image_url)
        if not url:
            raise HumbirdError("订单里没有商品图")
        suffix = Path(url.split("?")[0]).suffix or ".jpg"
        src = download(url, store.root / "原图" / f"{task.key}{suffix}")
        rec.source = str(src)

        ref = load_rgb(src)
        result, m = _best_cut(proc, src, ref)
        rec.cut_score = result.score
        rec.reasons.extend(result.reasons)
        if result.error or result.output is None:
            rec.status = ST_FAILED
            rec.reasons.append(result.error or "没有生成印花")
            return rec
        rec.output = str(result.output)
        if m is None:
            rec.status = ST_MANUAL
            rec.reasons.append("印花是空的")
            return rec
        rec.match_score = round(m.score, 3)
        rec.overlay = str(save_overlay(ref, result.output, m, store.root / "比对" / f"{task.key}.jpg"))
        if m.score < MATCH_THRESHOLD:
            rec.reasons.insert(0, f"回贴比对只有 {m.score:.0%}，可能缺笔画或带了衣服底色")
        ok = not mismatches and m.score >= MATCH_THRESHOLD and result.score >= proc.opts.review_threshold
        rec.status = ST_REVIEW if ok else ST_MANUAL
    except Exception as exc:
        rec.status = ST_FAILED
        rec.reasons.append(str(exc) if isinstance(exc, HumbirdError) else f"{type(exc).__name__}: {exc}")
    finally:
        store.put(rec)
    return rec
