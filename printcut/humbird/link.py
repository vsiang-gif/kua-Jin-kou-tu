"""Auto 关联商品: bind each 待设计 order to the library product whose 主题名称 is its <WB订单号>_<SKU>.

Only exact name matches are bound, and only after Humbird's own spec matching confirms the
order's colour and size, so a wrong product is never linked. Everything else is reported for
manual handling.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .client import DesignTask, HumbirdClient, HumbirdError
from .imgmatch import SHOW_MIN_INLIERS, Candidate, LibraryIndex, decide, load_order_image
from .job import CACHE


def norm(name: str) -> str:
    """Library names are typed by hand ("WB123 456", "wb123_456"), so compare letters/digits only."""
    return re.sub(r"[^0-9A-Za-z]", "", name or "").upper()


def load_aliases(root: Path) -> dict[str, str]:
    """Orders whose print duplicated an earlier download share that image's name (from 下载清单.csv)."""
    aliases: dict[str, str] = {}
    for sheet in sorted((Path(root) / "商品图").glob("*/下载清单.csv")):
        try:
            with sheet.open(encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    key = f"{row.get('商户订单号', '')}_{row.get('SKU', '')}"
                    image = Path(row.get("使用图片（同一印花）") or "").stem
                    if image and image != key:
                        aliases[key] = image
        except (OSError, csv.Error):
            continue
    return aliases


def _same(a: str, b: str) -> bool:
    return norm(a) == norm(b)


def link_one(client: HumbirdClient, task: DesignTask, product: dict) -> None:
    m = client.match_spec(task.sku, str(product.get("spu_id") or ""))
    if str(m.get("third_sku") or "") != task.sku:
        raise HumbirdError("蜂鸟返回的 SKU 和订单不一致")
    if not m.get("sku_id") or not m.get("id"):
        raise HumbirdError("蜂鸟没识别出对应的颜色/尺码，请手动关联")
    if task.color and not _same(m.get("third_color") or "", task.color):
        raise HumbirdError(f"颜色不一致：订单 {task.color}，识别 {m.get('third_color')}")
    if task.size and not _same(m.get("third_size") or "", task.size):
        raise HumbirdError(f"尺码不一致：订单 {task.size}，识别 {m.get('third_size')}")
    item = m.get("third_order_item") or {}
    item_id = task.item_id or str(item.get("item_id") or "")
    client.bind_product(str(m["id"]), {
        "order_id": task.order_id,
        "order_item_id": item_id,
        "product_id": str(product["product_id"]),
        "sku_id": str(m["sku_id"]),
        "spu_sales_type": m.get("spu_sales_type") or product.get("sales_type") or 2,
        "type": m.get("type") or 1,
        "origin_item_id": task.origin_item_id or item_id,
        "origin_order_id": task.origin_order_id or task.order_id,
    })


class Memory:
    """Pairs the user (or an unmistakable image match) confirmed, so the same listing links instantly next time."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "关联记录" / "已确认配对.json"
        self.sku: dict[str, str] = {}
        self.image: dict[str, str] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.sku, self.image = dict(data.get("sku") or {}), dict(data.get("image") or {})
        except (OSError, ValueError):
            pass

    @staticmethod
    def image_key(url: str) -> str:
        return (url or "").split("?")[0]

    def lookup(self, task: DesignTask) -> str | None:
        return self.sku.get(task.sku) or self.image.get(self.image_key(task.image_url))

    def remember(self, task: DesignTask, product_id: str) -> None:
        self.sku[task.sku] = product_id
        if task.image_url:
            self.image[self.image_key(task.image_url)] = product_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"sku": self.sku, "image": self.image}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)


@dataclass
class Review:
    task: DesignTask
    order_image: Path | None
    candidates: list[Candidate] = field(default_factory=list)
    reason: str = ""


def link_and_remember(client: HumbirdClient, memory: Memory, task: DesignTask, product: dict) -> None:
    link_one(client, task, product)
    memory.remember(task, str(product["product_id"]))


def auto_link(client: HumbirdClient, root: Path, log=print, stop=None, use_images: bool = True
              ) -> tuple[Path, dict[str, int], list[Review]]:
    """Name match → remembered pair → image recognition. Returns (report, counts, items to review)."""
    log("正在拉取「待设计」订单…")
    tasks = client.design_tasks()
    library = [p for p in client.library_products(log) if p.get("is_valid") is not False]
    by_name: dict[str, list[dict]] = {}
    by_id: dict[str, dict] = {}
    for p in library:
        by_name.setdefault(norm(p.get("theme") or ""), []).append(p)
        by_id[str(p["product_id"])] = p
    aliases = load_aliases(root)
    memory = Memory(root)

    counts = {"name": 0, "memory": 0, "image": 0, "review": 0, "failed": 0}
    rows: list[list[str]] = []
    leftovers: list[tuple[DesignTask, str]] = []

    def record(t: DesignTask, how: str, product: dict | None, result: str) -> None:
        rows.append([t.third_order_id, t.store_name, t.sku, t.spec, how,
                     str(product.get("theme") if product else ""), result])

    def try_link(t: DesignTask, product: dict, kind: str, how: str) -> str:
        """Empty string on success, otherwise the reason it could not be linked."""
        try:
            link_and_remember(client, memory, t, product)
        except HumbirdError as exc:
            return str(exc)
        counts[kind] += 1
        record(t, how, product, "已关联")
        return ""

    for i, t in enumerate(tasks, 1):
        if stop is not None and stop():
            break
        log(f"按名称关联 {i}/{len(tasks)}  {t.third_order_id}")
        names = [t.key] + ([aliases[t.key]] if t.key in aliases else [])
        found: dict[str, dict] = {}
        used = ""
        for name in names:
            for p in by_name.get(norm(name), []):
                found[str(p["product_id"])] = p
            if found:
                used = name
                break
        why = f"图库里有 {len(found)} 个同名商品" if len(found) > 1 else ""
        if len(found) == 1:
            why = try_link(t, next(iter(found.values())), "name",
                           "名称一致" + ("（同一印花）" if used != t.key else ""))
            if not why:
                continue
        remembered = memory.lookup(t)
        if not why and remembered and remembered in by_id:
            why = try_link(t, by_id[remembered], "memory", "之前确认过的配对")
            if not why:
                continue
        leftovers.append((t, why))

    reviews: list[Review] = []
    if leftovers and use_images and not (stop is not None and stop()):
        log(f"还有 {len(leftovers)} 个订单名称对不上，开始识图…")
        index = LibraryIndex(library, CACHE)
        index.build(log, stop)
        paths, imgs, whys = [], [], []
        for i, (t, why) in enumerate(leftovers, 1):
            if stop is not None and stop():
                break
            log(f"下载订单图 {i}/{len(leftovers)}  {t.third_order_id}")
            img_path, img = None, None
            try:
                img_path, img = load_order_image(t.image_url, CACHE)
            except Exception as exc:
                why = why or f"订单图下载失败：{exc}"
            paths.append(img_path)
            imgs.append(img)
            whys.append(why)
        log(f"开始比对 {len(imgs)} 个订单和 {len(index.products)} 个图库商品…")
        all_cands = index.search_many(imgs, log=log, stop=stop) if imgs else []
        for (t, _), img_path, why, cands in zip(leftovers, paths, whys, all_cands):
            best = decide(cands)
            if best is not None:
                err = try_link(t, best.product, "image", f"识图一致（{best.inliers} 个特征点吻合）")
                if not err:
                    continue
                why = f"识图找到了，但{err}"
            if not why:
                why = "识图结果不够确定，请人工确认" if cands else "识图没找到相似的图库商品"
            reviews.append(Review(t, img_path, [c for c in cands if c.inliers >= SHOW_MIN_INLIERS][:3], why))
    else:
        reviews = [Review(t, None, [], why or "图库里没有同名商品") for t, why in leftovers]

    for r in reviews:
        counts["review"] += 1
        top = r.candidates[0] if r.candidates else None
        record(r.task, f"待人工确认（最高 {top.inliers} 点）" if top else "待人工确认",
               top.product if top else None, r.reason)

    folder = Path(root) / "关联记录"
    folder.mkdir(parents=True, exist_ok=True)
    report = folder / f"关联结果_{datetime.now():%Y-%m-%d_%H%M%S}.csv"
    with report.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["商户订单号", "店铺", "SKU", "规格", "关联方式", "图库主题名称", "结果"])
        w.writerows(sorted(rows, key=lambda r: (r[6] != "已关联", r[0])))
    return report, counts, reviews
