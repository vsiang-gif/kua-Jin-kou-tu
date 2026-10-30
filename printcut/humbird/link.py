"""Auto 关联商品: bind each 待设计 order to the library product whose 主题名称 is its <WB订单号>_<SKU>.

Only exact name matches are bound, and only after Humbird's own spec matching confirms the
order's colour and size, so a wrong product is never linked. Everything else is reported for
manual handling.
"""
from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

from .client import DesignTask, HumbirdClient, HumbirdError


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


def auto_link(client: HumbirdClient, root: Path, log=print, stop=None) -> tuple[Path, int, int, int]:
    """Returns (report path, linked, not found, needs manual)."""
    log("正在拉取「待设计」订单…")
    tasks = client.design_tasks()
    library = client.library_products(log)
    by_name: dict[str, list[dict]] = {}
    for p in library:
        if p.get("is_valid") is False:
            continue
        by_name.setdefault(norm(p.get("theme") or ""), []).append(p)
    aliases = load_aliases(root)

    rows: list[list[str]] = []
    linked = missing = manual = 0
    for i, t in enumerate(tasks, 1):
        if stop is not None and stop():
            break
        names = [t.key] + ([aliases[t.key]] if t.key in aliases else [])
        found: dict[str, dict] = {}
        used = ""
        for name in names:
            for p in by_name.get(norm(name), []):
                found[str(p["product_id"])] = p
            if found:
                used = name
                break
        product, result = None, ""
        if not found:
            missing += 1
            result = "图库里没有同名商品"
        elif len(found) > 1:
            manual += 1
            result = f"图库里有 {len(found)} 个同名商品，请手动选择"
        else:
            product = next(iter(found.values()))
            try:
                link_one(client, t, product)
                linked += 1
                result = "已关联" + ("（同一印花）" if used != t.key else "")
            except HumbirdError as exc:
                manual += 1
                result = str(exc)
        rows.append([t.third_order_id, t.store_name, t.sku, t.spec, used or t.key,
                     str(product.get("theme") if product else ""), result])
        log(f"关联中 {i}/{len(tasks)}  {t.third_order_id}  {result}")

    folder = Path(root) / "关联记录"
    folder.mkdir(parents=True, exist_ok=True)
    report = folder / f"关联结果_{datetime.now():%Y-%m-%d_%H%M%S}.csv"
    with report.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["商户订单号", "店铺", "SKU", "规格", "查找名称", "图库主题名称", "结果"])
        w.writerows(sorted(rows, key=lambda r: (r[6] != "已关联", r[0])))
    return report, linked, missing, manual
