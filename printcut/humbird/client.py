from __future__ import annotations

import hashlib
import hmac
import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

API = "https://apigw.hihumbird.com"
SITE = "https://www.ymtpod.com"
STATUS_WAIT_DESIGN = "-2"
PAGE_SIZE = 20
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
)
_JWT = re.compile(r"eyJ[\w-]+\.[\w-]+\.[\w-]+")


class HumbirdError(Exception):
    pass


def parse_token(text: str) -> str:
    """Accepts a bare token, 'Bearer …', or a whole pasted cURL command."""
    m = _JWT.search(text or "")
    if not m:
        if re.fullmatch(r"\s*[A-Za-z0-9+/]{16,}={0,2}\s*", text or ""):
            raise HumbirdError(
                "这是「开放平台 API Key」，不是登录令牌。\n"
                "请在蜂鸟后台按 F12 → Network，点任意一个请求 → Headers，"
                "复制 authorization 后面以 eyJ 开头的那一长串粘贴进来。"
            )
        raise HumbirdError("没找到登录令牌（应该是以 eyJ 开头的一长串）")
    return m.group(0)


def sign(token: str, method: str, url: str, body: str, stamp: str, nonce: str) -> str:
    """Same scheme as the Humbird web app: HmacSHA256 over the lexically sorted pieces."""
    query = sorted(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    params = "&".join(f"{k}={v}" for k, v in query)
    parts = [stamp, nonce, params] if method.upper() == "GET" else [stamp, nonce, params, body]
    message = "".join(sorted(parts))
    return hmac.new(token.encode(), message.encode(), hashlib.sha256).hexdigest()


@dataclass
class DesignTask:
    order_id: str
    third_order_id: str
    store_name: str
    sku: str
    color: str
    size: str
    spec: str
    name: str
    image_url: str
    remaining_seconds: int

    @property
    def key(self) -> str:
        return f"{self.third_order_id}_{self.sku}"


class HumbirdClient:
    def __init__(self, token: str, timeout: float = 30) -> None:
        self.token = parse_token(token)
        self.timeout = timeout

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = API + path
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) if payload is not None else ""
        stamp = str(int(time.time() * 1000))
        nonce = str(random.randint(0, 999998))
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json;charset=UTF-8",
            "Origin": SITE,
            "Referer": SITE + "/",
            "User-Agent": USER_AGENT,
            "sign": sign(self.token, method, url, body, stamp, nonce),
            "stamp": stamp,
            "nonce": nonce,
        }
        req = urllib.request.Request(url, data=body.encode() if body else None, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise HumbirdError("登录令牌已失效，请在蜂鸟后台重新登录后复制新的令牌") from exc
            raise HumbirdError(f"蜂鸟接口返回 HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise HumbirdError(f"连不上蜂鸟：{exc.reason}") from exc
        code = data.get("result_code")
        if code != 200:
            if code in (401, 403):
                raise HumbirdError("登录令牌已失效，请在蜂鸟后台重新登录后复制新的令牌")
            raise HumbirdError(f"蜂鸟接口报错：{data.get('msg') or code}")
        return data.get("data") or {}

    def pending_orders(self, limit_pages: int = 50) -> list[dict]:
        orders: list[dict] = []
        for page in range(1, limit_pages + 1):
            data = self.request("POST", "/oc/v2/orders/page", {
                "page": page, "page_size": PAGE_SIZE, "expand": False, "history_order": True,
                "is_fba": False, "platform_list": [], "sorting_center_app_id_list": [],
                "receiver_warehouse_id_list": [], "store_list": [], "site_id_list": [],
                "status_list": [STATUS_WAIT_DESIGN], "order_model_list": [], "third_status": [],
                "performance_status_list": [], "logistics_method_id_list": [], "third_tag_list": [],
                "sort": [{"sort_by": "import_time", "sort_type": 2}],
            })
            batch = data.get("list") or []
            orders.extend(batch)
            if not batch or len(orders) >= int(data.get("total") or 0):
                break
        return orders

    def design_tasks(self) -> list[DesignTask]:
        tasks: list[DesignTask] = []
        for o in self.pending_orders():
            wanted = set(o.get("wait_design_sku_list") or [])
            seen: set[str] = set()
            for item in o.get("order_items") or []:
                t = item.get("third_order_item") or {}
                sku = str(item.get("third_sku") or t.get("goods_sku_id") or "")
                if not sku or (wanted and sku not in wanted) or sku in seen:
                    continue
                seen.add(sku)
                tasks.append(DesignTask(
                    order_id=str(o.get("id", "")),
                    third_order_id=str(o.get("third_order_id", "")),
                    store_name=str(o.get("store_name") or ""),
                    sku=sku,
                    color=str(t.get("color") or ""),
                    size=str(t.get("size") or ""),
                    spec=str(t.get("spec") or ""),
                    name=str(t.get("sku_name") or ""),
                    image_url=str(t.get("image_path") or ""),
                    remaining_seconds=_int(o.get("remaining_deadline_delivery_time")),
                ))
        tasks.sort(key=lambda x: x.remaining_seconds)
        return tasks

    def match_sku(self, sku: str) -> dict:
        return self.request("PATCH", "/oc/v1/product_relation/matching", {"third_sku": sku})


def _int(value, default: int = 10**9) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def download(url: str, target: Path, timeout: float = 60) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Referer": SITE + "/"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target
