"""Find which 定制商品 in the library carries the same print as an order's product photo.

The library thumbnails are mockups of the same model wearing the shirt; the order photo is the
seller's own picture. Global similarity (hashes, embeddings) confuses prints that share colours or
layout, so this uses SIFT keypoints + RANSAC geometric verification instead: only an identical
artwork produces dozens of keypoints that agree on one consistent transform. A true match usually
scores 50-150 inliers while unrelated prints stay below ~10, which leaves a wide safety margin.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .client import USER_AGENT, download

LIB_SIDE = 500
ORDER_SIDE = 1000
LIB_FEATURES = 600
ORDER_FEATURES = 2500
RATIO = 0.75
RANSAC_PX = 8.0
MIN_GOOD = 8
CHUNK = 150

# Auto-link only when the best product is unmistakable.
AUTO_MIN_INLIERS = 25
AUTO_MAX_RUNNER_UP = 0.4
SHOW_MIN_INLIERS = 8


def library_image_url(product: dict, side: int = LIB_SIDE) -> str:
    path = str(product.get("product_image") or "")
    if not path:
        return ""
    if path.startswith("http"):
        base = path.split("?")[0]
    else:
        path = path.lstrip("/")
        bucket = path.split("/", 1)[0]
        base = f"https://obs-{bucket}.hihumbird.com/{path}"
    return f"{base}?x-image-process=style/s{side}"


def _read_gray(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def _resize_long(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    k = side / max(h, w)
    if abs(k - 1) < 0.02:
        return img
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)


def library_features(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints on the torsos only: face, legs and background are identical across all mockups."""
    img = _resize_long(img, LIB_SIDE)
    h, w = img.shape
    mask = np.zeros_like(img)
    mask[int(h * 0.15):int(h * 0.82), :] = 255
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    big_mask = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    kps, desc = cv2.SIFT_create(nfeatures=LIB_FEATURES).detectAndCompute(big, big_mask)
    if desc is None or not kps:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    pts = np.float32([k.pt for k in kps])
    return pts, np.clip(desc, 0, 255).astype(np.uint8)


def order_features(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = _resize_long(img, ORDER_SIDE)
    kps, desc = cv2.SIFT_create(nfeatures=ORDER_FEATURES).detectAndCompute(img, None)
    if desc is None or not kps:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.float32([k.pt for k in kps]), desc.astype(np.float32)


@dataclass
class Candidate:
    product: dict
    inliers: int
    image: Path


class LibraryIndex:
    """SIFT features of every library thumbnail, cached on disk so only new products are fetched."""

    def __init__(self, products: list[dict], cache: Path) -> None:
        self.products = [p for p in products if p.get("product_image") and p.get("is_valid") is not False]
        self.cache = Path(cache)
        self.images = self.cache / "图库缩略图"
        self.feats = self.cache / "图库特征"
        self.pts: list[np.ndarray] = []
        self.desc: list[np.ndarray] = []

    def _stem(self, p: dict) -> str:
        digest = hashlib.md5(str(p.get("product_image")).encode()).hexdigest()[:10]
        return f"{p['product_id']}_{digest}"

    def image_path(self, p: dict) -> Path:
        return self.images / f"{self._stem(p)}.jpg"

    def _load_one(self, p: dict) -> tuple[np.ndarray, np.ndarray]:
        feat = self.feats / f"{self._stem(p)}.npz"
        if feat.is_file():
            try:
                with np.load(feat) as z:
                    return z["pts"], z["desc"]
            except (OSError, ValueError, KeyError):
                pass
        img_path = self.image_path(p)
        if not img_path.is_file():
            download(library_image_url(p), img_path, timeout=30)
        img = _read_gray(img_path)
        if img is None:
            raise ValueError("bad image")
        pts, desc = library_features(img)
        feat.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(feat, pts=pts, desc=desc)
        return pts, desc

    def build(self, log=print, stop=None) -> None:
        self.images.mkdir(parents=True, exist_ok=True)
        empty = (np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))
        results: list[tuple[np.ndarray, np.ndarray]] = [empty] * len(self.products)
        failed = 0

        def work(i: int) -> None:
            nonlocal failed
            if stop is not None and stop():
                return
            try:
                results[i] = self._load_one(self.products[i])
            except Exception:
                failed += 1

        total = len(self.products)
        with ThreadPoolExecutor(max_workers=8) as ex:
            for n, _ in enumerate(ex.map(work, range(total)), 1):
                if n % 50 == 0 or n == total:
                    log(f"正在读取图库印花特征 {n}/{total}（第一次较慢，之后会用缓存）…")
        self.pts = [r[0] for r in results]
        self.desc = [r[1] for r in results]
        if failed:
            log(f"有 {failed} 个图库缩略图下载失败，这些商品不参与识图")

    def search(self, order_img: np.ndarray, top: int = 3) -> list[Candidate]:
        return self.search_many([order_img], top)[0]

    def search_many(self, order_imgs: list[np.ndarray | None], top: int = 3, log=None, stop=None) -> list[list[Candidate]]:
        """Candidates for every order image, scanning the library only once for the whole batch."""
        n_orders = len(order_imgs)
        feats = [order_features(img) if img is not None else (np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
                 for img in order_imgs]
        usable = [k for k, (_, d) in enumerate(feats) if len(d) >= MIN_GOOD]
        if not usable:
            return [[] for _ in range(n_orders)]
        all_desc = np.concatenate([feats[k][1] for k in usable]).astype(np.float32)
        o_owner = np.concatenate([np.full(len(feats[k][1]), k, np.int32) for k in usable])
        o_local = np.concatenate([np.arange(len(feats[k][1]), dtype=np.int32) for k in usable])
        flann = cv2.flann_Index(all_desc, dict(algorithm=1, trees=4))
        K = min(6, len(all_desc))

        scores = np.zeros((len(self.products), n_orders), dtype=np.int32)
        total = len(self.products)
        for start in range(0, total, CHUNK):
            if stop is not None and stop():
                break
            if log is not None and start and start % (CHUNK * 4) == 0:
                log(f"识图比对中 {start}/{total}…")
            idx = [i for i in range(start, min(start + CHUNK, total)) if len(self.desc[i]) >= MIN_GOOD]
            if not idx:
                continue
            query = np.concatenate([self.desc[i] for i in idx]).astype(np.float32)
            owner = np.concatenate([np.full(len(self.desc[i]), i, np.int32) for i in idx])
            local = np.concatenate([np.arange(len(self.desc[i]), dtype=np.int32) for i in idx])
            nn, dist = flann.knnSearch(query, K, params=dict(checks=32))
            own = o_owner[nn]
            rows = np.arange(len(query))
            keys, qs, ts = [], [], []
            # Ratio test per order: the second-best neighbour must come from the same order image,
            # otherwise two orders with the same print would cancel each other out.
            for c in range(K):
                oc = own[:, c]
                first = ~(own[:, :c] == oc[:, None]).any(1) if c else np.ones(len(query), bool)
                later = own[:, c + 1:] == oc[:, None]
                nxt = np.where(later.any(1), later.argmax(1) + c + 1, K - 1) if c < K - 1 else np.full(len(query), K - 1)
                # distances are squared L2, so the ratio is squared too
                ok = first & (dist[:, c] < (RATIO ** 2) * dist[rows, nxt])
                if ok.any():
                    keys.append(owner[ok].astype(np.int64) * n_orders + oc[ok])
                    qs.append(local[ok])
                    ts.append(o_local[nn[ok, c]])
            if not keys:
                continue
            key, q, t = np.concatenate(keys), np.concatenate(qs), np.concatenate(ts)
            sort = np.argsort(key, kind="stable")
            key, q, t = key[sort], q[sort], t[sort]
            uniq, begin, count = np.unique(key, return_index=True, return_counts=True)
            for kk, b, cnt in zip(uniq.tolist(), begin.tolist(), count.tolist()):
                if cnt < MIN_GOOD:
                    continue
                i, o = divmod(kk, n_orders)
                pairs = list(zip(q[b:b + cnt].tolist(), t[b:b + cnt].tolist()))
                scores[i, o] = _verify(self.pts[i], feats[o][0], pairs)

        result: list[list[Candidate]] = []
        for o in range(n_orders):
            col = scores[:, o]
            best = np.argsort(-col)[:max(top, 2)]
            result.append([Candidate(self.products[i], int(col[i]), self.image_path(self.products[i]))
                           for i in best if col[i] > 0])
        return result


def _verify(lib_pts: np.ndarray, order_pts: np.ndarray, pairs: list[tuple[int, int]]) -> int:
    """Number of matches consistent with a single plausible homography (0 if degenerate)."""
    src = lib_pts[[a for a, _ in pairs]]
    dst = order_pts[[b for _, b in pairs]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if H is None or mask is None:
        return 0
    det = np.linalg.det(H[:2, :2])
    if not (0.02 < abs(det) < 50) or det < 0:
        return 0
    inl = mask.ravel().astype(bool)
    n = int(inl.sum())
    if n < MIN_GOOD:
        return n
    # Inliers bunched into one tiny spot are usually a shared logo/texture, not the whole artwork.
    spread = src[inl].max(0) - src[inl].min(0)
    if min(spread) < 20:
        return min(n, MIN_GOOD - 1)
    return n


def decide(cands: list[Candidate]) -> Candidate | None:
    """The best candidate only when it is unmistakable; otherwise None (needs a human)."""
    if not cands or cands[0].inliers < AUTO_MIN_INLIERS:
        return None
    runner_up = cands[1].inliers if len(cands) > 1 else 0
    if runner_up > cands[0].inliers * AUTO_MAX_RUNNER_UP:
        return None
    return cands[0]


def load_order_image(url: str, cache: Path) -> tuple[Path, np.ndarray | None]:
    clean = url.split("?")[0]
    suffix = Path(clean).suffix or ".jpg"
    path = Path(cache) / "订单图" / f"{hashlib.md5(clean.encode()).hexdigest()[:16]}{suffix}"
    if not path.is_file():
        download(url, path, timeout=60)
    return path, _read_gray(path)


__all__ = ["Candidate", "LibraryIndex", "decide", "load_order_image", "library_image_url", "USER_AGENT"]
