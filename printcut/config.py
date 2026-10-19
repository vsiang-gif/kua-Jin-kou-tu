from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

PRINT_AUTO = "auto"
PRINT_GRAPHIC = "graphic"
PRINT_PHOTO = "photo"

BG_TRANSPARENT = "transparent"
BG_WHITE = "white"

SIZE_PRESETS: list[tuple[str, int, int]] = [
    ("4000 × 4000（正方形，默认）", 4000, 4000),
    ("4500 × 5400（Printful / Merch 常用）", 4500, 5400),
    ("3600 × 4800（12×16 英寸 @300）", 3600, 4800),
    ("2400 × 3200（小尺寸）", 2400, 3200),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

REVIEW_DIR_NAME = "待检查"
LOG_FILE_NAME = "处理记录.csv"


@dataclass
class Options:
    print_type: str = PRINT_AUTO
    colors: int = 0  # 0 = 自动
    edge_smooth: int = 3  # 1..5
    despeckle: int = 2  # 0..3
    background: str = BG_TRANSPARENT
    dpi: int = 1200
    width: int = 4000
    height: int = 4000
    margin: float = 0.02
    deskew: bool = True
    dewrinkle: bool = True
    save_svg: bool = True
    skip_existing: bool = True
    review_threshold: int = 60
    workers: int = 3

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str | None) -> "Options":
        opts = cls()
        if not text:
            return opts
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return opts
        valid = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key in valid:
                setattr(opts, key, value)
        return opts


def print_size_cm(pixels: int, dpi: int) -> float:
    return pixels / dpi * 2.54
