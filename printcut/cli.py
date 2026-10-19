"""Headless batch mode: python -m printcut.cli <图片或文件夹> -o <输出文件夹>"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import IMAGE_EXTENSIONS, Options
from .pipeline import Processor, get_parser, get_segmenter, get_upscaler


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        elif p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return files


def _force_utf8_console() -> None:
    # Windows consoles default to cp1252/gbk; Chinese log lines would raise UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _force_utf8_console()
    ap = argparse.ArgumentParser(description="PrintCut 批量抠印花")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--type", choices=["auto", "graphic", "photo"], default="auto")
    ap.add_argument("--colors", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=1200)
    ap.add_argument("--size", default="4000x4000")
    ap.add_argument("--white", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    opts = Options(print_type=args.type, colors=args.colors, dpi=args.dpi, width=w, height=h,
                   background="white" if args.white else "transparent", skip_existing=False)
    seg = get_segmenter()
    seg.load(print)
    parser = get_parser()
    parser.load(print)
    proc = Processor(opts, Path(args.output), seg, get_upscaler(), parser if parser.available else None)
    files = collect(args.inputs)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(proc.process, files), 1):
            status = r.error or f"{r.score} 分"
            print(f"[{i}/{len(files)}] {r.source.name} -> {status} {r.print_type} {r.seconds:.1f}s {'；'.join(r.reasons)}")


if __name__ == "__main__":
    main()
