from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ..config import Options
from ..pipeline import Processor, Result, get_parser, get_segmenter, get_upscaler


class BatchWorker(QObject):
    log = Signal(str)
    item_done = Signal(object)  # Result
    finished = Signal(int, int, int)  # ok, review, failed

    def __init__(self, files: list[Path], opts: Options, out_dir: Path) -> None:
        super().__init__()
        self.files = files
        self.opts = opts
        self.out_dir = out_dir
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _one(self, proc: Processor, path: Path) -> Result | None:
        if self._stop.is_set():
            return None
        return proc.process(path)

    @Slot()
    def run(self) -> None:
        ok = review = failed = 0
        try:
            seg = get_segmenter()
            seg.load(self.log.emit)
            parser = get_parser()
            parser.load(self.log.emit)
            up = get_upscaler()
            if not up.available:
                self.log.emit("未找到 Real-ESRGAN，照片类印花将用普通放大")
            self.out_dir.mkdir(parents=True, exist_ok=True)
            proc = Processor(self.opts, self.out_dir, seg, up, parser if parser.available else None)

            todo: list[Path] = []
            for f in self.files:
                existing = proc.existing_output(f) if self.opts.skip_existing else None
                if existing is not None:
                    self.item_done.emit(Result(source=f, output=existing, skipped=True))
                else:
                    todo.append(f)
            if len(todo) < len(self.files):
                self.log.emit(f"跳过已处理的 {len(self.files) - len(todo)} 张")

            workers = max(1, int(self.opts.workers))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(self._one, proc, f) for f in todo]
                for fut in as_completed(futures):
                    r = fut.result()
                    if r is None:
                        continue
                    if r.error:
                        failed += 1
                    elif r.needs_review:
                        review += 1
                    else:
                        ok += 1
                    self.item_done.emit(r)
            if self._stop.is_set():
                self.log.emit("已停止")
        except Exception as exc:
            traceback.print_exc()
            self.log.emit(f"出错：{exc}")
        finally:
            self.finished.emit(ok, review, failed)
