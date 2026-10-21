from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import Options
from ..humbird import HumbirdClient, HumbirdError, Record, Store, parse_token, process_task
from ..humbird.job import FINAL, ST_APPROVED, ST_FAILED, ST_MANUAL, ST_REJECTED, ST_REVIEW
from ..pipeline import Processor, get_parser, get_segmenter, get_upscaler
from .widgets import PreviewPane, load_preview

KEY_ROLE = Qt.UserRole
STATUS_COLORS = {
    ST_REVIEW: "#1F6FEB",
    ST_MANUAL: "#B26A00",
    ST_FAILED: "#C62828",
    ST_APPROVED: "#1E7F43",
    ST_REJECTED: "#66707A",
}
COLUMNS = ["剩余发货", "商户订单号", "店铺", "SKU", "规格", "回贴比对", "抠图分", "状态"]


def _remaining(seconds: int) -> str:
    if seconds >= 10**8:
        return "-"
    if seconds <= 0:
        return "已超时"
    return f"{seconds // 3600}时{seconds % 3600 // 60:02d}分"


class OrderWorker(QObject):
    log = Signal(str)
    record = Signal(object)
    finished = Signal(str)

    def __init__(self, token: str, opts: Options, store: Store, redo: bool) -> None:
        super().__init__()
        self.token, self.opts, self.store, self.redo = token, opts, store, redo
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @Slot()
    def run(self) -> None:
        summary = ""
        try:
            client = HumbirdClient(self.token)
            self.log.emit("正在拉取「待设计」订单…")
            tasks = client.design_tasks()
            todo = [t for t in tasks if self.redo or t.key not in self.store.records
                    or self.store.records[t.key].status == ST_FAILED]
            self.log.emit(f"共 {len(tasks)} 个待设计商品，本次处理 {len(todo)} 个。正在加载 AI 模型…")
            if not todo:
                summary = f"没有新订单（共 {len(tasks)} 个，都已处理过）"
                return
            seg = get_segmenter()
            seg.load(self.log.emit)
            parser = get_parser()
            parser.load(self.log.emit)
            proc = Processor(self.opts, self.store.root / "印花", seg, get_upscaler(),
                             parser if parser.available else None)
            counts = {ST_REVIEW: 0, ST_MANUAL: 0, ST_FAILED: 0}
            with ThreadPoolExecutor(max_workers=max(1, int(self.opts.workers))) as ex:
                futures = {ex.submit(self._one, client, proc, t): t for t in todo}
                for i, fut in enumerate(as_completed(futures), 1):
                    rec = fut.result()
                    if rec is None:
                        continue
                    counts[rec.status] = counts.get(rec.status, 0) + 1
                    self.record.emit(rec)
                    self.log.emit(f"{i}/{len(todo)}  {rec.third_order_id}  {rec.status}")
            summary = (f"完成：可审核 {counts[ST_REVIEW]} 个，需人工处理 {counts[ST_MANUAL]} 个，"
                       f"失败 {counts[ST_FAILED]} 个")
            if self._stop.is_set():
                summary = "已停止。" + summary
        except HumbirdError as exc:
            summary = str(exc)
        except Exception as exc:
            traceback.print_exc()
            summary = f"出错：{exc}"
        finally:
            self.finished.emit(summary)

    def _one(self, client: HumbirdClient, proc: Processor, task) -> Record | None:
        if self._stop.is_set():
            return None
        return process_task(client, proc, self.store, task)


class HumbirdWindow(QDialog):
    def __init__(self, opts: Options, out_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("蜂鸟订单：自动抠印花 + 人工审核")
        self.resize(1500, 900)
        self.opts = opts
        self.store = Store(Path(out_root) / "蜂鸟订单")
        self.settings = QSettings("PrintCut", "PrintCut")
        self.thread: QThread | None = None
        self.worker: OrderWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        top = QHBoxLayout()
        top.addWidget(QLabel("登录令牌"))
        self.token_edit = QLineEdit(self.settings.value("humbird_token", "", str))
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setPlaceholderText("粘贴 authorization 的值（eyJ 开头），或整段 cURL 也可以")
        top.addWidget(self.token_edit, 1)
        self.redo_box = QCheckBox("重新处理已做过的")
        top.addWidget(self.redo_box)
        self.run_btn = QPushButton("拉取待设计订单并抠图")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.toggle_run)
        top.addWidget(self.run_btn)
        root.addLayout(top)

        self.status = QLabel(f"结果保存在：{self.store.root}")
        self.status.setObjectName("muted")
        root.addWidget(self.status)

        split = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.show_selected)
        split.addWidget(self.table)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        panes = QHBoxLayout()
        self.cut_pane = PreviewPane("抠出的印花", checker=True)
        self.overlay_pane = PreviewPane("左：商品图　右：印花贴回去的效果")
        panes.addWidget(self.cut_pane, 2)
        panes.addWidget(self.overlay_pane, 3)
        rl.addLayout(panes, 1)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rl.addWidget(self.detail)
        buttons = QHBoxLayout()
        self.folder_btn = QPushButton("打开文件夹")
        self.folder_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.root))))
        self.reject_btn = QPushButton("退回（不用）")
        self.reject_btn.clicked.connect(lambda: self.decide(False))
        self.approve_btn = QPushButton("通过")
        self.approve_btn.setObjectName("primary")
        self.approve_btn.clicked.connect(lambda: self.decide(True))
        buttons.addWidget(self.folder_btn)
        self.compare_btn = QPushButton("对比灵图结果")
        self.compare_btn.setToolTip("选择存放灵图抠图结果的文件夹，生成对比报告（文件名要和订单号对应）")
        self.compare_btn.clicked.connect(self.compare_reference)
        buttons.addWidget(self.compare_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.reject_btn)
        buttons.addWidget(self.approve_btn)
        rl.addLayout(buttons)
        split.addWidget(right)
        split.setSizes([620, 880])
        root.addWidget(split, 1)

        for rec in sorted(self.store.records.values(), key=lambda r: r.remaining_seconds):
            self.upsert(rec)
        self._set_buttons(None)

    # ---------- table ----------
    def compare_reference(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择灵图抠图结果所在的文件夹")
        if not folder:
            return
        from ..compare import run as run_compare

        self.compare_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            report = run_compare(self.store.root, Path(folder), log=lambda _: QApplication.processEvents())
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.warning(self, "提示", f"对比失败：{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.compare_btn.setEnabled(True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(report)))

    def _row_of(self, key: str) -> int:
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(KEY_ROLE) == key:
                return row
        return -1

    def upsert(self, rec: Record) -> None:
        row = self._row_of(rec.key)
        if row < 0:
            row = self.table.rowCount()
            self.table.insertRow(row)
        values = [
            _remaining(rec.remaining_seconds), rec.third_order_id, rec.store_name, rec.sku, rec.spec,
            f"{rec.match_score:.0%}" if rec.match_score else "-", str(rec.cut_score or "-"), rec.status,
        ]
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            if col == 0:
                item.setData(KEY_ROLE, rec.key)
            if col == len(values) - 1:
                item.setForeground(QColor(STATUS_COLORS.get(rec.status, "#1C2128")))
            self.table.setItem(row, col, item)
        if row == self.table.currentRow():
            self.show_selected()

    def current(self) -> Record | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        return self.store.records.get(self.table.item(row, 0).data(KEY_ROLE))

    def show_selected(self) -> None:
        rec = self.current()
        self._set_buttons(rec)
        if rec is None:
            return
        self.cut_pane.set_pixmap(load_preview(Path(rec.output)) if rec.output else None, "没有生成印花")
        self.overlay_pane.set_pixmap(
            load_preview(Path(rec.overlay or rec.source)) if (rec.overlay or rec.source) else None, "没有商品图")
        why = "；".join(rec.reasons) or "无"
        self.detail.setText(f"{rec.status} · {rec.third_order_id} · SKU {rec.sku} · {rec.spec}\n{rec.name}\n说明：{why}")

    def _set_buttons(self, rec: Record | None) -> None:
        can = rec is not None and bool(rec.output) and rec.status not in FINAL
        self.approve_btn.setEnabled(can)
        self.reject_btn.setEnabled(rec is not None and rec.status not in FINAL)

    def decide(self, approve: bool) -> None:
        rec = self.current()
        if rec is None:
            return
        if approve and rec.status == ST_MANUAL:
            ok = QMessageBox.question(self, "确认", "这张没有通过自动比对，确定印花没问题吗？")
            if ok != QMessageBox.Yes:
                return
        self.upsert(self.store.decide(rec.key, approve))
        row = self.table.currentRow()
        if row + 1 < self.table.rowCount():
            self.table.selectRow(row + 1)

    # ---------- run ----------
    def toggle_run(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.run_btn.setEnabled(False)
            self.status.setText("正在停止，等当前几张做完…")
            return
        try:
            token = parse_token(self.token_edit.text())
        except HumbirdError as exc:
            QMessageBox.warning(self, "提示", str(exc))
            return
        self.token_edit.setText(token)
        self.settings.setValue("humbird_token", token)
        self.run_btn.setText("停止")
        self.thread = QThread(self)
        self.worker = OrderWorker(token, self.opts, self.store, self.redo_box.isChecked())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.status.setText)
        self.worker.record.connect(self.upsert)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_finished(self, summary: str) -> None:
        self.worker = None
        self.thread = None
        self.run_btn.setText("拉取待设计订单并抠图")
        self.run_btn.setEnabled(True)
        self.status.setText(summary)
        if "令牌" in summary:
            QMessageBox.warning(self, "登录失效", summary)

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.stop()
            if self.thread is not None:
                self.thread.quit()
                self.thread.wait(3000)
        super().closeEvent(event)
