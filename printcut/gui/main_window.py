from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..cli import collect
from ..config import (
    BG_TRANSPARENT,
    BG_WHITE,
    PRINT_AUTO,
    PRINT_GRAPHIC,
    PRINT_PHOTO,
    SIZE_PRESETS,
    Options,
    print_size_cm,
)
from ..pipeline import Result
from .widgets import DropList, PreviewPane, load_preview
from .worker import BatchWorker

PATH_ROLE = Qt.UserRole
PRIMARY = "#1F6FEB"

STYLE = f"""
QMainWindow, QWidget#root {{ background: #F3F4F6; }}
QFrame#panel {{ background: #FFFFFF; border: 1px solid #DDE1E6; border-radius: 8px; }}
QLabel#title {{ font-size: 15px; font-weight: 600; color: #1C2128; }}
QLabel#muted {{ color: #66707A; }}
QListWidget {{ border: 1px dashed #C4CAD1; border-radius: 6px; background: #FAFBFC; padding: 4px; }}
QListWidget::item {{ padding: 5px 4px; }}
QListWidget::item:selected {{ background: #E3EEFD; color: #1C2128; }}
QPushButton {{ padding: 6px 12px; border: 1px solid #C4CAD1; border-radius: 6px; background: #FFFFFF; }}
QPushButton:hover {{ background: #F0F2F4; }}
QPushButton#primary {{ background: {PRIMARY}; color: #FFFFFF; border: none; font-size: 15px;
    font-weight: 600; padding: 10px 28px; }}
QPushButton#primary:hover {{ background: #1A5FCC; }}
QPushButton#primary:disabled {{ background: #9DB8E6; }}
QProgressBar {{ border: 1px solid #DDE1E6; border-radius: 6px; background: #FFFFFF; height: 22px;
    text-align: center; }}
QProgressBar::chunk {{ background: {PRIMARY}; border-radius: 5px; }}
QComboBox, QSpinBox, QLineEdit {{ padding: 4px 6px; border: 1px solid #C4CAD1; border-radius: 5px;
    background: #FFFFFF; }}
"""


def _panel() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 14, 14, 14)
    layout.setSpacing(10)
    return frame, layout


def _label(text: str, name: str = "") -> QLabel:
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(True)
    return lab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"PrintCut 印花抠图  v{__version__}")
        self.resize(1440, 860)
        self.settings = QSettings("PrintCut", "PrintCut")
        self.results: dict[str, Result] = {}
        self.thread: QThread | None = None
        self.worker: BatchWorker | None = None
        self.total = 0
        self.done = 0

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_left(), 0)
        body.addWidget(self._build_center(), 1)
        body.addWidget(self._build_right(), 0)
        outer.addLayout(body, 1)
        outer.addWidget(self._build_bottom())

        self._load_settings()
        self._update_size_hint()

    # ---------- layout ----------
    def _build_left(self) -> QWidget:
        frame, lay = _panel()
        frame.setFixedWidth(300)
        lay.addWidget(_label("1. 添加图片", "title"))
        self.file_list = DropList()
        self.file_list.paths_dropped.connect(self.add_paths)
        self.file_list.currentItemChanged.connect(lambda cur, _prev: self.show_item(cur))
        lay.addWidget(self.file_list, 1)

        row = QHBoxLayout()
        add_files = QPushButton("添加图片")
        add_files.clicked.connect(self.pick_files)
        add_dir = QPushButton("添加文件夹")
        add_dir.clicked.connect(self.pick_folder)
        clear = QPushButton("清空")
        clear.clicked.connect(self.clear_files)
        for b in (add_files, add_dir, clear):
            row.addWidget(b)
        lay.addLayout(row)
        self.count_label = _label("共 0 张", "muted")
        lay.addWidget(self.count_label)

        lay.addWidget(_label("输出文件夹", "title"))
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("默认：原图旁边的「印花输出」")
        pick_out = QPushButton("选择")
        pick_out.clicked.connect(self.pick_output)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(pick_out)
        lay.addLayout(out_row)
        return frame

    def _build_center(self) -> QWidget:
        frame, lay = _panel()
        lay.addWidget(_label("2. 效果对比", "title"))
        panes = QHBoxLayout()
        panes.setSpacing(12)
        left = QVBoxLayout()
        left.addWidget(_label("原图", "muted"))
        self.src_pane = PreviewPane("点击左侧图片查看")
        left.addWidget(self.src_pane, 1)
        right = QVBoxLayout()
        right.addWidget(_label("结果（灰白格子 = 透明）", "muted"))
        self.out_pane = PreviewPane("处理后显示在这里", checker=True)
        right.addWidget(self.out_pane, 1)
        panes.addLayout(left, 1)
        panes.addLayout(right, 1)
        lay.addLayout(panes, 1)
        self.detail_label = _label("", "muted")
        lay.addWidget(self.detail_label)
        return frame

    def _build_right(self) -> QWidget:
        frame, lay = _panel()
        frame.setFixedWidth(330)
        lay.addWidget(_label("3. 选项", "title"))

        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(12)
        form.setLabelAlignment(Qt.AlignLeft)

        self.type_combo = QComboBox()
        for text, val in (("自动识别", PRINT_AUTO), ("文字图形", PRINT_GRAPHIC), ("照片插画", PRINT_PHOTO)):
            self.type_combo.addItem(text, val)
        form.addRow("印花类型", self.type_combo)

        self.colors_combo = QComboBox()
        self.colors_combo.addItem("自动", 0)
        for n in range(1, 17):
            self.colors_combo.addItem(f"{n} 色", n)
        form.addRow("颜色数量", self.colors_combo)

        smooth_box = QWidget()
        sl = QHBoxLayout(smooth_box)
        sl.setContentsMargins(0, 0, 0, 0)
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(1, 5)
        self.smooth_slider.setPageStep(1)
        sl.addWidget(_label("锐利", "muted"))
        sl.addWidget(self.smooth_slider, 1)
        sl.addWidget(_label("圆滑", "muted"))
        form.addRow("边缘平滑", smooth_box)

        self.speck_combo = QComboBox()
        for i, text in enumerate(("关闭", "弱", "中（推荐）", "强")):
            self.speck_combo.addItem(text, i)
        form.addRow("去杂点", self.speck_combo)

        bg_box = QWidget()
        bl = QHBoxLayout(bg_box)
        bl.setContentsMargins(0, 0, 0, 0)
        self.bg_transparent = QRadioButton("透明")
        self.bg_white = QRadioButton("白底")
        self.bg_group = QButtonGroup(self)
        self.bg_group.addButton(self.bg_transparent)
        self.bg_group.addButton(self.bg_white)
        bl.addWidget(self.bg_transparent)
        bl.addWidget(self.bg_white)
        bl.addStretch(1)
        form.addRow("背景", bg_box)

        self.dpi_combo = QComboBox()
        self.dpi_combo.addItem("1200 DPI", 1200)
        self.dpi_combo.addItem("300 DPI", 300)
        self.dpi_combo.currentIndexChanged.connect(self._update_size_hint)
        form.addRow("DPI", self.dpi_combo)

        self.size_combo = QComboBox()
        for text, w, h in SIZE_PRESETS:
            self.size_combo.addItem(text, (w, h))
        self.size_combo.currentIndexChanged.connect(self._update_size_hint)
        form.addRow("尺寸", self.size_combo)
        lay.addWidget(form_host)

        self.size_hint = _label("", "muted")
        lay.addWidget(self.size_hint)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #DDE1E6;")
        lay.addWidget(line)

        self.dewrinkle_cb = QCheckBox("去除褶皱阴影")
        self.deskew_cb = QCheckBox("自动拉正")
        self.svg_cb = QCheckBox("文字图形类同时保存 SVG 矢量图")
        self.skip_cb = QCheckBox("跳过已处理过的图片")
        for cb in (self.dewrinkle_cb, self.deskew_cb, self.svg_cb, self.skip_cb):
            lay.addWidget(cb)

        review_row = QHBoxLayout()
        review_row.addWidget(QLabel("低于"))
        self.review_spin = QSpinBox()
        self.review_spin.setRange(0, 100)
        self.review_spin.setSuffix(" 分")
        review_row.addWidget(self.review_spin)
        review_row.addWidget(QLabel("放进「待检查」"))
        review_row.addStretch(1)
        lay.addLayout(review_row)

        workers_row = QHBoxLayout()
        workers_row.addWidget(QLabel("同时处理"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 8)
        self.workers_spin.setSuffix(" 张")
        workers_row.addWidget(self.workers_spin)
        workers_row.addStretch(1)
        lay.addLayout(workers_row)

        reset = QPushButton("恢复默认设置")
        reset.clicked.connect(lambda: self._apply_options(Options()))
        lay.addWidget(reset)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(frame)
        scroll.setFixedWidth(346)
        return scroll

    def _build_bottom(self) -> QWidget:
        frame, _ = _panel()
        lay = QHBoxLayout()
        frame.layout().addLayout(lay)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m")
        self.progress.setValue(0)
        self.status_label = _label("就绪", "muted")
        self.status_label.setMinimumWidth(320)
        self.open_btn = QPushButton("打开输出文件夹")
        self.open_btn.clicked.connect(self.open_output)
        self.start_btn = QPushButton("开始处理")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.toggle_run)
        lay.addWidget(self.progress, 1)
        lay.addWidget(self.status_label)
        lay.addWidget(self.open_btn)
        lay.addWidget(self.start_btn)
        return frame

    # ---------- options ----------
    def _set_combo(self, combo: QComboBox, value) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _apply_options(self, o: Options) -> None:
        self._set_combo(self.type_combo, o.print_type)
        self._set_combo(self.colors_combo, o.colors)
        self.smooth_slider.setValue(int(o.edge_smooth))
        self._set_combo(self.speck_combo, o.despeckle)
        (self.bg_white if o.background == BG_WHITE else self.bg_transparent).setChecked(True)
        self._set_combo(self.dpi_combo, o.dpi)
        self._set_combo(self.size_combo, (o.width, o.height))
        self.dewrinkle_cb.setChecked(o.dewrinkle)
        self.deskew_cb.setChecked(o.deskew)
        self.svg_cb.setChecked(o.save_svg)
        self.skip_cb.setChecked(o.skip_existing)
        self.review_spin.setValue(int(o.review_threshold))
        self.workers_spin.setValue(int(o.workers))
        self._update_size_hint()

    def current_options(self) -> Options:
        w, h = self.size_combo.currentData()
        return Options(
            print_type=self.type_combo.currentData(),
            colors=self.colors_combo.currentData(),
            edge_smooth=self.smooth_slider.value(),
            despeckle=self.speck_combo.currentData(),
            background=BG_WHITE if self.bg_white.isChecked() else BG_TRANSPARENT,
            dpi=self.dpi_combo.currentData(),
            width=w,
            height=h,
            dewrinkle=self.dewrinkle_cb.isChecked(),
            deskew=self.deskew_cb.isChecked(),
            save_svg=self.svg_cb.isChecked(),
            skip_existing=self.skip_cb.isChecked(),
            review_threshold=self.review_spin.value(),
            workers=self.workers_spin.value(),
        )

    def _update_size_hint(self) -> None:
        data = self.size_combo.currentData()
        dpi = self.dpi_combo.currentData()
        if not data or not dpi:
            return
        w, h = data
        self.size_hint.setText(
            f"按 {dpi} DPI 打印约 {print_size_cm(w, dpi):.1f} × {print_size_cm(h, dpi):.1f} cm"
            "（像素不变，DPI 只影响印厂理解的尺寸）"
        )

    def _load_settings(self) -> None:
        self._apply_options(Options.from_json(self.settings.value("options", "", str)))
        self.out_edit.setText(self.settings.value("output", "", str))

    def _save_settings(self) -> None:
        self.settings.setValue("options", self.current_options().to_json())
        self.settings.setValue("output", self.out_edit.text().strip())

    # ---------- files ----------
    def add_paths(self, paths: list[str]) -> None:
        existing = {self.file_list.item(i).data(PATH_ROLE) for i in range(self.file_list.count())}
        added = 0
        for f in collect(paths):
            key = str(f)
            if key in existing:
                continue
            existing.add(key)
            item = QListWidgetItem(f.name)
            item.setData(PATH_ROLE, key)
            item.setToolTip(key)
            self.file_list.addItem(item)
            added += 1
        self.count_label.setText(f"共 {self.file_list.count()} 张")
        if added and self.file_list.currentRow() < 0:
            self.file_list.setCurrentRow(0)

    def pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)"
        )
        if files:
            self.add_paths(files)

    def pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            self.add_paths([folder])

    def pick_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self.out_edit.text())
        if folder:
            self.out_edit.setText(folder)

    def clear_files(self) -> None:
        if self.worker is not None:
            return
        self.file_list.clear()
        self.results.clear()
        self.count_label.setText("共 0 张")
        self.src_pane.set_pixmap(None)
        self.out_pane.set_pixmap(None, "处理后显示在这里")
        self.detail_label.setText("")

    def output_dir(self) -> Path | None:
        text = self.out_edit.text().strip()
        if text:
            return Path(text)
        if self.file_list.count():
            return Path(self.file_list.item(0).data(PATH_ROLE)).parent / "印花输出"
        return None

    def show_item(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        key = item.data(PATH_ROLE)
        self.src_pane.set_pixmap(load_preview(Path(key)), "无法读取这张图片")
        r = self.results.get(key)
        if r is None:
            self.out_pane.set_pixmap(None, "还没处理")
            self.detail_label.setText("")
            return
        if r.error:
            self.out_pane.set_pixmap(None, "处理失败")
            self.detail_label.setText(f"失败：{r.error}  {'；'.join(r.reasons)}")
            return
        self.out_pane.set_pixmap(load_preview(r.output), "找不到结果文件")
        if r.skipped:
            self.detail_label.setText(f"之前已处理：{r.output}")
        else:
            where = "（已放进「待检查」）" if r.needs_review else ""
            self.detail_label.setText(
                f"{r.score} 分{where} · {r.print_type} · {r.seconds:.1f} 秒 · {'；'.join(r.reasons)}"
            )

    def _mark_item(self, r: Result) -> None:
        key = str(r.source)
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(PATH_ROLE) != key:
                continue
            if r.error:
                tag, color = "失败", "#C62828"
            elif r.skipped:
                tag, color = "已跳过", "#66707A"
            elif r.needs_review:
                tag, color = f"待检查 {r.score}分", "#B26A00"
            else:
                tag, color = f"完成 {r.score}分", "#1E7F43"
            item.setText(f"{r.source.name}   [{tag}]")
            item.setForeground(QColor(color))
            if item is self.file_list.currentItem():
                self.show_item(item)
            break

    # ---------- run ----------
    def toggle_run(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.start_btn.setEnabled(False)
            self.status_label.setText("正在停止，等当前几张做完…")
            return
        files = [Path(self.file_list.item(i).data(PATH_ROLE)) for i in range(self.file_list.count())]
        if not files:
            QMessageBox.information(self, "提示", "请先添加图片或文件夹。")
            return
        out = self.output_dir()
        assert out is not None
        self._save_settings()
        self.total, self.done = len(files), 0
        self.progress.setRange(0, self.total)
        self.progress.setValue(0)
        self.status_label.setText("正在加载 AI 模型…")
        self.start_btn.setText("停止")

        self.thread = QThread(self)
        self.worker = BatchWorker(files, self.current_options(), out)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.status_label.setText)
        self.worker.item_done.connect(self._on_item)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_item(self, r: Result) -> None:
        self.results[str(r.source)] = r
        self.done += 1
        self.progress.setValue(self.done)
        self.status_label.setText(f"{r.source.name}  {r.seconds:.1f} 秒")
        self._mark_item(r)

    def _on_finished(self, ok: int, review: int, failed: int) -> None:
        self.worker = None
        self.thread = None
        self.start_btn.setText("开始处理")
        self.start_btn.setEnabled(True)
        self.status_label.setText(f"完成：合格 {ok} 张，待检查 {review} 张，失败 {failed} 张")

    def open_output(self) -> None:
        out = self.output_dir()
        if out is None:
            return
        out.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def closeEvent(self, event) -> None:
        self._save_settings()
        if self.worker is not None:
            self.worker.stop()
            if self.thread is not None:
                self.thread.quit()
                self.thread.wait(3000)
        super().closeEvent(event)


def main() -> None:
    if sys.platform == "win32":
        os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setApplicationName("PrintCut")
    font = app.font()
    font.setFamilies(["Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC"])
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
