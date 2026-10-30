from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..humbird import HumbirdClient, HumbirdError
from ..humbird.link import Memory, Review, link_and_remember

ORDER_SIDE = 420
CAND_SIDE = 260


def _pixmap(path: Path | None, side: int) -> QPixmap | None:
    if not path or not Path(path).exists():
        return None
    pm = QPixmap(str(path))
    if pm.isNull():
        return None
    return pm.scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _image_label(path: Path | None, side: int) -> QLabel:
    label = QLabel()
    label.setAlignment(Qt.AlignCenter)
    label.setFixedSize(side, side)
    label.setStyleSheet("background:#F2F2F2;border:1px solid #DDD;")
    pm = _pixmap(path, side)
    if pm is not None:
        label.setPixmap(pm)
    else:
        label.setText("没有图片")
    return label


class LinkReviewDialog(QDialog):
    """Show each uncertain order next to its closest library products; one click links and remembers it."""

    def __init__(self, token: str, root: Path, reviews: list[Review], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("识图待确认：点选正确的图库商品")
        self.client = HumbirdClient(token)
        self.memory = Memory(root)
        self.reviews = [r for r in reviews if r.candidates]
        self.index = 0
        self.linked = 0
        self.skipped = 0

        self.header = QLabel()
        self.header.setWordWrap(True)
        self.reason = QLabel()
        self.reason.setStyleSheet("color:#B26A00;")
        self.reason.setWordWrap(True)

        self.body = QHBoxLayout()
        self.body.setSpacing(16)

        skip_btn = QPushButton("都不是，跳过")
        skip_btn.clicked.connect(self._skip)
        close_btn = QPushButton("结束")
        close_btn.clicked.connect(self.accept)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(skip_btn)
        footer.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addWidget(self.reason)
        layout.addLayout(self.body, 1)
        layout.addLayout(footer)
        self._show()

    def summary(self) -> str:
        left = len(self.reviews) - self.linked - self.skipped
        return f"人工确认：关联 {self.linked} 个，跳过 {self.skipped} 个，未处理 {left} 个。"

    def _clear(self) -> None:
        while self.body.count():
            item = self.body.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def _show(self) -> None:
        self._clear()
        if self.index >= len(self.reviews):
            self.accept()
            return
        r = self.reviews[self.index]
        t = r.task
        self.header.setText(
            f"<b>第 {self.index + 1}/{len(self.reviews)} 个</b>　订单号 {t.third_order_id}　"
            f"店铺 {t.store_name}　SKU {t.sku}　规格 {t.spec}")
        self.reason.setText(r.reason)

        order_box = QVBoxLayout()
        order_box.addWidget(QLabel("<b>订单图</b>"))
        order_box.addWidget(_image_label(r.order_image, ORDER_SIDE))
        order_box.addStretch(1)
        order_w = QWidget()
        order_w.setLayout(order_box)
        self.body.addWidget(order_w)

        for c in r.candidates:
            box = QVBoxLayout()
            box.addWidget(_image_label(c.image, CAND_SIDE))
            name = QLabel(f"{c.product.get('theme') or '(无名称)'}\n{c.inliers} 个特征点吻合")
            name.setWordWrap(True)
            name.setMaximumWidth(CAND_SIDE)
            box.addWidget(name)
            btn = QPushButton("就是这个")
            btn.clicked.connect(lambda _=False, p=c.product: self._pick(p))
            box.addWidget(btn)
            box.addStretch(1)
            w = QWidget()
            w.setLayout(box)
            self.body.addWidget(w)
        self.body.addStretch(1)

    def _pick(self, product: dict) -> None:
        try:
            link_and_remember(self.client, self.memory, self.reviews[self.index].task, product)
        except HumbirdError as exc:
            QMessageBox.warning(self, "关联失败", str(exc))
            return
        self.linked += 1
        self.index += 1
        self._show()

    def _skip(self) -> None:
        self.skipped += 1
        self.index += 1
        self._show()
