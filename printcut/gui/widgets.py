from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QListWidget, QWidget

PREVIEW_MAX_SIDE = 1400


def load_preview(path: Path | None) -> QPixmap | None:
    if path is None or not Path(path).is_file():
        return None
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > PREVIEW_MAX_SIDE:
        reader.setScaledSize(size.scaled(PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE, Qt.KeepAspectRatio))
    image = reader.read()
    return None if image.isNull() else QPixmap.fromImage(image)


def _checker_brush() -> QBrush:
    tile = QPixmap(16, 16)
    tile.fill(QColor("#FFFFFF"))
    p = QPainter(tile)
    p.fillRect(0, 0, 8, 8, QColor("#E3E6EA"))
    p.fillRect(8, 8, 8, 8, QColor("#E3E6EA"))
    p.end()
    return QBrush(tile)


class DropList(QListWidget):
    """File list that accepts dropped files and folders."""

    paths_dropped = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setUniformItemSizes(True)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.paths_dropped.emit([u.toLocalFile() for u in urls if u.isLocalFile()])
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.count() == 0:
            p = QPainter(self.viewport())
            p.setPen(QColor("#8A929C"))
            p.drawText(self.viewport().rect(), Qt.AlignCenter, "把图片或文件夹\n拖到这里")


class PreviewPane(QWidget):
    def __init__(self, placeholder: str, checker: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder
        self._checker = checker
        self._brush = _checker_brush()
        self.setMinimumSize(240, 240)

    def sizeHint(self) -> QSize:
        return QSize(480, 480)

    def set_pixmap(self, pixmap: QPixmap | None, placeholder: str | None = None) -> None:
        self._pixmap = pixmap
        if placeholder is not None:
            self._placeholder = placeholder
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        area = self.rect().adjusted(12, 12, -12, -12)
        if self._pixmap is None:
            p.setPen(QColor("#8A929C"))
            p.drawText(area, Qt.AlignCenter, self._placeholder)
            return
        size = self._pixmap.size().scaled(area.size(), Qt.KeepAspectRatio)
        target = QRect(0, 0, size.width(), size.height())
        target.moveCenter(area.center())
        if self._checker:
            p.fillRect(target, self._brush)
        p.drawPixmap(target, self._pixmap)
        p.setPen(QColor("#D5D9DE"))
        p.drawRect(target.adjusted(0, 0, -1, -1))
