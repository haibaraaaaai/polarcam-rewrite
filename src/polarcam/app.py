# ------------------------------------------
# File: src/polarcam/app.py
# (Minimal + ROI v2 — four buttons + Mono12 preview + ROI dock; non‑disruptive refresh; no refresh button)
# ------------------------------------------

import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton, QStatusBar,
    QDockWidget, QFormLayout, QLineEdit, QHBoxLayout
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QMetaObject

from .ids_backend import IDSCamera


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PolarCam — Minimal + ROI v2")
        self.resize(1200, 720)

        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.video = QLabel("No video"); self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#777;")
        self.video.setMinimumSize(640, 480)

        self.btn_open = QPushButton("Open"); self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop"); self.btn_close = QPushButton("Close")

        # initial state
        self.btn_open.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_close.setEnabled(False)

        col = QVBoxLayout(); col.addWidget(self.video)
        col.addWidget(self.btn_open); col.addWidget(self.btn_start)
        col.addWidget(self.btn_stop); col.addWidget(self.btn_close)
        central = QWidget(); central.setLayout(col); self.setCentralWidget(central)

        # backend
        self.cam = IDSCamera()
        self.btn_open.clicked.connect(self._open_clicked)
        self.btn_start.clicked.connect(self._start_clicked)
        self.btn_stop.clicked.connect(self._stop_clicked)
        self.btn_close.clicked.connect(self._close_clicked)

        self.cam.opened.connect(self._on_open)
        self.cam.started.connect(self._on_started)
        self.cam.stopped.connect(self._on_stopped)
        self.cam.closed.connect(self._on_closed)
        self.cam.error.connect(self._on_error)
        self.cam.frame.connect(self._on_frame)

        # ROI dock
        self._make_roi_dock()
        self.cam.roi.connect(self._on_roi)

    # ROI dock UI
    def _make_roi_dock(self):
        dock = QDockWidget("ROI", self)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.ed_w = QLineEdit(); self.ed_h = QLineEdit(); self.ed_x = QLineEdit(); self.ed_y = QLineEdit()
        btns = QWidget(); hb = QHBoxLayout(btns); hb.setContentsMargins(0,0,0,0)
        self.btn_apply = QPushButton("Apply ROI")
        self.btn_full = QPushButton("Full sensor")
        hb.addWidget(self.btn_apply); hb.addWidget(self.btn_full)
        f.addRow("Width", self.ed_w); f.addRow("Height", self.ed_h)
        f.addRow("OffsetX", self.ed_x); f.addRow("OffsetY", self.ed_y)
        f.addRow(btns)
        dock.setWidget(w)
        # wire
        self.btn_apply.clicked.connect(self._apply_roi)
        self.btn_full.clicked.connect(self._full_roi)

    # button handlers (with debug)
    def _open_clicked(self):
        print("[BTN] Open clicked"); self.cam.open()

    def _start_clicked(self):
        print("[BTN] Start clicked"); self.cam.start()

    def _stop_clicked(self):
        print("[BTN] Stop clicked"); self.status.showMessage("Stop requested…", 0); self.cam.stop()

    def _close_clicked(self):
        print("[BTN] Close clicked"); self.cam.close()

    # ROI handlers
    def _apply_roi(self):
        try:
            w = float(self.ed_w.text()); h = float(self.ed_h.text())
            x = float(self.ed_x.text()); y = float(self.ed_y.text())
        except Exception as e:
            self._on_error(f"Bad ROI values: {e}"); return
        self.cam.set_roi(w, h, x, y)

    def _full_roi(self):
        # Set offsets to 0 and make width/height large; worker snaps to limits
        self.cam.set_roi(1e9, 1e9, 0.0, 0.0)

    # signal handlers
    def _on_open(self, name: str) -> None:
        self.status.showMessage(f"Opened: {name}", 1500)
        self.btn_open.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_close.setEnabled(True)

    def _on_started(self) -> None:
        self.status.showMessage("Acquisition started", 1500)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_close.setEnabled(False)

    def _on_stopped(self) -> None:
        self.status.showMessage("Acquisition stopped", 1500)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_close.setEnabled(True)

    def _on_closed(self) -> None:
        self.status.showMessage("Closed", 1500)
        self.btn_open.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_close.setEnabled(False)
        self.video.setPixmap(QPixmap())

    def _on_error(self, msg: str) -> None:
        print("[Camera Error]", msg)
        self.status.showMessage(f"Error: {msg}", 0)

    def _on_frame(self, arr_obj: object) -> None:
        a = np.asarray(arr_obj)
        if a.ndim != 2:
            return
        h, w = a.shape
        # Very fast 12->8 preview: right-shift; keep raw 16-bit internally
        if a.dtype == np.uint16:
            a8 = (a >> 4).astype(np.uint8)
        else:
            a8 = a.astype(np.uint8, copy=False)
        if not a8.flags.c_contiguous:
            a8 = np.ascontiguousarray(a8)
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        self.video.setPixmap(QPixmap.fromImage(qimg.copy()))

    def _on_roi(self, d: dict):
        # Update fields if present
        for name, widget in [("Width", self.ed_w), ("Height", self.ed_h), ("OffsetX", self.ed_x), ("OffsetY", self.ed_y)]:
            v = d.get(name)
            if v is not None:
                try:
                    widget.setText(str(int(v)))
                except Exception:
                    widget.setText(str(v))


def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
