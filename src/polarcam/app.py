# ------------------------------------------
# File: src/polarcam/app.py
# (Clean build — Minimal UI + ROI + Timing docks)
# ------------------------------------------

import sys
from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QMetaObject

from .ids_backend import IDSCamera


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PolarCam — Minimal + ROI + Timing")
        self.resize(1200, 720)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.video = QLabel("No video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#777;")
        self.video.setMinimumSize(640, 480)

        self.btn_open = QPushButton("Open")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_close = QPushButton("Close")

        # initial state
        self.btn_open.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_close.setEnabled(False)

        col = QVBoxLayout()
        col.addWidget(self.video)
        col.addWidget(self.btn_open)
        col.addWidget(self.btn_start)
        col.addWidget(self.btn_stop)
        col.addWidget(self.btn_close)
        central = QWidget()
        central.setLayout(col)
        self.setCentralWidget(central)

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

        # Docks
        self._make_roi_dock()
        self._make_timing_dock()

        self.cam.roi.connect(self._on_roi)
        self.cam.timing.connect(self._on_timing)

    # ROI dock UI
    def _make_roi_dock(self) -> None:
        dock = QDockWidget("ROI", self)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget()
        f = QFormLayout(w)
        self.ed_w = QLineEdit()
        self.ed_h = QLineEdit()
        self.ed_x = QLineEdit()
        self.ed_y = QLineEdit()
        btns = QWidget()
        hb = QHBoxLayout(btns)
        hb.setContentsMargins(0, 0, 0, 0)
        self.btn_apply_roi = QPushButton("Apply ROI")
        self.btn_full = QPushButton("Full sensor")
        hb.addWidget(self.btn_apply_roi)
        hb.addWidget(self.btn_full)
        f.addRow("Width", self.ed_w)
        f.addRow("Height", self.ed_h)
        f.addRow("OffsetX", self.ed_x)
        f.addRow("OffsetY", self.ed_y)
        f.addRow(btns)
        dock.setWidget(w)
        # wire
        self.btn_apply_roi.clicked.connect(self._apply_roi)
        self.btn_full.clicked.connect(self._full_roi)

    # Timing dock UI
    def _make_timing_dock(self) -> None:
        dock = QDockWidget("Timing", self)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget()
        f = QFormLayout(w)
        self.ed_fps = QLineEdit()
        self.ed_exp = QLineEdit()
        self.ed_fps.setPlaceholderText("FPS (e.g. 200.0)")
        self.ed_exp.setPlaceholderText("Exposure ms (e.g. 2.5)")
        btns = QWidget()
        hb = QHBoxLayout(btns)
        hb.setContentsMargins(0, 0, 0, 0)
        self.btn_timing = QPushButton("Apply Timing")
        hb.addWidget(self.btn_timing)
        f.addRow("FrameRate", self.ed_fps)
        f.addRow("Exposure (ms)", self.ed_exp)
        f.addRow(btns)
        dock.setWidget(w)
        # wire
        self.btn_timing.clicked.connect(self._apply_timing)

    # button handlers (with debug)
    def _open_clicked(self) -> None:
        print("[BTN] Open clicked")
        self.cam.open()

    def _start_clicked(self) -> None:
        print("[BTN] Start clicked")
        self.cam.start()

    def _stop_clicked(self) -> None:
        print("[BTN] Stop clicked")
        self.status.showMessage("Stop requested…", 0)
        self.cam.stop()

    def _close_clicked(self) -> None:
        print("[BTN] Close clicked")
        self.cam.close()

    # ROI handlers
    def _apply_roi(self) -> None:
        try:
            w = float(self.ed_w.text())
            h = float(self.ed_h.text())
            x = float(self.ed_x.text())
            y = float(self.ed_y.text())
        except Exception as e:
            self._on_error(f"Bad ROI values: {e}")
            return
        self.cam.set_roi(w, h, x, y)

    def _full_roi(self) -> None:
        # Set offsets to 0 and make width/height large; worker snaps to limits
        self.cam.set_roi(1e9, 1e9, 0.0, 0.0)

    # Timing handlers
    def _apply_timing(self) -> None:
        fps_txt = self.ed_fps.text().strip()
        exp_txt = self.ed_exp.text().strip()
        fps: Optional[float] = float(fps_txt) if fps_txt else None
        exp: Optional[float] = float(exp_txt) if exp_txt else None
        self.cam.set_timing(fps, exp)

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

    def _on_roi(self, d: dict) -> None:
        # Update ROI fields if present
        for name, widget in [("Width", self.ed_w), ("Height", self.ed_h), ("OffsetX", self.ed_x), ("OffsetY", self.ed_y)]:
            v = d.get(name)
            if v is not None:
                try:
                    widget.setText(str(int(v)))
                except Exception:
                    widget.setText(str(v))

    def _on_timing(self, d: dict) -> None:
        fps = d.get("fps")
        if fps is not None:
            try:
                self.ed_fps.setText(f"{float(fps):.3f}")
            except Exception:
                self.ed_fps.setText(str(fps))
        exp_us = d.get("exposure_us")
        if exp_us is not None:
            try:
                self.ed_exp.setText(f"{float(exp_us) / 1000.0:.3f}")
            except Exception:
                self.ed_exp.setText(str(exp_us))
        rf = d.get("resulting_fps")
        if rf is not None:
            self.status.showMessage(f"Resulting FPS: {float(rf):.3f}", 1500)


def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
