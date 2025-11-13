# src/polarcam/app/main_window.py
from __future__ import annotations

from typing import Optional
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QIntValidator, QDoubleValidator
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QStatusBar,
    QPushButton, QHBoxLayout, QVBoxLayout, QFormLayout,
    QLineEdit, QMessageBox, QApplication
)


class MainWindow(QMainWindow):
    """Lean GUI: Open/Start/Stop/Close + simple ROI/Timing apply. Live video."""

    def __init__(self, ctrl) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.setWindowTitle("PolarCam (lean)")
        self.resize(1200, 720)

        # status bar
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

        # build UI
        self._build_video()
        self._build_toolbar()
        self._build_forms()
        self._assemble_layout()

        # start with conservative button states
        self._set_buttons(open_enabled=True, start=False, stop=False, close=False)

        # wire controller/backend signals if present
        cam = getattr(self.ctrl, "cam", None)
        if cam is not None:
            cam.opened.connect(self._on_open)
            cam.started.connect(self._on_started)
            cam.stopped.connect(self._on_stopped)
            cam.closed.connect(self._on_closed)
            cam.error.connect(self._on_error)
            cam.frame.connect(self._on_frame)
            cam.timing.connect(self._on_timing)
            cam.roi.connect(self._on_roi)
        else:
            # still allow UI launch without a backend for layout testing
            self.status.showMessage("No camera backend attached.", 4000)

        # wire buttons (UI → controller)
        self.btn_open.clicked.connect(self._open_clicked)
        self.btn_start.clicked.connect(self._start_clicked)
        self.btn_stop.clicked.connect(self._stop_clicked)
        self.btn_close.clicked.connect(self._close_clicked)
        self.btn_apply_roi.clicked.connect(self._apply_roi)
        self.btn_full_roi.clicked.connect(self._full_roi_clicked)
        self.btn_apply_tim.clicked.connect(self._apply_timing)

    # ---------- builders ----------
    def _build_video(self) -> None:
        self.video = QLabel("No video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setStyleSheet("background:#111; color:#777;")

    def _build_toolbar(self) -> None:
        self.btn_open  = QPushButton("Open")
        self.btn_start = QPushButton("Start")
        self.btn_stop  = QPushButton("Stop")
        self.btn_close = QPushButton("Close")

        self._toolbar = QHBoxLayout()
        for b in (self.btn_open, self.btn_start, self.btn_stop, self.btn_close):
            self._toolbar.addWidget(b)
        self._toolbar.addStretch(1)

    def _build_forms(self) -> None:
        # ROI form
        self.ed_w = QLineEdit("2464"); self.ed_h = QLineEdit("2056")
        self.ed_x = QLineEdit("0");    self.ed_y = QLineEdit("0")
        intv = QIntValidator(0, 1_000_000, self)
        for ed in (self.ed_w, self.ed_h, self.ed_x, self.ed_y):
            ed.setValidator(intv)
        self.btn_apply_roi = QPushButton("Apply ROI")
        self.btn_full_roi  = QPushButton("Full sensor")

        # Timing form
        self.ed_fps = QLineEdit("20.0")
        self.ed_exp = QLineEdit("50.0")
        fpsv = QDoubleValidator(0.01, 1_000.0, 3, self); fpsv.setNotation(QDoubleValidator.StandardNotation)
        expv = QDoubleValidator(0.01, 2_000.0, 3, self); expv.setNotation(QDoubleValidator.StandardNotation)
        self.ed_fps.setValidator(fpsv)
        self.ed_exp.setValidator(expv)
        self.btn_apply_tim = QPushButton("Apply Timing")

        self._form = QFormLayout()
        self._form.addRow("Width", self.ed_w)
        self._form.addRow("Height", self.ed_h)
        self._form.addRow("OffsetX", self.ed_x)
        self._form.addRow("OffsetY", self.ed_y)
        row = QHBoxLayout()
        row.addWidget(self.btn_apply_roi)
        row.addWidget(self.btn_full_roi)
        rw = QWidget(); rw.setLayout(row)
        self._form.addRow(rw)
        self._form.addRow("FPS", self.ed_fps)
        self._form.addRow("Exposure (ms)", self.ed_exp)
        self._form.addRow(self.btn_apply_tim)

    def _assemble_layout(self) -> None:
        right = QWidget(); right.setLayout(self._form)

        leftcol = QVBoxLayout()
        leftcol.addLayout(self._toolbar)
        leftcol.addWidget(self.video, 1)

        root = QHBoxLayout()
        root.addLayout(leftcol, 2)
        root.addWidget(right, 1)

        cw = QWidget(); cw.setLayout(root)
        self.setCentralWidget(cw)

    # ---------- button state helper ----------
    def _set_buttons(self, *, open_enabled: bool, start: bool, stop: bool, close: bool) -> None:
        self.btn_open.setEnabled(open_enabled)
        self.btn_start.setEnabled(start)
        self.btn_stop.setEnabled(stop)
        self.btn_close.setEnabled(close)

    # ---------- backend signal handlers ----------
    def _on_open(self, name: str) -> None:
        self.status.showMessage(f"Opened: {name}", 1500)
        self._set_buttons(open_enabled=False, start=True, stop=False, close=True)

    def _on_started(self) -> None:
        self.status.showMessage("Started", 1500)
        self._set_buttons(open_enabled=False, start=False, stop=True, close=False)

    def _on_stopped(self) -> None:
        self.status.showMessage("Stopped", 1500)
        self._set_buttons(open_enabled=False, start=True, stop=False, close=True)

    def _on_closed(self) -> None:
        self.status.showMessage("Closed", 1500)
        self.video.setPixmap(QPixmap())  # clear view
        self._set_buttons(open_enabled=True, start=False, stop=False, close=False)

    def _on_error(self, msg: str) -> None:
        # show and also toast for visibility
        self.status.showMessage(f"Error: {msg}", 5000)
        QMessageBox.warning(self, "Camera error", msg)

    def _on_frame(self, arr_obj: object) -> None:
        """Assume uint16 0..4095; downscale to 8-bit for display."""
        try:
            a = np.asarray(arr_obj)
            if a.ndim != 2:
                return
            h, w = a.shape

            if a.dtype == np.uint16:
                a8 = (a >> 4).astype(np.uint8, copy=False)  # 12→8 bit
            elif a.dtype == np.uint8:
                a8 = a
            else:
                # be forgiving in early testing
                a8 = np.clip(a.astype(np.float32), 0, 255).astype(np.uint8)

            if not a8.flags.c_contiguous:
                a8 = np.ascontiguousarray(a8)

            qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
            self.video.setPixmap(QPixmap.fromImage(qimg.copy()))
        except Exception as e:
            self.status.showMessage(f"Frame error: {e}", 2000)

    def _on_timing(self, d: dict) -> None:
        try:
            fps = d.get("fps"); exp_us = d.get("exposure_us")
            if fps is not None:   self.ed_fps.setText(f"{float(fps):.3f}")
            if exp_us is not None:self.ed_exp.setText(f"{float(exp_us)/1000.0:.3f}")
            msg = []
            if fps is not None:   msg.append(f"FPS={float(fps):.3f}")
            if exp_us is not None:msg.append(f"EXP={float(exp_us)/1000.0:.3f} ms")
            if msg: self.status.showMessage("Timing applied: " + "  ".join(msg), 2000)
        except Exception:
            pass

    def _on_roi(self, d: dict) -> None:
        """Refresh ROI line-edits from backend snapshot (snapped values)."""
        try:
            mapping = [("Width", self.ed_w), ("Height", self.ed_h),
                       ("OffsetX", self.ed_x), ("OffsetY", self.ed_y)]
            for key, edit in mapping:
                v = d.get(key)
                if v is not None:
                    edit.setText(str(int(round(float(v)))))
        except Exception:
            pass

    # ---------- UI handlers (UI → controller) ----------
    def _open_clicked(self) -> None:
        if not hasattr(self.ctrl, "open"): return
        self.ctrl.open()

    def _start_clicked(self) -> None:
        if not self.btn_start.isEnabled() or not hasattr(self.ctrl, "start"): return
        self.ctrl.start()

    def _stop_clicked(self) -> None:
        if not self.btn_stop.isEnabled() or not hasattr(self.ctrl, "stop"): return
        self.ctrl.stop()

    def _close_clicked(self) -> None:
        if not self.btn_close.isEnabled() or not hasattr(self.ctrl, "close"): return
        self.ctrl.close()

    def _apply_roi(self) -> None:
        try:
            w = int(float(self.ed_w.text() or "0"))
            h = int(float(self.ed_h.text() or "0"))
            x = int(float(self.ed_x.text() or "0"))
            y = int(float(self.ed_y.text() or "0"))
        except Exception:
            self.status.showMessage("ROI: invalid numbers", 2000); return

        if w <= 0 or h <= 0:
            self.status.showMessage("ROI: width/height must be > 0. Use Full sensor if unsure.", 3000)
            return

        self.ctrl.set_roi(w, h, x, y)

    def _full_roi_clicked(self) -> None:
        self.ctrl.full_sensor()

    def _apply_timing(self) -> None:
        fps = float(self.ed_fps.text()) if self.ed_fps.hasAcceptableInput() else None
        exp = float(self.ed_exp.text()) if self.ed_exp.hasAcceptableInput() else None
        if hasattr(self.ctrl, "set_timing"):
            self.ctrl.set_timing(fps, exp)

    # ---------- optional cleanup ----------
    def safe_shutdown(self) -> None:
        try:
            if hasattr(self.ctrl, "stop"):
                self.ctrl.stop()
        except Exception:
            pass
        try:
            if hasattr(self.ctrl, "close"):
                self.ctrl.close()
        except Exception:
            pass

    def closeEvent(self, e) -> None:
        # If we’re acquiring, the Stop button will be enabled — stop first.
        try:
            if self.btn_stop.isEnabled() and hasattr(self.ctrl, "stop"):
                self.ctrl.stop()
                # Let the worker unwind any WaitForFinishedBuffer()
                QApplication.processEvents()
        except Exception:
            pass

        # Then close the device/session.
        try:
            if hasattr(self.ctrl, "close"):
                self.ctrl.close()
        except Exception:
            pass

        # Let the normal close proceed.
        super().closeEvent(e)