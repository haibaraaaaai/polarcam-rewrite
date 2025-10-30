# ------------------------------------------
# File: src/polarcam/app.py
# (UI: PixelFormat dropdown + quick buttons; sticky errors; 12->8 fast preview)
# ------------------------------------------

import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton,
    QStatusBar, QDockWidget, QFormLayout, QLineEdit, QMessageBox,
    QComboBox, QHBoxLayout
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

from .ids_backend import IDSCamera


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PolarCam — Params step 2 (pixel format)")
        self.resize(1250, 820)

        # central video
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.video = QLabel("No video"); self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#777;")
        self.video.setMinimumSize(800, 600)

        self.btn_open = QPushButton("Open"); self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop"); self.btn_close = QPushButton("Close")
        self.btn_open.setEnabled(True); self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False); self.btn_close.setEnabled(False)

        col = QVBoxLayout(); col.addWidget(self.video)
        col.addWidget(self.btn_open); col.addWidget(self.btn_start)
        col.addWidget(self.btn_stop); col.addWidget(self.btn_close)
        central = QWidget(); central.setLayout(col); self.setCentralWidget(central)

        # backend
        self.cam = IDSCamera()
        self.btn_open.clicked.connect(self.cam.open)
        self.btn_start.clicked.connect(self.cam.start)
        self.btn_stop.clicked.connect(self.cam.stop)
        self.btn_close.clicked.connect(self.cam.close)

        self.cam.opened.connect(self._on_open)
        self.cam.started.connect(self._on_started)
        self.cam.stopped.connect(self._on_stopped)
        self.cam.closed.connect(self._on_closed)
        self.cam.error.connect(self._on_error)
        self.cam.frame.connect(self._on_frame)
        self.cam.parameters_updated.connect(self._on_params)
        self.cam.pixel_format_changed.connect(self._on_pf_changed)

        # params dock (ROI/Exposure/FPS/Gain + PixelFormat)
        self._make_param_dock()

    def _make_param_dock(self):
        dock = QDockWidget("Parameters", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        # ROI / Exposure / FPS / Gain
        self.ed_w = QLineEdit(); self.ed_h = QLineEdit(); self.ed_x = QLineEdit(); self.ed_y = QLineEdit()
        self.ed_exp = QLineEdit(); self.ed_fps = QLineEdit(); self.ed_ag = QLineEdit(); self.ed_dg = QLineEdit()
        self.btn_set_roi = QPushButton("Set ROI"); self.btn_set_exp = QPushButton("Set Exposure (µs)")
        self.btn_set_fps = QPushButton("Set FPS"); self.btn_set_gain = QPushButton("Set Gains")
        f.addRow("Width", self.ed_w); f.addRow("Height", self.ed_h)
        f.addRow("OffsetX", self.ed_x); f.addRow("OffsetY", self.ed_y); f.addRow(self.btn_set_roi)
        f.addRow("ExposureTime [µs]", self.ed_exp); f.addRow(self.btn_set_exp)
        f.addRow("AcquisitionFrameRate", self.ed_fps); f.addRow(self.btn_set_fps)
        f.addRow("AnalogGain", self.ed_ag); f.addRow("DigitalGain", self.ed_dg); f.addRow(self.btn_set_gain)
        # Pixel format chooser
        row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0)
        self.cmb_pf = QComboBox(); self.cmb_pf.addItems(["Mono12","Mono10","Mono8","Mono12p","Mono10p"])  # conservative default list
        self.btn_apply_pf = QPushButton("Apply PixelFormat")
        self.btn_quick_12 = QPushButton("Use Mono12 (Full)")
        self.btn_quick_8 = QPushButton("Use Mono8 (Fast)")
        h.addWidget(self.cmb_pf); h.addWidget(self.btn_apply_pf)
        h2 = QHBoxLayout();
        row2 = QWidget(); row2.setLayout(h2); h2.addWidget(self.btn_quick_12); h2.addWidget(self.btn_quick_8)
        f.addRow("PixelFormat", row); f.addRow(row2)
        dock.setWidget(w)
        # wire buttons
        self.btn_set_roi.clicked.connect(self._apply_roi)
        self.btn_set_exp.clicked.connect(self._apply_exp)
        self.btn_set_fps.clicked.connect(self._apply_fps)
        self.btn_set_gain.clicked.connect(self._apply_gain)
        self.btn_apply_pf.clicked.connect(self._apply_pf)
        self.btn_quick_12.clicked.connect(lambda: self._apply_pf_fixed("Mono12"))
        self.btn_quick_8.clicked.connect(lambda: self._apply_pf_fixed("Mono8"))

    # ---- slots ----
    def _on_open(self, name: str) -> None:
        self.status.showMessage(f"Opened: {name}", 1500)
        self.btn_open.setEnabled(False); self.btn_close.setEnabled(True)
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.cam.refresh_parameters()

    def _on_started(self) -> None:
        self.status.showMessage("Acquisition started", 1500)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True); self.btn_close.setEnabled(False)

    def _on_stopped(self) -> None:
        self.status.showMessage("Acquisition stopped", 1500)
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); self.btn_close.setEnabled(True)
        self.cam.refresh_parameters()

    def _on_closed(self) -> None:
        self.status.showMessage("Closed", 1500)
        self.btn_open.setEnabled(True); self.btn_start.setEnabled(False); self.btn_stop.setEnabled(False); self.btn_close.setEnabled(False)
        self.video.setPixmap(QPixmap())

    def _on_error(self, msg: str) -> None:
        # Print to terminal, show sticky status, and pop a modal dialog
        try:
            print("[Camera Error]", msg)
        except Exception:
            pass
        self.status.showMessage(f"Error: {msg}", 0)  # 0 = persist until replaced
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Camera Error", str(msg))
        except Exception:
            pass

    def _on_frame(self, arr_obj: object) -> None:
        a = np.asarray(arr_obj)
        if a.ndim != 2:
            return
        h, w = a.shape
        # Very fast 12->8 bit preview: right-shift by 4 (keep raw 12-bit internally)
        if a.dtype == np.uint16:
            a8 = (a >> 4).astype(np.uint8)
        elif a.dtype == np.uint8:
            a8 = a
        else:
            # generic fallback (rare)
            a8 = np.clip(a.astype(np.float32), 0, 65535)
            a8 = (a8 / 257.0).astype(np.uint8)
        if not a8.flags.c_contiguous:
            a8 = np.ascontiguousarray(a8)
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        self.video.setPixmap(QPixmap.fromImage(qimg.copy()))

    def _on_params(self, p: dict) -> None:
        def tip(name):
            d = p.get(name, {}); mn, mx, inc = d.get("min"), d.get("max"), d.get("increment")
            return f"min={mn} max={mx} step={inc}"
        for name, ed in [("Width", self.ed_w), ("Height", self.ed_h), ("OffsetX", self.ed_x), ("OffsetY", self.ed_y), ("ExposureTime", self.ed_exp), ("AcquisitionFrameRate", self.ed_fps), ("AnalogGain", self.ed_ag), ("DigitalGain", self.ed_dg)]:
            d = p.get(name, {}); cur = d.get("current")
            if cur is not None: ed.setText(str(int(cur) if name in ("Width","Height","OffsetX","OffsetY") else round(cur, 4)))
            ed.setToolTip(tip(name))
        # show current PixelFormat if known
        pf = p.get("PixelFormat", {}).get("current")
        if isinstance(pf, str):
            idx = self.cmb_pf.findText(pf)
            if idx >= 0: self.cmb_pf.setCurrentIndex(idx)
        # show fps range
        fr = p.get("AcquisitionFrameRate", {}); self.status.showMessage(f"FPS range: {fr.get('min')}–{fr.get('max')}")

    def _on_pf_changed(self, pf: str) -> None:
        # reflect in combo box (in case set programmatically)
        idx = self.cmb_pf.findText(pf)
        if idx >= 0: self.cmb_pf.setCurrentIndex(idx)
        self.status.showMessage(f"PixelFormat set to {pf}", 1500)

    # ---- apply buttons ----
    def _apply_roi(self):
        try:
            w = float(self.ed_w.text()); h = float(self.ed_h.text())
            x = float(self.ed_x.text()); y = float(self.ed_y.text())
            self.cam.set_parameters({"Width": {"current": w}, "Height": {"current": h}, "OffsetX": {"current": x}, "OffsetY": {"current": y}})
        except Exception as e:
            QMessageBox.critical(self, "ROI error", str(e))

    def _apply_exp(self):
        try:
            exp_us = float(self.ed_exp.text()); self.cam.set_parameters({"ExposureTime": {"current": exp_us}})
        except Exception as e:
            QMessageBox.critical(self, "Exposure error", str(e))

    def _apply_fps(self):
        try:
            fps = float(self.ed_fps.text()); self.cam.set_parameters({"AcquisitionFrameRate": {"current": fps}})
        except Exception as e:
            QMessageBox.critical(self, "FPS error", str(e))

    def _apply_gain(self):
        try:
            ag = float(self.ed_ag.text()); dg = float(self.ed_dg.text())
            self.cam.set_parameters({"AnalogGain": {"current": ag}, "DigitalGain": {"current": dg}})
        except Exception as e:
            QMessageBox.critical(self, "Gain error", str(e))

    def _apply_pf(self):
        pf = self.cmb_pf.currentText().strip()
        if pf:
            self.cam.set_pixel_format(pf)

    def _apply_pf_fixed(self, pf: str):
        self.cam.set_pixel_format(pf)


def main() -> None:
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())


if __name__ == "__main__":
    main()
