import sys, time
from typing import Optional, List, Tuple

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QStatusBar, QVBoxLayout, QWidget, QCheckBox
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from PySide6.QtCore import Qt, QMetaObject, QThread

from .ids_backend import IDSCamera
from .plugins.motion_spots import MotionSpotDetectorPlugin  # plugin (runs in its own QThread)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PolarCam — ROI + Timing + Gains + One-Shot Detect")
        self.resize(1200, 780)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.video = QLabel("No video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#777;")
        self.video.setMinimumSize(640, 480)

        # Top buttons
        self.btn_open = QPushButton("Open")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_close = QPushButton("Close")

        self.btn_open.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_close.setEnabled(False)

        col = QVBoxLayout()
        col.addWidget(self.video)
        row = QHBoxLayout()
        row.addWidget(self.btn_open); row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop); row.addWidget(self.btn_close)
        col.addLayout(row)
        central = QWidget(); central.setLayout(col)
        self.setCentralWidget(central)

        # Backend
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
        self._make_gains_dock()     # restored
        self._make_detect_dock()    # one-shot detect + auto-peak

        self.cam.roi.connect(self._on_roi)
        self.cam.timing.connect(self._on_timing)
        self.cam.gains.connect(self._on_gains)

        # Frozen overlays
        self._frozen_spots: List[Tuple[float, float, float, int]] = []

        # --- Detector plugin in its own QThread ---
        self.det_thread = QThread(self)
        self.det = MotionSpotDetectorPlugin(hz=12, delta_frames=3,
                                            diff_thresh=200, intensity_thresh=1200,
                                            min_area=3, max_area=500)
        self.det.moveToThread(self.det_thread)
        self.det.status.connect(lambda s: self.status.showMessage(s, 2000))
        self.det.detections.connect(self._on_plugin_detections, Qt.QueuedConnection)
        self.det_thread.start()
        # We feed frames always; the plugin only works when its internal timer is started.
        self.cam.frame.connect(self.det.enqueue, Qt.QueuedConnection)

        # Detect one-shot control
        self._detect_inflight = False

    # ---------- ROI dock ----------
    def _make_roi_dock(self) -> None:
        dock = QDockWidget("ROI", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.ed_w = QLineEdit(); self.ed_h = QLineEdit()
        self.ed_x = QLineEdit(); self.ed_y = QLineEdit()
        btns = QWidget(); hb = QHBoxLayout(btns); hb.setContentsMargins(0,0,0,0)
        self.btn_apply_roi = QPushButton("Apply ROI")
        self.btn_full = QPushButton("Full sensor")
        hb.addWidget(self.btn_apply_roi); hb.addWidget(self.btn_full)
        f.addRow("Width", self.ed_w); f.addRow("Height", self.ed_h)
        f.addRow("OffsetX", self.ed_x); f.addRow("OffsetY", self.ed_y)
        f.addRow(btns); dock.setWidget(w)
        self.btn_apply_roi.clicked.connect(self._apply_roi)
        self.btn_full.clicked.connect(self._full_roi)

    # ---------- Timing dock ----------
    def _make_timing_dock(self) -> None:
        dock = QDockWidget("Timing", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.ed_fps = QLineEdit(); self.ed_exp = QLineEdit()
        self.ed_fps.setPlaceholderText("FPS (e.g. 200.0)")
        self.ed_exp.setPlaceholderText("Exposure ms (e.g. 2.5)")
        btns = QWidget(); hb = QHBoxLayout(btns); hb.setContentsMargins(0,0,0,0)
        self.btn_timing = QPushButton("Apply Timing"); hb.addWidget(self.btn_timing)
        f.addRow("FrameRate", self.ed_fps); f.addRow("Exposure (ms)", self.ed_exp); f.addRow(btns)
        dock.setWidget(w)
        self.btn_timing.clicked.connect(self._apply_timing)

    # ---------- Gains dock (restored) ----------
    def _make_gains_dock(self) -> None:
        dock = QDockWidget("Gains", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.ed_gain_ana = QLineEdit(); self.ed_gain_dig = QLineEdit()
        self.ed_gain_ana.setPlaceholderText("Analog gain (e.g. 1.0)")
        self.ed_gain_dig.setPlaceholderText("Digital gain (e.g. 1.0)")
        btns = QWidget(); hb = QHBoxLayout(btns); hb.setContentsMargins(0,0,0,0)
        self.btn_apply_gains = QPushButton("Apply Gains")
        self.btn_refresh_gains = QPushButton("Refresh")
        hb.addWidget(self.btn_apply_gains); hb.addWidget(self.btn_refresh_gains)
        f.addRow("Analog", self.ed_gain_ana); f.addRow("Digital", self.ed_gain_dig); f.addRow(btns)
        dock.setWidget(w)
        self.btn_apply_gains.clicked.connect(self._apply_gains)
        self.btn_refresh_gains.clicked.connect(self.cam.refresh_gains)

    # ---------- Detect dock (one-shot) ----------
    def _make_detect_dock(self) -> None:
        dock = QDockWidget("Detect (one-shot)", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.ed_dt = QLineEdit(); self.ed_dt.setText("10")               # ms (only used if FPS low)
        self.ed_diff = QLineEdit(); self.ed_diff.setText("200")          # Mono12 ADU
        self.ed_intens = QLineEdit(); self.ed_intens.setText("1200")     # Mono12 ADU
        self.ed_minarea = QLineEdit(); self.ed_minarea.setText("3")
        self.ed_maxarea = QLineEdit(); self.ed_maxarea.setText("500")
        self.chk_auto_peak = QCheckBox("Auto-peak before Detect (aim 90% FS)")
        btns = QWidget(); hb = QHBoxLayout(btns); hb.setContentsMargins(0,0,0,0)
        self.btn_detect = QPushButton("Detect"); self.btn_clear = QPushButton("Clear")
        hb.addWidget(self.btn_detect); hb.addWidget(self.btn_clear)
        f.addRow("Δt (ms)", self.ed_dt); f.addRow("Diff ≥", self.ed_diff)
        f.addRow("Intensity ≥", self.ed_intens)
        f.addRow("Min area", self.ed_minarea); f.addRow("Max area", self.ed_maxarea)
        f.addRow(self.chk_auto_peak); f.addRow(btns); dock.setWidget(w)
        self.btn_detect.clicked.connect(self._detect_once)
        self.btn_clear.clicked.connect(self._clear_detections)

    # ---------- Buttons ----------
    def _open_clicked(self) -> None:
        print("[BTN] Open clicked"); self.cam.open()

    def _start_clicked(self) -> None:
        print("[BTN] Start clicked"); self.cam.start()

    def _stop_clicked(self) -> None:
        print("[BTN] Stop clicked"); self.status.showMessage("Stop requested…", 0); self.cam.stop()

    def _close_clicked(self) -> None:
        print("[BTN] Close clicked"); self.cam.close()

    # ---------- ROI / Timing / Gains handlers ----------
    def _apply_roi(self) -> None:
        try:
            w = float(self.ed_w.text()); h = float(self.ed_h.text())
            x = float(self.ed_x.text()); y = float(self.ed_y.text())
        except Exception as e:
            self._on_error(f"Bad ROI values: {e}"); return
        self.cam.set_roi(w, h, x, y)

    def _full_roi(self) -> None:
        self.cam.set_roi(1e9, 1e9, 0.0, 0.0)

    def _apply_timing(self) -> None:
        fps_txt, exp_txt = self.ed_fps.text().strip(), self.ed_exp.text().strip()
        fps: Optional[float] = float(fps_txt) if fps_txt else None
        exp: Optional[float] = float(exp_txt) if exp_txt else None
        self.cam.set_timing(fps, exp)

    def _apply_gains(self) -> None:
        a_txt, d_txt = self.ed_gain_ana.text().strip(), self.ed_gain_dig.text().strip()
        a = float(a_txt) if a_txt else None
        d = float(d_txt) if d_txt else None
        self.cam.set_gains(a, d)

    # ---------- One-shot Detect flow ----------
    def _detect_once(self) -> None:
        if self._detect_inflight:
            return
        if not self.btn_stop.isEnabled():  # not acquiring
            self._on_error("Open and Start the camera first."); return

        # read UI
        try:
            delta_ms = max(0.0, float(self.ed_dt.text()))
            self.det.delta = max(1, int(round(max(1.0, delta_ms / 1000.0 * (float(self.ed_fps.text() or "0") or 60.0)))))
            self.det.diff_thresh = int(float(self.ed_diff.text()))
            self.det.intensity_thresh = int(float(self.ed_intens.text()))
            self.det.min_area = int(float(self.ed_minarea.text()))
            self.det.max_area = int(float(self.ed_maxarea.text()))
        except Exception as e:
            self._on_error(f"Bad detect params: {e}"); return

        self._frozen_spots = []
        self._detect_inflight = True
        self.btn_detect.setEnabled(False)
        self.status.showMessage("Detect: sampling frames…", 0)

        # optional auto-peak (exposure only) to push brightest ~90% FS (Mono12 → 4095)
        if self.chk_auto_peak.isChecked():
            try:
                self._auto_peak_once(target_level=0.90)
            except Exception as e:
                self._on_error(f"Auto-peak failed: {e}")

        # start plugin timer; it will emit once it has enough frames (delta)
        QMetaObject.invokeMethod(self.det, "start", Qt.QueuedConnection)

    def _on_plugin_detections(self, dets: list) -> None:
        """Freeze the first emission, then stop the plugin timer."""
        if not self._detect_inflight:
            return
        self._frozen_spots = [(float(x), float(y), float(r), int(a)) for (x, y, r, a) in dets]
        # stop plugin timer
        QMetaObject.invokeMethod(self.det, "stop", Qt.QueuedConnection)
        self._detect_inflight = False
        self.btn_detect.setEnabled(True)
        self.status.showMessage(f"Detect: {len(self._frozen_spots)} spots", 2500)

    def _clear_detections(self) -> None:
        self._frozen_spots = []
        if self._detect_inflight:
            QMetaObject.invokeMethod(self.det, "stop", Qt.QueuedConnection)
            self._detect_inflight = False
            self.btn_detect.setEnabled(True)
        self.status.showMessage("Detections cleared.", 1500)

    # ---- simple auto-peak (exposure) ----
    def _auto_peak_once(self, target_level: float = 0.90) -> None:
        # grab a single frame synchronously via a temporary slot
        fut: dict = {"arr": None}
        def grab_once(obj: object) -> None:
            a = np.asarray(obj)
            if a.ndim == 2:
                fut["arr"] = a.copy()
        self.cam.frame.connect(grab_once, Qt.QueuedConnection)
        t0 = time.perf_counter()
        while fut["arr"] is None and (time.perf_counter() - t0) < 0.5:
            QApplication.processEvents()
        try:
            self.cam.frame.disconnect(grab_once)
        except Exception:
            pass
        arr = fut["arr"]
        if arr is None:
            return
        # estimate peak and set exposure to land near target_level
        peak = float(np.percentile(arr, 99.9))
        if arr.dtype == np.uint16:
            full = 4095.0  # Mono12 in 16-bit container (<<4 already handled in preview)
        else:
            full = 255.0
        if peak <= 1.0:
            return
        # scale current exposure by ratio
        cur_ms = float(self.ed_exp.text() or "0")  # current shown in UI
        if cur_ms <= 0:
            return
        scale = (target_level * full) / peak
        new_ms = max(0.02, cur_ms * scale)
        self.cam.set_timing(None, new_ms)

    # ---------- signal handlers ----------
    def _on_open(self, name: str) -> None:
        self.status.showMessage(f"Opened: {name}", 1500)
        self.btn_open.setEnabled(False); self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False); self.btn_close.setEnabled(True)

    def _on_started(self) -> None:
        self.status.showMessage("Acquisition started", 1500)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.btn_close.setEnabled(False)
        # refresh current gains/timing once stream is live
        self.cam.refresh_timing(); self.cam.refresh_gains()

    def _on_stopped(self) -> None:
        self.status.showMessage("Acquisition stopped", 1500)
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.btn_close.setEnabled(True)
        self._clear_detections()

    def _on_closed(self) -> None:
        self.status.showMessage("Closed", 1500)
        self.btn_open.setEnabled(True); self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False); self.btn_close.setEnabled(False)
        self.video.setPixmap(QPixmap()); self._clear_detections()

    def _on_error(self, msg: str) -> None:
        print("[Camera Error]", msg); self.status.showMessage(f"Error: {msg}", 0)

    def _on_frame(self, arr_obj: object) -> None:
        a = np.asarray(arr_obj)
        if a.ndim != 2: return
        h, w = a.shape
        a8 = (a >> 4).astype(np.uint8) if a.dtype == np.uint16 else a.astype(np.uint8, copy=False)
        if not a8.flags.c_contiguous: a8 = np.ascontiguousarray(a8)
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        img = qimg.copy()

        # overlays
        if self._frozen_spots:
            p = QPainter(img); p.setRenderHint(QPainter.Antialiasing, True); p.setPen(QPen(Qt.green, 2))
            for (x, y, r, _area) in self._frozen_spots:
                p.drawEllipse(int(x - r), int(y - r), int(2 * r), int(2 * r))
            p.end()

        self.video.setPixmap(QPixmap.fromImage(img))

    def _on_roi(self, d: dict) -> None:
        for name, widget in [("Width", self.ed_w), ("Height", self.ed_h),
                             ("OffsetX", self.ed_x), ("OffsetY", self.ed_y)]:
            v = d.get(name)
            if v is not None:
                try: widget.setText(str(int(v)))
                except Exception: widget.setText(str(v))

    def _on_timing(self, d: dict) -> None:
        fps = d.get("fps")
        if fps is not None:
            try: self.ed_fps.setText(f"{float(fps):.3f}")
            except Exception: self.ed_fps.setText(str(fps))
        exp_us = d.get("exposure_us")
        if exp_us is not None:
            try: self.ed_exp.setText(f"{float(exp_us)/1000.0:.3f}")
            except Exception: self.ed_exp.setText(str(exp_us))
        rf = d.get("resulting_fps")
        if rf is not None: self.status.showMessage(f"Resulting FPS: {float(rf):.3f}", 1500)

    def _on_gains(self, g: dict) -> None:
        ana = g.get("analog", {}); dig = g.get("digital", {})
        av = ana.get("val"); dv = dig.get("val")
        if av is not None: self.ed_gain_ana.setText(f"{float(av):.3f}")
        if dv is not None: self.ed_gain_dig.setText(f"{float(dv):.3f}")


def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
