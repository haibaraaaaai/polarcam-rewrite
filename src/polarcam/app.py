import sys, time
from typing import Optional, List, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future

from PySide6.QtWidgets import (
    QApplication, QDockWidget, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QStatusBar, QVBoxLayout, QWidget, QCheckBox
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from PySide6.QtCore import Qt, QMetaObject, QEventLoop, QTimer, Signal

from .ids_backend import IDSCamera

# ---- Optional scikit-image (for detection); UI disables Detect if missing ----
try:
    from skimage.morphology import binary_closing, binary_opening, disk
    from skimage.measure import label, regionprops
    _SK_OK = True
except Exception:
    _SK_OK = False


# ---------------- Worker compute (runs off the GUI thread) ----------------
def compute_detections(f0: np.ndarray, f1: np.ndarray, *,
                       intens_thr: int, diff_thr: int,
                       min_area: int, max_area: int,
                       downscale: int = 2,
                       clutter_max_px: int = 200_000,
                       top_n: int = 100) -> List[Tuple[float, float, float, int, float]]:
    # Fast decimation for speed
    ds = max(1, int(downscale))
    if ds > 1:
        f0d = f0[::ds, ::ds]
        f1d = f1[::ds, ::ds]
    else:
        f0d, f1d = f0, f1

    # Prefilter by intensity
    f0f = np.where(f0d >= intens_thr, f0d, 0)
    f1f = np.where(f1d >= intens_thr, f1d, 0)

    # Absolute difference
    diff = np.abs(f1f.astype(np.int32) - f0f.astype(np.int32))
    mask = diff >= int(diff_thr)

    # Clutter guard: tighten once; if still huge, bail
    if mask.sum() > clutter_max_px:
        diff2 = np.abs(f1f.astype(np.int32) - f0f.astype(np.int32))
        mask = (diff2 >= int(diff_thr * 1.5)) & (f1f >= int(intens_thr * 1.2))
        if mask.sum() > clutter_max_px:
            return []

    if not mask.any():
        return []

    se = disk(1)
    mask = binary_closing(mask, se)
    mask = binary_opening(mask, se)

    lbl = label(mask, connectivity=2)
    spots: List[Tuple[float, float, float, int, float]] = []
    for r in regionprops(lbl, intensity_image=f1f):
        area_ds = int(r.area)
        if area_ds < max(1, int(min_area / (ds * ds))) or area_ds > int(max_area / (ds * ds)):
            continue
        cy, cx = r.centroid
        peak = float(r.max_intensity)
        # scale back to full-res coords/area
        cx *= ds; cy *= ds
        area = int(area_ds * ds * ds)
        radius = float(np.sqrt(max(1.0, area) / np.pi))
        spots.append((float(cx), float(cy), radius, area, peak))

    spots.sort(key=lambda t: t[4], reverse=True)
    if len(spots) > top_n:
        spots = spots[:top_n]
    return spots


class MainWindow(QMainWindow):
    # Thread-safe bridge: worker callback emits this to the UI thread
    detectReady = Signal(object)  # payload: List[(x,y,r,area,peak)]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PolarCam — ROI / Timing / One-shot Detect (stops during compute)")
        self.resize(1200, 780)

        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.video = QLabel("No video"); self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#777;"); self.video.setMinimumSize(640, 480)

        self.btn_open = QPushButton("Open"); self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop"); self.btn_close = QPushButton("Close")

        # initial state
        self.btn_open.setEnabled(True);  self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False); self.btn_close.setEnabled(False)

        col = QVBoxLayout()
        col.addWidget(self.video)
        row = QHBoxLayout()
        row.addWidget(self.btn_open); row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop);  row.addWidget(self.btn_close)
        col.addLayout(row)
        central = QWidget(); central.setLayout(col)
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
        self._make_detect_dock()

        self.cam.roi.connect(self._on_roi)
        self.cam.timing.connect(self._on_timing)

        # detection state / infra
        self._frozen_spots: List[Tuple[float, float, float, int, float]] = []
        self._last_frame: Optional[np.ndarray] = None
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._det_future: Optional[Future] = None
        self._restart_after_detect = False
        self.detectReady.connect(self._on_detect_ready)  # deliver results to UI thread

    # ---------- ROI dock ----------
    def _make_roi_dock(self) -> None:
        dock = QDockWidget("ROI", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.ed_w = QLineEdit(); self.ed_h = QLineEdit()
        self.ed_x = QLineEdit(); self.ed_y = QLineEdit()
        btns = QWidget(); hb = QHBoxLayout(btns); hb.setContentsMargins(0,0,0,0)
        self.btn_apply_roi = QPushButton("Apply ROI"); self.btn_full = QPushButton("Full sensor")
        hb.addWidget(self.btn_apply_roi); hb.addWidget(self.btn_full)
        f.addRow("Width", self.ed_w); f.addRow("Height", self.ed_h)
        f.addRow("OffsetX", self.ed_x); f.addRow("OffsetY", self.ed_y); f.addRow(btns)
        dock.setWidget(w)
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

    # ---------- Detect dock ----------
    def _make_detect_dock(self) -> None:
        dock = QDockWidget("Detect (one-shot)", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)

        self.ed_intens = QLineEdit(); self.ed_intens.setText("1200")
        self.ed_diff   = QLineEdit(); self.ed_diff.setText("200")
        self.ed_minarea = QLineEdit(); self.ed_minarea.setText("3")
        self.ed_maxarea = QLineEdit(); self.ed_maxarea.setText("500")
        self.ed_dt = QLineEdit(); self.ed_dt.setText("10")  # ms

        self.chk_autodesat = QCheckBox("Auto-desaturate before Detect")
        self.ed_sat   = QLineEdit(); self.ed_sat.setText("98")   # %FS
        self.ed_step  = QLineEdit(); self.ed_step.setText("20")  # % cut
        self.ed_iters = QLineEdit(); self.ed_iters.setText("6")  # max steps

        self.chk_stop_during = QCheckBox("Stop camera during compute")
        self.chk_stop_during.setChecked(True)

        btns = QWidget(); hb = QHBoxLayout(btns); hb.setContentsMargins(0,0,0,0)
        self.btn_detect = QPushButton("Detect"); self.btn_clear = QPushButton("Clear")
        hb.addWidget(self.btn_detect); hb.addWidget(self.btn_clear)

        f.addRow("Intensity ≥ (ADU)", self.ed_intens)
        f.addRow("Diff ≥ (ADU)", self.ed_diff)
        f.addRow("Δt (ms)", self.ed_dt)
        f.addRow("Min area (px)", self.ed_minarea)
        f.addRow("Max area (px)", self.ed_maxarea)
        f.addRow(self.chk_autodesat)
        f.addRow("Near-sat %FS", self.ed_sat)
        f.addRow("Exposure step %", self.ed_step)
        f.addRow("Max steps", self.ed_iters)
        f.addRow(self.chk_stop_during)
        f.addRow(btns)
        dock.setWidget(w)

        self.btn_detect.clicked.connect(self._detect_once)
        self.btn_clear.clicked.connect(self._clear_detections)

        if not _SK_OK:
            self.status.showMessage("scikit-image not installed — Detect disabled", 4000)
            self.btn_detect.setEnabled(False)

    # ---------- Buttons ----------
    def _open_clicked(self) -> None:
        print("[BTN] Open clicked"); self.cam.open()

    def _start_clicked(self) -> None:
        print("[BTN] Start clicked"); self.cam.start()

    def _stop_clicked(self) -> None:
        print("[BTN] Stop clicked"); self.status.showMessage("Stop requested…", 0); self.cam.stop()

    def _close_clicked(self) -> None:
        print("[BTN] Close clicked"); self.cam.close()

    # ---------- ROI / Timing ----------
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
        fps_txt = self.ed_fps.text().strip()
        exp_txt = self.ed_exp.text().strip()
        fps: Optional[float] = float(fps_txt) if fps_txt else None
        exp: Optional[float] = float(exp_txt) if exp_txt else None
        self.cam.set_timing(fps, exp)

    # ---------- One-shot detect (grabs A/B, optionally stops, computes off-UI, restarts) ----------
    def _detect_once(self) -> None:
        if not self.btn_stop.isEnabled():
            self._on_error("Open and Start the camera first."); return
        if not _SK_OK:
            self._on_error("Install scikit-image to enable detection."); return

        # read params
        try:
            intens_thr = int(float(self.ed_intens.text()))
            diff_thr   = int(float(self.ed_diff.text()))
            min_area   = int(float(self.ed_minarea.text()))
            max_area   = int(float(self.ed_maxarea.text()))
            dt_ms      = max(0.0, float(self.ed_dt.text()))
            do_desat   = bool(self.chk_autodesat.isChecked())
            near_pct   = float(self.ed_sat.text())
            step_pct   = float(self.ed_step.text())
            max_steps  = int(float(self.ed_iters.text()))
            self._restart_after_detect = bool(self.chk_stop_during.isChecked())
        except Exception as e:
            self._on_error(f"Bad detect params: {e}"); return

        # optional reduce-only auto-desaturate (while acquiring)
        if do_desat:
            self._auto_desat_until_safe(near_sat_pct=near_pct, step_cut_pct=step_pct, max_steps=max_steps)

        # grab two frames Δt apart
        f0 = self._grab_frame(timeout=0.6)
        if f0 is None: self._on_error("Could not grab frame A"); return
        t_end = time.perf_counter() + (dt_ms / 1000.0)
        while time.perf_counter() < t_end:
            QApplication.processEvents()
        f1 = self._grab_frame(timeout=0.6)
        if f1 is None: self._on_error("Could not grab frame B"); return

        self._last_frame = f1.copy()

        # stop stream during compute if requested
        if self._restart_after_detect:
            self.cam.stop()
            self._await_stopped(1200)

        # launch compute (off the GUI thread)
        self.btn_detect.setEnabled(False)
        self.status.showMessage("Detecting…", 0)
        kwargs = dict(intens_thr=intens_thr, diff_thr=diff_thr,
                      min_area=min_area, max_area=max_area,
                      downscale=2, clutter_max_px=200_000, top_n=100)
        self._det_future = self._pool.submit(compute_detections, f0, f1, **kwargs)

        # bridge results back to UI thread when finished
        def _done_cb(fut: Future):
            try:
                spots = fut.result()
            except Exception as e:
                spots = []
                print("[Detect] worker failed:", e)
            # emit to UI thread
            self.detectReady.emit(spots)

        self._det_future.add_done_callback(_done_cb)

    def _on_detect_ready(self, spots_obj: object) -> None:
        spots = spots_obj if isinstance(spots_obj, list) else []
        self._frozen_spots = spots
        self.btn_detect.setEnabled(True)
        self.status.showMessage(f"Detect: {len(spots)} spots", 2500)

        # render still with overlays (even if stream is stopped)
        if self._last_frame is not None:
            self._render_still_with_overlays(self._last_frame)

        # restart if we stopped earlier
        if self._restart_after_detect and not self.btn_stop.isEnabled():
            self.cam.start()

    def _clear_detections(self) -> None:
        self._frozen_spots = []
        self.status.showMessage("Detections cleared.", 1500)

    # ---------- helpers ----------
    def _grab_frame(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        fut = {"arr": None}
        def cb(obj: object) -> None:
            a = np.asarray(obj)
            if a.ndim == 2:
                fut["arr"] = a.copy()
        self.cam.frame.connect(cb, Qt.QueuedConnection)
        t0 = time.perf_counter()
        while fut["arr"] is None and (time.perf_counter() - t0) < timeout:
            QApplication.processEvents()
        try:
            self.cam.frame.disconnect(cb)
        except Exception:
            pass
        return fut["arr"]

    def _await_stopped(self, timeout_ms: int = 1000) -> bool:
        loop = QEventLoop()
        timer = QTimer(self); timer.setSingleShot(True)

        def done():
            if loop.isRunning():
                loop.quit()

        self.cam.stopped.connect(done)
        timer.timeout.connect(done)
        timer.start(int(max(1, timeout_ms)))
        loop.exec()

        try: self.cam.stopped.disconnect(done)
        except Exception: pass
        try: timer.timeout.disconnect(done)
        except Exception: pass

        # If timer is active, we quit early because 'stopped' fired.
        return timer.isActive()

    def _auto_desat_until_safe(self, near_sat_pct: float = 98.0, step_cut_pct: float = 20.0, max_steps: int = 6) -> None:
        """Reduce exposure by fixed % steps until max pixel < near-sat threshold."""
        try:
            cur_ms = float(self.ed_exp.text() or "0")
        except Exception:
            cur_ms = 0.0
        if cur_ms <= 0:
            return
        step = max(1.0, float(step_cut_pct)) / 100.0
        near = max(50.0, min(100.0, float(near_sat_pct))) / 100.0
        for _ in range(max(1, int(max_steps))):
            frame = self._grab_frame(timeout=0.6)
            if frame is None:
                break
            full = 4095.0 if frame.dtype == np.uint16 else 255.0
            if float(frame.max()) < near * full:
                break
            cur_ms = max(0.02, cur_ms * (1.0 - step))
            self.cam.set_timing(None, cur_ms)
            # settle a couple frames
            t_end = time.perf_counter() + 0.05
            while time.perf_counter() < t_end:
                QApplication.processEvents()
        self.cam.refresh_timing()

    def _render_still_with_overlays(self, frame: np.ndarray) -> None:
        a = frame
        if a.ndim != 2:
            return
        h, w = a.shape
        a8 = (a >> 4).astype(np.uint8) if a.dtype == np.uint16 else a.astype(np.uint8, copy=False)
        if not a8.flags.c_contiguous:
            a8 = np.ascontiguousarray(a8)
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        img = qimg.copy()
        if self._frozen_spots:
            # ensure painter always ends, even on exceptions
            p = QPainter(img)
            try:
                p.setRenderHint(QPainter.Antialiasing, True)
                p.setPen(QPen(Qt.green, 2))
                full_scale = 4095.0 if a.dtype == np.uint16 else 255.0
                for i, (x, y, r, area, peak) in enumerate(self._frozen_spots, start=1):
                    p.drawEllipse(int(x - r), int(y - r), int(2*r), int(2*r))
                    pct = 100.0 * (float(peak) / full_scale)
                    p.drawText(int(x + r + 3), int(y), f"{i}: {int(peak)} ({pct:.1f}%) • A={area}")
            finally:
                p.end()
        self.video.setPixmap(QPixmap.fromImage(img))

    # ---------- signal handlers ----------
    def _on_open(self, name: str) -> None:
        self.status.showMessage(f"Opened: {name}", 1500)
        self.btn_open.setEnabled(False); self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False); self.btn_close.setEnabled(True)

    def _on_started(self) -> None:
        self.status.showMessage("Acquisition started", 1500)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.btn_close.setEnabled(False)

    def _on_stopped(self) -> None:
        self.status.showMessage("Acquisition stopped", 1500)
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.btn_close.setEnabled(True)

    def _on_closed(self) -> None:
        self.status.showMessage("Closed", 1500)
        self.btn_open.setEnabled(True); self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False); self.btn_close.setEnabled(False)
        self.video.setPixmap(QPixmap()); self._frozen_spots = []

    def _on_error(self, msg: str) -> None:
        print("[Camera Error]", msg)
        self.status.showMessage(f"Error: {msg}", 0)

    def _on_frame(self, arr_obj: object) -> None:
        a = np.asarray(arr_obj)
        if a.ndim != 2:
            return
        self._last_frame = a.copy()
        h, w = a.shape
        a8 = (a >> 4).astype(np.uint8) if a.dtype == np.uint16 else a.astype(np.uint8, copy=False)
        if not a8.flags.c_contiguous:
            a8 = np.ascontiguousarray(a8)
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        img = qimg.copy()
        if self._frozen_spots:
            p = QPainter(img)
            try:
                p.setRenderHint(QPainter.Antialiasing, True)
                p.setPen(QPen(Qt.green, 2))
                full_scale = 4095.0 if a.dtype == np.uint16 else 255.0
                for i, (x, y, r, area, peak) in enumerate(self._frozen_spots, start=1):
                    p.drawEllipse(int(x - r), int(y - r), int(2*r), int(2*r))
                    pct = 100.0 * (float(peak) / full_scale)
                    p.drawText(int(x + r + 3), int(y), f"{i}: {int(peak)} ({pct:.1f}%) • A={area}")
            finally:
                p.end()
        self.video.setPixmap(QPixmap.fromImage(img))

    def _on_roi(self, d: dict) -> None:
        """Update ROI fields when the worker publishes a snapshot."""
        for name, widget in [
            ("Width", self.ed_w),
            ("Height", self.ed_h),
            ("OffsetX", self.ed_x),
            ("OffsetY", self.ed_y),
        ]:
            v = d.get(name)
            if v is not None:
                try:
                    widget.setText(str(int(v)))
                except Exception:
                    widget.setText(str(v))

    def _on_timing(self, d: dict) -> None:
        """Update timing fields (fps, exposure) and status."""
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

    def _await_signal(self, signal, timeout_ms: int = 2000) -> bool:
        """Return True if the signal arrived before timeout."""
        loop = QEventLoop(self)
        timer = QTimer(self)
        timer.setSingleShot(True)

        def done(*_):
            if loop.isRunning():
                loop.quit()

        signal.connect(done)
        timer.timeout.connect(done)
        timer.start(int(max(1, timeout_ms)))
        loop.exec()

        # if timer is still active, we exited because the signal fired first
        success = timer.isActive()

        # cleanup
        try: signal.disconnect(done)
        except Exception: pass
        try: timer.timeout.disconnect(done)
        except Exception: pass
        return success

    def safe_shutdown(self):
        """Ensure acquisition stops and the device closes before the window dies."""
        try:
            # If acquiring, stop and wait for 'stopped'
            if self.btn_stop.isEnabled():
                self.cam.stop()
                self._await_signal(self.cam.stopped, 3000)
        except Exception:
            pass
        try:
            # If device is open, close and wait for 'closed'
            if self.btn_close.isEnabled():
                self.cam.close()
                self._await_signal(self.cam.closed, 3000)
        except Exception:
            pass

    def closeEvent(self, e):
        # Last-ditch: clean shutdown even if user clicks the window X while running
        self.safe_shutdown()
        super().closeEvent(e)

def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.aboutToQuit.connect(w.safe_shutdown)
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
