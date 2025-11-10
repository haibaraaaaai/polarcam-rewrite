# ------------------------------------------
# File: src/polarcam/app.py
# ------------------------------------------

import sys
import time
from typing import Optional, List, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import Qt, QTimer, QEventLoop, QPointF, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
    QLabel,
    QStatusBar,
    QPushButton,
    QFormLayout,
    QWidget,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QDoubleSpinBox,
    QComboBox,
    QMessageBox,
)

# scikit-image bits for one-shot detection
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.measure import label, regionprops

from .ids_backend import IDSCamera


# ============================================================
# Helper pop-up window for polar-variance image
# ============================================================
class PolVarWindow(QMainWindow):
    """Simple window that shows a single 8-bit grayscale image and fits it to the window."""
    def __init__(self, img8: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Polar variance (Ix/Iy range magnitude)")
        self.resize(900, 700)
        self._img8 = img8
        self.label = QLabel("", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background:#000; color:#999;")
        self.label.setScaledContents(False)
        self.setCentralWidget(self.label)
        self._orig_pm = self._mk_pixmap(img8)
        self._set_scaled()

    def _mk_pixmap(self, a8: np.ndarray) -> QPixmap:
        h, w = a8.shape
        if not a8.flags.c_contiguous:
            a8 = np.ascontiguousarray(a8)
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

    def _set_scaled(self):
        if self._orig_pm.isNull():
            self.label.setText("Empty image")
            return
        self.label.setPixmap(self._orig_pm.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._set_scaled()


# ============================================================
# Spot viewer (zoomed ROI + live 4-pol FFT numbers placeholder)
# ============================================================
class SpotViewerWindow(QMainWindow):
    """Pop-up window that zooms ROI around a chosen spot and will compute live FFTs."""
    def __init__(self, cam: IDSCamera, spots, index: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spot viewer — zoomed ROI + 4×pol FFT")
        self.resize(1100, 640)
        self.cam = cam
        self.spots = list(spots)
        self.idx = int(max(0, min(index, len(self.spots) - 1)))

        # UI
        self.video = QLabel("Waiting for frames…")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#777;")

        self.lbl_fft0 = QLabel("0°: –")
        self.lbl_fft45 = QLabel("45°: –")
        self.lbl_fft90 = QLabel("90°: –")
        self.lbl_fft135 = QLabel("135°: –")
        for l in (self.lbl_fft0, self.lbl_fft45, self.lbl_fft90, self.lbl_fft135):
            l.setStyleSheet("font: 14px 'Consolas','Courier New',monospace;")

        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)

        left = QVBoxLayout(); left.addWidget(self.video, 1)
        right = QVBoxLayout()
        gb = QGroupBox("Live FFT peaks (placeholder)")
        g = QGridLayout(gb)
        g.addWidget(self.lbl_fft0, 0, 0); g.addWidget(self.lbl_fft45, 1, 0)
        g.addWidget(self.lbl_fft90, 2, 0); g.addWidget(self.lbl_fft135, 3, 0)
        right.addWidget(gb)
        nav = QHBoxLayout(); nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next); nav.addStretch(1); nav.addWidget(self.btn_close)
        right.addLayout(nav)
        w = QWidget(); h = QHBoxLayout(w); h.addLayout(left, 2); h.addLayout(right, 1)
        self.setCentralWidget(w)

        # wiring
        self.btn_prev.clicked.connect(lambda: self._jump(-1))
        self.btn_next.clicked.connect(lambda: self._jump(+1))

        QTimer.singleShot(0, self._show_current)

    def _jump(self, d: int):
        if not self.spots: return
        self.idx = (self.idx + d) % len(self.spots)
        self._show_current()

    def _show_current(self):
        if not self.spots: return
        cx, cy, r, area, inten = self.spots[self.idx]
        roi_w = roi_h = max(64, int(round(2.2 * r)))
        x0 = int(max(0, round(cx - roi_w/2))); y0 = int(max(0, round(cy - roi_h/2)))
        self.cam.set_zoom_roi((x0, y0, roi_w, roi_h))
        try:
            self.cam.frame.disconnect(self._on_zoom_frame)
        except Exception:
            pass
        self.cam.frame.connect(self._on_zoom_frame, Qt.QueuedConnection)

    def closeEvent(self, e):
        try:
            self.cam.clear_zoom_roi()
            self.cam.frame.disconnect(self._on_zoom_frame)
        except Exception:
            pass
        super().closeEvent(e)

    def _on_zoom_frame(self, arr_obj: object) -> None:
        a = np.asarray(arr_obj)
        if a.ndim != 2: return
        h, w = a.shape
        a8 = (a >> 4).astype(np.uint8) if a.dtype == np.uint16 else a.astype(np.uint8, copy=False)
        if not a8.flags.c_contiguous: a8 = np.ascontiguousarray(a8)
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        pm = QPixmap.fromImage(qimg)
        self.video.setPixmap(pm.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # placeholder "peaks"
        self.lbl_fft0.setText(f"0°: {int(a.mean())}")
        self.lbl_fft45.setText(f"45°: {int(a[h//2:, :].mean())}")
        self.lbl_fft90.setText(f"90°: {int(a[:, w//2:].mean())}")
        self.lbl_fft135.setText(f"135°: {int(a[:h//2, :w//2].mean())}")


# ============================================================
# Main window
# ============================================================
class MainWindow(QMainWindow):
    detect_done = Signal(object)  # detect results back to GUI thread

    # --------------------------
    # Construction / UI wiring
    # --------------------------
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PolarCam — ROI + Timing + Detect + Spots")
        self.resize(1280, 800)

        self.status = QStatusBar(); self.setStatusBar(self.status)

        # central video
        self.video = QLabel("No video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#777;")
        self.video.setMinimumSize(640, 480)

        # top buttons
        self.btn_open = QPushButton("Open")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_close = QPushButton("Close")

        row = QHBoxLayout()
        for b in (self.btn_open, self.btn_start, self.btn_stop, self.btn_close): row.addWidget(b)
        row.addStretch(1)
        col = QVBoxLayout(); col.addLayout(row); col.addWidget(self.video, 1)
        central = QWidget(); central.setLayout(col); self.setCentralWidget(central)

        # backend + connections
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

        # docks
        self._make_roi_dock()
        self._make_timing_dock()
        self._make_detect_dock()

        self.cam.roi.connect(self._on_roi)
        self.cam.timing.connect(self._on_timing)
        self.detect_done.connect(self._handle_detect_done, Qt.QueuedConnection)

        # initial button states
        self.btn_open.setEnabled(True); self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False); self.btn_close.setEnabled(False)

        # state
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._spots: List[Tuple[float, float, float, int, int]] = []
        self._collecting = False
        self._collect_gap = 1
        self._collect_frames: List[np.ndarray] = []
        self._detect_watchdog = None
        self._detect_t0 = 0.0
        self._last_fps = 20.0

        # auto-desat
        self._desat_active = False
        self._desat_iters = 0
        self._desat_cooldown = 0
        self._desat_target = 3900

        # live freeze while showing popups
        self._freeze_video = False
        self._frozen_qimg: Optional[QImage] = None

        # pol-variance capture
        self._pv_active = False
        self._pv_connected = False
        self._pv_frames: List[np.ndarray] = []
        self._pv_target_n = 0
        self._pv_watchdog: Optional[QTimer] = None

    # --------------------------
    # UI docks
    # --------------------------
    def _make_roi_dock(self) -> None:
        dock = QDockWidget("ROI", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.ed_w = QLineEdit("2464"); self.ed_h = QLineEdit("2056")
        self.ed_x = QLineEdit("0"); self.ed_y = QLineEdit("0")
        self.btn_apply_roi = QPushButton("Apply"); self.btn_full_roi = QPushButton("Full sensor")
        f.addRow("Width", self.ed_w); f.addRow("Height", self.ed_h)
        f.addRow("OffsetX", self.ed_x); f.addRow("OffsetY", self.ed_y)
        row = QHBoxLayout(); row.addWidget(self.btn_apply_roi); row.addWidget(self.btn_full_roi)
        roww = QWidget(); roww.setLayout(row); f.addRow(roww); dock.setWidget(w)
        self.btn_apply_roi.clicked.connect(self._apply_roi_clicked)
        self.btn_full_roi.clicked.connect(self._full_roi_clicked)

    def _make_timing_dock(self) -> None:
        dock = QDockWidget("Timing", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.lbl_fps = QLabel("–"); self.ed_fps = QLineEdit("19.97"); self.ed_exp = QLineEdit("50.0")
        self.btn_apply_timing = QPushButton("Apply timing")
        f.addRow("FPS (max)", self.lbl_fps); f.addRow("FPS (target)", self.ed_fps); f.addRow("Exposure (ms)", self.ed_exp)
        f.addRow(self.btn_apply_timing); dock.setWidget(w)
        self.btn_apply_timing.clicked.connect(self._apply_timing_clicked)

    def _make_detect_dock(self) -> None:
        dock = QDockWidget("Detect & Spots", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)

        # params
        self.ed_dt = QLineEdit("10.0")     # ms between frames
        self.ed_diff = QLineEdit("50")     # |Δ| threshold
        self.ed_int = QLineEdit("200")     # intensity floor (Mono12 units)
        self.ed_minA = QLineEdit("6")      # min area
        self.ed_maxA = QLineEdit("5000")   # max area

        # auto-desat
        self.spn_peak = QDoubleSpinBox(); self.spn_peak.setRange(50.0, 99.9)
        self.spn_peak.setSingleStep(0.5); self.spn_peak.setDecimals(1); self.spn_peak.setValue(95.0)
        self.btn_autodesat = QPushButton("Auto Desaturate")

        # actions
        self.btn_detect = QPushButton("Detect")
        self.btn_clear = QPushButton("Clear overlays")
        self.btn_polvar = QPushButton("Pol-Var 1s")  # new pop-up

        f.addRow("Δt (ms)", self.ed_dt)
        f.addRow("|Δ| threshold", self.ed_diff)
        f.addRow("Intensity floor", self.ed_int)
        f.addRow("Min area", self.ed_minA)
        f.addRow("Max area", self.ed_maxA)
        f.addRow("Peak target (%FS)", self.spn_peak)
        f.addRow(self.btn_autodesat)
        f.addRow(self.btn_detect)
        f.addRow(self.btn_clear)
        f.addRow(self.btn_polvar)

        # spots row
        self.spot_combo = QComboBox()
        self.btn_go_spot = QPushButton("Go to spot")
        self.btn_add_spot = QPushButton("Add spot")
        self.btn_rm_spot = QPushButton("Remove spot")
        row2 = QHBoxLayout(); row2.addWidget(self.btn_go_spot); row2.addWidget(self.btn_add_spot); row2.addWidget(self.btn_rm_spot)
        temp = QWidget(); temp.setLayout(row2)
        f.addRow("Spots", self.spot_combo); f.addRow(temp)
        dock.setWidget(w)

        # wiring
        self.btn_detect.clicked.connect(self._detect_clicked)
        self.btn_clear.clicked.connect(self._clear_overlays)
        self.btn_autodesat.clicked.connect(self._auto_desat_clicked)
        self.btn_polvar.clicked.connect(self._polvar_clicked)
        self.btn_go_spot.clicked.connect(self._go_to_selected_spot)
        self.btn_add_spot.clicked.connect(lambda: QMessageBox.information(self, "Add spot", "Manual add via mouse-pick will come later."))
        self.btn_rm_spot.clicked.connect(self._remove_spot)

    # --------------------------
    # Button handlers
    # --------------------------
    def _open_clicked(self) -> None:
        print("[BTN] Open clicked"); self.cam.open()

    def _start_clicked(self) -> None:
        print("[BTN] Start clicked"); self.cam.start()

    def _stop_clicked(self) -> None:
        print("[BTN] Stop clicked"); self.cam.stop()

    def _close_clicked(self) -> None:
        print("[BTN] Close clicked"); self.cam.close()

    def _apply_roi_clicked(self) -> None:
        try:
            w = int(float(self.ed_w.text())); h = int(float(self.ed_h.text()))
            x = int(float(self.ed_x.text())); y = int(float(self.ed_y.text()))
        except Exception:
            self.status.showMessage("ROI: invalid numbers", 2000); return
        self.cam.enqueue_roi({"Width": w, "Height": h, "OffsetX": x, "OffsetY": y})
        self.cam.process_pending_roi()

    def _full_roi_clicked(self) -> None:
        self.cam.process_full_roi()

    def _apply_timing_clicked(self) -> None:
        try:
            fps = float(self.ed_fps.text())
            exp_ms = float(self.ed_exp.text())
        except Exception:
            self.status.showMessage("Timing: invalid numbers", 2000); return
        self.cam.enqueue_timing({"fps": fps, "exposure_ms": exp_ms})
        self.cam.process_pending_timing()

    # --------------------------
    # Detect flow
    # --------------------------
    def _detect_clicked(self) -> None:
        try:
            dt_ms = float(self.ed_dt.text())
        except Exception:
            dt_ms = 10.0
        self._begin_collect(dt_ms)

    def _begin_collect(self, dt_ms: float) -> None:
        if self._collecting:
            print("[Detect] Ignored: already collecting"); return
        self._detect_t0 = time.perf_counter()
        fps_now = getattr(self, "_last_fps", 20.0)
        try:
            txt = self.lbl_fps.text()
            if txt not in ("-", "–", "", None): fps_now = float(txt)
        except Exception:
            pass
        self._collect_gap = max(1, int(round(dt_ms * fps_now / 1000.0)))
        need = self._collect_gap + 1
        print(f"[Detect] Begin capture: dt_ms={dt_ms:.2f}  fps≈{fps_now:.3f}  gap={self._collect_gap}  need={need} frames")
        self._collect_frames.clear(); self._collecting = True
        self.btn_detect.setEnabled(False)
        self.status.showMessage(f"Detect: capturing (need {need} frames, gap={self._collect_gap})…", 0)
        self.cam.frame.connect(self._collect_for_detect, Qt.QueuedConnection)
        self._stop_detect_watchdog()
        self._detect_watchdog = QTimer(self); self._detect_watchdog.setSingleShot(True)
        self._detect_watchdog.timeout.connect(self._on_detect_timeout)
        self._detect_watchdog.start(8000)
        print("[Detect] Watchdog armed: 8000 ms")

    def _collect_for_detect(self, arr_obj: object) -> None:
        a = np.asarray(arr_obj)

        # feed auto-desat if active
        if self._desat_active:
            try: self._on_auto_desat_frame(a)
            except Exception as e: print(f"[AutoDesat] ERROR: {e}")
            return

        if a.ndim != 2: return

        self._collect_frames.append(a.copy())
        need = self._collect_gap + 1
        have = len(self._collect_frames)
        if have < need:
            self.status.showMessage(f"Detect: capturing {have}/{need}…", 0)
            print(f"[Detect] Captured {have}/{need} frames…")
            return

        # got enough frames — stop listening and process
        try: self.cam.frame.disconnect(self._collect_for_detect)
        except Exception: pass
        self._collecting = False

        prev = self._collect_frames[0]; cur = self._collect_frames[-1]
        print(f"[Detect] Processing pair: prev shape={prev.shape} dtype={prev.dtype}  cur shape={cur.shape} dtype={cur.dtype}")
        print(f"[Detect] Stats: prev[min={prev.min()} max={prev.max()}]  cur[min={cur.min()} max={cur.max()}]")

        # thresholds
        def _num(le: QLineEdit, default: float) -> float:
            try: return float(le.text())
            except Exception: return default
        diff_thr = _num(self.ed_diff, 50.0)
        inten_thr = _num(self.ed_int, 200.0)
        try: minA = int(float(self.ed_minA.text()))
        except Exception: minA = 6
        try: maxA = int(float(self.ed_maxA.text()))
        except Exception: maxA = 5000
        print(f"[Detect] Thresholds: |Δ|≥{diff_thr}  I≥{inten_thr}  area∈[{minA},{maxA}]")

        def _work(prev, cur, diff_thr, inten_thr, minA, maxA):
            t0 = time.perf_counter()
            # (current version keeps pre-floor; easy to flip off if needed)
            if inten_thr > 0:
                prev32 = prev.astype(np.int32, copy=False); cur32 = cur.astype(np.int32, copy=False)
                prev32 = np.where(prev32 >= inten_thr, prev32, 0)
                cur32 = np.where(cur32 >= inten_thr, cur32, 0)
            else:
                prev32 = prev.astype(np.int32, copy=False); cur32 = cur.astype(np.int32, copy=False)
            diff = np.abs(cur32 - prev32); t1 = time.perf_counter()
            mask = (diff >= diff_thr); n_true0 = int(mask.sum())
            if not n_true0: print(f"[Detect] mask empty (diff {1000*(t1-t0):.1f} ms)."); return []
            se = disk(1)
            mask = binary_closing(mask, se); t2 = time.perf_counter()
            mask = binary_opening(mask, se); t3 = time.perf_counter()
            lbl = label(mask, connectivity=2); t4 = time.perf_counter()
            nlab = int(lbl.max()); dets = []
            for r in regionprops(lbl):
                area = int(r.area)
                if area < minA or area > maxA: continue
                cy, cx = r.centroid
                rad = float((area / np.pi) ** 0.5)
                iy, ix = int(round(cy)), int(round(cx))
                iy = max(0, min(cur.shape[0]-1, iy)); ix = max(0, min(cur.shape[1]-1, ix))
                inten = int(cur[iy, ix])
                dets.append((float(cx), float(cy), rad, area, inten))
            t5 = time.perf_counter()
            print("[Detect] timings: diff={:.1f}ms  close={:.1f}ms  open={:.1f}ms  label={:.1f}ms  props={:.1f}ms  mask_true={}  labels={}"
                  .format(1000*(t1-t0), 1000*(t2-t1), 1000*(t3-t2), 1000*(t4-t3), 1000*(t5-t4), n_true0, nlab))
            dets.sort(key=lambda t: -t[4]); return dets

        fut = self._executor.submit(_work, prev, cur, diff_thr, inten_thr, minA, maxA)
        print("[Detect] Worker submitted")

        def _emit_from_future(f):
            try: dets = f.result(); self.detect_done.emit(("ok", dets, self._detect_t0))
            except Exception as e: self.detect_done.emit(("err", str(e), self._detect_t0))
        fut.add_done_callback(_emit_from_future)

    # --------------------------
    # Detect results → UI thread
    # --------------------------
    def _on_detect_ready(self, dets: list) -> None:
        self._spots = dets or []
        self.btn_detect.setEnabled(True)
        if not dets:
            self.status.showMessage("Detect: no spots.", 3000); return
        self.status.showMessage(f"Detect: {len(dets)} spot(s).", 3000)
        self._refresh_spot_combo()

    def _clear_overlays(self):
        self._spots = []; self._refresh_spot_combo()

    def _refresh_spot_combo(self):
        self.spot_combo.clear()
        for i, (cx, cy, r, area, inten) in enumerate(self._spots):
            self.spot_combo.addItem(f"{i+1}: (x={cx:.1f}, y={cy:.1f}) A={area}")

    def _remove_spot(self):
        idx = self.spot_combo.currentIndex()
        if 0 <= idx < len(self._spots):
            self._spots.pop(idx); self._refresh_spot_combo()

    def _go_to_selected_spot(self):
        if not self._spots:
            QMessageBox.information(self, "Go to spot", "No spots available. Run Detect first."); return
        idx = self.spot_combo.currentIndex()
        if idx < 0 or idx >= len(self._spots):
            QMessageBox.warning(self, "Go to spot", "Invalid selection."); return
        self._spot_view = SpotViewerWindow(self.cam, self._spots, idx, parent=self); self._spot_view.show()

    # --------------------------
    # Signal handlers
    # --------------------------
    def _on_open(self, name: str) -> None:
        self.status.showMessage(f"Opened: {name}", 1500)
        self.btn_open.setEnabled(False); self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False); self.btn_close.setEnabled(True)

    def _on_started(self) -> None:
        self.status.showMessage("Acquisition started", 1500)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True); self.btn_close.setEnabled(False)

    def _on_stopped(self) -> None:
        self.status.showMessage("Acquisition stopped", 1500)
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); self.btn_close.setEnabled(True)

    def _on_closed(self) -> None:
        self.status.showMessage("Closed", 1500)
        self.btn_open.setEnabled(True); self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False); self.btn_close.setEnabled(False)
        self.video.setPixmap(QPixmap()); self._freeze_video = False; self._frozen_qimg = None

    def _on_error(self, msg: str) -> None:
        print("[Camera Error]", msg); self.status.showMessage(f"Error: {msg}", 0)

    def _on_frame(self, arr_obj: object) -> None:
        # if frozen, keep showing frozen image
        if self._freeze_video and self._frozen_qimg is not None:
            pm = QPixmap.fromImage(self._frozen_qimg)
            self.video.setPixmap(pm.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            return

        a = np.asarray(arr_obj)
        if a.ndim != 2: return
        h, w = a.shape
        a8 = (a >> 4).astype(np.uint8) if a.dtype == np.uint16 else a.astype(np.uint8, copy=False)
        if not a8.flags.c_contiguous: a8 = np.ascontiguousarray(a8)
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        img = qimg.copy()

        if self._spots:
            p = QPainter(img); pen = QPen(Qt.green); pen.setWidth(2); p.setPen(pen); p.setFont(QFont("", 10))
            full_scale = 4095.0 if a.dtype == np.uint16 else 255.0
            for i, (cx, cy, r, area, inten) in enumerate(self._spots):
                p.drawEllipse(QPointF(cx, cy), r, r)
                p.drawText(int(cx + 6), int(cy - 6), f"{i+1}  I={inten}  A={area}  {100.0*inten/full_scale:.1f}%")
            p.end()

        pm = QPixmap.fromImage(img)
        self.video.setPixmap(pm.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_roi(self, d: dict) -> None:
        try:
            for name, widget in (("Width", self.ed_w), ("Height", self.ed_h), ("OffsetX", self.ed_x), ("OffsetY", self.ed_y)):
                v = d.get(name)
                if v is not None:
                    try: widget.setText(str(int(v)))
                    except Exception: widget.setText(str(v))
        except Exception:
            pass

    def _on_timing(self, d: dict) -> None:
        try:
            fps_max = d.get("fps_max")
            if fps_max is not None and hasattr(self, "lbl_fps"):
                try: self.lbl_fps.setText(f"{float(fps_max):.3f}")
                except Exception: self.lbl_fps.setText(str(fps_max))
            fps = d.get("fps")
            if fps is not None and hasattr(self, "ed_fps"):
                try: self.ed_fps.setText(f"{float(fps):.3f}")
                except Exception: self.ed_fps.setText(str(fps))
            exp_us = d.get("exposure_us")
            if exp_us is not None and hasattr(self, "ed_exp"):
                try: self.ed_exp.setText(f"{float(exp_us)/1000.0:.3f}")
                except Exception: self.ed_exp.setText(str(exp_us))
            rf = d.get("resulting_fps") or d.get("fps")
            if rf is not None:
                try: self._last_fps = float(rf)
                except Exception: pass
        except Exception:
            pass

    def _on_detect_timeout(self):
        try: have = len(self._collect_frames)
        except Exception: have = 0
        elapsed_ms = (time.perf_counter() - getattr(self, "_detect_t0", time.perf_counter())) * 1000.0
        print(f"[Detect] Watchdog TIMEOUT after {elapsed_ms:.1f} ms (frames collected={have}).")
        try: self.cam.frame.disconnect(self._collect_for_detect)
        except Exception: pass
        self._collecting = False; self.btn_detect.setEnabled(True)
        try: self.status.showMessage(f"Detect timed out after {elapsed_ms:.0f} ms.", 4000)
        except Exception: pass

    def _stop_detect_watchdog(self):
        try:
            if self._detect_watchdog is not None:
                if self._detect_watchdog.isActive(): self._detect_watchdog.stop()
                self._detect_watchdog.deleteLater()
        except Exception:
            pass
        self._detect_watchdog = None

    def _handle_detect_done(self, payload: object) -> None:
        self._stop_detect_watchdog()
        try: kind, data, t0 = payload
        except Exception:
            self.btn_detect.setEnabled(True); self.status.showMessage("Detect failed: bad payload", 4000)
            print("[Detect] ERROR: bad payload", payload); return
        if kind == "ok":
            elapsed_ms = (time.perf_counter() - (t0 or time.perf_counter())) * 1000.0
            dets = data or []; print(f"[Detect] Done: {len(dets)} spot(s) in {elapsed_ms:.1f} ms")
            self._on_detect_ready(dets)
            try: self.status.showMessage(f"Detect: {len(dets)} spot(s) in {elapsed_ms:.0f} ms.", 3000)
            except Exception: pass
        else:
            try: self.btn_detect.setEnabled(True)
            except Exception: pass
            try: self.status.showMessage(f"Detect failed: {data}", 5000)
            except Exception: pass
            print(f"[Detect] ERROR: {data}")

    # --------------------------
    # Auto-desaturate
    # --------------------------
    def _auto_desat_clicked(self) -> None:
        if not self.btn_stop.isEnabled():
            self.status.showMessage("Start acquisition first.", 2000); return
        full_scale = 4095.0; pct = float(self.spn_peak.value())
        self._desat_target = int(round(full_scale * pct / 100.0))
        self._desat_active = True; self._desat_iters = 0; self._desat_cooldown = 0
        self.status.showMessage(f"Auto-desaturate: targeting ≤ {self._desat_target} (≈{pct:.1f}% FS)", 0)
        print(f"[AutoDesat] Start: target={self._desat_target} DN")

    def _on_auto_desat_frame(self, a: np.ndarray) -> None:
        if a.ndim != 2: return
        if self._desat_cooldown > 0:
            self._desat_cooldown -= 1; return
        cur_max = int(a.max())
        print(f"[AutoDesat] Frame max={cur_max} target={self._desat_target} iter={self._desat_iters}")
        if cur_max <= self._desat_target or self._desat_iters >= 6:
            self._desat_active = False
            self.status.showMessage(f"Auto-desaturate: peak={cur_max} (done in {self._desat_iters} step(s)).", 3000)
            print(f"[AutoDesat] Done: peak={cur_max}"); return
        try: exp_ms = float(self.ed_exp.text())
        except Exception: exp_ms = 50.0
        factor = max(0.05, min(1.0, (self._desat_target / max(1.0, cur_max)) * 0.9))
        new_exp_ms = max(0.05, exp_ms * factor)
        self.ed_exp.setText(f"{new_exp_ms:.3f}")
        self.cam.enqueue_timing({"exposure_ms": new_exp_ms}); self.cam.process_pending_timing()
        self._desat_iters += 1; self._desat_cooldown = 3
        self.status.showMessage(f"Auto-desaturate: set exposure {new_exp_ms:.3f} ms (iter {self._desat_iters}).", 0)
        print(f"[AutoDesat] Set exposure {new_exp_ms:.3f} ms")

    # --------------------------
    # Polar-variance map (pop-up)
    # --------------------------
    def _polvar_clicked(self) -> None:
        """Start a ~1s capture to build the polar-variance map."""
        if not self.btn_stop.isEnabled():  # not acquiring
            self.status.showMessage("Start acquisition first.", 2500)
            return

        # target ~1s worth of frames (bounded)
        try:
            fps = float(self.lbl_fps.text())
        except Exception:
            fps = max(1.0, float(getattr(self, "_last_fps", 20.0)))
        self._pv_target_n = int(max(24, min(180, round(fps * 1.0))))

        # (re)arm state
        self._pv_frames.clear()
        self._pv_active = True

        # Connect exactly once (avoid the RuntimeWarning)
        if self._pv_connected:
            try:
                self.cam.frame.disconnect(self._collect_polvar)
            except Exception:
                pass
            self._pv_connected = False
        self.cam.frame.connect(self._collect_polvar, Qt.QueuedConnection)
        self._pv_connected = True

        # Watchdog ~5s
        if self._pv_watchdog is not None:
            try:
                if self._pv_watchdog.isActive():
                    self._pv_watchdog.stop()
                self._pv_watchdog.deleteLater()
            except Exception:
                pass
        self._pv_watchdog = QTimer(self)
        self._pv_watchdog.setSingleShot(True)
        self._pv_watchdog.timeout.connect(self._pv_timeout)
        self._pv_watchdog.start(5000)

        self.status.showMessage(f"[PolVar] Begin 1 s capture — target {self._pv_target_n} frames", 0)
        print(f"[PolVar] Begin 1 s capture — target {self._pv_target_n} frames")


    def _collect_polvar(self, arr_obj: object) -> None:
        """Frame handler for polar-variance capture."""
        if not self._pv_active:
            return
        a = np.asarray(arr_obj)
        if a.ndim != 2:
            return

        self._pv_frames.append(a.copy())
        n, tgt = len(self._pv_frames), self._pv_target_n
        if n < tgt:
            return

        # Done collecting — disconnect safely
        if self._pv_connected:
            try:
                self.cam.frame.disconnect(self._collect_polvar)
            except Exception:
                pass
            self._pv_connected = False
        self._pv_active = False

        if self._pv_watchdog is not None:
            try:
                if self._pv_watchdog.isActive():
                    self._pv_watchdog.stop()
                self._pv_watchdog.deleteLater()
            except Exception:
                pass
            self._pv_watchdog = None

        frames = self._pv_frames
        print(f"[PolVar] Collected {n} frames. Computing map…")

        # Convert “Intensity floor” (Mono12) -> 8-bit floor
        try:
            floor16 = float(self.ed_int.text())
        except Exception:
            floor16 = 0.0
        floor8 = int(max(0, min(255, round(floor16 / 16.0))))

        # Build (ΔIx, ΔIy) ranges in 8-bit, then amplitude = sqrt(ΔIx^2 + ΔIy^2)
        ix_min = ix_max = iy_min = iy_max = None
        for f in frames:
            p90, p45, p135, p0 = self._split_pol4_u8(f)
            if floor8 > 0:
                strong = (np.maximum.reduce([p0, p45, p90, p135]) >= floor8)
            else:
                strong = None
            ix = p90.astype(np.int16) - p0.astype(np.int16)
            iy = p135.astype(np.int16) - p45.astype(np.int16)
            if strong is not None:
                ix = np.where(strong, ix, 0)
                iy = np.where(strong, iy, 0)
            ix_min = ix if ix_min is None else np.minimum(ix_min, ix)
            ix_max = ix if ix_max is None else np.maximum(ix_max, ix)
            iy_min = iy if iy_min is None else np.minimum(iy_min, iy)
            iy_max = iy if iy_max is None else np.maximum(iy_max, iy)

        amp = np.sqrt((ix_max - ix_min).astype(np.float32)**2 +
                    (iy_max - iy_min).astype(np.float32)**2)

        # Percentile clip for display; guard p99>0
        p99 = float(np.percentile(amp, 99.5))
        if not np.isfinite(p99) or p99 <= 0:
            p99 = float(amp.max() or 1.0)
        scale = 255.0 / max(1.0, p99)
        amp8 = np.clip(amp * scale, 0, 255).astype(np.uint8)

        # Pop-up window (fits to size)
        try:
            win = PolVarWindow(amp8, parent=self)
            win.show()
        except NameError:
            # If you don't have PolVarWindow class in this file, fall back to a simple label
            qimg = QImage(amp8.data, amp8.shape[1], amp8.shape[0], amp8.shape[1], QImage.Format_Grayscale8)
            lbl = QLabel(); lbl.setAlignment(Qt.AlignCenter); lbl.setPixmap(QPixmap.fromImage(qimg))
            w = QMainWindow(self); w.setWindowTitle("Polar variance"); w.setCentralWidget(lbl); w.resize(900, 700); w.show()

        self.status.showMessage(f"[PolVar] Finished: n={n} frames; pop-up shown", 4000)
        print("[PolVar] Finished: pop-up shown")


    def _pv_timeout(self):
        n = len(self._pv_frames) if self._pv_frames is not None else 0
        print(f"[PolVar] Timeout while collecting (n={n}).")
        if self._pv_connected:
            try:
                self.cam.frame.disconnect(self._collect_polvar)
            except Exception:
                pass
            self._pv_connected = False
        self._pv_active = False
        self.status.showMessage("[PolVar] Timeout while collecting.", 3000)

    def _collect_polvar(self, arr_obj: object) -> None:
        a = np.asarray(arr_obj)
        if a.ndim != 2: return
        self._pv_frames.append(a.copy())
        if len(self._pv_frames) < self._pv_target_n:
            return

        # stop capture
        try: self.cam.frame.disconnect(self._collect_polvar)
        except Exception: pass
        self._pv_active = False
        if self._pv_watchdog is not None:
            try: self._pv_watchdog.stop(); self._pv_watchdog.deleteLater()
            except Exception: pass
            self._pv_watchdog = None

        frames = self._pv_frames
        # freeze main view to the last collected frame (scaled)
        last = frames[-1]
        la8 = (last >> 4).astype(np.uint8) if last.dtype == np.uint16 else last.astype(np.uint8, copy=False)
        if not la8.flags.c_contiguous: la8 = np.ascontiguousarray(la8)
        h, w = la8.shape
        self._frozen_qimg = QImage(la8.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._freeze_video = True

        # build polar variance map (8-bit scaled, with floor)
        try: floor16 = float(self.ed_int.text())
        except Exception: floor16 = 0.0
        floor8 = int(max(0, min(255, round(floor16 / 16.0))))

        ix_min = ix_max = iy_min = iy_max = None
        for f in frames:
            p90, p45, p135, p0 = self._split_pol4_u8(f)
            if floor8 > 0:
                strong = (np.maximum.reduce([p0, p45, p90, p135]) >= floor8)
            else:
                strong = None
            ix = p90.astype(np.int16) - p0.astype(np.int16)
            iy = p135.astype(np.int16) - p45.astype(np.int16)
            if strong is not None:
                ix = np.where(strong, ix, 0); iy = np.where(strong, iy, 0)
            ix_min = ix if ix_min is None else np.minimum(ix_min, ix)
            ix_max = ix if ix_max is None else np.maximum(ix_max, ix)
            iy_min = iy if iy_min is None else np.minimum(iy_min, iy)
            iy_max = iy if iy_max is None else np.maximum(iy_max, iy)
        amp = np.sqrt((ix_max - ix_min).astype(np.float32)**2 + (iy_max - iy_min).astype(np.float32)**2)
        p99 = float(np.percentile(amp, 99.5)) or 1.0
        scale = 255.0 / max(1.0, p99)
        amp8 = np.clip(amp * scale, 0, 255).astype(np.uint8)

        # show popup
        win = PolVarWindow(amp8, parent=self)
        win.show()
        self.status.showMessage(f"[PolVar] Finished: n={len(frames)} frames; pop-up shown", 4000)

    def _pv_timeout(self):
        print("[PolVar] Timeout while collecting.")
        try: self.cam.frame.disconnect(self._collect_polvar)
        except Exception: pass
        self._pv_active = False

    # --------------------------
    # Helpers (grouped)
    # --------------------------
    def _split_pol4_u8(self, a: np.ndarray):
        """Return 8-bit subimages for 90, 45, 135, 0 degrees. Layout 2×2:
           (0,0)=90, (0,1)=45, (1,0)=135, (1,1)=0
        """
        if a.dtype == np.uint16:
            a8 = (a >> 4).astype(np.uint8, copy=False)
        else:
            a8 = a.astype(np.uint8, copy=False)
        p90  = a8[0::2, 0::2]
        p45  = a8[0::2, 1::2]
        p135 = a8[1::2, 0::2]
        p0   = a8[1::2, 1::2]
        return p90, p45, p135, p0

    # --------------------------
    # Shutdown
    # --------------------------
    def safe_shutdown(self):
        try:
            if self.btn_stop.isEnabled():
                self.cam.stop(); QApplication.processEvents()
        except Exception:
            pass
        try:
            if self.btn_close.isEnabled():
                self.cam.close()
        except Exception:
            pass
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


# ============================================================
# Entrypoint
# ============================================================
def main(argv: Optional[List[str]] = None) -> int:
    app = QApplication(argv or sys.argv)
    w = MainWindow(); w.show()
    ret = app.exec()
    try: w.safe_shutdown()
    except Exception: pass
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
