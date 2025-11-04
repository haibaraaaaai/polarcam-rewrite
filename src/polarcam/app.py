# ------------------------------------------
# File: src/polarcam/app.py
# ------------------------------------------

import sys
from collections import deque
from typing import Optional, List, Tuple

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import Qt, QTimer, QEventLoop, QPointF, Signal, Slot
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
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QMessageBox,
)

# scikit-image bits for one-shot detection
from skimage.morphology import binary_closing, binary_opening, remove_small_objects, disk
from skimage.measure import label, regionprops

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .ids_backend import IDSCamera


# ============================================================
# Spot viewer (zoomed ROI + live 4-pol FFT numbers)
# ============================================================
class SpotViewerWindow(QMainWindow):
    """Pop-up window that zooms ROI around a chosen spot and shows a live 4-pol spectrum."""
    def __init__(self, cam: IDSCamera, spots, index: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spot viewer — zoomed ROI + 4×pol spectrum")
        self.resize(1220, 720)
        self.cam = cam
        self.spots = list(spots)
        self.idx = int(max(0, min(index, len(self.spots) - 1)))

        # ==== LEFT: zoomed video ====
        self.video = QLabel("Waiting for frames…")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#777;")

        # ==== RIGHT: spectrum plot + controls ====
        self.fig = Figure(figsize=(5, 4), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Power")
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvas(self.fig)

        # four line objects (created on first update)
        self._line0 = self._line45 = self._line90 = self._line135 = None

        # simple controls
        self.spn_win = QSpinBox()
        self.spn_win.setRange(64, 65536)
        self.spn_win.setSingleStep(64)
        self.spn_win.setValue(2048)     # default window length (samples)

        self.spn_nfft = QSpinBox()
        self.spn_nfft.setRange(256, 65536)
        self.spn_nfft.setSingleStep(256)
        self.spn_nfft.setValue(4096)    # default FFT length

        self.lbl_win = QLabel("Window (samples)")
        self.lbl_nfft = QLabel("NFFT")

        ctl = QGridLayout()
        ctl.addWidget(self.lbl_win, 0, 0); ctl.addWidget(self.spn_win, 0, 1)
        ctl.addWidget(self.lbl_nfft, 1, 0); ctl.addWidget(self.spn_nfft, 1, 1)

        # nav row
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        nav = QHBoxLayout()
        nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next); nav.addStretch(1); nav.addWidget(self.btn_close)

        # right layout
        right = QVBoxLayout()
        right.addWidget(self.canvas, 1)
        gb = QGroupBox("Spectrum settings")
        gb.setLayout(ctl)
        right.addWidget(gb)
        right.addLayout(nav)

        # main split
        left = QVBoxLayout(); left.addWidget(self.video, 1)
        w = QWidget(); h = QHBoxLayout(w)
        h.addLayout(left, 2); h.addLayout(right, 3)
        self.setCentralWidget(w)

        # ==== FFT state ====
        from collections import deque
        self.WINDOW = int(self.spn_win.value())
        self.NFFT = int(self.spn_nfft.value())
        L = max(self.NFFT, self.WINDOW, 4096)
        self._c0 = deque(maxlen=L); self._c45 = deque(maxlen=L)
        self._c90 = deque(maxlen=L); self._c135 = deque(maxlen=L)
        self._zoom_roi = None            # requested (x, y, w, h)
        self._applied_roi = None         # as reported by camera (snapped): (x, y, w, h)
        self._fps = 20.0                 # updated from cam.timing
        self._frame_count = 0            # throttle spectrum refresh

        # wiring
        self.btn_prev.clicked.connect(lambda: self._jump(-1))
        self.btn_next.clicked.connect(lambda: self._jump(+1))
        self.spn_win.valueChanged.connect(self._on_win_changed)
        self.spn_nfft.valueChanged.connect(self._on_nfft_changed)

        # listen to ROI + timing so we get parity + fps
        try:
            self.cam.roi.connect(self._on_roi_update, Qt.QueuedConnection)
        except Exception:
            pass
        try:
            self.cam.timing.connect(self._on_timing_update, Qt.QueuedConnection)
        except Exception:
            pass

        # start feed
        QTimer.singleShot(0, self._show_current)

    # ----------------- controls -----------------
    def _on_win_changed(self, v: int):
        self.WINDOW = int(v)
        # update deque maxlen if needed (grow only)
        newL = max(self.NFFT, self.WINDOW, len(self._c0), 4096)
        if newL > self._c0.maxlen:
            self._resize_deques(newL)

    def _on_nfft_changed(self, v: int):
        self.NFFT = int(v)
        newL = max(self.NFFT, self.WINDOW, len(self._c0), 4096)
        if newL > self._c0.maxlen:
            self._resize_deques(newL)

    def _resize_deques(self, newL: int):
        def grow(dq, L):
            tmp = list(dq)[-L:]
            dq.clear()
            dq.extend(tmp)
            dq.maxlen = L
        grow(self._c0, newL); grow(self._c45, newL)
        grow(self._c90, newL); grow(self._c135, newL)

    # ----------------- ROI + timing -----------------
    @Slot(dict)
    def _on_roi_update(self, d: dict) -> None:
        try:
            w = int(round(float(d.get("Width", 0))))
            h = int(round(float(d.get("Height", 0))))
            x = int(round(float(d.get("OffsetX", 0))))
            y = int(round(float(d.get("OffsetY", 0))))
            if w and h:
                self._applied_roi = (x, y, w, h)
        except Exception:
            pass

    @Slot(dict)
    def _on_timing_update(self, d: dict) -> None:
        try:
            rf = d.get("resulting_fps") or d.get("fps")
            if rf is not None:
                self._fps = float(rf)
        except Exception:
            pass

    # ----------------- navigation -----------------
    def _jump(self, d: int):
        if not self.spots:
            return
        self.idx = (self.idx + d) % len(self.spots)
        self._show_current()

    def _show_current(self):
        if not self.spots:
            return
        cx, cy, r, area, inten = self.spots[self.idx]
        # build a small ROI around the spot (will be snapped by driver)
        roi_w = roi_h = max(64, int(round(2.2 * r)))
        x0 = int(max(0, round(cx - roi_w/2)))
        y0 = int(max(0, round(cy - roi_h/2)))
        self._zoom_roi = (x0, y0, roi_w, roi_h)
        self.cam.set_zoom_roi(self._zoom_roi)
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

    # ----------------- per-frame -----------------
    def _on_zoom_frame(self, arr_obj: object) -> None:
        a = np.asarray(arr_obj)
        if a.ndim != 2:
            return
        H, W = a.shape

        # 1) square crop around the current spot (software, independent of snapped ROI stripes)
        try:
            cx, cy, r, _area, _inten = self.spots[self.idx]
        except Exception:
            return

        # absolute origin of the current view (for parity)
        if self._applied_roi is not None:
            ax, ay, aw, ah = self._applied_roi
        elif self._zoom_roi is not None:
            ax, ay, aw, ah = self._zoom_roi
        else:
            ax = ay = 0

        rel_cx = float(cx) - float(ax)
        rel_cy = float(cy) - float(ay)

        side = int(max(16, round(2.2 * max(4.0, float(r)))))
        half = side // 2
        x1 = max(0, min(W - 1, int(round(rel_cx)) - half))
        y1 = max(0, min(H - 1, int(round(rel_cy)) - half))
        x2 = min(W, x1 + side)
        y2 = min(H, y1 + side)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, W, H
        a_crop = a[y1:y2, x1:x2]

        # 2) show crop in the left panel
        ch, cw = a_crop.shape
        a8 = (a_crop >> 4).astype(np.uint8) if a_crop.dtype == np.uint16 else a_crop.astype(np.uint8, copy=False)
        if not a8.flags.c_contiguous:
            a8 = np.ascontiguousarray(a8)
        qimg = QImage(a8.data, cw, ch, cw, QImage.Format_Grayscale8)
        self.video.setPixmap(QPixmap.fromImage(qimg))

        # 3) accumulate per-pol means respecting the 2×2 mosaic parity
        # Layout assumption (as documented in app): (0,0)=90°, (0,1)=45°, (1,0)=135°, (1,1)=0°
        row0 = (ay + y1) & 1   # parity of the top row of the crop in sensor coords
        col0 = (ax + x1) & 1   # parity of the left col of the crop in sensor coords

        # channel slices
        s90  = a_crop[row0::2,          col0::2]          # (0,0) -> 90°
        s45  = a_crop[row0::2,          (col0 ^ 1)::2]    # (0,1) -> 45°
        s135 = a_crop[(row0 ^ 1)::2,    col0::2]          # (1,0) -> 135°
        s0   = a_crop[(row0 ^ 1)::2,    (col0 ^ 1)::2]    # (1,1) -> 0°

        # per-frame means
        m0   = float(s0.mean())   if s0.size   else 0.0
        m45  = float(s45.mean())  if s45.size  else 0.0
        m90  = float(s90.mean())  if s90.size  else 0.0
        m135 = float(s135.mean()) if s135.size else 0.0

        self._c0.append(m0); self._c45.append(m45); self._c90.append(m90); self._c135.append(m135)

        # 4) update spectrum at ~4 Hz to keep UI snappy
        self._frame_count = (self._frame_count + 1) % 4
        if self._frame_count == 0:
            self._update_spectrum()

    # ----------------- spectrum -----------------
    def _update_spectrum(self):
        # choose window length and nfft
        n = min(self.WINDOW, len(self._c0))
        if n < 64 or self._fps <= 0.1:
            return

        # collect last n samples per channel
        import numpy as _np
        c0   = _np.asarray(list(self._c0)[-n:], dtype=_np.float64)
        c45  = _np.asarray(list(self._c45)[-n:], dtype=_np.float64)
        c90  = _np.asarray(list(self._c90)[-n:], dtype=_np.float64)
        c135 = _np.asarray(list(self._c135)[-n:], dtype=_np.float64)

        # detrend & window
        def pre(x):
            x = x - x.mean()
            w = _np.hanning(len(x))
            return x * w, (w**2).sum()

        x0, w2_0 = pre(c0);   x45, w2_45 = pre(c45)
        x90, w2_90 = pre(c90); x135, w2_135 = pre(c135)

        # pick nfft (power of 2 up to NFFT)
        nfft = 1 << int(_np.floor(_np.log2(max(256, min(self.NFFT, 4*n)))))
        freqs = _np.fft.rfftfreq(nfft, d=1.0 / float(self._fps))

        def ps(x, w2):
            X = _np.fft.rfft(x, nfft)
            P = (X.real**2 + X.imag**2) / max(1.0, w2)   # scaled power
            return P

        P0   = ps(x0,   w2_0)
        P45  = ps(x45,  w2_45)
        P90  = ps(x90,  w2_90)
        P135 = ps(x135, w2_135)

        # update plot
        if self._line0 is None:
            (self._line0,)   = self.ax.plot(freqs, P0,  label="0°")
            (self._line45,)  = self.ax.plot(freqs, P45, label="45°")
            (self._line90,)  = self.ax.plot(freqs, P90, label="90°")
            (self._line135,) = self.ax.plot(freqs, P135,label="135°")
            self.ax.legend(loc="upper right")
        else:
            self._line0.set_data(freqs, P0)
            self._line45.set_data(freqs, P45)
            self._line90.set_data(freqs, P90)
            self._line135.set_data(freqs, P135)

        # keep x/ y limits sensible
        self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw_idle()

# ============================================================
# Main window
# ============================================================
class MainWindow(QMainWindow):
    # Qt signal used to ferry detect results safely back to GUI thread
    detect_done = Signal(object)
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PolarCam — ROI + Timing + Detect + Spots")
        self.resize(1280, 760)

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

        row = QHBoxLayout()
        for b in (self.btn_open, self.btn_start, self.btn_stop, self.btn_close):
            row.addWidget(b)
        row.addStretch(1)

        col = QVBoxLayout()
        col.addLayout(row)
        col.addWidget(self.video, 1)

        row = QHBoxLayout()
        col.addLayout(row)
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
        self._make_detect_dock()

        self.cam.roi.connect(self._on_roi)
        self.cam.timing.connect(self._on_timing)
        # detect result signal -> GUI thread
        self.detect_done.connect(self._handle_detect_done, Qt.QueuedConnection)

        self.btn_open.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_close.setEnabled(False)

        # detection & state
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._spots: List[Tuple[float, float, float, int, int]] = []
        self._collecting = False
        self._collect_gap = 1
        self._collect_frames: List[np.ndarray] = []
        self._detect_watchdog = None
        self._detect_t0 = 0.0
        self._last_fps = 20.0
        # auto-desat state
        self._desat_active = False
        self._desat_iters = 0
        self._desat_cooldown = 0
        self._desat_target = 3900  # 95% of 4095 by default
        self._desat_connected = False

    # ---------- ROI dock ----------
    def _make_roi_dock(self) -> None:
        dock = QDockWidget("ROI", self)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.ed_w = QLineEdit("2464")
        self.ed_h = QLineEdit("2056")
        self.ed_x = QLineEdit("0")
        self.ed_y = QLineEdit("0")
        self.btn_apply_roi = QPushButton("Apply")
        self.btn_full_roi = QPushButton("Full sensor")
        f.addRow("Width", self.ed_w)
        f.addRow("Height", self.ed_h)
        f.addRow("OffsetX", self.ed_x)
        f.addRow("OffsetY", self.ed_y)
        row = QWidget(); hb = QHBoxLayout(row); hb.setContentsMargins(0,0,0,0)
        hb.addWidget(self.btn_apply_roi); hb.addWidget(self.btn_full_roi)
        f.addRow(row)
        dock.setWidget(w)
        # wire
        self.btn_apply_roi.clicked.connect(self._apply_roi_clicked)
        self.btn_full_roi.clicked.connect(self._full_roi_clicked)

    # ---------- Timing dock ----------
    def _make_timing_dock(self) -> None:
        dock = QDockWidget("Timing", self)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.lbl_fps = QLabel("–")
        self.ed_fps = QLineEdit("19.97")
        self.ed_exp = QLineEdit("50.0")
        f.addRow("FPS (max)", self.lbl_fps)
        f.addRow("FPS (target)", self.ed_fps)
        f.addRow("Exposure (ms)", self.ed_exp)
        self.btn_apply_timing = QPushButton("Apply timing")
        f.addRow(self.btn_apply_timing)
        dock.setWidget(w)
        # wire
        self.btn_apply_timing.clicked.connect(self._apply_timing_clicked)

    # ---------- Detect & Spots dock ----------
    def _make_detect_dock(self) -> None:
        # Detect & spots dock (now with Auto-Desaturate)
        dock = QDockWidget("Detect & Spots", self)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)

        # params
        self.ed_dt = QLineEdit("10.0")     # ms between frames
        self.ed_diff = QLineEdit("50")     # |Δ| threshold
        self.ed_int = QLineEdit("200")     # intensity floor
        self.ed_minA = QLineEdit("6")      # min area
        self.ed_maxA = QLineEdit("5000")   # max area

        # auto-desaturate controls
        self.spn_peak = QDoubleSpinBox(); self.spn_peak.setRange(50.0, 99.9); self.spn_peak.setSingleStep(0.5); self.spn_peak.setDecimals(1); self.spn_peak.setValue(95.0)
        self.btn_autodesat = QPushButton("Auto Desaturate")

        self.btn_detect = QPushButton("Detect")
        self.btn_clear = QPushButton("Clear overlays")

        f.addRow("Δt (ms)", self.ed_dt)
        f.addRow("|Δ| threshold", self.ed_diff)
        f.addRow("Intensity floor", self.ed_int)
        f.addRow("Min area", self.ed_minA)
        f.addRow("Max area", self.ed_maxA)
        f.addRow("Peak target (%FS)", self.spn_peak)
        f.addRow(self.btn_autodesat)
        f.addRow(self.btn_detect)
        f.addRow(self.btn_clear)

        # spots row
        self.spot_combo = QComboBox()
        self.btn_go_spot = QPushButton("Go to spot")
        self.btn_add_spot = QPushButton("Add spot")
        self.btn_rm_spot = QPushButton("Remove spot")
        row2 = QWidget(); hb2 = QHBoxLayout(row2); hb2.setContentsMargins(0,0,0,0)
        hb2.addWidget(self.btn_go_spot); hb2.addWidget(self.btn_add_spot); hb2.addWidget(self.btn_rm_spot)
        f.addRow("Spots", self.spot_combo)
        f.addRow(row2)

        dock.setWidget(w)

        # wire
        self.btn_detect.clicked.connect(self._detect_clicked)
        self.btn_clear.clicked.connect(self._clear_overlays)
        self.btn_autodesat.clicked.connect(self._auto_desat_clicked)
        self.btn_go_spot.clicked.connect(self._go_to_selected_spot)
        self.btn_add_spot.clicked.connect(lambda: QMessageBox.information(self, "Add spot", "Manual add via mouse-pick will come later."))
        self.btn_rm_spot.clicked.connect(self._remove_spot)

    # ---------- Buttons ----------
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
        self.cam.set_roi(w, h, x, y)

    def _full_roi_clicked(self) -> None:
        self.cam.set_roi(1e9, 1e9, 0.0, 0.0)

    def _apply_timing_clicked(self) -> None:
        try:
            fps = float(self.ed_fps.text())
            exp_ms = float(self.ed_exp.text())
        except Exception:
            self.status.showMessage("Timing: invalid numbers", 2000); return

        self.cam.set_timing(fps, exp_ms)

        # optional status hint
        self.status.showMessage(f"Timing queued: fps={fps:.3f}, exp={exp_ms:.3f} ms", 1500)

    # ---------- Detect flow ----------
    def _detect_clicked(self) -> None:
        # If desat is running, cancel it so Detect can proceed cleanly
        if self._desat_active:
            print("[Detect] Cancelling Auto-Desaturate (detect requested)")
            self._stop_auto_desat("detect-requested")

        if self._collecting:
            print("[Detect] Ignored: already collecting")
            return

        try:
            dt_ms = float(self.ed_dt.text())
        except Exception:
            dt_ms = 10.0
        self._begin_collect(dt_ms)

    def _begin_collect(self, dt_ms: float) -> None:
        if self._collecting:
            print("[Detect] Ignored: already collecting")
            return
        # choose a frame gap ~ dt_ms using last known fps
        self._detect_t0 = time.perf_counter()
        fps_now = getattr(self, "_last_fps", 20.0)
        try:
            # prefer the label if it contains a parseable number
            txt = self.lbl_fps.text()
            if txt not in ("-", "–", "", None):
                fps_now = float(txt)
        except Exception:
            pass
        self._collect_gap = max(1, int(round(dt_ms * fps_now / 1000.0)))
        need = self._collect_gap + 1
        print(f"[Detect] Begin capture: dt_ms={dt_ms:.2f}  fps≈{fps_now:.3f}  gap={self._collect_gap}  need={need} frames")
        self._collect_frames.clear()
        self._collecting = True
        self.btn_detect.setEnabled(False)
        self.status.showMessage(f"Detect: capturing (need {need} frames, gap={self._collect_gap})…", 0)
        # connect capture handler
        self.cam.frame.connect(self._collect_for_detect, Qt.QueuedConnection)
        # arm watchdog (8 s)
        try:
            self._stop_detect_watchdog()
        except Exception:
            pass
        self._detect_watchdog = QTimer(self)
        self._detect_watchdog.setSingleShot(True)
        self._detect_watchdog.timeout.connect(self._on_detect_timeout)
        self._detect_watchdog.start(8000)
        print("[Detect] Watchdog armed: 8000 ms")

    def _collect_for_detect(self, arr_obj: object) -> None:
        a = np.asarray(arr_obj)
        # if auto-desat is running, feed that state machine and skip detection capture
        if self._desat_active:
            try:
                self._on_auto_desat_frame(a)
            except Exception as e:
                print(f"[AutoDesat] ERROR: {e}")
            return
        if a.ndim != 2:
            return
        if a.ndim != 2:
            return
        self._collect_frames.append(a.copy())
        need = self._collect_gap + 1
        have = len(self._collect_frames)
        if have < need:
            self.status.showMessage(f"Detect: capturing {have}/{need}…", 0)
            print(f"[Detect] Captured {have}/{need} frames…")
            return
        # got enough frames — stop listening and process
        try:
            self.cam.frame.disconnect(self._collect_for_detect)
        except Exception:
            pass
        self._collecting = False
        prev = self._collect_frames[0]
        cur = self._collect_frames[-1]
        print(f"[Detect] Processing pair: prev shape={prev.shape} dtype={prev.dtype}  cur shape={cur.shape} dtype={cur.dtype}")
        print(f"[Detect] Stats: prev[min={prev.min()} max={prev.max()}]  cur[min={cur.min()} max={cur.max()}]")
        # parse thresholds
        def _num(le: QLineEdit, default: float) -> float:
            try:
                return float(le.text())
            except Exception:
                return default
        diff_thr = _num(self.ed_diff, 50.0)
        inten_thr = _num(self.ed_int, 200.0)
        try: minA = int(float(self.ed_minA.text()))
        except Exception: minA = 6
        try: maxA = int(float(self.ed_maxA.text()))
        except Exception: maxA = 5000
        print(f"[Detect] Thresholds: |Δ|≥{diff_thr}  I≥{inten_thr}  area∈[{minA},{maxA}]")

        def _work(prev, cur, diff_thr, inten_thr, minA, maxA):
            # NOTE: inten_thr is currently unused in the pure-diff pipeline (by request)
            t0 = time.perf_counter()

            # ---- pure |Δ| (no pre-flooring) ----
            prev32 = prev.astype(np.int32, copy=False)
            cur32  = cur.astype(np.int32, copy=False)
            diff   = np.abs(cur32 - prev32)
            t1 = time.perf_counter()

            # threshold on diff only
            mask0 = (diff >= diff_thr)
            n_true0 = int(mask0.sum())
            if not n_true0:
                print(f"[Detect] mask empty (diff {1000*(t1-t0):.1f} ms).")
                return []

            # ---- morphology: Opening -> remove_small_objects -> Closing ----
            # Opening first to de-speckle, use a slightly larger SE for robustness
            se_open  = disk(2)
            se_close = disk(1)

            mask1 = binary_opening(mask0, se_open)
            t2 = time.perf_counter()

            # Early cull dust before labeling; use minA as size cutoff
            mask2 = remove_small_objects(mask1, min_size=max(1, int(minA)), connectivity=2)
            t3 = time.perf_counter()

            # Mend tiny gaps after de-speckle
            mask3 = binary_closing(mask2, se_close)
            t4 = time.perf_counter()

            # Label with 4-connectivity to reduce diagonal merging
            lbl = label(mask3, connectivity=1)
            t5 = time.perf_counter()

            nlab = int(lbl.max())
            dets = []
            for r in regionprops(lbl):
                area = int(r.area)
                if area < minA or area > maxA:
                    continue
                cy, cx = r.centroid
                rad = float((area / np.pi) ** 0.5)
                iy, ix = int(round(cy)), int(round(cx))
                iy = max(0, min(cur.shape[0] - 1, iy))
                ix = max(0, min(cur.shape[1] - 1, ix))
                inten = int(cur[iy, ix])  # intensity from 'cur' frame
                dets.append((float(cx), float(cy), rad, area, inten))
            t6 = time.perf_counter()

            print(
                "[Detect] timings: diff={:.1f}ms  open={:.1f}ms  smallobj={:.1f}ms  close={:.1f}ms  label={:.1f}ms  props={:.1f}ms  true0={}  true3={}  labels={}".format(
                    1000 * (t1 - t0),
                    1000 * (t2 - t1),
                    1000 * (t3 - t2),
                    1000 * (t4 - t3),
                    1000 * (t5 - t4),
                    1000 * (t6 - t5),
                    n_true0,
                    int(mask3.sum()),
                    nlab,
                )
            )
            dets.sort(key=lambda t: -t[4])
            return dets

        fut = self._executor.submit(_work, prev, cur, diff_thr, inten_thr, minA, maxA)
        print("[Detect] Worker submitted")

        def _post_detect_result(f):
            try:
                dets = f.result()
            except Exception as e:
                self._stop_detect_watchdog()
                # recover UI and show message instead of hanging
                try:
                    self.btn_detect.setEnabled(True)
                except Exception:
                    pass
                try:
                    self.status.showMessage(f"Detect failed: {e}", 5000)
                except Exception:
                    pass
                print(f"[Detect] ERROR: {e}")
                return
            self._stop_detect_watchdog()
            elapsed_ms = (time.perf_counter() - self._detect_t0) * 1000.0
            print(f"[Detect] Done: {len(dets)} spot(s) in {elapsed_ms:.1f} ms")
            self._on_detect_ready(dets)
            try:
                self.status.showMessage(f"Detect: {len(dets)} spot(s) in {elapsed_ms:.0f} ms.", 3000)
            except Exception:
                pass

        def _emit_from_future(f):
            try:
                dets = f.result()
                self.detect_done.emit(("ok", dets, self._detect_t0))
            except Exception as e:
                self.detect_done.emit(("err", str(e), self._detect_t0))
        fut.add_done_callback(_emit_from_future)

    def _on_detect_ready(self, dets: list) -> None:
        self._spots = dets or []
        self.btn_detect.setEnabled(True)
        if not dets:
            self.status.showMessage("Detect: no spots.", 3000)
            return
        self.status.showMessage(f"Detect: {len(dets)} spot(s).", 3000)
        self._refresh_spot_combo()

    def _clear_overlays(self):
        self._spots = []
        self._refresh_spot_combo()
        # next frame draw will show no overlays

    def _refresh_spot_combo(self):
        self.spot_combo.clear()
        for i, (cx, cy, r, area, inten) in enumerate(self._spots):
            self.spot_combo.addItem(f"{i+1}: (x={cx:.1f}, y={cy:.1f}) A={area}")

    def _remove_spot(self):
        idx = self.spot_combo.currentIndex()
        if 0 <= idx < len(self._spots):
            self._spots.pop(idx)
            self._refresh_spot_combo()

    def _go_to_selected_spot(self):
        if not self._spots:
            QMessageBox.information(self, "Go to spot", "No spots available. Run Detect first."); return
        idx = self.spot_combo.currentIndex()
        try:
            _ = self._spots[idx]
        except Exception:
            QMessageBox.warning(self, "Go to spot", "Invalid selection."); return
        self._spot_view = SpotViewerWindow(self.cam, self._spots, idx, parent=self)
        self._spot_view.show()

    # ---------- signal handlers ----------
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
        a8 = (a >> 4).astype(np.uint8) if a.dtype == np.uint16 else a.astype(np.uint8, copy=False)
        if not a8.flags.c_contiguous:
            a8 = np.ascontiguousarray(a8)
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        img = qimg.copy()

        # overlays on a copy
        if self._spots:
            p = QPainter(img)
            pen = QPen(Qt.green); pen.setWidth(2)
            p.setPen(pen); p.setFont(QFont("", 10))
            full_scale = 4095.0 if a.dtype == np.uint16 else 255.0
            for i, (cx, cy, r, area, inten) in enumerate(self._spots):
                p.drawEllipse(QPointF(cx, cy), r, r)
                p.drawText(int(cx + 6), int(cy - 6), f"{i+1}  I={inten}  A={area}  {100.0*inten/full_scale:.1f}%")
            p.end()

        self.video.setPixmap(QPixmap.fromImage(img))
    def _on_roi(self, d: dict) -> None:
        """Update ROI fields from worker snapshot."""
        try:
            pairs = [("Width", self.ed_w), ("Height", self.ed_h), ("OffsetX", self.ed_x), ("OffsetY", self.ed_y)]
            for name, widget in pairs:
                v = d.get(name)
                if v is not None:
                    try:
                        widget.setText(str(int(v)))
                    except Exception:
                        widget.setText(str(v))
        except Exception:
            pass

    def _on_timing(self, d: dict) -> None:
        """Update timing fields and keep track of current FPS."""
        try:
            fps_max = d.get("fps_max")
            if fps_max is not None and hasattr(self, "lbl_fps"):
                try:
                    self.lbl_fps.setText(f"{float(fps_max):.3f}")
                except Exception:
                    self.lbl_fps.setText(str(fps_max))
            fps = d.get("fps")
            if fps is not None and hasattr(self, "ed_fps"):
                try:
                    self.ed_fps.setText(f"{float(fps):.3f}")
                except Exception:
                    self.ed_fps.setText(str(fps))
            exp_us = d.get("exposure_us")
            if exp_us is not None and hasattr(self, "ed_exp"):
                try:
                    self.ed_exp.setText(f"{float(exp_us)/1000.0:.3f}")
                except Exception:
                    self.ed_exp.setText(str(exp_us))
            rf = d.get("resulting_fps") or d.get("fps")
            if rf is not None:
                try:
                    self._last_fps = float(rf)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_detect_timeout(self):
        """Watchdog fired: detection took too long. Recover UI and disconnect."""
        try:
            have = len(self._collect_frames)
        except Exception:
            have = 0
        elapsed_ms = (time.perf_counter() - getattr(self, "_detect_t0", time.perf_counter())) * 1000.0
        print(f"[Detect] Watchdog TIMEOUT after {elapsed_ms:.1f} ms (frames collected={have}).")
        try:
            self.cam.frame.disconnect(self._collect_for_detect)
        except Exception:
            pass
        self._collecting = False
        self.btn_detect.setEnabled(True)
        try:
            self.status.showMessage(f"Detect timed out after {elapsed_ms:.0f} ms.", 4000)
        except Exception:
            pass

    def _stop_detect_watchdog(self):
        try:
            if self._detect_watchdog is not None:
                if self._detect_watchdog.isActive():
                    self._detect_watchdog.stop()
                self._detect_watchdog.deleteLater()
        except Exception:
            pass
        self._detect_watchdog = None
    def _handle_detect_done(self, payload: object) -> None:
        """Receive detect results on the GUI thread via signal."""
        self._stop_detect_watchdog()
        try:
            kind, data, t0 = payload
        except Exception:
            # defensive
            self.btn_detect.setEnabled(True)
            self.status.showMessage("Detect failed: bad payload", 4000)
            print("[Detect] ERROR: bad payload", payload)
            return
        if kind == "ok":
            elapsed_ms = (time.perf_counter() - (t0 or time.perf_counter())) * 1000.0
            dets = data or []
            print(f"[Detect] Done: {len(dets)} spot(s) in {elapsed_ms:.1f} ms")
            self._on_detect_ready(dets)
            try:
                self.status.showMessage(f"Detect: {len(dets)} spot(s) in {elapsed_ms:.0f} ms.", 3000)
            except Exception:
                pass
        else:
            try:
                self.btn_detect.setEnabled(True)
            except Exception:
                pass
            try:
                self.status.showMessage(f"Detect failed: {data}", 5000)
            except Exception:
                pass
            print(f"[Detect] ERROR: {data}")
    # ---------- Auto-desaturate ----------
    def _stop_auto_desat(self, reason: str = "done") -> None:
        """Disconnect from frame stream and reset desat state."""
        if self._desat_connected:
            try:
                self.cam.frame.disconnect(self._on_auto_desat_frame)
                print(f"[AutoDesat] Disconnected from frame signal ({reason})")
            except Exception as e:
                print(f"[AutoDesat] Disconnect warn: {e}")
        self._desat_connected = False
        self._desat_active = False

    def _auto_desat_clicked(self) -> None:
        # toggle: click again to cancel
        if self._desat_active:
            self._stop_auto_desat("cancelled by user")
            self.status.showMessage("Auto-desaturate: cancelled.", 2000)
            return

        if not self.btn_stop.isEnabled():
            self.status.showMessage("Start acquisition first.", 2000)
            print("[AutoDesat] Ignored: not acquiring")
            return

        full_scale = 4095.0
        pct = float(self.spn_peak.value())
        self._desat_target = int(round(full_scale * pct / 100.0))
        self._desat_active = True
        self._desat_iters = 0
        self._desat_cooldown = 0

        # subscribe to live frames
        if not self._desat_connected:
            self.cam.frame.connect(self._on_auto_desat_frame, Qt.QueuedConnection)
            self._desat_connected = True
            print("[AutoDesat] Connected to frame signal")

        self.status.showMessage(
            f"Auto-desaturate: targeting ≤ {self._desat_target} (≈{pct:.1f}% FS)", 0
        )
        print(f"[AutoDesat] Start: target={self._desat_target} DN (pct={pct:.1f})")

    def _on_auto_desat_frame(self, a: np.ndarray) -> None:
        if not getattr(self, "_desat_active", False):
            return
        if a is None or getattr(a, "ndim", 0) != 2:
            print("[AutoDesat] Skip: bad frame"); return
        if self._desat_cooldown > 0:
            self._desat_cooldown -= 1
            print(f"[AutoDesat] Cooldown… {self._desat_cooldown} frame(s) left")
            return

        full = 4095.0
        cur_max = int(a.max())
        p995 = float(np.percentile(a, 99.5))
        sat_cnt = int((a >= full - 5).sum())
        sat_pct = 100.0 * sat_cnt / a.size

        try:
            exp_ms = float(self.ed_exp.text())
        except Exception:
            exp_ms = 50.0

        print(f"[AutoDesat] stats: max={cur_max}  p99.5={p995:.0f}  sat={sat_pct:.3f}%  exp={exp_ms:.3f} ms  target={self._desat_target}  iter={self._desat_iters}")

        # Done if the *percentile* is under target (robust to 1–2 hot pixels)
        if p995 <= self._desat_target or (cur_max <= self._desat_target and sat_pct < 0.05):
            msg = f"Auto-desaturate: peak~p99.5={p995:.0f} (done in {self._desat_iters} step(s))."
            self.status.showMessage(msg, 3000)
            print("[AutoDesat] Done (percentile criterion).")
            try: self._stop_auto_desat("done")
            except Exception: self._desat_active = False
            return

        # Allow more attempts (the old code used 6)
        if self._desat_iters >= 12:
            self.status.showMessage(f"Auto-desaturate: stopping after {self._desat_iters} steps (p99.5={p995:.0f}).", 3000)
            print(f"[AutoDesat] Max steps reached (p99.5={p995:.0f}).")
            try: self._stop_auto_desat("max-steps")
            except Exception: self._desat_active = False
            return

        # Compute next exposure — be more aggressive if many pixels are pinned
        base = self._desat_target / max(1.0, float(cur_max))
        if sat_pct > 1.0 or cur_max >= full - 1:
            # heavy saturation — take a bigger bite
            factor = max(0.05, min(0.80, 0.70 * base))
        else:
            # normal case
            factor = max(0.05, min(0.95, 0.90 * base))

        new_exp_ms = max(0.05, exp_ms * factor)
        print(f"[AutoDesat] Adjust: factor={factor:.3f}  new_exp={new_exp_ms:.3f} ms")

        self.ed_exp.setText(f"{new_exp_ms:.3f}")
        try:
            self.cam.set_timing(None, new_exp_ms)
            print(f"[AutoDesat] set_timing(None, {new_exp_ms:.3f}) sent")
        except Exception as e:
            print(f"[AutoDesat] ERROR applying exposure: {e}")
            self.status.showMessage(f"Auto-desaturate: failed to set exposure ({e})", 3000)
            try: self._stop_auto_desat("error")
            except Exception: self._desat_active = False
            return

        self._desat_iters += 1
        self._desat_cooldown = 3
        self.status.showMessage(f"Auto-desaturate: set exposure {new_exp_ms:.3f} ms (iter {self._desat_iters}).", 0)

    def _await_signal(self, signal, timeout_ms: int = 3000) -> bool:
        """Block the GUI event loop until `signal` fires or `timeout_ms` elapses.
        Returns True if signal arrived, False on timeout."""
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

        success = timer.isActive()  # if timer still active, the signal fired first
        # cleanup
        try: signal.disconnect(done)
        except Exception: pass
        try: timer.timeout.disconnect(done)
        except Exception: pass
        return success

    # ---------- safe shutdown ----------
    def safe_shutdown(self):
        """Ensure acquisition stops and the device closes before the window/app die."""
        # If a detect capture is active, disconnect it and stop the watchdog
        try:
            if self._collecting:
                try:
                    self.cam.frame.disconnect(self._collect_for_detect)
                except Exception:
                    pass
                self._collecting = False
            self._stop_detect_watchdog()
        except Exception:
            pass

        # Stop acquisition if running, and wait for 'stopped'
        try:
            if self.btn_stop.isEnabled():
                self.cam.stop()
                self._await_signal(self.cam.stopped, 3000)
        except Exception:
            pass

        # Close device if open, and wait for 'closed'
        try:
            if self.btn_close.isEnabled():
                self.cam.close()
                self._await_signal(self.cam.closed, 3000)
        except Exception:
            pass

        # Shut down any detection worker
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
    app.aboutToQuit.connect(w.safe_shutdown)  # <-- add this
    ret = app.exec()
    try:
        w.safe_shutdown()  # extra belt-and-braces
    except Exception:
        pass
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
