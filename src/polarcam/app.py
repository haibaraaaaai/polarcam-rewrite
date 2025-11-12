# ------------------------------------------
# File: src/polarcam/app.py
# ------------------------------------------

import sys
from collections import deque
from typing import Optional, List, Tuple

import numpy as np
import time
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
    QCheckBox,
)

# scikit-image bits for labeling
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.measure import label, regionprops

from .ids_backend import IDSCamera


# =========================
# Helpers: tone LUT (12→8)
# =========================
def make_tone_lut12_to_8(
    black_adu: int = 48,
    white_adu: int = 3800,
    mid: float = 0.55,
    contrast: float = 7.0,  # 4..10
    shadow_cut: float = 0.08,
    knee: float = 1.7
) -> np.ndarray:
    """Fast, display-only tone curve: shadow crush + S-curve + soft knee."""
    try:
        black_adu = int(np.clip(black_adu, 0, 4094))
        white_adu = int(np.clip(white_adu, black_adu + 1, 4095))
        x = np.arange(4096, dtype=np.float32)
        t = (x - black_adu) / max(1.0, (white_adu - black_adu))
        t = np.clip(t, 0.0, 1.0)
        # dead zone near black
        t = np.where(t <= shadow_cut, 0.0, (t - shadow_cut) / (1.0 - shadow_cut))
        # logistic S-curve centered at `mid`
        width = max(1e-3, 1.0 / contrast)
        s = 1.0 / (1.0 + np.exp(-(t - mid) / width))
        s0 = 1.0 / (1.0 + np.exp(-(0.5 - mid) / width))
        s = np.clip((s - s0) / (1.0 - s0), 0.0, 1.0)
        # highlight knee
        y = 1.0 - np.power(1.0 - s, knee)
        lut = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
        print(f"[Tone] LUT built: black={black_adu} white={white_adu} mid={mid} contrast={contrast} knee={knee} shadow_cut={shadow_cut}")
        return lut
    except Exception as e:
        print(f"[Tone] ERROR building LUT: {e}. Falling back to >>4.")
        return (np.arange(4096, dtype=np.uint16) >> 4).astype(np.uint8)


# ============================================================
# Spot viewer (zoom ROI + simple FFT placeholders)
# ============================================================
class SpotViewerWindow(QMainWindow):
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
        gb = QGroupBox("Live FFT peaks"); g = QGridLayout(gb)
        g.addWidget(self.lbl_fft0, 0, 0); g.addWidget(self.lbl_fft45, 1, 0)
        g.addWidget(self.lbl_fft90, 2, 0); g.addWidget(self.lbl_fft135, 3, 0)
        right.addWidget(gb)

        nav = QHBoxLayout()
        nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next); nav.addStretch(1); nav.addWidget(self.btn_close)
        right.addLayout(nav)

        w = QWidget(); h = QHBoxLayout(w)
        h.addLayout(left, 2); h.addLayout(right, 1)
        self.setCentralWidget(w)

        # FFT state (placeholder)
        self.WINDOW = 8000; self.NFFT = 32000
        L = max(self.NFFT, self.WINDOW)
        self._c0 = deque(maxlen=L); self._c45 = deque(maxlen=L)
        self._c90 = deque(maxlen=L); self._c135 = deque(maxlen=L)

        self._zoom_roi = None
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
        # 2x2 block radius to sensor pixels — just use a reasonable small ROI
        roi_w = roi_h = max(64, int(round(2.2 * r)))
        x0 = int(max(0, round(cx - roi_w/2))); y0 = int(max(0, round(cy - roi_h/2)))
        self._zoom_roi = (x0, y0, roi_w, roi_h)
        try:
            self.cam.set_zoom_roi(self._zoom_roi)
            print(f"[SpotView] set_zoom_roi -> {self._zoom_roi}")
        except Exception as e:
            print(f"[SpotView] set_zoom_roi not supported: {e}")
        try:
            self.cam.frame.connect(self._on_zoom_frame, Qt.QueuedConnection)
        except Exception as e:
            print(f"[SpotView] ERROR connect frame: {e}")

    def closeEvent(self, e):
        try:
            self.cam.clear_zoom_roi()
        except Exception:
            pass
        try:
            self.cam.frame.disconnect(self._on_zoom_frame)
        except Exception:
            pass
        super().closeEvent(e)

    def _on_zoom_frame(self, arr_obj: object) -> None:
        a = np.asarray(arr_obj)
        if a.ndim != 2:
            return
        h, w = a.shape
        a8 = (a >> 4).astype(np.uint8) if a.dtype == np.uint16 else a.astype(np.uint8, copy=False)
        if not a8.flags.c_contiguous:
            a8 = np.ascontiguousarray(a8)
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        self.video.setPixmap(QPixmap.fromImage(qimg))
        # placeholder "FFT peaks"
        self._c0.append(int(a.mean()))
        self._c45.append(int(a[h//2:, :].mean()))
        self._c90.append(int(a[:, w//2:].mean()))
        self._c135.append(int(a[:h//2, :w//2].mean()))
        self.lbl_fft0.setText(f"0°: {self._c0[-1]}")
        self.lbl_fft45.setText(f"45°: {self._c45[-1]}")
        self.lbl_fft90.setText(f"90°: {self._c90[-1]}")
        self.lbl_fft135.setText(f"135°: {self._c135[-1]}")


# ============================================================
# Main window
# ============================================================
class MainWindow(QMainWindow):
    detect_done = Signal(object)  # (kind, data, t0)
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PolarCam — ROI · Timing · Detect · VarMap · Tone")
        self.resize(1280, 780)

        # --- central video ---
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.video = QLabel("No video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#777;")
        self.video.setMinimumSize(640, 480)

        self.btn_open = QPushButton("Open")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_close = QPushButton("Close")

        row = QHBoxLayout()
        for b in (self.btn_open, self.btn_start, self.btn_stop, self.btn_close): row.addWidget(b)
        row.addStretch(1)

        col = QVBoxLayout(); col.addLayout(row); col.addWidget(self.video, 1)
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
        self._make_tone_dock()

        self.cam.roi.connect(self._on_roi)
        self.cam.timing.connect(self._on_timing)
        self.detect_done.connect(self._handle_detect_done, Qt.QueuedConnection)

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
        self._desat_target = 3900

        # tone preview
        self._tone_mode = "off"
        self._tone_lut = (np.arange(4096, dtype=np.uint16) >> 4).astype(np.uint8)

        # varmap state
        self._pv_collecting = False
        self._pv_frames: List[np.ndarray] = []
        self._pv_watchdog: Optional[QTimer] = None
        self._pv_showing = False   # when True: freeze live and show varmap
        self._last_qimage: Optional[QImage] = None

        # initial buttons
        self.btn_open.setEnabled(True); self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False); self.btn_close.setEnabled(False)

    # -------------------------
    # Docks
    # -------------------------
    def _make_roi_dock(self) -> None:
        dock = QDockWidget("ROI", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.ed_w = QLineEdit("2464"); self.ed_h = QLineEdit("2056")
        self.ed_x = QLineEdit("0"); self.ed_y = QLineEdit("0")
        self.btn_apply_roi = QPushButton("Apply")
        self.btn_full_roi = QPushButton("Full sensor")
        f.addRow("Width", self.ed_w); f.addRow("Height", self.ed_h)
        f.addRow("OffsetX", self.ed_x); f.addRow("OffsetY", self.ed_y)
        row = QHBoxLayout(); row.addWidget(self.btn_apply_roi); row.addWidget(self.btn_full_roi)
        box = QWidget(); box.setLayout(row); f.addRow(box)
        dock.setWidget(w)
        self.btn_apply_roi.clicked.connect(self._apply_roi_clicked)
        self.btn_full_roi.clicked.connect(self._full_roi_clicked)

    def _make_timing_dock(self) -> None:
        dock = QDockWidget("Timing", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.lbl_fps = QLabel("–"); self.ed_fps = QLineEdit("19.97"); self.ed_exp = QLineEdit("50.0")
        self.btn_apply_timing = QPushButton("Apply timing")
        f.addRow("FPS (max)", self.lbl_fps); f.addRow("FPS (target)", self.ed_fps)
        f.addRow("Exposure (ms)", self.ed_exp); f.addRow(self.btn_apply_timing)
        dock.setWidget(w)
        self.btn_apply_timing.clicked.connect(self._apply_timing_clicked)

    def _make_detect_dock(self) -> None:
        dock = QDockWidget("Detect / VarMap", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)

        # 1) frame diff detection params
        self.ed_dt = QLineEdit("10.0")
        self.ed_diff = QLineEdit("50")
        self.ed_int = QLineEdit("200")
        self.ed_minA = QLineEdit("6")
        self.ed_maxA = QLineEdit("5000")

        # auto-desat
        self.spn_peak = QDoubleSpinBox(); self.spn_peak.setRange(50.0, 99.9); self.spn_peak.setSingleStep(0.5); self.spn_peak.setDecimals(1); self.spn_peak.setValue(95.0)
        self.btn_autodesat = QPushButton("Auto Desaturate")

        self.btn_detect = QPushButton("Detect (Δ frames)")
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

        # 2) variance-map controls
        self.btn_varmap = QPushButton("VarMap Detect (≈1 s)")
        self.ed_Mfloor = QLineEdit("40")      # post-variance floor (ADU)
        self.ed_balance = QLineEdit("0.30")   # min(min(Dx,Dy)/max(Dx,Dy))
        self.ed_Mscale = QLineEdit("384")     # fixed display scale if not auto
        self.chk_Mauto = QCheckBox("Auto-scale (99th pct)"); self.chk_Mauto.setChecked(True)
        self.btn_back_live = QPushButton("Back to live")

        f.addRow(self.btn_varmap)
        f.addRow("M floor (ADU)", self.ed_Mfloor)
        f.addRow("Balance min", self.ed_balance)
        f.addRow("Display scale", self.ed_Mscale)
        f.addRow(self.chk_Mauto)
        f.addRow(self.btn_back_live)

        # spots row (shared)
        self.spot_combo = QComboBox()
        self.btn_go_spot = QPushButton("Go to spot")
        self.btn_rm_spot = QPushButton("Remove spot")
        row2 = QHBoxLayout(); row2.addWidget(self.btn_go_spot); row2.addWidget(self.btn_rm_spot)
        box = QWidget(); box.setLayout(row2)
        f.addRow("Spots", self.spot_combo); f.addRow(box)

        w.setLayout(f); dock.setWidget(w)

        # wire
        self.btn_detect.clicked.connect(self._detect_clicked)
        self.btn_clear.clicked.connect(self._clear_overlays)
        self.btn_autodesat.clicked.connect(self._auto_desat_clicked)
        self.btn_varmap.clicked.connect(self._polvar_clicked)
        self.btn_back_live.clicked.connect(self._back_to_live)
        self.btn_go_spot.clicked.connect(self._go_to_selected_spot)
        self.btn_rm_spot.clicked.connect(self._remove_spot)

    def _make_tone_dock(self) -> None:
        dock = QDockWidget("Tone (preview only)", self); self.addDockWidget(Qt.RightDockWidgetArea, dock)
        w = QWidget(); f = QFormLayout(w)
        self.cmb_tone = QComboBox(); self.cmb_tone.addItems(["off", "crush"])
        self.btn_rebuild_tone = QPushButton("Rebuild LUT")
        f.addRow("Mode", self.cmb_tone); f.addRow(self.btn_rebuild_tone)
        dock.setWidget(w)
        self.cmb_tone.currentTextChanged.connect(self._tone_changed)
        self.btn_rebuild_tone.clicked.connect(self._rebuild_tone)

    # -------------------------
    # Buttons
    # -------------------------
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
        print(f"[UI] set_roi(w={w}, h={h}, x={x}, y={y}); acquiring={self.btn_stop.isEnabled()}")
        self.cam.enqueue_roi({"Width": w, "Height": h, "OffsetX": x, "OffsetY": y})
        self.cam.process_pending_roi()

    def _full_roi_clicked(self) -> None:
        if hasattr(self.cam, "process_full_roi"):
            print("[UI] Full sensor requested"); self.cam.process_full_roi()
        else:
            print("[UI] Full sensor not supported by backend")
            self.status.showMessage("Full sensor not supported by backend.", 3000)

    def _apply_timing_clicked(self) -> None:
        try:
            fps = float(self.ed_fps.text()); exp_ms = float(self.ed_exp.text())
        except Exception:
            self.status.showMessage("Timing: invalid numbers", 2000); return
        print(f"[UI] set_timing(fps={fps:.3f}, exposure_ms={exp_ms}); acquiring={self.btn_stop.isEnabled()}")
        self.cam.enqueue_timing({"fps": fps, "exposure_ms": exp_ms})
        self.cam.process_pending_timing()

    # -------------------------
    # Live frame display
    # -------------------------
    def _to_u8_preview(self, a: np.ndarray) -> np.ndarray:
        if a.dtype == np.uint16:
            if self._tone_mode == "crush":
                return self._tone_lut[a]
            return (a >> 4).astype(np.uint8)
        return a.astype(np.uint8, copy=False)

    def _show_qimage(self, qimg: QImage):
        """Fit-to-label with aspect ratio; store last for resize."""
        try:
            target = self.video.size()
            pm = QPixmap.fromImage(qimg).scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video.setPixmap(pm)
            self._last_qimage = qimg
        except Exception as e:
            print(f"[UI] _show_qimage ERROR: {e}")

    def resizeEvent(self, e):
        # re-fit last image on resize
        try:
            if self._last_qimage is not None:
                self._show_qimage(self._last_qimage)
        except Exception:
            pass
        super().resizeEvent(e)

    def _on_frame(self, arr_obj: object) -> None:
        if self._pv_showing:
            # showing varmap; ignore live frames
            return
        a = np.asarray(arr_obj)
        if a.ndim != 2:
            return
        h, w = a.shape
        a8 = self._to_u8_preview(a)
        if not a8.flags.c_contiguous:
            a8 = np.ascontiguousarray(a8)
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        img = qimg.copy()

        # overlays (from last detect-by-Δ)
        if self._spots:
            p = QPainter(img); pen = QPen(Qt.green); pen.setWidth(2)
            p.setPen(pen); p.setFont(QFont("", 10))
            full_scale = 4095.0 if a.dtype == np.uint16 else 255.0
            for i, (cx, cy, r, area, inten) in enumerate(self._spots):
                p.drawEllipse(QPointF(cx, cy), r, r)
                p.drawText(int(cx + 6), int(cy - 6), f"{i+1}  I={inten}  A={area}  {100.0*inten/full_scale:.1f}%")
            p.end()

        self._show_qimage(img)

    # -------------------------
    # Frame-diff detection
    # -------------------------
    def _detect_clicked(self) -> None:
        if self._pv_showing:
            self.status.showMessage("Currently showing VarMap. Click 'Back to live' first.", 3000)
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
        self._detect_t0 = time.perf_counter()
        fps_now = getattr(self, "_last_fps", 20.0)
        try:
            txt = self.lbl_fps.text()
            if txt not in ("-", "–", "", None): fps_now = float(txt)
        except Exception: pass
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
        if self._desat_active:
            try: self._on_auto_desat_frame(a)
            except Exception as e: print(f"[AutoDesat] ERROR: {e}")
            return
        if a.ndim != 2: return

        self._collect_frames.append(a.copy())
        need = self._collect_gap + 1; have = len(self._collect_frames)
        if have < need:
            self.status.showMessage(f"Detect: capturing {have}/{need}…", 0)
            print(f"[Detect] Captured {have}/{need} frames…")
            return

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
            prev32 = prev.astype(np.int32, copy=False)
            cur32  = cur.astype(np.int32, copy=False)
            # No pre-flooring now; rely on diff_thr
            diff = np.abs(cur32 - prev32); t1 = time.perf_counter()
            mask = (diff >= diff_thr)
            n_true0 = int(mask.sum())
            if not n_true0:
                print(f"[Detect] mask empty (diff {1000*(t1-t0):.1f} ms).")
                return []
            se = disk(1)
            # small object suppression
            mask = binary_opening(mask, se); t2 = time.perf_counter()
            mask = binary_closing(mask, se);  t3 = time.perf_counter()
            lbl = label(mask, connectivity=2); t4 = time.perf_counter()
            nlab = int(lbl.max())
            dets = []
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
            print("[Detect] timings: diff={:.1f}ms  open={:.1f}ms  close={:.1f}ms  label={:.1f}ms  props={:.1f}ms  true={}  labels={}".format(
                1000*(t1-t0), 1000*(t2-t1), 1000*(t3-t2), 1000*(t4-t3), 1000*(t5-t4), n_true0, nlab))
            dets.sort(key=lambda t: -t[4])
            return dets

        fut = self._executor.submit(_work, prev, cur, diff_thr, inten_thr, minA, maxA)
        print("[Detect] Worker submitted")
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
        try: _ = self._spots[idx]
        except Exception:
            QMessageBox.warning(self, "Go to spot", "Invalid selection."); return
        self._spot_view = SpotViewerWindow(self.cam, self._spots, idx, parent=self)
        self._spot_view.show()

    # -------------------------
    # VarMap (pol variance) capture + detect on varmap
    # -------------------------
    def _polvar_clicked(self):
        if not self.btn_stop.isEnabled():
            self.status.showMessage("Start acquisition first.", 3000); return
        if self._pv_collecting:
            print("[PolVar] Already collecting — ignored"); return
        # estimate ~1s worth of frames
        fps = max(1.0, float(self._last_fps or 30.0))
        try:
            txt = self.lbl_fps.text()
            if txt not in ("-", "–", "", None):
                fps = max(1.0, float(txt))
        except Exception: pass
        target = int(round(fps))
        target = max(12, min(120, target))
        self._pv_frames.clear(); self._pv_collecting = True
        self._pv_showing = False  # still live until we show varmap
        print(f"[PolVar] Begin 1s capture — target {target} frames (fps≈{fps:.2f})")
        self.status.showMessage(f"VarMap: capturing ~1 s ({target} frames)…", 0)
        self._pv_target = target
        # frame hook
        try:
            self.cam.frame.connect(self._collect_polvar, Qt.QueuedConnection)
        except Exception as e:
            print(f"[PolVar] ERROR connecting frame: {e}")
        # watchdog 2.5s
        if self._pv_watchdog:
            try: self._pv_watchdog.stop(); self._pv_watchdog.deleteLater()
            except Exception: pass
        self._pv_watchdog = QTimer(self); self._pv_watchdog.setSingleShot(True)
        self._pv_watchdog.timeout.connect(self._polvar_timeout)
        self._pv_watchdog.start(2500)
        print("[PolVar] Watchdog armed: 2500 ms")

    def _polvar_timeout(self):
        print("[PolVar] Timeout while collecting.")
        try: self.cam.frame.disconnect(self._collect_polvar)
        except Exception: pass
        self._pv_collecting = False
        self.status.showMessage("VarMap: timeout collecting frames.", 4000)

    def _collect_polvar(self, arr_obj: object):
        if not self._pv_collecting: return
        a = np.asarray(arr_obj)
        if a.ndim != 2: return
        self._pv_frames.append(a.copy())
        have = len(self._pv_frames); need = int(getattr(self, "_pv_target", 60))
        if have < need:
            if have % 5 == 0:
                print(f"[PolVar] Captured {have}/{need}…")
            return
        # Enough frames
        try: self.cam.frame.disconnect(self._collect_polvar)
        except Exception: pass
        self._pv_collecting = False
        if self._pv_watchdog:
            try: self._pv_watchdog.stop(); self._pv_watchdog.deleteLater()
            except Exception: pass
            self._pv_watchdog = None

        frames = np.stack(self._pv_frames, axis=0)  # N,H,W
        print(f"[PolVar] Stack ready: {frames.shape} dtype={frames.dtype} min={frames.min()} max={frames.max()}")
        t0 = time.perf_counter()
        try:
            var_img8, dets = self._compute_varmap_and_spots(frames)
        except Exception as e:
            print(f"[PolVar] ERROR compute varmap: {e}")
            self.status.showMessage(f"VarMap error: {e}", 5000); return

        # Paint varmap + overlays, freeze live
        h, w = var_img8.shape
        if not var_img8.flags.c_contiguous:
            var_img8 = np.ascontiguousarray(var_img8)
        qimg = QImage(var_img8.data, w, h, w, QImage.Format_Grayscale8)
        img = qimg.copy()
        if dets:
            p = QPainter(img); pen = QPen(Qt.yellow); pen.setWidth(2); p.setPen(pen); p.setFont(QFont("", 10))
            for i, (cx, cy, r, area, inten_dummy) in enumerate(dets):
                p.drawEllipse(QPointF(cx, cy), r, r)
                p.drawText(int(cx + 6), int(cy - 6), f"{i+1}  A={area}")
            p.end()
        self._pv_showing = True
        self._spots = dets or []  # store, though these are on varmap coords (already in sensor px)
        self._show_qimage(img)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[PolVar] Done: {len(dets)} spot(s); compute+render={dt:.1f} ms; showing VarMap (live paused)")
        self.status.showMessage(f"VarMap: {len(dets)} spot(s). Click 'Back to live' to resume.", 5000)

    def _compute_varmap_and_spots(self, frames: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float,float,float,int,int]]]:
        """Compute variance magnitude on 2x2 polarization blocks and detect spots.
        Returns (8-bit display image, spots in SENSOR pixels)."""
        assert frames.ndim == 3, "frames should be (N,H,W)"
        N, H, W = frames.shape
        if H < 2 or W < 2:
            raise ValueError("Frame too small for 2x2 pol blocks.")
        # 2x2 layout assumed: (0,0)=90, (0,1)=45, (1,0)=135, (1,1)=0
        I90  = frames[:, 0::2, 0::2].astype(np.int32, copy=False)
        I45  = frames[:, 0::2, 1::2].astype(np.int32, copy=False)
        I135 = frames[:, 1::2, 0::2].astype(np.int32, copy=False)
        I0   = frames[:, 1::2, 1::2].astype(np.int32, copy=False)
        # per-frame Ix/Iy
        Ix = I90 - I0
        Iy = I135 - I45
        # per-block spans across time
        Dx = Ix.max(axis=0) - Ix.min(axis=0)  # (Hb,Wb)
        Dy = Iy.max(axis=0) - Iy.min(axis=0)
        Dx = np.maximum(Dx, 0); Dy = np.maximum(Dy, 0)
        mag = np.hypot(Dx, Dy).astype(np.float32)  # ADU in 12-bit domain

        # thresholds / display scaling
        def _num(le: QLineEdit, default: float) -> float:
            try: return float(le.text())
            except Exception: return default
        M_floor = _num(self.ed_Mfloor, 40.0)
        balance_min = _num(self.ed_balance, 0.30)
        use_auto = bool(self.chk_Mauto.isChecked())
        M_scale = _num(self.ed_Mscale, 384.0)

        # post-variance mask
        mask = (mag >= M_floor)
        # balance gate to suppress one-axis flicker
        mn = np.minimum(Dx, Dy).astype(np.float32)
        mx = np.maximum(Dx, Dy).astype(np.float32)
        bal = mn / (mx + 1e-6)
        mask &= (bal >= balance_min)

        kept = mag[mask]
        if use_auto and kept.size:
            scale = float(np.percentile(kept, 99.0))
            print(f"[PolVar] Auto scale (p99 of kept): {scale:.1f}")
        else:
            scale = max(1.0, M_scale)
            print(f"[PolVar] Fixed scale: {scale:.1f}")

        Mdisp = (mag / scale) * mask.astype(np.float32)
        M8_block = np.clip(Mdisp * 255.0, 0, 255).astype(np.uint8)  # in block grid

        # upscale 2x for display to sensor pixels (nearest)
        Hb, Wb = M8_block.shape
        var8 = np.repeat(np.repeat(M8_block, 2, axis=0), 2, axis=1)  # (2Hb,2Wb) ≈ (H_align, W_align)
        var8 = var8[:H, :W]  # crop if odd dims
        # --- Spot detect on block mask ---
        # Build a binary mask on block grid for labeling (smoother on blocks)
        se = disk(1)
        mask2 = binary_opening(mask, se)
        mask2 = binary_closing(mask2, se)
        lbl = label(mask2, connectivity=2)
        dets_block = []
        for r in regionprops(lbl):
            area = int(r.area)
            cy, cx = r.centroid  # in block coords
            rad_block = float((area / np.pi) ** 0.5)
            dets_block.append((float(cx), float(cy), rad_block, area, 0))

        # convert to SENSOR pixel coords
        dets_sensor = []
        for (cx_b, cy_b, r_b, area_b, _inten) in dets_block:
            cx_s = 2.0 * cx_b + 1.0  # center of 2x2 block
            cy_s = 2.0 * cy_b + 1.0
            r_s  = 2.0 * r_b
            dets_sensor.append((cx_s, cy_s, r_s, area_b, 0))

        print(f"[PolVar] Spots: blocks={len(dets_block)} -> sensor={len(dets_sensor)}")
        return var8, dets_sensor

    def _back_to_live(self):
        if not self._pv_showing:
            self.status.showMessage("Already live.", 2000); return
        self._pv_showing = False
        self._last_qimage = None
        self.status.showMessage("Live feed resumed.", 2000)
        print("[PolVar] Back to live; live frames will render again.")

    # -------------------------
    # Signals & housekeeping
    # -------------------------
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
        self.video.setPixmap(QPixmap()); self._last_qimage = None
        self._pv_showing = False

    def _on_error(self, msg: str) -> None:
        print("[Camera Error]", msg)
        self.status.showMessage(f"Error: {msg}", 0)

    def _on_roi(self, d: dict) -> None:
        try:
            for name, widget in (("Width", self.ed_w), ("Height", self.ed_h), ("OffsetX", self.ed_x), ("OffsetY", self.ed_y)):
                v = d.get(name)
                if v is not None:
                    try: widget.setText(str(int(v)))
                    except Exception: widget.setText(str(v))
        except Exception: pass

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
        except Exception: pass

    def _on_detect_timeout(self):
        try: have = len(self._collect_frames)
        except Exception: have = 0
        elapsed_ms = (time.perf_counter() - getattr(self, "_detect_t0", time.perf_counter())) * 1000.0
        print(f"[Detect] Watchdog TIMEOUT after {elapsed_ms:.1f} ms (frames collected={have}).")
        try: self.cam.frame.disconnect(self._collect_for_detect)
        except Exception: pass
        self._collecting = False; self.btn_detect.setEnabled(True)
        self.status.showMessage(f"Detect timed out after {elapsed_ms:.0f} ms.", 4000)

    def _stop_detect_watchdog(self):
        try:
            if self._detect_watchdog is not None:
                if self._detect_watchdog.isActive(): self._detect_watchdog.stop()
                self._detect_watchdog.deleteLater()
        except Exception: pass
        self._detect_watchdog = None

    def _handle_detect_done(self, payload: object) -> None:
        self._stop_detect_watchdog()
        try: kind, data, t0 = payload
        except Exception:
            self.btn_detect.setEnabled(True)
            self.status.showMessage("Detect failed: bad payload", 4000)
            print("[Detect] ERROR: bad payload", payload); return
        if kind == "ok":
            elapsed_ms = (time.perf_counter() - (t0 or time.perf_counter())) * 1000.0
            dets = data or []
            print(f"[Detect] Done: {len(dets)} spot(s) in {elapsed_ms:.1f} ms")
            self._on_detect_ready(dets)
        else:
            self.btn_detect.setEnabled(True)
            self.status.showMessage(f"Detect failed: {data}", 5000)
            print(f"[Detect] ERROR: {data}")

    # -------------------------
    # Auto-desaturate
    # -------------------------
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
            self._desat_cooldown -= 1; print(f"[AutoDesat] Cooldown… {self._desat_cooldown} frame(s) left"); return
        cur_max = int(a.max())
        print(f"[AutoDesat] Frame max={cur_max}  target={self._desat_target}  iter={self._desat_iters}")
        if cur_max <= self._desat_target or self._desat_iters >= 6:
            self._desat_active = False
            self.status.showMessage(f"Auto-desaturate: peak={cur_max} (done in {self._desat_iters} step(s)).", 3000)
            print(f"[AutoDesat] Done: peak={cur_max} ≤ target={self._desat_target}")
            return
        try: exp_ms = float(self.ed_exp.text())
        except Exception: exp_ms = 50.0
        factor = max(0.05, min(1.0, (self._desat_target / max(1.0, cur_max)) * 0.855))
        new_exp_ms = max(0.05, exp_ms * factor)
        self.ed_exp.setText(f"{new_exp_ms:.3f}")
        print(f"[UI] set_timing(fps=None, exposure_ms={new_exp_ms:.3f}); acquiring={self.btn_stop.isEnabled()}")
        self.cam.enqueue_timing({"exposure_ms": new_exp_ms})
        self.cam.process_pending_timing()
        self._desat_iters += 1; self._desat_cooldown = 3
        self.status.showMessage(f"Auto-desaturate: set exposure {new_exp_ms:.3f} ms (iter {self._desat_iters}).", 0)

    # -------------------------
    # Tone dock callbacks
    # -------------------------
    def _tone_changed(self, mode: str):
        self._tone_mode = mode or "off"
        print(f"[Tone] Mode -> {self._tone_mode}")
        if self._tone_mode == "crush":
            self._tone_lut = make_tone_lut12_to_8()
        else:
            self._tone_lut = (np.arange(4096, dtype=np.uint16) >> 4).astype(np.uint8)

    def _rebuild_tone(self):
        if self._tone_mode == "crush":
            self._tone_lut = make_tone_lut12_to_8()
            self.status.showMessage("Tone LUT rebuilt (crush).", 1500)
        else:
            self.status.showMessage("Tone is OFF; nothing to rebuild.", 1500)

    # -------------------------
    # Safe shutdown
    # -------------------------
    def safe_shutdown(self):
        # stop varmap hooks
        try:
            if self._pv_collecting:
                try: self.cam.frame.disconnect(self._collect_polvar)
                except Exception: pass
                self._pv_collecting = False
            if self._pv_watchdog:
                try: self._pv_watchdog.stop(); self._pv_watchdog.deleteLater()
                except Exception: pass
                self._pv_watchdog = None
        except Exception: pass

        # stop detect hooks
        try:
            if self._collecting:
                try: self.cam.frame.disconnect(self._collect_for_detect)
                except Exception: pass
                self._collecting = False
            self._stop_detect_watchdog()
        except Exception: pass

        # stop acquisition then close
        try:
            if self.btn_stop.isEnabled():
                print("[UI] safe_shutdown: stopping…"); self.cam.stop(); QApplication.processEvents()
        except Exception: pass
        try:
            if self.btn_close.isEnabled():
                print("[UI] safe_shutdown: closing…"); self.cam.close()
        except Exception: pass
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception: pass


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
