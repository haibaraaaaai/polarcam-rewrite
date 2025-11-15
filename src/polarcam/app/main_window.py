from __future__ import annotations

import os
import time
import math
from typing import Optional, List, Tuple
import numpy as np

from PySide6.QtCore import Qt, Signal, QThread, Slot
from PySide6.QtGui import QImage, QPixmap, QIntValidator, QDoubleValidator, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QGroupBox,
)

from polarcam.app.lut_widget import HighlightLUTWidget
from polarcam.app.spot_detect import detect_spots_oneframe
from polarcam.app.spot_viewer import SpotViewerDialog

# Cycling recorder
from polarcam.app.spot_cycler import MultiSpotCycler, CycleConfig

# Old/desired spot tuple: (cx, cy, r, area, inten)
Spot = Tuple[float, float, float, int, int]


class MainWindow(QMainWindow):
    """Lean GUI: Open/Start/Stop/Close + ROI/Timing + Gains + Highlight LUT + Spot tools. Live video.
       Adds: Cycle Spots recorder that hops across selected spots (1 s/spot), saves 4×pol means from
       the displayed/software crop only, records at max camera FPS while preview is capped at 20 FPS.
    """
    detect_done = Signal(object)

    def __init__(self, ctrl) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.setWindowTitle("PolarCam (lean)")
        self.resize(1200, 750)

        # status bar
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

        # tone/LUT control
        self._lut: Optional[np.ndarray] = None
        self.tone = HighlightLUTWidget(self)
        self.tone.paramsChanged.connect(self._on_tone_params)
        self._on_tone_params(*self.tone.params())

        # detection / overlay state
        self._spots: List[Spot] = []
        self._collecting = False
        self._collect_frames: List[np.ndarray] = []
        self._detect_params: Tuple = tuple()
        self._detect_conn_active = False
        self._video_paused = False

        # preview throttle (used during Cycle mode)
        self._cycle_active = False
        self._preview_cap_fps = 20.0  # cap preview to 20 fps while cycling
        self._last_preview_t = 0.0

        # build UI
        self._build_video()
        self._build_toolbar()
        self._build_forms()
        self._build_detect_ui()
        self._build_spot_list_ui()
        self._assemble_layout()

        # starting button states
        self._set_buttons(open_enabled=True, start=False, stop=False, close=False)

        # wire controller/backend signals if present
        cam = getattr(self.ctrl, "cam", None)
        if cam is not None:
            try: cam.opened.connect(self._on_open)
            except Exception: pass
            try: cam.started.connect(self._on_started)
            except Exception: pass
            try: cam.stopped.connect(self._on_stopped)
            except Exception: pass
            try: cam.closed.connect(self._on_closed)
            except Exception: pass
            try: cam.error.connect(self._on_error)
            except Exception: pass
            try: cam.frame.connect(self._on_frame)
            except Exception: pass
            try: cam.timing.connect(self._on_timing)
            except Exception: pass
            try: cam.roi.connect(self._on_roi)
            except Exception: pass
            try: cam.gains.connect(self._on_gains)
            except Exception: pass
            if hasattr(cam, "desaturated"):
                try: cam.desaturated.connect(self._on_desaturated)
                except Exception: pass
            if hasattr(cam, "auto_desat_started"):
                try: cam.auto_desat_started.connect(self._on_desat_started)
                except Exception: pass
            if hasattr(cam, "auto_desat_finished"):
                try: cam.auto_desat_finished.connect(self._on_desat_finished)
                except Exception: pass
        else:
            self.status.showMessage("No camera backend attached.", 4000)

        # wire buttons
        self.btn_open.clicked.connect(self._open_clicked)
        self.btn_start.clicked.connect(self._start_clicked)
        self.btn_stop.clicked.connect(self._stop_clicked)
        self.btn_close.clicked.connect(self._close_clicked)
        self.btn_apply_roi.clicked.connect(self._apply_roi)
        self.btn_full_roi.clicked.connect(self._full_roi_clicked)
        self.btn_apply_tim.clicked.connect(self._apply_timing)
        self.btn_desat.clicked.connect(self._desaturate_clicked)
        self.btn_apply_gains.clicked.connect(self._apply_gains)
        self.btn_refresh_gains.clicked.connect(self._refresh_gains)
        self.btn_varmap.clicked.connect(self._open_varmap)
        self.btn_detect.clicked.connect(self._detect_clicked)
        self.btn_clear_overlays.clicked.connect(self._clear_overlays)
        self.btn_view_spot.clicked.connect(self._view_selected_spots)
        self.btn_remove_spot.clicked.connect(self._remove_selected_spots)

        # Cycle buttons
        self.btn_cycle_start.clicked.connect(self._start_cycle_clicked)
        self.btn_cycle_stop.clicked.connect(self._stop_cycle_clicked)

        self.detect_done.connect(self._handle_detect_done, Qt.QueuedConnection)

        self._varmap = None
        self._last_pm: Optional[QPixmap] = None

        self._roi_offset = (0, 0)   # (OffsetX, OffsetY)
        self._roi_size = (0, 0)     # (Width, Height)

        # histogram rate-limiting
        self._hist_t_last = 0.0

        # cycle worker state
        self._cycle_thread: Optional[QThread] = None
        self._cycler = None  # MultiSpotCycler instance

    # ---------- builders ----------
    def _build_video(self) -> None:
        self.video = QLabel("No video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setStyleSheet("background:#111; color:#777;")

    def _build_toolbar(self) -> None:
        self.btn_open = QPushButton("Open")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_close = QPushButton("Close")
        self.btn_varmap = QPushButton("Variance…")
        self._toolbar = QHBoxLayout()
        for b in (self.btn_open, self.btn_start, self.btn_stop, self.btn_close, self.btn_varmap):
            self._toolbar.addWidget(b)
        self._toolbar.addStretch(1)

    def _build_forms(self) -> None:
        # ROI form
        self.ed_w = QLineEdit("2464")
        self.ed_h = QLineEdit("2056")
        self.ed_x = QLineEdit("0")
        self.ed_y = QLineEdit("0")
        intv = QIntValidator(0, 1_000_000, self)
        for ed in (self.ed_w, self.ed_h, self.ed_x, self.ed_y):
            ed.setValidator(intv)
        self.btn_apply_roi = QPushButton("Apply ROI")
        self.btn_full_roi = QPushButton("Full sensor")

        # Timing form (EXPOSURE IN MILLISECONDS)
        self.ed_fps = QLineEdit("20.0")
        self.ed_exp = QLineEdit("50.0")  # ms
        fpsv = QDoubleValidator(0.01, 100000.0, 3, self); fpsv.setNotation(QDoubleValidator.StandardNotation)
        expv = QDoubleValidator(0.01, 2_000.0, 3, self); expv.setNotation(QDoubleValidator.StandardNotation)
        self.ed_fps.setValidator(fpsv)
        self.ed_exp.setValidator(expv)
        self.btn_apply_tim = QPushButton("Apply Timing")
        self.btn_desat = QPushButton("Desaturate")

        # Gains
        self.ed_gain_ana = QLineEdit("")
        self.ed_gain_dig = QLineEdit("")
        gval = QDoubleValidator(0.0, 1000.0, 3, self); gval.setNotation(QDoubleValidator.StandardNotation)
        self.ed_gain_ana.setValidator(gval)
        self.ed_gain_dig.setValidator(gval)
        self.ed_gain_ana.setPlaceholderText("—")
        self.ed_gain_dig.setPlaceholderText("—")
        self.btn_apply_gains = QPushButton("Apply gains")
        self.btn_refresh_gains = QPushButton("Refresh gains")
        self._lbl_gain_ana = QLabel("")
        self._lbl_gain_dig = QLabel("")

        # Layout
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
        rowt = QHBoxLayout(); rowt.addWidget(self.btn_apply_tim); rowt.addWidget(self.btn_desat)
        wt = QWidget(); wt.setLayout(rowt)
        self._form.addRow(wt)

        self._form.addRow("Analog gain", self.ed_gain_ana); self._form.addRow("", self._lbl_gain_ana)
        self._form.addRow("Digital gain", self.ed_gain_dig); self._form.addRow("", self._lbl_gain_dig)
        rowg = QHBoxLayout(); rowg.addWidget(self.btn_apply_gains); rowg.addWidget(self.btn_refresh_gains)
        rg = QWidget(); rg.setLayout(rowg)
        self._form.addRow(rg)

    def _build_detect_ui(self) -> None:
        box = QGroupBox("Detect spots (single frame, old logic)")
        f = QFormLayout(box)

        self.cmb_thr_mode = QComboBox()
        self.cmb_thr_mode.addItems(["absolute", "percentile"])
        self.ed_thr_val = QLineEdit("1200")
        self.ed_minA = QLineEdit("6")
        self.ed_maxA = QLineEdit("5000")

        self.btn_detect = QPushButton("Detect")
        self.btn_clear_overlays = QPushButton("Clear overlays")

        f.addRow("Threshold mode", self.cmb_thr_mode)
        f.addRow("Threshold value", self.ed_thr_val)
        f.addRow("Min area", self.ed_minA)
        f.addRow("Max area", self.ed_maxA)
        row = QHBoxLayout(); row.addWidget(self.btn_detect); row.addWidget(self.btn_clear_overlays)
        w = QWidget(); w.setLayout(row); f.addRow(w)

        self._form.addRow(box)

    def _build_spot_list_ui(self) -> None:
        box = QGroupBox("Spots")
        v = QVBoxLayout(box)
        self.spot_list = QListWidget()
        self.spot_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.btn_view_spot = QPushButton("View spot…")
        self.btn_remove_spot = QPushButton("Remove selected")
        # Cycle controls
        self.btn_cycle_start = QPushButton("Start Cycle (1 s / spot)")
        self.btn_cycle_stop  = QPushButton("Stop Cycle")
        self.btn_cycle_stop.setEnabled(False)

        row = QHBoxLayout()
        row.addWidget(self.btn_view_spot)
        row.addWidget(self.btn_remove_spot)
        v.addWidget(self.spot_list, 1)
        v.addLayout(row)
        v.addWidget(self.btn_cycle_start)
        v.addWidget(self.btn_cycle_stop)
        self._form.addRow(box)

    def _assemble_layout(self) -> None:
        right = QWidget(); right.setLayout(self._form)
        leftcol = QVBoxLayout()
        leftcol.addLayout(self._toolbar)
        leftcol.addWidget(self.video, 1)
        leftcol.addWidget(self.tone)
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

    # Centralize UI lock/unlock while cycling
    def _set_cycle_ui_state(self, active: bool) -> None:
        self._cycle_active = bool(active)
        # Buttons that could fight with cycling
        for w in [
            self.btn_detect, self.btn_clear_overlays,
            self.btn_apply_roi, self.btn_full_roi,
            self.btn_apply_tim, self.btn_desat,
            self.btn_apply_gains, self.btn_refresh_gains,
            self.btn_view_spot, self.btn_remove_spot,
            self.spot_list,
        ]:
            try:
                w.setEnabled(not active)
            except Exception:
                pass
        self.btn_cycle_start.setEnabled(not active)
        self.btn_cycle_stop.setEnabled(active)
        if active:
            self._last_preview_t = 0.0  # reset throttle timer

    # ---------- backend signal handlers ----------
    def _on_open(self, name: str) -> None:
        self.status.showMessage(f"Opened: {name}", 1500)
        self._set_buttons(open_enabled=False, start=True, stop=False, close=True)
        self._refresh_gains()

    def _on_started(self) -> None:
        self.status.showMessage("Started", 1500)
        self._set_buttons(open_enabled=False, start=False, stop=True, close=False)

    def _on_stopped(self) -> None:
        self.status.showMessage("Stopped", 1500)
        self._set_buttons(open_enabled=False, start=True, stop=False, close=True)

    def _on_closed(self) -> None:
        self.status.showMessage("Closed", 1500)
        self.video.setPixmap(QPixmap())
        self._last_pm = None
        self._set_buttons(open_enabled=True, start=False, stop=False, close=False)

    def _on_error(self, msg: str) -> None:
        self.status.showMessage(f"Error: {msg}", 5000)
        QMessageBox.warning(self, "Camera error", msg)

    def _on_frame(self, arr_obj: object) -> None:
        try:
            a16 = np.asarray(arr_obj)
            if a16.ndim != 2:
                return

            # histogram ~2 Hz for LUT widget
            if time.time() - self._hist_t_last > 0.5:
                self._hist_t_last = time.time()
                vals = (a16.astype(np.uint16, copy=False) >> 4).ravel()
                hist = np.bincount(vals, minlength=256)
                if hist.size > 256: hist = hist[:256]
                self.tone.setHistogram256(hist)

            # PREVIEW THROTTLE while cycling: cap to ~20 fps
            if self._cycle_active and self._preview_cap_fps and self._preview_cap_fps > 0:
                now = time.perf_counter()
                if now - self._last_preview_t < (1.0 / float(self._preview_cap_fps)):
                    return
                self._last_preview_t = now

            if self._lut is not None:
                if a16.dtype != np.uint16: a16 = a16.astype(np.uint16, copy=False)
                a8 = self._lut[a16]
            else:
                if a16.dtype != np.uint16: a16 = a16.astype(np.uint16, copy=False)
                a8 = ((a16 + 8) >> 4).astype(np.uint8, copy=False)

            if not a8.flags.c_contiguous:
                a8 = np.ascontiguousarray(a8)

            pm = self._compose_pixmap_with_overlays(a8)
            self._last_pm = pm
            self._refresh_video_view()
        except Exception as e:
            self.status.showMessage(f"Frame error: {e}", 2000)

    def _compose_pixmap_with_overlays(self, a8: np.ndarray) -> QPixmap:
        """Draw spot overlays on the unscaled frame, then return a QPixmap."""
        h, w = a8.shape
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        pm = QPixmap.fromImage(qimg.copy())  # own memory for safety

        if not self._spots:
            return pm

        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(Qt.green, 2)
        p.setPen(pen)
        font = QFont()
        font.setPointSize(9)
        p.setFont(font)

        ox, oy = self._roi_offset
        for i, s in enumerate(self._spots):
            try:
                cx, cy, r, area, inten = s
                cx = int(round(cx - ox))
                cy = int(round(cy - oy))
                rr = max(1, min(40, int(round(r))))   # radius in px
            except Exception:
                continue

            if 0 <= cx < w and 0 <= cy < h:
                p.drawEllipse(cx - rr, cy - rr, 2 * rr, 2 * rr)
                p.drawText(cx + rr + 3, cy - rr - 2, f"{i + 1}")
        p.end()
        return pm

    def _on_timing(self, d: dict) -> None:
        """Keep exposure displayed in *milliseconds* (label says ms)."""
        try:
            fps = d.get("fps")
            exp_us = d.get("exposure_us")
            if fps is not None:
                self.ed_fps.setText(f"{float(fps):.3f}")
            if exp_us is not None:
                self.ed_exp.setText(f"{float(exp_us) / 1000.0:.3f}")  # μs → ms
            msg = []
            if fps is not None:   msg.append(f"FPS={float(fps):.3f}")
            if exp_us is not None:msg.append(f"EXP={float(exp_us)/1000.0:.3f} ms")
            if msg: self.status.showMessage("Timing applied: " + "  ".join(msg), 2000)
        except Exception:
            pass

    def _on_roi(self, d: dict) -> None:
        try:
            mapping = [
                ("Width", self.ed_w),
                ("Height", self.ed_h),
                ("OffsetX", self.ed_x),
                ("OffsetY", self.ed_y),
            ]
            for key, edit in mapping:
                v = d.get(key)
                if v is not None:
                    edit.setText(str(int(round(float(v)))))
            # keep numeric copies for overlays
            W = d.get("Width"); H = d.get("Height")
            X = d.get("OffsetX"); Y = d.get("OffsetY")
            if None not in (W, H, X, Y):
                self._roi_size = (int(round(float(W))), int(round(float(H))))
                self._roi_offset = (int(round(float(X))), int(round(float(Y))))
        except Exception:
            pass

    def _on_gains(self, g: dict) -> None:
        try:
            ana = g.get("analog", {}) or {}
            dig = g.get("digital", {}) or {}
            if ana.get("val") is not None:
                self.ed_gain_ana.setText(f"{float(ana['val']):.3f}")
            if dig.get("val") is not None:
                self.ed_gain_dig.setText(f"{float(dig['val']):.3f}")
            def rng(dct: dict) -> str:
                mn = dct.get("min"); mx = dct.get("max"); inc = dct.get("inc")
                bits = []
                if mn is not None and mx is not None: bits.append(f"{float(mn):.3f} … {float(mx):.3f}")
                if inc is not None: bits.append(f"inc {float(inc):.3f}")
                return "  ".join(bits)
            self._lbl_gain_ana.setText(rng(ana)); self._lbl_gain_dig.setText(rng(dig))
        except Exception:
            pass

    # ---------- detect ----------
    def _detect_clicked(self) -> None:
        if self._collecting:
            return
        mode = self.cmb_thr_mode.currentText().strip().lower()
        try: thr_val = float(self.ed_thr_val.text())
        except Exception: thr_val = 1200.0
        try: minA = int(float(self.ed_minA.text()))
        except Exception: minA = 6
        try: maxA = int(float(self.ed_maxA.text()))
        except Exception: maxA = 5000

        self._detect_params = (mode, thr_val, minA, maxA)
        self._collecting = True
        self._collect_frames = []
        self.btn_detect.setEnabled(False)
        self.status.showMessage("Detect: grabbing one frame…", 0)

        if not self._detect_conn_active:
            try:
                self.ctrl.cam.frame.connect(self._collect_for_detect_1f, Qt.QueuedConnection)
                self._detect_conn_active = True
            except Exception:
                self._detect_conn_active = False
                self.detect_done.emit(("err", "Camera not available for detect."))

    def _collect_for_detect_1f(self, arr_obj: object) -> None:
        if not self._detect_conn_active:
            return
        try:
            self.ctrl.cam.frame.disconnect(self._collect_for_detect_1f)
        except Exception:
            pass
        self._detect_conn_active = False

        try:
            img = np.asarray(arr_obj)
            if img.ndim != 2:
                raise RuntimeError("Frame is not 2D.")
            mode, thr_val, minA, maxA = self._detect_params

            spots = detect_spots_oneframe(
                img.astype(np.uint16, copy=False),
                thr_mode="percentile" if mode == "percentile" else "absolute",
                thr_value=thr_val,
                min_area=minA,
                max_area=maxA,
                open_radius=2,
                close_radius=1,
                remove_border=True,
                debug=True,
            )

            # show overlays on the exact frame used for detection
            self._spots = list(spots)
            a16 = img.astype(np.uint16, copy=False)
            a8 = self._lut[a16] if self._lut is not None else ((a16 + 8) >> 4).astype(np.uint8, copy=False)
            if not a8.flags.c_contiguous: a8 = np.ascontiguousarray(a8)
            pm = self._compose_pixmap_with_overlays(a8)
            self._last_pm = pm
            self._refresh_video_view()

            self.detect_done.emit(("ok", spots))
        except Exception as e:
            self.detect_done.emit(("err", str(e)))

    def _handle_detect_done(self, payload: object) -> None:
        self._collecting = False
        self.btn_detect.setEnabled(True)
        if not isinstance(payload, (list, tuple)) or len(payload) < 2:
            self._spots = []
            self._refresh_spot_list()
            self.status.showMessage("Detect failed: bad payload.", 3000)
            return
        kind, data = payload
        if kind == "ok":
            spots: List[Spot] = list(data or [])
            self._spots = spots
            self._refresh_spot_list()
            self.status.showMessage(f"Detect: {len(self._spots)} spot(s).", 3000)
        else:
            self._spots = []
            self._refresh_spot_list()
            self.status.showMessage(f"Detect failed: {data}", 4000)

    def _refresh_spot_list(self) -> None:
        self.spot_list.clear()
        for i, s in enumerate(self._spots):
            cx, cy, r, area, inten = s
            item = QListWidgetItem(f"#{i+1}  (x={cx:.1f}, y={cy:.1f})  r≈{r:.1f}px  A={area}  I={inten}")
            self.spot_list.addItem(item)

    def _clear_overlays(self) -> None:
        self._spots = []
        self._refresh_spot_list()
        self.status.showMessage("Overlays cleared.", 1500)

    # ---------- view/remove spots ----------
    def _selected_spot_indices(self) -> List[int]:
        return sorted({it.row() for it in self.spot_list.selectedIndexes()})

    def _resume_main_video(self) -> None:
        if self._video_paused:
            try:
                self.ctrl.cam.frame.connect(self._on_frame, Qt.QueuedConnection)
                print("[MainWindow] Live preview resumed.")
            except Exception:
                pass
            self._video_paused = False

    def _view_selected_spots(self) -> None:
        idxs = self._selected_spot_indices()
        if not self._spots:
            self.status.showMessage("No spots to view.", 2000)
            return
        if not idxs:
            idxs = [0]
        sel_spots = [self._spots[i] for i in idxs]

        # Pause main-window live preview to reduce UI load while viewer is open
        if not self._video_paused:
            try:
                self.ctrl.cam.frame.disconnect(self._on_frame)
                self._video_paused = True
                print("[MainWindow] Live preview paused for Spot Viewer.")
            except Exception:
                pass

        # snapshot current ROI & FPS so the viewer can restore exactly
        W, H = self._roi_size
        X, Y = self._roi_offset
        try:
            fps_saved = float(self.ed_fps.text()) if self.ed_fps.text().strip() else None
        except Exception:
            fps_saved = None

        dlg = SpotViewerDialog(self.ctrl, sel_spots, self, saved_roi=(W, H, X, Y), saved_fps=fps_saved)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.destroyed.connect(self._resume_main_video)  # resume when the dialog closes
        dlg.show()

    def _remove_selected_spots(self) -> None:
        idxs = self._selected_spot_indices()
        if not idxs:
            return
        keep = [s for i, s in enumerate(self._spots) if i not in idxs]
        self._spots = keep
        self._refresh_spot_list()

    # ---------- CYCLE: start/stop threads ----------
    def _start_cycle_clicked(self) -> None:
        if not self._spots:
            QMessageBox.information(self, "Cycle", "No spots selected/detected.")
            return
        out_dir = os.path.join(os.getcwd(), "cycles")
        cfg = CycleConfig(
            out_dir=out_dir,
            base_name="cycle",
            dwell_sec=1.0,
            max_duration_sec=3600,
            chunk_len=20000,
            maximize_camera_fps=True,
        )

        if self._cycle_thread is not None:
            QMessageBox.information(self, "Cycle", "Cycler already running.")
            return

        # Build cycler with error guard — if __init__ throws, self._cycler would stay None.
        try:
            cycler = MultiSpotCycler(self.ctrl, self._spots, cfg)
        except Exception as e:
            self.status.showMessage(f"Cycle init error: {e}", 6000)
            return

        self._cycle_thread = QThread(self)
        self._cycler = cycler
        self._cycler.moveToThread(self._cycle_thread)

        # signals
        try:
            self._cycler.advise_ui_cap.connect(self._on_cycle_ui_cap)
        except Exception:
            pass  # older builds without the signal

        # signals
        self._cycle_thread.started.connect(self._cycler.start,       Qt.QueuedConnection)
        self._cycler.progress.connect(self._on_cycler_progress,      Qt.QueuedConnection)
        self._cycler.error.connect(self._on_cycler_error,            Qt.QueuedConnection)
        self._cycler.stopped.connect(self._cycle_thread.quit,        Qt.QueuedConnection)
        self._cycler.advise_ui_cap.connect(self._on_cycle_ui_cap,    Qt.QueuedConnection)
        self._cycle_thread.finished.connect(self._cycle_finished,    Qt.QueuedConnection)

        # Enable preview throttle on our side; worker will also advise via signal
        self._cycle_active = True
        self._last_preview_t = 0.0

        self._cycle_thread.start()
        self.btn_cycle_start.setEnabled(False)
        self.btn_cycle_stop.setEnabled(True)
        self.status.showMessage("Cycle started — recording at max FPS, preview capped at 20 FPS.", 4000)

    def _stop_cycle_clicked(self) -> None:
        # User stop
        if self._cycler:
            try: self._cycler.stop()
            except Exception: pass
        if self._cycle_thread:
            self._cycle_thread.wait(3000)  # join

    def _cycle_finished(self) -> None:
        # Thread finished
        self._cycler = None
        self._cycle_thread = None
        self._set_cycle_ui_state(False)
        self.status.showMessage("Cycle stopped.", 2500)

    @Slot(str)
    def _on_cycler_progress(self, s: str) -> None:
        # Safe: runs on GUI thread
        self.status.showMessage(s, 1200)

    @Slot(str)
    def _on_cycler_error(self, msg: str) -> None:
        # Safe: runs on GUI thread
        self.status.showMessage(f"Cycle error: {msg}", 5000)

    @Slot(float)
    def _on_cycle_ui_cap(self, cap: float) -> None:
        # 0.0 = uncap (disable throttle); >0 = cap to that FPS
        if cap and cap > 0.0:
            self._preview_cap_fps = float(cap)
            self._cycle_active = True
        else:
            self._preview_cap_fps = 0.0
            self._cycle_active = False
        self._last_preview_t = 0.0  # reset throttle phase

    # ---------- UI handlers (UI → controller) ----------
    def _open_clicked(self) -> None:
        if hasattr(self.ctrl, "open"): self.ctrl.open()

    def _start_clicked(self) -> None:
        if self.btn_start.isEnabled() and hasattr(self.ctrl, "start"): self.ctrl.start()

    def _stop_clicked(self) -> None:
        if self.btn_stop.isEnabled() and hasattr(self.ctrl, "stop"): self.ctrl.stop()

    def _close_clicked(self) -> None:
        if self.btn_close.isEnabled() and hasattr(self.ctrl, "close"): self.ctrl.close()

    def _apply_roi(self) -> None:
        try:
            w = int(float(self.ed_w.text() or "0"))
            h = int(float(self.ed_h.text() or "0"))
            x = int(float(self.ed_x.text() or "0"))
            y = int(float(self.ed_y.text() or "0"))
        except Exception:
            self.status.showMessage("ROI: invalid numbers", 2000); return
        if w <= 0 or h <= 0:
            self.status.showMessage("ROI: width/height must be > 0. Use Full sensor if unsure.", 3000); return
        if hasattr(self.ctrl, "set_roi"): self.ctrl.set_roi(w, h, x, y)

    def _full_roi_clicked(self) -> None:
        if hasattr(self.ctrl, "full_sensor"): self.ctrl.full_sensor()

    def _apply_timing(self) -> None:
        # exposure textbox is *milliseconds*
        fps = float(self.ed_fps.text()) if self.ed_fps.hasAcceptableInput() else None
        exp_ms = float(self.ed_exp.text()) if self.ed_exp.hasAcceptableInput() else None
        if hasattr(self.ctrl, "set_timing"):
            self.ctrl.set_timing(fps, exp_ms)

    def _apply_gains(self) -> None:
        ana = float(self.ed_gain_ana.text()) if self.ed_gain_ana.text().strip() else None
        dig = float(self.ed_gain_dig.text()) if self.ed_gain_dig.text().strip() else None
        if hasattr(self.ctrl, "set_gains"): self.ctrl.set_gains(ana, dig)

    def _refresh_gains(self) -> None:
        if hasattr(self.ctrl, "refresh_gains"):
            self.ctrl.refresh_gains(); self.status.showMessage("Refreshing gains…", 800)
        elif hasattr(self.ctrl, "cam") and hasattr(self.ctrl.cam, "refresh_gains"):
            self.ctrl.cam.refresh_gains(); self.status.showMessage("Refreshing gains…", 800)

    def _desaturate_clicked(self) -> None:
        if hasattr(self.ctrl, "desaturate"):
            self.status.showMessage("Auto-desaturating…", 2000)
            self.ctrl.desaturate(0.85, 5)

    def _on_desaturated(self, d: dict) -> None:
        it = int(d.get("iterations", 0))
        mx = int(d.get("final_max", -1))
        ok = bool(d.get("success", False))
        exp_us = d.get("exposure_us", None)
        tgt = int(d.get("target", 0))
        if ok:
            self.status.showMessage(f"Desaturate done: iters={it}, max={mx}, target={tgt}, exposure={(exp_us or 0)/1000.0:.1f} ms", 3000)
        else:
            QMessageBox.warning(
                self, "Desaturate",
                f"Couldn’t reach target after {max(it, 5)} tries.\n"
                f"Final max={mx}, target={tgt}, exposure={(exp_us or 0)/1000.0:.1f} ms"
            )

    def _on_desat_started(self) -> None:
        if hasattr(self, "btn_desat"): self.btn_desat.setEnabled(False)
        self.status.showMessage("Auto-desaturating…", 2000)

    def _on_desat_finished(self) -> None:
        if hasattr(self, "btn_desat"): self.btn_desat.setEnabled(True)
        self.status.showMessage("Auto-desaturate complete.", 1500)

    def _open_varmap(self) -> None:
        if self._varmap_alive():
            self._varmap.raise_(); self._varmap.activateWindow(); return
        from polarcam.app.varmap_dialog import VarMapDialog
        dlg = VarMapDialog(self.ctrl, self)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.destroyed.connect(lambda _=None: setattr(self, "_varmap", None))
        self._varmap = dlg; dlg.show()

    # ---------- LUT plumbing ----------
    def _on_tone_params(self, floor: int, cap: int, gamma: float) -> None:
        self._lut = self._build_lut(floor, cap, gamma)

    @staticmethod
    def _build_lut(floor: int, cap: int, gamma: float) -> np.ndarray:
        cap = max(floor + 1, min(cap, 4095)); floor = max(0, min(floor, 4094))
        gamma = max(0.05, min(gamma, 10.0))
        x = np.arange(4096, dtype=np.float32)
        t = (x - float(floor)) / float(cap - floor)
        t = np.clip(t, 0.0, 1.0)
        y = np.power(t, gamma) * 255.0
        return np.clip(np.rint(y), 0, 255).astype(np.uint8)

    def _varmap_alive(self) -> bool:
        vm = getattr(self, "_varmap", None)
        if vm is None: return False
        try: return vm.isVisible()
        except RuntimeError:
            self._varmap = None; return False

    def _refresh_video_view(self) -> None:
        pm = getattr(self, "_last_pm", None)
        if pm is None: return
        self.video.setPixmap(pm.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._refresh_video_view()

    @Slot(float)
    def _on_cycle_ui_cap(self, hz: float) -> None:
        """Failsafe UI preview cap while the cycler runs (recording stays max FPS)."""
        if hz and hz > 0:
            self._preview_cap_fps = float(hz)
            self._last_preview_t = 0.0
            self._cycle_active = True
        else:
            self._cycle_active = False

    # ---------- cleanup ----------
    def safe_shutdown(self) -> None:
        try:
            # stop cycler if running
            if self._cycler:
                try: self._cycler.stop()
                except Exception: pass
            if self._cycle_thread:
                self._cycle_thread.wait(2000)
        except Exception:
            pass
        try:
            if hasattr(self.ctrl, "stop"): self.ctrl.stop()
        except Exception: pass
        try:
            if hasattr(self.ctrl, "close"): self.ctrl.close()
        except Exception: pass

    def closeEvent(self, e) -> None:
        try:
            if self._cycler:
                try: self._cycler.stop()
                except Exception: pass
            if self._cycle_thread:
                self._cycle_thread.wait(2000)
        except Exception:
            pass
        try:
            if self.btn_stop.isEnabled() and hasattr(self.ctrl, "stop"):
                self.ctrl.stop(); QApplication.processEvents()
        except Exception:
            pass
        try:
            if hasattr(self.ctrl, "close"): self.ctrl.close()
        except Exception:
            pass
        super().closeEvent(e)
