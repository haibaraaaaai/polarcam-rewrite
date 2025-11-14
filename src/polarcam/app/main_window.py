# src/polarcam/app/main_window.py
from __future__ import annotations

import time
from typing import Optional, List, Tuple
import numpy as np

from PySide6.QtCore import Qt, Signal
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
    QDialog,
)

from polarcam.app.lut_widget import HighlightLUTWidget
from polarcam.app.spot_detect import detect_spots_oneframe
from polarcam.app.spot_viewer import SpotViewerDialog

# Spot tuple: (cx_abs, cy_abs, area, bbox_w, bbox_h)
Spot = Tuple[float, float, float, int, int]


class MainWindow(QMainWindow):
    """Lean GUI with ROI/Timing/Gains + LUT + spot tools + varmap. Live video."""
    detect_done = Signal(object)

    # ------------------------------------------
    # init / build
    # ------------------------------------------
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

        # state
        self._spots: List[Spot] = []
        self._collecting = False
        self._detect_params: Tuple = tuple()
        self._detect_conn_active = False

        self._varmap = None
        self._last_pm: Optional[QPixmap] = None
        self._roi_offset = (0, 0)   # (OffsetX, OffsetY)
        self._roi_size = (0, 0)     # (Width, Height)
        self._hist_t_last = 0.0

        # UI
        self._build_video()
        self._build_toolbar()
        self._build_forms()
        self._build_detect_ui()
        self._build_spot_list_ui()
        self._assemble_layout()

        self._set_buttons(open_enabled=True, start=False, stop=False, close=False)

        # wire controller/backend signals
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
            cam.gains.connect(self._on_gains)
            cam.desaturated.connect(self._on_desaturated)
            if hasattr(cam, "auto_desat_started"):
                cam.auto_desat_started.connect(self._on_desat_started)
            if hasattr(cam, "auto_desat_finished"):
                cam.auto_desat_finished.connect(self._on_desat_finished)
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

        self.detect_done.connect(self._handle_detect_done, Qt.QueuedConnection)

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

        # Timing form — exposure shown/edited in MILLISECONDS
        self.ed_fps = QLineEdit("20.0")
        self.ed_exp = QLineEdit("50.0")  # ms
        fpsv = QDoubleValidator(0.01, 1_000.0, 3, self); fpsv.setNotation(QDoubleValidator.StandardNotation)
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
        box = QGroupBox("Detect spots (single frame)")
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
        row = QHBoxLayout()
        row.addWidget(self.btn_view_spot)
        row.addWidget(self.btn_remove_spot)
        v.addWidget(self.spot_list, 1)
        v.addLayout(row)
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

    # ------------------------------------------
    # backend signal handlers
    # ------------------------------------------
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
        self.status.showMessage(f"Error: { msg }", 5000)
        QMessageBox.warning(self, "Camera error", msg)

    def _on_frame(self, arr_obj: object) -> None:
        """Apply LUT (if any) and draw current overlays; update histogram ~2 Hz."""
        try:
            a16 = np.asarray(arr_obj)
            if a16.ndim != 2:
                return
            h, w = a16.shape

            # histogram ~2 Hz for LUT widget
            if time.time() - self._hist_t_last > 0.5:
                self._hist_t_last = time.time()
                vals = (a16.astype(np.uint16, copy=False) >> 4).ravel()
                hist = np.bincount(vals, minlength=256)
                if hist.size > 256:
                    hist = hist[:256]
                self.tone.setHistogram256(hist)

            if self._lut is not None:
                if a16.dtype != np.uint16:
                    a16 = a16.astype(np.uint16, copy=False)
                a8 = self._lut[a16]
            else:
                if a16.dtype != np.uint16:
                    a16 = a16.astype(np.uint16, copy=False)
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
        pm = QPixmap.fromImage(qimg.copy())

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
                # s are absolute coords; bring into current frame coords
                cx = int(round(float(s[0]) - ox))
                cy = int(round(float(s[1]) - oy))
                r = 6
            except Exception:
                continue

            if 0 <= cx < w and 0 <= cy < h:
                p.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)
                p.drawText(cx + r + 3, cy - r - 2, f"{i + 1}")
        p.end()
        return pm

    def _on_timing(self, d: dict) -> None:
        """Keep exposure displayed in *milliseconds*."""
        try:
            fps = d.get("fps")
            exp_us = d.get("exposure_us")
            if fps is not None:
                self.ed_fps.setText(f"{float(fps):.3f}")
            if exp_us is not None:
                self.ed_exp.setText(f"{float(exp_us) / 1000.0:.3f}")  # μs → ms
            msg = []
            if fps is not None:
                msg.append(f"FPS={float(fps):.3f}")
            if exp_us is not None:
                msg.append(f"EXP={float(exp_us) / 1000.0:.3f} ms")
            if msg:
                self.status.showMessage("Timing applied: " + "  ".join(msg), 2000)
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
                if mn is not None and mx is not None:
                    bits.append(f"{float(mn):.3f} … {float(mx):.3f}")
                if inc is not None:
                    bits.append(f"inc {float(inc):.3f}")
                return "  ".join(bits)
            self._lbl_gain_ana.setText(rng(ana)); self._lbl_gain_dig.setText(rng(dig))
        except Exception:
            pass

    # ------------------------------------------
    # detect
    # ------------------------------------------
    def _detect_clicked(self) -> None:
        if self._collecting:
            return
        mode = self.cmb_thr_mode.currentText().strip().lower()
        try:
            thr_val = float(self.ed_thr_val.text())
        except Exception:
            thr_val = 1200.0
        try:
            minA = int(float(self.ed_minA.text()))
        except Exception:
            minA = 6
        try:
            maxA = int(float(self.ed_maxA.text()))
        except Exception:
            maxA = 5000

        self._detect_params = (mode, thr_val, minA, maxA)
        self._collecting = True
        self.btn_detect.setEnabled(False)
        self.status.showMessage("Detect: grabbing one frame…", 0)

        if not self._detect_conn_active:
            self.ctrl.cam.frame.connect(self._collect_for_detect_1f, Qt.QueuedConnection)
            self._detect_conn_active = True

    def _debug_cluster_summary(self, spots: List[Spot], join_px: float = 8.0) -> None:
        """Quick-and-dirty cluster sizes: count how many spots lie within join_px of a seed."""
        try:
            if not spots:
                print("[Detect] cluster summary: no spots.")
                return
            pts = np.array([(float(s[0]), float(s[1])) for s in spots], dtype=np.float64)
            used = np.zeros(len(pts), dtype=bool)
            sizes = []
            for i in range(len(pts)):
                if used[i]:
                    continue
                d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
                idx = np.where(d <= float(join_px))[0]
                used[idx] = True
                sizes.append(len(idx))
            sizes.sort(reverse=True)
            print(f"[Detect] cluster summary (join={join_px}px): top sizes {sizes[:10]}")
        except Exception as e:
            print(f"[Detect] cluster summary error: {e}")

    def _merge_close_spots(self, spots: List[Spot], join_px: float = 8.0) -> List[Spot]:
        """Greedy merge: within join_px, keep the spot with the largest area."""
        n = len(spots)
        if n < 2:
            return spots
        pts = np.array([(s[0], s[1]) for s in spots], dtype=np.float64)
        used = np.zeros(n, dtype=bool)
        merged: List[Spot] = []
        for i in range(n):
            if used[i]:
                continue
            d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
            idx = np.where(d <= float(join_px))[0]
            used[idx] = True
            # choose max area within cluster
            best = max(idx, key=lambda k: float(spots[k][2]) if len(spots[k]) > 2 else 0.0)
            merged.append(spots[best])
        return merged

    def _popup_detect_preview(self, frame8: np.ndarray, spots_xy: List[Tuple[float, float]]) -> None:
        """Show the exact frame used for detection with spot centers overlaid (frame coords)."""
        h, w = frame8.shape
        qimg = QImage(frame8.data, w, h, w, QImage.Format_Grayscale8)
        pm = QPixmap.fromImage(qimg.copy())

        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(Qt.red, 2)
        p.setPen(pen)
        for x, y in spots_xy:
            cx = int(round(x))
            cy = int(round(y))
            r = 6
            if 0 <= cx < w and 0 <= cy < h:
                p.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)
        p.end()

        class _ImgDlg(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Detect preview (exact frame)")
                self.label = QLabel(self); self.label.setAlignment(Qt.AlignCenter)
                lay = QVBoxLayout(self); lay.addWidget(self.label)
                self._pm = None

            def set_pixmap(self, pix):
                self._pm = pix
                self._resize_to_label()

            def resizeEvent(self, ev):
                super().resizeEvent(ev)
                self._resize_to_label()

            def _resize_to_label(self):
                if self._pm is None:
                    return
                self.label.setPixmap(self._pm.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        dlg = _ImgDlg(self)
        dlg.resize(900, 650)
        dlg.set_pixmap(pm)
        dlg.show()  # non-modal so the stream continues

    def _collect_for_detect_1f(self, arr_obj: object) -> None:
        # disconnect immediately; we need only one sample
        try:
            if self._detect_conn_active:
                self.ctrl.cam.frame.disconnect(self._collect_for_detect_1f)
                self._detect_conn_active = False
        except Exception:
            pass

        try:
            img = np.asarray(arr_obj)
            if img.ndim != 2:
                raise RuntimeError("Frame is not 2D.")
            H, W = img.shape
            ox, oy = self._roi_offset
            mode, thr_val, minA, maxA = self._detect_params

            # Compute absolute DN threshold (for logging)
            if (mode or "").lower() == "percentile":
                thr_abs = float(np.percentile(img, float(thr_val)))
                mode_str = f"percentile {thr_val:.3f}% -> DN {thr_abs:.1f}"
            else:
                thr_abs = float(thr_val)
                mode_str = f"absolute DN {thr_abs:.1f}"

            above = int((img.astype(np.uint16, copy=False) > thr_abs).sum())
            print(
                f"[Detect] HxW={H}x{W}  ROI@({ox},{oy})  "
                f"mode={mode_str}  min={int(img.min())} max={int(img.max())} "
                f"mean={float(img.mean()):.1f}  pixels>thr={above}"
            )

            # Run detector in frame coords
            spots_rel = detect_spots_oneframe(
                img.astype(np.uint16, copy=False),
                thr_mode=("percentile" if (mode or "").lower() == "percentile" else "absolute"),
                thr_value=float(thr_val),
                min_area=int(minA),
                max_area=int(maxA),
                open_radius=1,
                close_radius=1,
            )

            # Build absolute coords and filter OOB robustly
            spots_abs: List[Spot] = []
            preview_xy: List[Tuple[float, float]] = []  # for the preview (frame coords)
            for s in spots_rel or []:
                try:
                    cx, cy, area, bw, bh = s
                    cx_f = float(cx); cy_f = float(cy)
                    if not (0.0 <= cx_f < float(W) and 0.0 <= cy_f < float(H)):
                        continue
                    preview_xy.append((cx_f, cy_f))
                    spots_abs.append((
                        cx_f + float(ox),
                        cy_f + float(oy),
                        float(area),
                        int(max(1, round(bw))),
                        int(max(1, round(bh))),
                    ))
                except Exception:
                    continue

            n0 = len(spots_abs)
            # Merge near-duplicates
            spots_abs = self._merge_close_spots(spots_abs, join_px=8.0)
            n1 = len(spots_abs)

            if n0:
                areas = np.array([s[2] for s in spots_abs], dtype=float)
                q10, q50, q90 = np.percentile(areas, [10, 50, 90]).tolist()
                print(f"[Detect] spots(before/after)={n0}/{n1}  area q10/50/90 = {q10:.1f}/{q50:.1f}/{q90:.1f}")
                self._debug_cluster_summary(spots_abs, join_px=8.0)
            else:
                print("[Detect] spots=0")

            # Show exact frame + overlay so we know what was segmented
            # Use current LUT for readability if available, otherwise 12->8
            if self._lut is not None:
                frame8 = self._lut[img.astype(np.uint16, copy=False)]
            else:
                frame8 = ((img.astype(np.uint16, copy=False) + 8) >> 4).astype(np.uint8, copy=False)
            if not frame8.flags.c_contiguous:
                frame8 = np.ascontiguousarray(frame8)
            self._popup_detect_preview(frame8, preview_xy)

            self.detect_done.emit(("ok", spots_abs))

        except Exception as e:
            self.detect_done.emit(("err", str(e)))

    def _handle_detect_done(self, payload: object) -> None:
        kind, data = payload
        self._collecting = False
        self.btn_detect.setEnabled(True)

        if kind == "ok":
            spots = data or []
            if len(spots) > 800:
                print(f"[Detect] too many spots ({len(spots)}); showing first 800.")
                spots = spots[:800]
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
            cx, cy, a, w, h = s
            item = QListWidgetItem(
                f"#{i + 1}  (x={cx:.1f}, y={cy:.1f})  A={a:.1f}  wh≈({int(w)},{int(h)})"
            )
            self.spot_list.addItem(item)

    def _clear_overlays(self) -> None:
        self._spots = []
        self._refresh_spot_list()
        self.status.showMessage("Overlays cleared.", 1500)

    # ---------- view/remove spots ----------
    def _selected_spot_indices(self) -> List[int]:
        return sorted(set(idx.row() for idx in self.spot_list.selectedIndexes()))

    def _view_selected_spots(self) -> None:
        idxs = self._selected_spot_indices()
        if not self._spots:
            self.status.showMessage("No spots to view.", 2000)
            return
        if not idxs:
            idxs = [0]
        sel_spots = [self._spots[i] for i in idxs]

        # Snapshot current ROI + FPS so the viewer can restore them
        try:
            w = int(float(self.ed_w.text() or "0"))
            h = int(float(self.ed_h.text() or "0"))
            x = int(float(self.ed_x.text() or "0"))
            y = int(float(self.ed_y.text() or "0"))
            saved_roi = (w, h, x, y)
        except Exception:
            saved_roi = (0, 0, 0, 0)

        try:
            saved_fps = float(self.ed_fps.text()) if self.ed_fps.text().strip() else None
        except Exception:
            saved_fps = None

        from polarcam.app.spot_viewer import SpotViewerDialog
        dlg = SpotViewerDialog(self.ctrl, sel_spots, self, saved_roi=saved_roi, saved_fps=saved_fps)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.show()

    def _remove_selected_spots(self) -> None:
        idxs = set(self._selected_spot_indices())
        if not idxs:
            return
        self._spots = [s for i, s in enumerate(self._spots) if i not in idxs]
        self._refresh_spot_list()

    # ------------------------------------------
    # UI → controller
    # ------------------------------------------
    def _open_clicked(self) -> None:
        if hasattr(self.ctrl, "open"):
            self.ctrl.open()

    def _start_clicked(self) -> None:
        if self.btn_start.isEnabled() and hasattr(self.ctrl, "start"):
            self.ctrl.start()

    def _stop_clicked(self) -> None:
        if self.btn_stop.isEnabled() and hasattr(self.ctrl, "stop"):
            self.ctrl.stop()

    def _close_clicked(self) -> None:
        if self.btn_close.isEnabled() and hasattr(self.ctrl, "close"):
            self.ctrl.close()

    def _apply_roi(self) -> None:
        try:
            w = int(float(self.ed_w.text() or "0"))
            h = int(float(self.ed_h.text() or "0"))
            x = int(float(self.ed_x.text() or "0"))
            y = int(float(self.ed_y.text() or "0"))
        except Exception:
            self.status.showMessage("ROI: invalid numbers", 2000)
            return
        if w <= 0 or h <= 0:
            self.status.showMessage("ROI: width/height must be > 0. Use Full sensor if unsure.", 3000)
            return
        if hasattr(self.ctrl, "set_roi"):
            self.ctrl.set_roi(w, h, x, y)

    def _full_roi_clicked(self) -> None:
        if hasattr(self.ctrl, "full_sensor"):
            self.ctrl.full_sensor()

    def _apply_timing(self) -> None:
        # exposure textbox is *milliseconds*
        fps = float(self.ed_fps.text()) if self.ed_fps.hasAcceptableInput() else None
        exp_ms = float(self.ed_exp.text()) if self.ed_exp.hasAcceptableInput() else None
        if hasattr(self.ctrl, "set_timing"):
            self.ctrl.set_timing(fps, exp_ms)

    def _apply_gains(self) -> None:
        ana = float(self.ed_gain_ana.text()) if self.ed_gain_ana.text().strip() else None
        dig = float(self.ed_gain_dig.text()) if self.ed_gain_dig.text().strip() else None
        if hasattr(self.ctrl, "set_gains"):
            self.ctrl.set_gains(ana, dig)

    def _refresh_gains(self) -> None:
        if hasattr(self.ctrl, "refresh_gains"):
            self.ctrl.refresh_gains()
            self.status.showMessage("Refreshing gains…", 800)
        elif hasattr(self.ctrl, "cam") and hasattr(self.ctrl.cam, "refresh_gains"):
            self.ctrl.cam.refresh_gains()
            self.status.showMessage("Refreshing gains…", 800)

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
            self.status.showMessage(
                f"Desaturate done: iters={it}, max={mx}, target={tgt}, exposure={(exp_us or 0)/1000.0:.1f} ms", 3000
            )
        else:
            QMessageBox.warning(
                self,
                "Desaturate",
                f"Couldn’t reach target after {max(it, 5)} tries.\n"
                f"Final max={mx}, target={tgt}, exposure={(exp_us or 0)/1000.0:.1f} ms",
            )

    def _on_desat_started(self) -> None:
        if hasattr(self, "btn_desat"):
            self.btn_desat.setEnabled(False)
        self.status.showMessage("Auto-desaturating…", 2000)

    def _on_desat_finished(self) -> None:
        if hasattr(self, "btn_desat"):
            self.btn_desat.setEnabled(True)
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
        if vm is None:
            return False
        try:
            return vm.isVisible()
        except RuntimeError:
            self._varmap = None
            return False

    def _refresh_video_view(self) -> None:
        pm = getattr(self, "_last_pm", None)
        if pm is None:
            return
        self.video.setPixmap(pm.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._refresh_video_view()

    # ---------- cleanup ----------
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
        try:
            if self.btn_stop.isEnabled() and hasattr(self.ctrl, "stop"):
                self.ctrl.stop(); QApplication.processEvents()
        except Exception:
            pass
        try:
            if hasattr(self.ctrl, "close"):
                self.ctrl.close()
        except Exception:
            pass
        super().closeEvent(e)
