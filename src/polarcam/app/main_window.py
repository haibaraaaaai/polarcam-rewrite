# src/polarcam/app/main_window.py
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QIntValidator, QDoubleValidator
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
    QVBoxLayout,
    QWidget,
)

from polarcam.app.lut_widget import HighlightLUTWidget


class MainWindow(QMainWindow):
    """Lean GUI: Open/Start/Stop/Close + ROI/Timing + Gains + Highlight LUT. Live video."""

    def __init__(self, ctrl) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.setWindowTitle("PolarCam (lean)")
        self.resize(1200, 720)

        # status bar
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

        # tone/LUT control
        self._lut: Optional[np.ndarray] = None
        self.tone = HighlightLUTWidget(self)
        self.tone.paramsChanged.connect(self._on_tone_params)
        # seed LUT from widget defaults
        self._on_tone_params(*self.tone.params())

        # build UI
        self._build_video()
        self._build_toolbar()
        self._build_forms()
        self._assemble_layout()

        # starting button states
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
            cam.gains.connect(self._on_gains)
            cam.desaturated.connect(self._on_desaturated)
            cam.auto_desat_started.connect(self._on_desat_started)
            cam.auto_desat_finished.connect(self._on_desat_finished)
        else:
            self.status.showMessage("No camera backend attached.", 4000)

        # wire buttons (UI → controller)
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

        self._varmap = None
        self._last_pm = None  # type: Optional[QPixmap]

        # small helper for histogram rate-limiting
        self._hist_t_last = 0.0

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

        # Timing form
        self.ed_fps = QLineEdit("20.0")
        self.ed_exp = QLineEdit("50.0")
        fpsv = QDoubleValidator(0.01, 1_000.0, 3, self)
        fpsv.setNotation(QDoubleValidator.StandardNotation)
        expv = QDoubleValidator(0.01, 2_000.0, 3, self)
        expv.setNotation(QDoubleValidator.StandardNotation)
        self.ed_fps.setValidator(fpsv)
        self.ed_exp.setValidator(expv)
        self.btn_apply_tim = QPushButton("Apply Timing")
        self.btn_desat = QPushButton("Desaturate")

        # Gains form
        self.ed_gain_ana = QLineEdit("")  # blank => unchanged
        self.ed_gain_dig = QLineEdit("")
        gval = QDoubleValidator(0.0, 1000.0, 3, self)
        gval.setNotation(QDoubleValidator.StandardNotation)
        self.ed_gain_ana.setValidator(gval)
        self.ed_gain_dig.setValidator(gval)
        self.ed_gain_ana.setPlaceholderText("—")
        self.ed_gain_dig.setPlaceholderText("—")
        self.btn_apply_gains = QPushButton("Apply gains")
        self.btn_refresh_gains = QPushButton("Refresh gains")
        self._lbl_gain_ana = QLabel("")  # “min…max (inc …)”
        self._lbl_gain_dig = QLabel("")

        # master form layout
        self._form = QFormLayout()
        self._form.addRow("Width", self.ed_w)
        self._form.addRow("Height", self.ed_h)
        self._form.addRow("OffsetX", self.ed_x)
        self._form.addRow("OffsetY", self.ed_y)
        row = QHBoxLayout()
        row.addWidget(self.btn_apply_roi)
        row.addWidget(self.btn_full_roi)
        rw = QWidget()
        rw.setLayout(row)
        self._form.addRow(rw)
        self._form.addRow("FPS", self.ed_fps)
        self._form.addRow("Exposure (ms)", self.ed_exp)
        rowt = QHBoxLayout()
        rowt.addWidget(self.btn_apply_tim)
        rowt.addWidget(self.btn_desat)
        wt = QWidget(); wt.setLayout(rowt)
        self._form.addRow(wt)

        # gains rows
        self._form.addRow("Analog gain", self.ed_gain_ana)
        self._form.addRow("", self._lbl_gain_ana)
        self._form.addRow("Digital gain", self.ed_gain_dig)
        self._form.addRow("", self._lbl_gain_dig)
        rowg = QHBoxLayout()
        rowg.addWidget(self.btn_apply_gains)
        rowg.addWidget(self.btn_refresh_gains)
        rg = QWidget()
        rg.setLayout(rowg)
        self._form.addRow(rg)

    def _assemble_layout(self) -> None:
        right = QWidget()
        right.setLayout(self._form)

        leftcol = QVBoxLayout()
        leftcol.addLayout(self._toolbar)
        leftcol.addWidget(self.video, 1)
        leftcol.addWidget(self.tone)

        root = QHBoxLayout()
        root.addLayout(leftcol, 2)
        root.addWidget(right, 1)

        cw = QWidget()
        cw.setLayout(root)
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
        # pull a fresh gains snapshot on open
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
        """Apply highlight-LUT if present; otherwise do rounded 12→8. Update LUT histogram ~2 Hz."""
        try:
            a16 = np.asarray(arr_obj)
            if a16.ndim != 2:
                return
            h, w = a16.shape

            # update histogram for LUT widget at ~2 Hz
            if time.time() - self._hist_t_last > 0.5:
                self._hist_t_last = time.time()
                # 256 bins over 12-bit range: bucket by top 8 bits (>> 4)
                vals = (a16.astype(np.uint16, copy=False) >> 4).ravel()
                hist = np.bincount(vals, minlength=256)
                if hist.size > 256:  # safety (shouldn't happen)
                    hist = hist[:256]
                self.tone.setHistogram256(hist)

            # mapping
            if self._lut is not None:
                if a16.dtype != np.uint16:
                    a16 = a16.astype(np.uint16, copy=False)
                a8 = self._lut[a16]
            else:
                # rounded divide-by-16 (vs truncate)
                if a16.dtype != np.uint16:
                    a16 = a16.astype(np.uint16, copy=False)
                a8 = ((a16 + 8) >> 4).astype(np.uint8, copy=False)

            if not a8.flags.c_contiguous:
                a8 = np.ascontiguousarray(a8)

            qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
            self._last_pm = QPixmap.fromImage(qimg.copy())
            self._refresh_video_view()
        except Exception as e:
            self.status.showMessage(f"Frame error: {e}", 2000)

    def _on_timing(self, d: dict) -> None:
        try:
            fps = d.get("fps")
            exp_us = d.get("exposure_us")
            if fps is not None:
                self.ed_fps.setText(f"{float(fps):.3f}")
            if exp_us is not None:
                self.ed_exp.setText(f"{float(exp_us) / 1000.0:.3f}")
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
        except Exception:
            pass

    def _on_gains(self, g: dict) -> None:
        """Show current/limits for analog/digital gains."""
        try:
            ana = g.get("analog", {}) or {}
            dig = g.get("digital", {}) or {}

            if ana.get("val") is not None:
                self.ed_gain_ana.setText(f"{float(ana['val']):.3f}")
            if dig.get("val") is not None:
                self.ed_gain_dig.setText(f"{float(dig['val']):.3f}")

            def rng(dct: dict) -> str:
                mn = dct.get("min")
                mx = dct.get("max")
                inc = dct.get("inc")
                bits = []
                if mn is not None and mx is not None:
                    bits.append(f"{float(mn):.3f} … {float(mx):.3f}")
                if inc is not None:
                    bits.append(f"inc {float(inc):.3f}")
                return "  ".join(bits)

            self._lbl_gain_ana.setText(rng(ana))
            self._lbl_gain_dig.setText(rng(dig))
        except Exception:
            pass

    # ---------- UI handlers (UI → controller) ----------
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
        fps = float(self.ed_fps.text()) if self.ed_fps.hasAcceptableInput() else None
        exp = float(self.ed_exp.text()) if self.ed_exp.hasAcceptableInput() else None
        if hasattr(self.ctrl, "set_timing"):
            self.ctrl.set_timing(fps, exp)

    def _apply_gains(self) -> None:
        """Blank field = leave unchanged."""
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
        # Only really useful while streaming; still harmless if not
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
            self.status.showMessage(f"Desaturate done: iters={it}, max={mx}, target={tgt}, exposure={exp_us:.1f} µs", 3000)
        else:
            QMessageBox.warning(
                self, "Desaturate",
                f"Couldn’t reach target after {max(it, 5)} tries.\n"
                f"Final max={mx}, target={tgt}, exposure={exp_us:.1f} µs"
                if exp_us is not None else
                f"Couldn’t reach target after {max(it, 5)} tries.\nFinal max={mx}, target={tgt}"
            )

    def _on_desat_started(self) -> None:
        # defensive: button may not exist in some layouts
        if hasattr(self, "btn_desat"):
            self.btn_desat.setEnabled(False)
        self.status.showMessage("Auto-desaturating…", 2000)

    def _on_desat_finished(self) -> None:
        if hasattr(self, "btn_desat"):
            self.btn_desat.setEnabled(True)
        self.status.showMessage("Auto-desaturate complete.", 1500)

    def _open_varmap(self) -> None:
        # If an existing dialog is alive, just bring it to front
        if self._varmap_alive():
            self._varmap.raise_()
            self._varmap.activateWindow()
            return

        # Otherwise create a new one
        from polarcam.app.varmap_dialog import VarMapDialog
        dlg = VarMapDialog(self.ctrl, self)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)

        # When Qt deletes the C++ object, reset our Python ref to None
        dlg.destroyed.connect(lambda _=None: setattr(self, "_varmap", None))

        self._varmap = dlg
        dlg.show()

    # ---------- LUT plumbing ----------
    def _on_tone_params(self, floor: int, cap: int, gamma: float) -> None:
        """Rebuild 4096→8 LUT whenever the widget changes."""
        self._lut = self._build_lut(floor, cap, gamma)

    @staticmethod
    def _build_lut(floor: int, cap: int, gamma: float) -> np.ndarray:
        cap = max(floor + 1, min(cap, 4095))
        floor = max(0, min(floor, 4094))
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
            # this will raise RuntimeError if the C++ object was deleted
            return vm.isVisible()
        except RuntimeError:
            self._varmap = None
            return False
        
    def _refresh_video_view(self) -> None:
        """Scale the last frame to the label size, keeping aspect ratio."""
        pm = getattr(self, "_last_pm", None)
        if pm is None:
            return
        self.video.setPixmap(
            pm.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

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
        # graceful stop→close when user hits the window X
        try:
            if self.btn_stop.isEnabled() and hasattr(self.ctrl, "stop"):
                self.ctrl.stop()
                QApplication.processEvents()
        except Exception:
            pass
        try:
            if hasattr(self.ctrl, "close"):
                self.ctrl.close()
        except Exception:
            pass
        super().closeEvent(e)
