# src/polarcam/app/spot_viewer.py
from __future__ import annotations
import math, os, time
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread
from PySide6.QtWidgets import QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QStatusBar
from PySide6.QtGui import QImage, QPixmap

UI_CAP_MAX = 20.0  # preview cap (Hz)

# ---- Type-only imports ----
if TYPE_CHECKING:
    from .spot_recorder import SpotSignalRecorder, RecorderConfig, Spot  # type: ignore[misc]

# ---- Runtime-optional imports ----
try:
    from .spot_recorder import (
        SpotSignalRecorder as SpotSignalRecorderRT,
        RecorderConfig as RecorderConfigRT,
        Spot as SpotRT,
    )
    _RECORDER_AVAILABLE = True
except Exception:
    SpotSignalRecorderRT = None  # type: ignore[assignment]
    RecorderConfigRT = None      # type: ignore[assignment]
    SpotRT = None                # type: ignore[assignment]
    _RECORDER_AVAILABLE = False


class SpotViewerWindow(QMainWindow):
    """
    Passive spot viewer (UI preview capped at ≤20 Hz) with optional background
    recorder that runs the camera at max FPS. It tolerates being given either
    a Controller (with .cam) or a camera-like object (with .frame/.roi/.timing).
    """
    rec_progress = Signal(int)

    def __init__(
        self,
        dev,  # Controller or camera-like object
        spots: List[Tuple[float, float, float, int, int]],
        parent: Optional[QMainWindow] = None,
        index: int = 0,
        **_ignored_kwargs,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spot viewer — passive preview (≤20 Hz) + optional recorder")
        self.resize(900, 650)

        self.dev = dev
        self.spots = list(spots or [])
        self.idx = int(max(0, min(index, max(0, len(self.spots) - 1))))

        # UI
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.video = QLabel("Waiting for frames…")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#777;")

        left = QVBoxLayout(); left.addWidget(self.video, 1)
        lw = QWidget(); lw.setLayout(left)

        self.btn_start = QPushButton("Start Rec @ max FPS")
        self.btn_stop  = QPushButton("Stop Recording"); self.btn_stop.setEnabled(False)
        self.btn_prev  = QPushButton("◀ Prev")
        self.btn_next  = QPushButton("Next ▶")

        if not _RECORDER_AVAILABLE:
            self.btn_start.setEnabled(False)
            self.btn_start.setText("Recorder unavailable")
            self.status.showMessage("Recorder module not found — preview only.", 5000)

        right = QVBoxLayout()
        right.addWidget(self.btn_start)
        right.addWidget(self.btn_stop)
        right.addSpacing(24)
        nav = QHBoxLayout(); nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next)
        right.addLayout(nav)
        right.addStretch(1)
        rw = QWidget(); rw.setLayout(right)

        root = QHBoxLayout(); root.addWidget(lw, 3); root.addWidget(rw, 1)
        w = QWidget(); w.setLayout(root)
        self.setCentralWidget(w)

        # Signals (support dev or dev.cam)
        if hasattr(self.dev, "roi"):
            try: self.dev.roi.connect(self._on_roi_update, Qt.QueuedConnection)
            except Exception: pass
        if hasattr(self.dev, "timing"):
            try: self.dev.timing.connect(self._on_timing_update, Qt.QueuedConnection)
            except Exception: pass
        if hasattr(self.dev, "frame"):
            try: self.dev.frame.connect(self._on_frame_arrived, Qt.QueuedConnection)
            except Exception: pass
        elif hasattr(self.dev, "cam") and hasattr(self.dev.cam, "frame"):
            try: self.dev.cam.frame.connect(self._on_frame_arrived, Qt.QueuedConnection)
            except Exception: pass

        self.btn_prev.clicked.connect(lambda: self._jump(-1))
        self.btn_next.clicked.connect(lambda: self._jump(+1))
        self.btn_start.clicked.connect(self._start_rec)
        self.btn_stop.clicked.connect(self._stop_rec)

        # State
        self._applied_roi = (0, 0, 0, 0)  # (x, y, w, h)
        self._fps_now = 20.0
        self._ui_cap = UI_CAP_MAX
        self._latest = None  # np.ndarray

        # UI tick timer (decoupled from frame rate)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._ui_tick)
        self._timer.start(int(1000.0 / self._ui_cap))

        # Recorder bits
        self._rec_thread: Optional[QThread] = None
        self._rec: Optional['SpotSignalRecorder'] = None  # type-only name quoted

        self._update_cap()

    # ---- camera snapshots ----
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
                self._fps_now = float(rf)
                self._update_cap()
        except Exception:
            pass

    def _update_cap(self) -> None:
        cap = float(min(UI_CAP_MAX, max(1.0, self._fps_now)))
        self._timer.setInterval(int(1000.0 / cap))
        self._ui_cap = cap

    # ---- frame handling ----
    @Slot(object)
    def _on_frame_arrived(self, arr_obj: object) -> None:
        a = np.asarray(arr_obj)
        if a.ndim == 2:
            self._latest = a

    def _ui_tick(self) -> None:
        a = self._latest
        if a is None or a.ndim != 2 or not self.spots:
            return

        cx, cy, r, _area, _inten = self.spots[self.idx]

        ax, ay, aw, ah = self._applied_roi
        if aw <= 0 or ah <= 0:
            ah, aw = a.shape; ax = ay = 0

        rcx = float(cx) - float(ax)
        rcy = float(cy) - float(ay)

        r_eff = max(4.0, float(r))
        want = int(max(10, min(aw, ah, math.ceil(2.0 * r_eff + 6))))
        side = want if (want % 2 == 0) else want + 1

        ix = max(0, min(aw - side, int(round(rcx)) - side // 2))
        iy = max(0, min(ah - side, int(round(rcy)) - side // 2))

        H, W = a.shape
        ix = max(0, min(W - 2, ix)); iy = max(0, min(H - 2, iy))
        jx = max(ix + 1, min(W, ix + side))
        jy = max(iy + 1, min(H, iy + side))

        crop = a[iy:jy, ix:jx]
        a8 = (crop >> 4).astype(np.uint8) if crop.dtype == np.uint16 else crop.astype(np.uint8, copy=False)
        if not a8.flags.c_contiguous:
            a8 = np.ascontiguousarray(a8)
        h, w = a8.shape
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8).copy()  # copy to own memory
        self.video.setPixmap(QPixmap.fromImage(qimg))

    # ---- navigation ----
    def _jump(self, d: int) -> None:
        if not self.spots:
            return
        self.idx = (self.idx + d) % len(self.spots)

    # ---- recorder control ----
    def _start_rec(self) -> None:
        if not _RECORDER_AVAILABLE:
            return
        if not self.spots:
            self.status.showMessage("No spots available (run Detect first).", 2500)
            return

        cam_for_rec = getattr(self.dev, "cam", None)
        if cam_for_rec is None or not (hasattr(cam_for_rec, "roi") and hasattr(cam_for_rec, "frame")):
            cam_for_rec = self.dev

        if not (hasattr(cam_for_rec, "roi") and hasattr(cam_for_rec, "frame")):
            self.status.showMessage("Recorder can’t find a camera with .roi/.frame; preview only.", 4000)
            return

        cx, cy, r, area, inten = self.spots[self.idx]
        spot = SpotRT(cx, cy, r, int(area), int(inten))  # runtime class

        out_dir = os.path.join(os.getcwd(), "captures")
        cfg = RecorderConfigRT(
            out_dir=out_dir,
            base_name=f"spot{self.idx+1}",
            chunk_len=20000,
            maximize_camera_fps=True,
        )

        self._rec_thread = QThread(self)
        self._rec = SpotSignalRecorderRT(cam_for_rec, self.dev, spot, cfg)  # runtime class
        self._rec.moveToThread(self._rec_thread)

        self._rec_thread.started.connect(self._rec.start)
        self._rec.progress.connect(lambda n: (self.status.showMessage(f"Recording… {n} samples", 1000), self.rec_progress.emit(n)))
        self._rec.error.connect(lambda msg: self.status.showMessage(f"Recorder error: {msg}", 4000))
        self._rec.stopped.connect(self._rec_thread.quit)

        self._rec_thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.showMessage("Recording signals at max FPS (preview capped).", 3000)

    def _stop_rec(self) -> None:
        if not _RECORDER_AVAILABLE:
            return
        if self._rec:
            try: self._rec.stop()
            except Exception: pass
        if self._rec_thread:
            self._rec_thread.wait(3000)
        self._rec = None
        self._rec_thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.showMessage("Recorder stopped.", 2000)

    # ---- cleanup ----
    def closeEvent(self, e):
        try: self._timer.stop()
        except Exception: pass
        try:
            if hasattr(self.dev, "frame"):
                self.dev.frame.disconnect(self._on_frame_arrived)
        except Exception: pass
        try:
            if hasattr(self.dev, "cam") and hasattr(self.dev.cam, "frame"):
                self.dev.cam.frame.disconnect(self._on_frame_arrived)
        except Exception: pass
        try:
            if hasattr(self.dev, "roi"):
                self.dev.roi.disconnect(self._on_roi_update)
        except Exception: pass
        try:
            if hasattr(self.dev, "timing"):
                self.dev.timing.disconnect(self._on_timing_update)
        except Exception: pass
        try: self._stop_rec()
        except Exception: pass
        super().closeEvent(e)

# Back-compat alias
SpotViewerDialog = SpotViewerWindow
__all__ = ["SpotViewerWindow", "SpotViewerDialog"]
