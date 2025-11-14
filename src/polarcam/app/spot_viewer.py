# src/polarcam/app/spot_viewer.py
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget
)

# spot tuple: (cx, cy, area_px, bbox_w, bbox_h)


def _round_up(v: int, step: int) -> int:
    if step <= 1:
        return int(v)
    return int(((v + step - 1) // step) * step)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


class SpotViewerDialog(QDialog):
    """
    Automatically zooms to each selected spot:
      • chooses a minimal *hardware* ROI around the spot (respecting camera snapping);
      • then applies a *software* crop to hide any leftover padding.
    Restores prior ROI + FPS on close. Exposure is left untouched.
    """

    # Safe ROI constraints for IDS polarization camera (adjust if needed)
    W_STEP = 16     # width increment (px)
    H_STEP = 2      # height increment (px)
    MIN_W = 64      # minimal ROI width (px)
    MIN_H = 32      # minimal ROI height (px)

    PAD_HW = 12     # px padding around the spot before snapping (hardware ROI)
    PAD_SW = 8      # px padding shown around the spot in the software crop (display)

    def __init__(
        self,
        ctrl,
        spots: List[Tuple[float, float, float, int, int]],
        parent: Optional[QWidget] = None,
        saved_roi: Tuple[int, int, int, int] | None = None,  # (w, h, x, y)
        saved_fps: Optional[float] = None,
    ) -> None:
        super().__init__(parent, Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setWindowTitle("Spot viewer")
        self.resize(950, 700)

        self.ctrl = ctrl
        self.cam = getattr(ctrl, "cam", None)
        self.spots = list(spots)
        self.idx = 0

        # Remember original state to restore
        self._orig_roi = saved_roi or (0, 0, 0, 0)  # (w,h,x,y)
        self._orig_fps = saved_fps

        # Live camera reports
        self._applied_roi = (0, 0, 0, 0)  # (x,y,w,h)
        self._last_frame_shape = None     # (H, W)

        # UI
        self.video = QLabel("Waiting for frames…")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#777;")
        self.video.setMinimumSize(640, 480)

        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        self.btn_close = QPushButton("Close")

        nav = QHBoxLayout()
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addStretch(1)
        nav.addWidget(self.btn_close)

        root = QVBoxLayout(self)
        root.addWidget(self.video, 1)
        root.addLayout(nav)

        # Signals
        self.btn_prev.clicked.connect(lambda: self._jump(-1))
        self.btn_next.clicked.connect(lambda: self._jump(+1))
        self.btn_close.clicked.connect(self.close)

        if self.cam is not None:
            if hasattr(self.cam, "roi"):
                self.cam.roi.connect(self._on_roi_report, Qt.QueuedConnection)
            if hasattr(self.cam, "frame"):
                self.cam.frame.connect(self._on_frame, Qt.QueuedConnection)

        # First apply after the dialog shows
        QTimer.singleShot(0, self._apply_zoom)

    # -------- camera reports --------
    def _on_roi_report(self, d: dict) -> None:
        try:
            w = int(round(float(d.get("Width", 0))))
            h = int(round(float(d.get("Height", 0))))
            x = int(round(float(d.get("OffsetX", 0))))
            y = int(round(float(d.get("OffsetY", 0))))
            if w and h:
                self._applied_roi = (x, y, w, h)
        except Exception:
            pass

    # -------- navigation --------
    def _jump(self, delta: int) -> None:
        if not self.spots:
            return
        self.idx = (self.idx + delta) % len(self.spots)
        self._apply_zoom()

    # -------- zoom logic --------
    def _apply_zoom(self) -> None:
        """Compute minimal HW ROI around current spot and apply it; ask for max FPS."""
        if not self.spots or self.cam is None:
            return

        cx, cy, area, bw, bh = self.spots[self.idx]

        # Sensor shape from last frame if available
        H = W = None
        if self._last_frame_shape is not None:
            H, W = self._last_frame_shape

        # Minimal ROI around spot (+ padding), snapped to increments
        want_w = max(self.MIN_W, int(bw + 2 * self.PAD_HW))
        want_h = max(self.MIN_H, int(bh + 2 * self.PAD_HW))
        hw = _round_up(want_w, self.W_STEP)
        hh = _round_up(want_h, self.H_STEP)

        # Center on spot, clamp inside sensor if known
        x0 = int(round(cx - hw / 2))
        y0 = int(round(cy - hh / 2))
        if W is not None and H is not None:
            x0 = _clamp(x0, 0, max(0, W - hw))
            y0 = _clamp(y0, 0, max(0, H - hh))

        # Apply ROI; our roi slot will capture the snapped result
        self.ctrl.set_roi(hw, hh, x0, y0)

        # Ask for max FPS; backend will clamp to fps_max
        try:
            self.ctrl.set_timing(float("inf"), None)
        except Exception:
            pass

        self.setWindowTitle(f"Spot viewer — Spot {self.idx + 1}/{len(self.spots)} @ ({cx:.1f},{cy:.1f})")

    # -------- frame path / software crop --------
    def _on_frame(self, arr_obj: object) -> None:
        a16 = np.asarray(arr_obj)
        if a16.ndim != 2:
            return

        self._last_frame_shape = a16.shape  # update sensor/ROI knowledge
        ax, ay, aw, ah = self._applied_roi

        # Fallback if we haven't received a ROI report yet
        if aw <= 0 or ah <= 0:
            ah, aw = a16.shape
            ax = ay = 0

        # Crop centered on the spot, trimming ROI padding
        try:
            cx, cy, area, bw, bh = self.spots[self.idx]
        except Exception:
            cx = ax + aw / 2.0
            cy = ay + ah / 2.0
            bw = aw
            bh = ah

        rcx = float(cx) - float(ax)
        rcy = float(cy) - float(ay)

        crop_w = int(max(24, min(aw, bw + 2 * self.PAD_SW)))
        crop_h = int(max(24, min(ah, bh + 2 * self.PAD_SW)))

        ix = _clamp(int(round(rcx)) - crop_w // 2, 0, max(0, aw - crop_w))
        iy = _clamp(int(round(rcy)) - crop_h // 2, 0, max(0, ah - crop_h))
        jx = ix + crop_w
        jy = iy + crop_h

        # Bound by actual frame array
        ix = _clamp(ix, 0, a16.shape[1] - 1)
        iy = _clamp(iy, 0, a16.shape[0] - 1)
        jx = _clamp(jx, ix + 1, a16.shape[1])
        jy = _clamp(jy, iy + 1, a16.shape[0])

        crop = a16[iy:jy, ix:jx]
        self._show_u8(crop)

    def _show_u8(self, a16: np.ndarray) -> None:
        a8 = ((a16.astype(np.uint16, copy=False) + 8) >> 4).astype(np.uint8, copy=False)
        if not a8.flags.c_contiguous:
            a8 = np.ascontiguousarray(a8)
        h, w = a8.shape
        qimg = QImage(a8.data, w, h, w, QImage.Format_Grayscale8)
        pm = QPixmap.fromImage(qimg)
        self.video.setPixmap(pm.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # -------- restore on close --------
    def closeEvent(self, e) -> None:
        # Restore ROI
        try:
            w, h, x, y = self._orig_roi
            if w and h:
                self.ctrl.set_roi(int(w), int(h), int(x), int(y))
            else:
                if hasattr(self.ctrl, "full_sensor"):
                    self.ctrl.full_sensor()
        except Exception:
            pass

        # Restore FPS
        try:
            if self._orig_fps is not None:
                self.ctrl.set_timing(float(self._orig_fps), None)
        except Exception:
            pass

        # Disconnect
        try:
            if self.cam is not None and hasattr(self.cam, "frame"):
                self.cam.frame.disconnect(self._on_frame)
        except Exception:
            pass
        try:
            if self.cam is not None and hasattr(self.cam, "roi"):
                self.cam.roi.disconnect(self._on_roi_report)
        except Exception:
            pass

        super().closeEvent(e)
