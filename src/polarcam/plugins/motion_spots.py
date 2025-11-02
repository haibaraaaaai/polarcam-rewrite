from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from PySide6.QtCore import Signal, Slot
from .base import FramePlugin

# scikit-image equivalents for morphology/labeling
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.measure import label, regionprops

class MotionSpotDetectorPlugin(FramePlugin):
    """
    Variation-based detector:
      - Keep two frames Δ apart (by frame count)
      - diff = |f_t - f_{t-Δ}| >= diff_thresh  (motion)
      - AND f_t >= intensity_thresh            (bright like gold)
      - Light morphology; connected components; area filter.
    Emits list of (x, y, r, area) in image pixels.
    """
    detections = Signal(list)  # list[(x, y, r, area)]

    def __init__(self,
                 hz: float = 12.0,
                 delta_frames: int = 3,
                 diff_thresh: int = 200,        # Mono12 ADU (0..4095)
                 intensity_thresh: int = 1200,  # brightness gate for gold
                 min_area: int = 3,
                 max_area: int = 500) -> None:
        super().__init__(hz=hz)
        self.delta = int(max(1, delta_frames))
        self.diff_thresh = int(diff_thresh)
        self.intensity_thresh = int(intensity_thresh)
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self._ring: List[Optional[np.ndarray]] = [None] * (self.delta + 1)
        self._idx = 0

    @Slot()
    def process(self, frame: np.ndarray) -> None:
        if frame.ndim != 2:
            return

        # ring buffer push
        self._ring[self._idx] = frame
        prev = self._ring[(self._idx - self.delta) % len(self._ring)]
        self._idx = (self._idx + 1) % len(self._ring)
        if prev is None:
            self.detections.emit([])
            return

        # abs diff (use wider type to avoid wrap)
        diff = np.abs(frame.astype(np.int32) - prev.astype(np.int32)).astype(np.int32)
        # two-stage gate: motion AND bright
        mask = (diff >= self.diff_thresh) & (frame >= self.intensity_thresh)

        if not mask.any():
            self.detections.emit([])
            return

        # light morphology to compact blobs
        se = disk(1)
        mask = binary_closing(mask, se)
        mask = binary_opening(mask, se)

        # CC + size filter
        lbl = label(mask, connectivity=2)
        dets: List[Tuple[float, float, float, int]] = []
        for r in regionprops(lbl):
            area = int(r.area)
            if area < self.min_area or area > self.max_area:
                continue
            cy, cx = r.centroid  # (row, col)
            rad = float((area / np.pi) ** 0.5)
            dets.append((float(cx), float(cy), rad, area))

        self.detections.emit(dets)
