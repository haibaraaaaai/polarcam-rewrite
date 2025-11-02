from __future__ import annotations
from typing import Optional
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QTimer, Qt

class FramePlugin(QObject):
    """Base for frame-processing plugins. Runs in its own thread with a bounded 'latest frame' mailbox."""
    status = Signal(str)

    def __init__(self, hz: float = 8.0, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._latest: Optional[np.ndarray] = None
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.CoarseTimer)
        self._timer.setInterval(max(1, int(1000.0 / max(0.5, hz))))
        self._timer.timeout.connect(self._tick)

    @Slot(object)
    def enqueue(self, frame_obj: object) -> None:
        arr = np.asarray(frame_obj)
        if arr.ndim != 2:
            return
        # keep only the most recent frame
        self._latest = arr

    @Slot()
    def start(self) -> None:
        self._timer.start()

    @Slot()
    def stop(self) -> None:
        self._timer.stop()
        self._latest = None

    # override in subclass
    def process(self, frame: np.ndarray) -> None:  # pragma: no cover
        pass

    def _tick(self) -> None:
        if self._latest is None:
            return
        frame = self._latest
        self._latest = None
        try:
            self.process(frame)
        except Exception as e:
            self.status.emit(f"Plugin error: {e}")
