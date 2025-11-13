from __future__ import annotations

from PySide6.QtCore import QObject
from polarcam.backend.ids_backend import IDSCamera

class Controller(QObject):
    """Thin pass-through controller: keeps GUI logic out of the backend."""

    def __init__(self, cam: IDSCamera | None = None) -> None:
        super().__init__()
        self.cam = cam or IDSCamera()

    def open(self) -> None: self.cam.open()
    def start(self) -> None: self.cam.start()
    def stop(self) -> None: self.cam.stop()
    def close(self) -> None: self.cam.close()

    def set_roi(self, w: float, h: float, x: float, y: float) -> None:
        self.cam.set_roi(w, h, x, y)

    def full_sensor(self) -> None:
        self.cam.full_sensor()

    def set_timing(self, fps: float | None, exp_ms: float | None) -> None:
        self.cam.set_timing(fps, exp_ms)

    def set_gains(self, analog: float | None, digital: float | None) -> None:
        self.cam.set_gains(analog, digital)

    def shutdown(self) -> None:
        try: self.stop()
        except Exception: pass
        try: self.close()
        except Exception: pass
