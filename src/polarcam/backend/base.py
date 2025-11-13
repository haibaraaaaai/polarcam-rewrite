from abc import ABC, abstractmethod
from PySide6.QtCore import QObject, Signal

class ICamera(QObject, ABC):
    frame  = Signal(object)
    opened = Signal(str)
    started = Signal()
    stopped = Signal()
    closed = Signal()
    error = Signal(str)
    roi = Signal(dict)
    timing = Signal(dict)

    @abstractmethod
    def open(self) -> None: ...
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def set_roi(self, w:int, h:int, x:int, y:int) -> None: ...
    @abstractmethod
    def full_sensor(self) -> None: ...
    @abstractmethod
    def set_timing(self, fps:float|None=None, exposure_ms:float|None=None) -> None: ...
    @abstractmethod
    def set_zoom_roi(self, roi_xywh:tuple[int,int,int,int]|None) -> None: ...
