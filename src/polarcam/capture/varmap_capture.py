# src/polarcam/capture/varmap_capture.py
from __future__ import annotations
import json, os, time, uuid
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt

from polarcam.analysis.varmap import compute_varmap

class _ComputeWorker(QObject):
    finished = Signal(object, str)     # varmap array, varmap_path
    error = Signal(str)

    def __init__(self, stack_path: str, varmap_path: str, mode: str) -> None:
        super().__init__()
        self._stack_path = stack_path
        self._varmap_path = varmap_path
        self._mode = mode

    @Slot()
    def run(self) -> None:
        try:
            stack = np.load(self._stack_path, mmap_mode="r")
            varmap = compute_varmap(stack, mode=self._mode)
            np.save(self._varmap_path, varmap)
            self.finished.emit(varmap, self._varmap_path)
        except Exception as e:
            self.error.emit(f"VarMap compute failed: {e}")


class VarMapCapture(QObject):
    """
    Subscribe to cam.frame, capture N frames into a stack, save to disk,
    then compute a per-pixel variance/activity map in a background thread.
    """
    started  = Signal(int)                 # N frames target
    progress = Signal(int, int)            # k, N
    saved    = Signal(str, str)            # stack_path, varmap_path
    ready    = Signal(object)              # varmap ndarray
    error    = Signal(str)

    def __init__(self, cam: QObject, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._cam = cam
        self._want = 0
        self._have = 0
        self._stack: Optional[np.ndarray] = None
        self._shape: Optional[Tuple[int, int]] = None  # (H, W)
        self._running = False
        self._out_dir = os.path.abspath("varmap_runs")
        os.makedirs(self._out_dir, exist_ok=True)

        # compute thread references
        self._thread: Optional[QThread] = None
        self._worker: Optional[_ComputeWorker] = None

    # ---------- lifecycle ----------
    @Slot(int, str, bool)
    def start(self, n_frames: int = 20, mode: str = "intensity_range", memmap: bool = False) -> None:
        """
        Start a capture. Assumes camera is streaming and emitting cam.frame(arr).
        """
        if self._running:
            self.error.emit("VarMapCapture already running.")
            return

        self._mode = mode
        self._memmap = bool(memmap)
        self._want = max(1, int(n_frames))
        self._have = 0
        self._shape = None
        self._stack = None
        self._stack_path = ""
        self._varmap_path = ""

        # connect to frames
        try:
            self._cam.frame.connect(self._on_frame, Qt.QueuedConnection)
        except Exception:
            self.error.emit("VarMapCapture: camera does not expose 'frame' signal.")
            return

        self._running = True
        self.started.emit(self._want)

    @Slot()
    def cancel(self) -> None:
        if not self._running:
            return
        try:
            self._cam.frame.disconnect(self._on_frame)
        except Exception:
            pass
        self._running = False
        self._stack = None
        self._shape = None
        self._want = 0
        self._have = 0

    # ---------- frame intake ----------
    @Slot(object)
    def _on_frame(self, arr_obj: object) -> None:
        if not self._running:
            return
        try:
            a = np.asarray(arr_obj)
            if a.ndim != 2:
                return

            H, W = int(a.shape[0]), int(a.shape[1])
            if self._shape is None:
                # Allocate on first frame
                self._shape = (H, W)
                ts = time.strftime("%Y%m%d-%H%M%S")
                uid = uuid.uuid4().hex[:6]
                base = f"{ts}_{uid}"
                self._stack_path = os.path.join(self._out_dir, f"stack_{base}.npy")
                self._varmap_path = os.path.join(self._out_dir, f"varmap_{base}.npy")
                meta_path = os.path.join(self._out_dir, f"meta_{base}.json")

                if self._memmap:
                    # Create a memmap-backed .npy by pre-saving zeros then reopening memmap.
                    tmp = np.zeros((self._want, H, W), dtype=np.uint16)
                    np.save(self._stack_path, tmp)
                    # Reopen as memmap for in-place writing
                    self._stack = np.load(self._stack_path, mmap_mode="r+")
                else:
                    self._stack = np.empty((self._want, H, W), dtype=np.uint16)

                # Save lightweight metadata for reproducibility
                meta = {
                    "n_frames": self._want,
                    "shape": [H, W],
                    "created": ts,
                    "mode": self._mode,
                }
                try:
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                except Exception:
                    pass

            if self._stack is None:
                return

            if self._have < self._want:
                # copy into stack; ensure uint16
                if a.dtype != np.uint16:
                    a = a.astype(np.uint16, copy=False)
                self._stack[self._have, :, :] = a
                self._have += 1
                self.progress.emit(self._have, self._want)

            if self._have >= self._want:
                # Done capturing; disconnect and compute
                try:
                    self._cam.frame.disconnect(self._on_frame)
                except Exception:
                    pass
                self._running = False

                # Ensure stack is saved on disk if it lives in RAM
                if not self._memmap and self._stack is not None:
                    np.save(self._stack_path, self._stack)

                # Kick off background compute
                self._start_compute(self._stack_path, self._varmap_path, self._mode)

        except Exception as e:
            self.error.emit(f"VarMapCapture frame intake failed: {e}")

    # ---------- background compute ----------
    def _start_compute(self, stack_path: str, varmap_path: str, mode: str) -> None:
        self._thread = QThread(self)
        self._worker = _ComputeWorker(stack_path, varmap_path, mode)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_compute_finished, Qt.QueuedConnection)
        self._worker.error.connect(self.error, Qt.QueuedConnection)

        # cleanup thread
        def _cleanup():
            try:
                self._thread.quit()
                self._thread.wait(1500)
            except Exception:
                pass
            self._thread = None
            self._worker = None

        self._worker.finished.connect(lambda *_: _cleanup(), Qt.QueuedConnection)
        self._worker.error.connect(lambda *_: _cleanup(), Qt.QueuedConnection)

        self._thread.start()

    @Slot(object, str)
    def _on_compute_finished(self, varmap: object, varmap_path: str) -> None:
        try:
            self.saved.emit(self._stack_path, varmap_path)
        except Exception:
            pass
        self.ready.emit(varmap)
