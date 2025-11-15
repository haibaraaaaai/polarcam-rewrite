from __future__ import annotations
"""
Experimental: capture N frames → save stack → compute a quick varmap.

Notes
-----
- Keeps the same Qt signal API so existing dialogs keep working.
- Quiet by default: uses `logging` (configured in polarcam.__init__).
- Uses open_memmap for efficient on-disk stacks when memmap=True.
"""

import json
import os
import time
import uuid
import logging
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt

from polarcam.analysis.varmap import compute_varmap

log = logging.getLogger(__name__)


# ------------------------------- compute worker -------------------------------

class _ComputeWorker(QObject):
    finished = Signal(object, str)  # (varmap ndarray, varmap_path)
    error = Signal(str)

    def __init__(self, stack_path: str, varmap_path: str, mode: str) -> None:
        super().__init__()
        self._stack_path = stack_path
        self._varmap_path = varmap_path
        self._mode = mode

    @Slot()
    def run(self) -> None:
        """Run in background thread: load stack (.npy), compute varmap, save."""
        try:
            log.info("VarMap worker: loading stack %s", self._stack_path)
            stack = np.load(self._stack_path, mmap_mode="r")
            varmap = compute_varmap(stack, mode=self._mode)
            np.save(self._varmap_path, varmap)
            log.info("VarMap worker: saved %s", self._varmap_path)
            self.finished.emit(varmap, self._varmap_path)
        except Exception as e:
            log.exception("VarMap compute failed")
            self.error.emit(f"VarMap compute failed: {e}")


# ---------------------------------- capture -----------------------------------

class VarMapCapture(QObject):
    """
    Subscribe to cam.frame, capture N frames into a stack, save to disk,
    then compute a per-pixel variance/activity map in a background thread.
    """
    started  = Signal(int)           # N frames target
    progress = Signal(int, int)      # k, N
    saved    = Signal(str, str)      # stack_path, varmap_path
    ready    = Signal(object)        # varmap ndarray
    error    = Signal(str)

    def __init__(self, cam: QObject, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._cam = cam
        self._want = 0
        self._have = 0
        self._stack: Optional[np.ndarray] = None        # (N,H,W) in RAM or memmap
        self._shape: Optional[Tuple[int, int]] = None   # (H, W)
        self._running = False
        self._out_dir = os.path.abspath("varmap_runs")
        os.makedirs(self._out_dir, exist_ok=True)

        # compute thread references
        self._thread: Optional[QThread] = None
        self._worker: Optional[_ComputeWorker] = None

        # runtime params
        self._mode = "intensity_range"
        self._memmap = False
        self._stack_path = ""
        self._varmap_path = ""

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
            self._cam.frame.connect(self._on_frame, Qt.QueuedConnection)  # type: ignore[attr-defined]
        except Exception:
            self.error.emit("VarMapCapture: camera does not expose 'frame' signal.")
            return

        self._running = True
        log.info("VarMapCapture: started (N=%d, mode=%s, memmap=%s)", self._want, self._mode, self._memmap)
        self.started.emit(self._want)

    @Slot()
    def cancel(self) -> None:
        """Abort capture (no compute). Safe to call if idle."""
        if not self._running:
            return
        try:
            self._cam.frame.disconnect(self._on_frame)  # type: ignore[attr-defined]
        except Exception:
            pass
        self._running = False
        self._stack = None
        self._shape = None
        self._want = 0
        self._have = 0
        log.info("VarMapCapture: canceled")

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

            # First frame → allocate stack and metadata
            if self._shape is None:
                self._shape = (H, W)

                ts = time.strftime("%Y%m%d-%H%M%S")
                uid = uuid.uuid4().hex[:6]
                base = f"{ts}_{uid}"

                self._stack_path = os.path.join(self._out_dir, f"stack_{base}.npy")
                self._varmap_path = os.path.join(self._out_dir, f"varmap_{base}.npy")
                meta_path = os.path.join(self._out_dir, f"meta_{base}.json")

                if self._memmap:
                    # Allocate .npy memmap directly (no temporary zeros file)
                    self._stack = np.lib.format.open_memmap(
                        self._stack_path, mode="w+", dtype=np.uint16, shape=(self._want, H, W)
                    )
                else:
                    self._stack = np.empty((self._want, H, W), dtype=np.uint16)

                # metadata (best-effort)
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

            # Guard: ROI/shape changes mid-capture aren’t supported
            if self._shape != (H, W):
                self.error.emit("VarMapCapture: frame size changed during capture.")
                self.cancel()
                return

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
                    self._cam.frame.disconnect(self._on_frame)  # type: ignore[attr-defined]
                except Exception:
                    pass
                self._running = False

                # If stack lives in RAM, persist to disk now
                if not self._memmap and self._stack is not None:
                    np.save(self._stack_path, self._stack)

                log.info("VarMapCapture: saved stack → %s", self._stack_path)
                self._start_compute(self._stack_path, self._varmap_path, self._mode)

        except Exception as e:
            log.exception("VarMapCapture frame intake failed")
            self.error.emit(f"VarMapCapture frame intake failed: {e}")

    # ---------- background compute ----------

    def _start_compute(self, stack_path: str, varmap_path: str, mode: str) -> None:
        """Spin up a background thread to compute the varmap."""
        self._thread = QThread(self)
        self._worker = _ComputeWorker(stack_path, varmap_path, mode)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_compute_finished, Qt.QueuedConnection)
        self._worker.error.connect(self.error, Qt.QueuedConnection)

        # cleanup thread (always)
        def _cleanup() -> None:
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
        """Emit file paths and data once ready."""
        try:
            self.saved.emit(self._stack_path, varmap_path)
        except Exception:
            pass
        self.ready.emit(varmap)
