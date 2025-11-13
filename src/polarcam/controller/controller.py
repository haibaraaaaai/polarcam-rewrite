# src/polarcam/controller/controller.py
from __future__ import annotations
from PySide6.QtCore import QObject
from polarcam.backend.ids_backend import IDSCamera
import time
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
from typing import Optional

class Controller(QObject):
    def __init__(self, cam: IDSCamera | None = None) -> None:
        super().__init__()
        self.cam = cam or IDSCamera()

    # lifecycle
    def open(self) -> None: self.cam.open()
    def start(self) -> None: self.cam.start()
    def stop(self) -> None: self.cam.stop()
    def close(self) -> None: self.cam.close()

    # controls
    def set_roi(self, w: float, h: float, x: float, y: float) -> None:
        self.cam.set_roi(w, h, x, y)

    def full_sensor(self) -> None:
        self.cam.full_sensor()

    def set_timing(self, fps: float | None, exp_ms: float | None) -> None:
        self.cam.set_timing(fps, exp_ms)

    def set_gains(self, analog: float | None, digital: float | None) -> None:
        self.cam.set_gains(analog, digital)

    def refresh_gains(self) -> None:
        self.cam.refresh_gains()

    def desaturate(self, target_frac: float = 0.85, max_iters: int = 5) -> None:
        cam = getattr(self, "cam", None)
        if cam is None:
            return
        print(f"[Controller] desaturate target={target_frac} iters={max_iters}")
        cam.auto_desaturate(target_frac, max_iters)

    def shutdown(self) -> None:
        try: self.stop()
        except Exception: pass
        try: self.close()
        except Exception: pass

    def varmap_capture_and_compute(
        self,
        n_frames: int,
        mode: str = "intensity_range",
        use_memmap: Optional[bool] = None,
        memmap: Optional[bool] = None,
        on_progress=None,
        cancel_flag=None,
        **_ignore,   # <-- absorbs unexpected args like dialog=...
    ):
        """
        Minimal varmap capture+compute for the dialog.

        Returns:
        {
            "stack_path": str | None,
            "map_path": str,
            "map16": np.ndarray,   # HxW uint16
            "map8":  np.ndarray,   # HxW uint8 (quicklook)
        }
        """
        # normalize the flag
        if memmap is None and use_memmap is not None:
            memmap = bool(use_memmap)
        memmap = True if memmap is None else bool(memmap)

        n_frames = max(1, int(n_frames))
        if on_progress is None:
            on_progress = lambda p, msg="": None
        if cancel_flag is None:
            cancel_flag = lambda: False
        # Make sure we’re acquiring; start() is idempotent in your facade.
        try:
            if hasattr(self, "start"):
                self.start()
        except Exception:
            pass

        # One-time frame collector
        collected = []
        done = False

        def _on_frame(arr_obj):
            nonlocal done
            if done:
                return
            a16 = np.asarray(arr_obj, dtype=np.uint16, copy=True)
            collected.append(a16)
            on_progress(len(collected) / n_frames, f"Captured {len(collected)}/{n_frames}")
            if len(collected) >= n_frames:
                done = True

        # Connect, collect, pump events; allow cancel
        try:
            self.cam.frame.connect(_on_frame)
            while not done and not cancel_flag():
                QApplication.processEvents()
                time.sleep(0.002)
        finally:
            try:
                self.cam.frame.disconnect(_on_frame)
            except Exception:
                pass

        if not collected:
            raise RuntimeError("No frames captured.")

        # Stack (T,H,W) — optionally memmap to disk
        outdir = Path.cwd() / "varmap_runs"
        outdir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")

        H, W = collected[0].shape
        if memmap:
            stack_path = outdir / f"stack_{ts}.npy"
            stack = np.lib.format.open_memmap(stack_path, mode="w+", dtype=np.uint16, shape=(n_frames, H, W))
            for i, f in enumerate(collected):
                stack[i, :, :] = f
            # flush memmap; reopen read-only to be safe
            del stack
            stack = np.lib.format.open_memmap(stack_path, mode="r", dtype=np.uint16, shape=(n_frames, H, W))
        else:
            stack_path = None
            stack = np.stack(collected, axis=0).astype(np.uint16, copy=False)

        on_progress(1.0, "Computing map…")

        # Minimal modes (expand later)
        mode = (mode or "intensity_range").lower()
        if mode in ("intensity_range", "range", "ptp"):
            m16 = stack.max(axis=0) - stack.min(axis=0)
        else:
            # placeholder: stddev (uint16 scaled)
            m16f = stack.astype(np.float32).std(axis=0)
            # scale into 16-bit for saving
            scale = 4095.0 / max(1.0, float(m16f.max()))
            m16 = np.clip(np.rint(m16f * scale), 0, 4095).astype(np.uint16)

        # Save map16; also make an 8-bit quicklook
        map_path = outdir / f"varmap_{ts}.npy"
        np.save(map_path, m16)
        m8 = (np.clip(m16.astype(np.float32), 0, 4095) * (255.0 / 4095.0)).astype(np.uint8)

        return {
            "stack_path": str(stack_path) if stack_path is not None else None,
            "map_path": str(map_path),
            "map16": m16,
            "map8": m8,
        }