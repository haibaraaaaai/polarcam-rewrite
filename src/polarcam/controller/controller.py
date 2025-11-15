from __future__ import annotations
"""
Thin application-facing controller.

- Owns the camera facade (IDSCamera) but stays UI-agnostic.
- Provides small convenience helpers (auto-desaturate, quick varmap capture).
- Avoids terminal spam: uses `logging` (configured in `polarcam.__init__`).
"""

from pathlib import Path
from typing import Any, Callable, Optional, Dict

import logging
import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication

from polarcam.backend.ids_backend import IDSCamera

log = logging.getLogger(__name__)


class Controller(QObject):
    """High-level control surface used by the Qt widgets."""

    def __init__(self, cam: IDSCamera | None = None) -> None:
        super().__init__()
        self.cam = cam or IDSCamera()
        log.debug("Controller initialized with %s", type(self.cam).__name__)

    # ---------- lifecycle -------------------------------------------------
    def open(self) -> None:
        """Open the underlying camera device."""
        log.info("open()")
        self.cam.open()

    def start(self) -> None:
        """Start acquisition (idempotent on the facade)."""
        log.info("start()")
        self.cam.start()

    def stop(self) -> None:
        """Stop acquisition (safe to call when already stopped)."""
        log.info("stop()")
        self.cam.stop()

    def close(self) -> None:
        """Close the camera and tear down resources."""
        log.info("close()")
        self.cam.close()

    # ---------- controls --------------------------------------------------
    def set_roi(self, w: float, h: float, x: float, y: float) -> None:
        """Apply ROI (Width, Height, OffsetX, OffsetY) in sensor pixels."""
        log.debug("set_roi(w=%s, h=%s, x=%s, y=%s)", w, h, x, y)
        self.cam.set_roi(w, h, x, y)

    def full_sensor(self) -> None:
        """Reset ROI to the full sensor extents."""
        log.debug("full_sensor()")
        self.cam.full_sensor()

    def set_timing(self, fps: float | None, exp_ms: float | None) -> None:
        """Set frame rate (fps) and/or exposure (milliseconds)."""
        log.debug("set_timing(fps=%s, exp_ms=%s)", fps, exp_ms)
        self.cam.set_timing(fps, exp_ms)

    def set_gains(self, analog: float | None, digital: float | None) -> None:
        """Set analog and/or digital gains (None -> leave unchanged)."""
        log.debug("set_gains(analog=%s, digital=%s)", analog, digital)
        self.cam.set_gains(analog, digital)

    def refresh_gains(self) -> None:
        """Query current gain values and ranges."""
        log.debug("refresh_gains()")
        self.cam.refresh_gains()

    def refresh_timing(self) -> None:
        """Query current timing (fps/exposure)."""
        log.debug("refresh_timing()")
        if hasattr(self.cam, "refresh_timing"):
            self.cam.refresh_timing()

    def refresh_roi(self) -> None:
        """Query current ROI; only if the facade exposes it."""
        log.debug("refresh_roi()")
        if hasattr(self.cam, "refresh_roi"):
            self.cam.refresh_roi()

    # ---------- utilities -------------------------------------------------
    def desaturate(self, target_frac: float = 0.85, max_iters: int = 5) -> None:
        """
        Run the auto-desaturation loop on the camera until the brightest
        pixels settle below target_frac of full-scale (0..1).
        """
        log.info("desaturate(target=%.3f, iters=%d)", target_frac, max_iters)
        self.cam.auto_desaturate(target_frac, max_iters)

    def shutdown(self) -> None:
        """Best-effort orderly shutdown (used by the main window)."""
        log.info("shutdown()")
        try:
            self.stop()
        except Exception:
            log.exception("stop() during shutdown failed")
        try:
            self.close()
        except Exception:
            log.exception("close() during shutdown failed")

    # ---------- varmap capture -------------------------------------------
    def varmap_capture_and_compute(
        self,
        n_frames: int,
        mode: str = "intensity_range",
        use_memmap: Optional[bool] = None,   # legacy alias
        memmap: Optional[bool] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
        cancel_flag: Optional[Callable[[], bool]] = None,
        **_ignore,  # absorbs unexpected kwargs (e.g. dialog=...)
    ) -> Dict[str, Any]:
        """
        Capture a short burst and compute a simple variance-like map.

        Parameters
        ----------
        n_frames : int
            Number of frames to capture (>=1).
        mode : {"intensity_range","range","ptp", "std"}
            Computation: max-min (default) or per-pixel stddev.
        memmap : bool, optional
            If True, stack is persisted as a .npy memmap on disk.
        on_progress : callable(frac: float, msg: str), optional
            Called a few times during capture and once before compute.
        cancel_flag : callable() -> bool, optional
            If provided and returns True while capturing, the function aborts.

        Returns
        -------
        dict
            {
              "stack_path": str|None,   # path to saved memmap, if any
              "map_path":   str,        # path to saved map (.npy, uint16)
              "map16":      np.ndarray, # HxW uint16
              "map8":       np.ndarray, # HxW uint8 quicklook
            }
        """
        # normalize options
        if memmap is None and use_memmap is not None:
            memmap = bool(use_memmap)
        memmap = True if memmap is None else bool(memmap)

        n_frames = max(1, int(n_frames))
        on_progress = on_progress or (lambda _p, _m="": None)
        cancel_flag = cancel_flag or (lambda: False)

        # Start acquisition (facade is idempotent)
        try:
            self.start()
        except Exception:
            log.exception("start() before varmap capture failed (continuing)")

        # Collect frames
        collected: list[np.ndarray] = []
        done = False

        def _on_frame(arr_obj: object) -> None:
            nonlocal done
            if done:
                return
            a16 = np.asarray(arr_obj, dtype=np.uint16, copy=True)
            collected.append(a16)
            on_progress(len(collected) / n_frames, f"Captured {len(collected)}/{n_frames}")
            if len(collected) >= n_frames:
                done = True

        log.info("varmap: capturing %d frame(s)…", n_frames)
        try:
            self.cam.frame.connect(_on_frame)
            # Simple event loop while we wait; allow cancellation.
            while not done and not cancel_flag():
                QApplication.processEvents()
                # very short nap to avoid pegging CPU
                import time as _t
                _t.sleep(0.002)
        finally:
            try:
                self.cam.frame.disconnect(_on_frame)
            except Exception:
                pass

        if not collected:
            raise RuntimeError("No frames captured.")

        # Persist stack (optional) and compute map
        outdir = Path.cwd() / "varmap_runs"
        outdir.mkdir(parents=True, exist_ok=True)
        ts = __import__("time").strftime("%Y%m%d-%H%M%S")

        H, W = collected[0].shape
        stack_path: Optional[Path] = None

        if memmap:
            stack_path = outdir / f"stack_{ts}.npy"
            mm = np.lib.format.open_memmap(stack_path, mode="w+", dtype=np.uint16, shape=(n_frames, H, W))
            for i, f in enumerate(collected):
                mm[i, :, :] = f
            # flush and reopen read-only
            del mm
            mm = np.lib.format.open_memmap(stack_path, mode="r", dtype=np.uint16, shape=(n_frames, H, W))
            stack = mm  # type: ignore[assignment]
        else:
            stack = np.stack(collected, axis=0).astype(np.uint16, copy=False)

        on_progress(1.0, "Computing map…")

        mode = (mode or "intensity_range").lower()
        if mode in ("intensity_range", "range", "ptp"):
            m16 = stack.max(axis=0) - stack.min(axis=0)
        else:
            # stddev (float32 -> scaled into 12-bit range)
            m16f = stack.astype(np.float32).std(axis=0)
            scale = 4095.0 / max(1.0, float(m16f.max()))
            m16 = np.clip(np.rint(m16f * scale), 0, 4095).astype(np.uint16)

        map_path = outdir / f"varmap_{ts}.npy"
        np.save(map_path, m16)
        m8 = (np.clip(m16.astype(np.float32), 0, 4095) * (255.0 / 4095.0)).astype(np.uint8)

        log.info("varmap: saved map → %s%s",
                 map_path, f" (stack: {stack_path})" if stack_path else "")

        return {
            "stack_path": str(stack_path) if stack_path is not None else None,
            "map_path": str(map_path),
            "map16": m16,
            "map8": m8,
        }
