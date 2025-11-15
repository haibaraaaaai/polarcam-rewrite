# src/polarcam/app/spot_cycler.py
from __future__ import annotations
import json, math, os, time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, Qt
from PySide6.QtWidgets import QApplication

# Polar mosaic:
# (0,0)=90°, (0,1)=45°, (1,0)=135°, (1,1)=0°

SpotTuple = Tuple[float, float, float, int, int]  # (cx, cy, r, area, inten)


@dataclass
class CycleConfig:
    out_dir: str
    base_name: str = "cycle"
    dwell_sec: float = 1.0         # time per spot before hopping
    max_duration_sec: float = 3600 # safety cap (1 h)
    chunk_len: int = 20000         # flush every N samples per spot
    maximize_camera_fps: bool = True


class MultiSpotCycler(QObject):
    """
    Cycles a small HW-ROI across given spots; for each dwell, computes 4×pol mean
    signals from the software crop used for display and saves compressed shards (.npz).
    """
    started  = Signal()
    stopped  = Signal()
    error    = Signal(str)
    progress = Signal(str)  # human-readable progress line

    def __init__(self, cam, spots: List[SpotTuple], cfg: CycleConfig) -> None:
        super().__init__()
        self.cam = cam
        self.spots = list(spots or [])
        self.cfg = cfg
        self._want_stop = False

        # Snapshots updated via signals
        self._applied_roi = (0, 0, 0, 0)  # (x,y,w,h)
        self._fps_now = 20.0

        # per-spot accumulators
        self._acc_t: list[float] = []
        self._acc0: list[float] = []
        self._acc45: list[float] = []
        self._acc90: list[float] = []
        self._acc135: list[float] = []
        self._chunk_idx: dict[int, int] = {}  # spot_idx -> next chunk number

        # saved pre-run state to restore
        self._saved_roi: Optional[tuple[int,int,int,int]] = None
        self._saved_fps: Optional[float] = None

    # ---------- public control ----------
    def start(self) -> None:
        try:
            if not self.spots:
                raise RuntimeError("No spots to cycle.")
            self._setup_dirs()
            self._connect_signals()
            self._save_pre_state()

            if self.cfg.maximize_camera_fps:
                try:
                    self.cam.set_timing(float("inf"), None)  # go to camera fps_max
                except Exception:
                    pass

            self._want_stop = False
            self.started.emit()
            self._run_loop()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                self._restore_pre_state()
            except Exception:
                pass
            self._disconnect_signals()
            self.stopped.emit()

    def stop(self) -> None:
        self._want_stop = True

    # ---------- internals ----------
    def _setup_dirs(self) -> None:
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        for i, _ in enumerate(self.spots, start=1):
            os.makedirs(self._spot_dir(i), exist_ok=True)

    def _spot_dir(self, i: int) -> str:
        return os.path.join(self.cfg.out_dir, f"{self.cfg.base_name}_spot{i:02d}")

    def _save_pre_state(self) -> None:
        # Ask backend to refresh so snapshots are up-to-date (best effort).
        self._saved_roi = (0, 0, 0, 0)
        self._saved_fps = None
        try:
            if hasattr(self.cam, "refresh_roi"):
                self.cam.refresh_roi()
            if hasattr(self.cam, "refresh_timing"):
                self.cam.refresh_timing()
        except Exception:
            pass

    def _restore_pre_state(self) -> None:
        try:
            x, y, w, h = self._saved_roi or (0, 0, 0, 0)
            if w and h:
                self.cam.set_roi(w, h, x, y)
        except Exception:
            pass
        try:
            if self._saved_fps is not None:
                self.cam.set_timing(self._saved_fps, None)
        except Exception:
            pass

    def _connect_signals(self) -> None:
        try: self.cam.roi.connect(self._on_roi_update, Qt.QueuedConnection)
        except Exception: pass
        try: self.cam.timing.connect(self._on_timing_update, Qt.QueuedConnection)
        except Exception: pass
        try: self.cam.frame.connect(self._on_frame, Qt.QueuedConnection)
        except Exception: pass

    def _disconnect_signals(self) -> None:
        try: self.cam.roi.disconnect(self._on_roi_update)
        except Exception: pass
        try: self.cam.timing.disconnect(self._on_timing_update)
        except Exception: pass
        try: self.cam.frame.disconnect(self._on_frame)
        except Exception: pass

    @Slot(dict)
    def _on_roi_update(self, d: dict) -> None:
        try:
            w = int(round(float(d.get("Width", 0))))
            h = int(round(float(d.get("Height", 0))))
            x = int(round(float(d.get("OffsetX", 0))))
            y = int(round(float(d.get("OffsetY", 0))))
            if w and h:
                self._applied_roi = (x, y, w, h)
                if (self._saved_roi == (0, 0, 0, 0)) and (self._saved_roi is not None):
                    self._saved_roi = (x, y, w, h)
        except Exception:
            pass

    @Slot(dict)
    def _on_timing_update(self, d: dict) -> None:
        try:
            rf = d.get("resulting_fps") or d.get("fps")
            if rf is not None:
                self._fps_now = float(rf)
                if self._saved_fps is None:
                    self._saved_fps = self._fps_now
        except Exception:
            pass

    # ----- main loop -----
    def _run_loop(self) -> None:
        t0 = time.perf_counter()
        spot_idx = 0
        dwell_idx = 0
        self._chunk_idx = {i: 0 for i in range(len(self.spots))}

        try:
            if hasattr(self.cam, "start"):
                self.cam.start()
        except Exception:
            pass

        while not self._want_stop:
            if (time.perf_counter() - t0) >= max(1.0, float(self.cfg.max_duration_sec)):
                self.progress.emit("Max duration reached; stopping.")
                break

            i = spot_idx % len(self.spots)
            cx, cy, r, area, inten = self.spots[i]

            # Apply a small HW ROI around the spot (camera may snap to min steps).
            req_w, req_h, req_x, req_y = self._hw_roi_request_for_spot(cx, cy, r)
            try:
                self.cam.set_roi(req_w, req_h, req_x, req_y)
            except Exception as e:
                self.error.emit(f"Failed set_roi for spot {i+1}: {e}")
                return

            # Let ROI settle; a few event loop pumps ensure we see ROI snapshot.
            t_settle = time.perf_counter() + 0.030  # ~30 ms
            while time.perf_counter() < t_settle and not self._want_stop:
                QApplication.processEvents()
                time.sleep(0)

            # Dwell collection
            dwell_t0 = time.perf_counter()
            applied = self._applied_roi
            crop_abs = None  # (x, y, w, h) absolute top-left for software crop
            self._reset_acc()
            self.progress.emit(
                f"Spot {i+1}/{len(self.spots)} dwell {dwell_idx+1}: ROI={applied} r≈{r:.2f}"
            )

            while (time.perf_counter() - dwell_t0) < max(0.05, float(self.cfg.dwell_sec)) and not self._want_stop:
                QApplication.processEvents()
                time.sleep(0)  # yield a tick
                if crop_abs is None and hasattr(self, "_last_crop_abs"):
                    crop_abs = getattr(self, "_last_crop_abs")

                if len(self._acc_t) >= self.cfg.chunk_len:
                    self._flush_chunk(i, dwell_idx, (cx, cy, r, area, inten), applied, crop_abs)

            if len(self._acc_t):
                self._flush_chunk(i, dwell_idx, (cx, cy, r, area, inten), applied, crop_abs)

            spot_idx += 1
            if spot_idx % len(self.spots) == 0:
                dwell_idx += 1

    # ----- frame path -----
    @Slot(object)
    def _on_frame(self, arr_obj: object) -> None:
        if self._want_stop:
            return
        a = np.asarray(arr_obj)
        if a.ndim != 2 or a.size == 0:
            return

        try:
            ax, ay, aw, ah = self._applied_roi
            if aw <= 0 or ah <= 0:
                H, W = a.shape
                ax = ay = 0; aw = W; ah = H

            # choose closest configured spot to current view center
            vcx = ax + aw * 0.5
            vcy = ay + ah * 0.5
            idx = int(np.argmin([(s[0] - vcx) ** 2 + (s[1] - vcy) ** 2 for s in self.spots]))
            cx, cy, r, _area, _inten = self.spots[idx]

            # Relative within the HW ROI
            rcx = float(cx) - float(ax)
            rcy = float(cy) - float(ay)

            # display crop
            r_eff = max(4.0, float(r))
            want = int(max(10, min(aw, ah, math.ceil(2.0 * r_eff + 6.0))))
            side = want if (want % 2 == 0) else want + 1

            ix = max(0, min(aw - side, int(round(rcx)) - side // 2))
            iy = max(0, min(ah - side, int(round(rcy)) - side // 2))

            H, W = a.shape
            ix = max(0, min(W - 2, ix)); iy = max(0, min(H - 2, iy))
            jx = max(ix + 1, min(W, ix + side))
            jy = max(iy + 1, min(H, iy + side))

            crop = a[iy:jy, ix:jx]

            # track displayed crop absolute location
            crop_abs = (ax + ix, ay + iy, int(jx - ix), int(jy - iy))
            self._last_crop_abs = crop_abs

            # 4×pol means with correct parity
            row0 = (crop_abs[1]) & 1
            col0 = (crop_abs[0]) & 1

            s90  = crop[row0::2,          col0::2]
            s45  = crop[row0::2,          (col0 ^ 1)::2]
            s135 = crop[(row0 ^ 1)::2,    col0::2]
            s0   = crop[(row0 ^ 1)::2,    (col0 ^ 1)::2]

            def mean_safe(x: np.ndarray) -> float:
                return float(x.mean()) if x.size else 0.0

            t = time.perf_counter()
            self._acc_t.append(t)
            self._acc0.append(mean_safe(s0))
            self._acc45.append(mean_safe(s45))
            self._acc90.append(mean_safe(s90))
            self._acc135.append(mean_safe(s135))
        except Exception:
            # swallow; keep cycling resilient
            return

    # ----- helpers -----
    def _reset_acc(self) -> None:
        self._acc_t.clear()
        self._acc0.clear(); self._acc45.clear(); self._acc90.clear(); self._acc135.clear()

    def _flush_chunk(self, spot_i: int, dwell_i: int, spot: SpotTuple,
                     applied_roi: Tuple[int, int, int, int] | None,
                     crop_abs: Tuple[int, int, int, int] | None) -> None:
        """Write a chunk NPZ for the current spot."""
        if not self._acc_t:
            return
        t0 = float(self._acc_t[0])
        t_rel = np.asarray(self._acc_t, dtype=np.float64) - t0
        c0    = np.asarray(self._acc0, dtype=np.float64)
        c45   = np.asarray(self._acc45, dtype=np.float64)
        c90   = np.asarray(self._acc90, dtype=np.float64)
        c135  = np.asarray(self._acc135, dtype=np.float64)

        meta = {
            "spot": {"cx": spot[0], "cy": spot[1], "r": spot[2], "area": int(spot[3]), "inten": int(spot[4])},
            "applied_roi": {"x": int(applied_roi[0]), "y": int(applied_roi[1]),
                            "w": int(applied_roi[2]), "h": int(applied_roi[3])} if applied_roi else None,
            "crop_abs": {"x": int(crop_abs[0]), "y": int(crop_abs[1]),
                         "w": int(crop_abs[2]), "h": int(crop_abs[3])} if crop_abs else None,
            "t0_perf_counter": t0,
            "notes": "signals are per-frame means over the displayed/software crop only",
        }
        jmeta = json.dumps(meta)

        sd = self._spot_dir(spot_i + 1)
        idx = self._chunk_idx.get(spot_i, 0)
        self._chunk_idx[spot_i] = idx + 1
        fname = os.path.join(sd, f"{self.cfg.base_name}_s{spot_i+1:02d}_d{dwell_i+1:04d}_p{idx:04d}.npz")

        np.savez_compressed(
            fname,
            t=t_rel, c0=c0, c45=c45, c90=c90, c135=c135,
            meta=np.frombuffer(jmeta.encode("utf-8"), dtype=np.uint8),
        )
        self.progress.emit(f"Saved {fname}  ({len(t_rel)} samples)")
        self._reset_acc()

    def _hw_roi_request_for_spot(self, cx: float, cy: float, r: float) -> Tuple[int, int, int, int]:
        """Pick a *small* HW ROI; the camera will snap to supported steps."""
        side = int(max(32, math.ceil(2.2 * max(8.0, float(r)))))
        if side % 2: side += 1
        req_w = max(64, side)
        req_h = max(32, side // 2)
        x = max(0, int(round(cx - req_w / 2.0)))
        y = max(0, int(round(cy - req_h / 2.0)))
        return (req_w, req_h, x, y)
