# src/polarcam/app/spot_cycler.py
from __future__ import annotations
import json, math, os, time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, Qt

# Polar mosaic: (0,0)=90°, (0,1)=45°, (1,0)=135°, (1,1)=0°
SpotTuple = Tuple[float, float, float, int, int]  # (cx, cy, r, area, inten)

# ---- Hardware limits (your camera) ----
SENSOR_W, SENSOR_H = 2464, 2056
STEP_W, STEP_H     = 4, 2
MIN_W,  MIN_H      = 256, 2
MAX_W,  MAX_H      = SENSOR_W, SENSOR_H
OFFX_MIN, OFFX_MAX = 0, 2208                 # 2464 - 256
OFFY_MIN, OFFY_MAX = 0, 2054
OFFX_STEP, OFFY_STEP = 4, 2

def _snap_down(v: int, step: int) -> int:
    s = int(step) if step > 0 else 1
    return int((int(v) // s) * s)

@dataclass
class CycleConfig:
    out_dir: str
    base_name: str = "cycle"
    dwell_sec: float = 1.0
    max_duration_sec: float = 3600
    chunk_len: int = 20000
    maximize_camera_fps: bool = True
    reassert_timing_each_hop: bool = True

class MultiSpotCycler(QObject):
    """
    Cycle a tiny HW ROI across spots and save 4×pol mean signals of a software crop.
    All control calls are queued to the Controller/camera thread; no cross-thread timers.
    """
    # lifecycle / UI
    started       = Signal()
    stopped       = Signal()
    error         = Signal(str)
    progress      = Signal(str)
    advise_ui_cap = Signal(float)   # 20.0 while running, 0.0 when done

    # bridge commands → dev/dev.cam (in their thread)
    _req_set_roi        = Signal(int, int, int, int)  # w,h,x,y
    _req_set_timing     = Signal(object, object)      # fps, exposure_ms
    _req_start_acq      = Signal()
    _req_refresh_roi    = Signal()
    _req_refresh_timing = Signal()

    def __init__(self, dev, spots: List[SpotTuple], cfg: CycleConfig) -> None:
        super().__init__()
        self.dev = dev              # Controller or camera-like
        self.cam = getattr(dev, "cam", None) or dev   # whichever actually emits signals
        self.spots = list(spots or [])
        self.cfg = cfg
        self._want_stop = False

        # snapshots updated via signals
        self._applied_roi = (0, 0, 0, 0)  # x,y,w,h
        self._fps_now = 20.0

        # accumulators
        self._acc_t: list[float] = []
        self._acc0: list[float] = []
        self._acc45: list[float] = []
        self._acc90: list[float] = []
        self._acc135: list[float] = []
        self._chunk_idx: dict[int, int] = {}

        # saved state
        self._saved_roi: Optional[tuple[int,int,int,int]] = None
        self._saved_fps: Optional[float] = None

        # --- wire command bridge to safest target (Controller preferred) ---
        target = self.dev if hasattr(self.dev, "set_roi") else (self.cam if hasattr(self.cam, "set_roi") else None)
        if target is None:
            raise RuntimeError("Cycler cannot find a target with set_roi/set_timing.")
        self._req_set_roi.connect(target.set_roi, Qt.QueuedConnection)
        if hasattr(target, "set_timing"):  self._req_set_timing.connect(target.set_timing, Qt.QueuedConnection)
        if hasattr(target, "start"):       self._req_start_acq.connect(target.start, Qt.QueuedConnection)
        if hasattr(target, "refresh_roi"): self._req_refresh_roi.connect(target.refresh_roi, Qt.QueuedConnection)
        if hasattr(target, "refresh_timing"): self._req_refresh_timing.connect(target.refresh_timing, Qt.QueuedConnection)

    # ---------- public control ----------
    def start(self) -> None:
        try:
            if not self.spots:
                raise RuntimeError("No spots to cycle.")

            self._setup_dirs()
            self._connect_signals()
            self._save_pre_state()

            if self.cfg.maximize_camera_fps:
                try: self._req_set_timing.emit(float("inf"), None)
                except Exception: pass

            self._want_stop = False
            self.advise_ui_cap.emit(20.0)  # cap preview while cycling
            self.started.emit()
            self._run_loop()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try: self._restore_pre_state()
            except Exception: pass
            self._disconnect_signals()
            self.stopped.emit()
            self.advise_ui_cap.emit(0.0)   # uncap preview

    def stop(self) -> None:
        self._want_stop = True

    # ---------- internals ----------
    def _setup_dirs(self) -> None:
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        for i in range(1, len(self.spots)+1):
            os.makedirs(self._spot_dir(i), exist_ok=True)

    def _spot_dir(self, i: int) -> str:
        return os.path.join(self.cfg.out_dir, f"{self.cfg.base_name}_spot{i:02d}")

    def _save_pre_state(self) -> None:
        self._saved_roi = (0, 0, 0, 0)
        self._saved_fps = None
        try: self._req_refresh_roi.emit()
        except Exception: pass
        try: self._req_refresh_timing.emit()
        except Exception: pass

    def _restore_pre_state(self) -> None:
        try:
            x, y, w, h = self._saved_roi or (0, 0, 0, 0)
            if w and h: self._req_set_roi.emit(w, h, x, y)
        except Exception: pass
        try:
            if self._saved_fps is not None:
                self._req_set_timing.emit(self._saved_fps, None)
        except Exception: pass

    def _connect_signals(self) -> None:
        # Listen from whichever actually emits (cam or controller). Use Direct so
        # the slots run in the emitter's thread; they only touch plain Python data.
        src = self.cam
        try: src.roi.connect(self._on_roi_update, Qt.DirectConnection)
        except Exception: pass
        try: src.timing.connect(self._on_timing_update, Qt.DirectConnection)
        except Exception: pass
        try: src.frame.connect(self._on_frame, Qt.DirectConnection)
        except Exception: pass

    def _disconnect_signals(self) -> None:
        src = self.cam
        for sig, slot in ((getattr(src, "roi", None), self._on_roi_update),
                          (getattr(src, "timing", None), self._on_timing_update),
                          (getattr(src, "frame", None), self._on_frame)):
            try:
                if sig: sig.disconnect(slot)
            except Exception:
                pass

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

        try: self._req_start_acq.emit()
        except Exception: pass

        while not self._want_stop:
            if (time.perf_counter() - t0) >= max(1.0, float(self.cfg.max_duration_sec)):
                self.progress.emit("Max duration reached; stopping.")
                break

            i = spot_idx % len(self.spots)
            cx, cy, r, area, inten = self.spots[i]

            # Apply tiny HW ROI (queued to controller/camera thread)
            w, h, x, y = self._hw_roi_request_for_spot(cx, cy, r)
            try: self._req_set_roi.emit(w, h, x, y)
            except Exception as e:
                self.error.emit(f"Failed set_roi for spot {i+1}: {e}")
                return

            # Let ROI settle (~3 frames); DO NOT pump GUI here
            settle = max(0.03, 3.0 / max(1.0, float(self._fps_now)))
            time.sleep(settle)

            if self.cfg.maximize_camera_fps and self.cfg.reassert_timing_each_hop:
                try:
                    # keep exposure unchanged (None) but push FPS to hardware max
                    self.cam.set_timing(float("inf"), None)
                except Exception:
                    pass
                # brief settle again (1–2 frames) so the camera applies the new budget
                time.sleep(max(0.01, 2.0 / max(1.0, float(self._fps_now))))

            # Dwell
            dwell_t0 = time.perf_counter()
            applied = self._applied_roi
            crop_abs = None
            self._reset_acc()
            self.progress.emit(f"Spot {i+1}/{len(self.spots)} dwell {dwell_idx+1}: ROI={applied} r≈{r:.2f}")

            while (time.perf_counter() - dwell_t0) < max(0.05, float(self.cfg.dwell_sec)) and not self._want_stop:
                # _on_frame runs in emitter thread; we just yield
                time.sleep(0.001)
                if crop_abs is None:
                    crop_abs = getattr(self, "_last_crop_abs", None)
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
        if self._want_stop: return
        a = np.asarray(arr_obj)
        if a.ndim != 2 or a.size == 0: return
        try:
            ax, ay, aw, ah = self._applied_roi
            if aw <= 0 or ah <= 0:
                H, W = a.shape
                ax = ay = 0; aw = W; ah = H

            vcx = ax + aw * 0.5
            vcy = ay + ah * 0.5
            idx = int(np.argmin([(s[0]-vcx)**2 + (s[1]-vcy)**2 for s in self.spots]))
            cx, cy, r, _area, _inten = self.spots[idx]

            rcx = float(cx) - float(ax)
            rcy = float(cy) - float(ay)

            r_eff = max(4.0, float(r))
            want  = int(max(10, min(aw, ah, math.ceil(2.0 * r_eff + 6.0))))
            side  = want if (want % 2 == 0) else want + 1

            ix = max(0, min(aw - side, int(round(rcx)) - side // 2))
            iy = max(0, min(ah - side, int(round(rcy)) - side // 2))

            H, W = a.shape
            ix = max(0, min(W - 2, ix)); iy = max(0, min(H - 2, iy))
            jx = max(ix + 1, min(W, ix + side)); jy = max(iy + 1, min(H, iy + side))

            crop = a[iy:jy, ix:jx]

            crop_abs = (ax + ix, ay + iy, int(jx - ix), int(jy - iy))
            self._last_crop_abs = crop_abs

            row0 = (crop_abs[1]) & 1
            col0 = (crop_abs[0]) & 1
            s90  = crop[row0::2,          col0::2]
            s45  = crop[row0::2,          (col0 ^ 1)::2]
            s135 = crop[(row0 ^ 1)::2,    col0::2]
            s0   = crop[(row0 ^ 1)::2,    (col0 ^ 1)::2]

            def m(x: np.ndarray) -> float: return float(x.mean()) if x.size else 0.0

            t = time.perf_counter()
            self._acc_t.append(t)
            self._acc0.append(m(s0)); self._acc45.append(m(s45))
            self._acc90.append(m(s90)); self._acc135.append(m(s135))
        except Exception:
            return

    # ----- helpers -----
    def _reset_acc(self) -> None:
        self._acc_t.clear()
        self._acc0.clear(); self._acc45.clear(); self._acc90.clear(); self._acc135.clear()

    def _flush_chunk(self, spot_i: int, dwell_i: int, spot: SpotTuple,
                     applied_roi: Tuple[int, int, int, int] | None,
                     crop_abs: Tuple[int, int, int, int] | None) -> None:
        if not self._acc_t: return
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
        """
        Aggressive ROI:
          - Width fixed to 256 (step 4)
          - Height ≈ 2*r (step 2, min 2)
          - Offsets snapped (x step 4, y step 2) and clamped to sensor box.
        """
        h_want = max(MIN_H, int(math.ceil(2.0 * max(0.0, float(r)))))
        h = _snap_down(h_want + (STEP_H - 1), STEP_H)
        h = max(MIN_H, min(MAX_H, h))

        w = max(MIN_W, min(MAX_W, _snap_down(MIN_W + (STEP_W - 1), STEP_W)))

        x = int(round(cx - w / 2.0))
        y = int(round(cy - h / 2.0))
        x = _snap_down(max(OFFX_MIN, min(OFFX_MAX, x)), OFFX_STEP)
        y = _snap_down(max(OFFY_MIN, min(OFFY_MAX, y)), OFFY_STEP)

        if x + w > SENSOR_W:
            x = _snap_down(SENSOR_W - w, OFFX_STEP); x = max(OFFX_MIN, min(OFFX_MAX, x))
        if y + h > SENSOR_H:
            y = _snap_down(SENSOR_H - h, OFFY_STEP); y = max(OFFY_MIN, min(OFFY_MAX, y))

        return (w, h, x, y)
