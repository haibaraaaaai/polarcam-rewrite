from __future__ import annotations
import os, json, math, time, threading
from dataclasses import dataclass
from typing import Optional, Tuple, Deque
from collections import deque

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt

# Polar mosaic layout:
# (row%2, col%2): (0,0)=90°, (0,1)=45°, (1,0)=135°, (1,1)=0°

# --------------------------
# small utils
# --------------------------
def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _round_up(v: int, step: int) -> int:
    if step <= 1: return int(v)
    return int(((int(v) + step - 1)//step)*step)

def _snap_even(v: int) -> int:
    return int(v if (int(v) % 2 == 0) else int(v) + 1)

# --------------------------
# IDS-ish ROI constraints (edit if different)
# --------------------------
W_STEP = 16
H_STEP = 2
MIN_W  = 64
MIN_H  = 32

PAD_SW   = 2      # software padding around diameter
MIN_CROP = 10     # minimum crop side (even)

DEFAULT_CHUNK = 20000  # samples per shard

# --------------------------
# dataclasses
# --------------------------
@dataclass
class Spot:
    cx: float
    cy: float
    r: float
    area: int
    inten: int

@dataclass
class RecorderConfig:
    out_dir: str
    base_name: str = "spot"
    chunk_len: int = DEFAULT_CHUNK
    maximize_camera_fps: bool = True   # set_timing(inf, None)

# --------------------------
# shard writer (own thread)
# --------------------------
class _ShardWriter(QObject):
    finished = Signal()
    error = Signal(str)

    def __init__(self, out_dir: str, prefix: str):
        super().__init__()
        self._out_dir = out_dir
        self._prefix  = prefix
        self._q: Deque[np.ndarray] = deque()
        self._cond = threading.Condition()
        self._closing = False
        self._idx = 0

    @Slot(object)
    def enqueue(self, arr: np.ndarray) -> None:
        with self._cond:
            self._q.append(arr)
            self._cond.notify()

    @Slot()
    def close(self) -> None:
        with self._cond:
            self._closing = True
            self._cond.notify()

    @Slot()
    def run(self) -> None:
        try:
            os.makedirs(self._out_dir, exist_ok=True)
            while True:
                with self._cond:
                    while not self._q and not self._closing:
                        self._cond.wait()
                    if not self._q and self._closing:
                        break
                    chunk = self._q.popleft()
                fn = os.path.join(self._out_dir, f"{self._prefix}_chunk{self._idx:04d}.npy")
                # shape (N,5): [t, I0, I45, I90, I135] float32
                np.save(fn, chunk)
                self._idx += 1
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

# --------------------------
# headless recorder
# --------------------------
class SpotSignalRecorder(QObject):
    started = Signal()
    stopped = Signal()
    error = Signal(str)
    progress = Signal(int)   # total samples written

    def __init__(self, cam, ctrl, spot: Spot, cfg: RecorderConfig):
        """
        cam  : camera object (emits .frame(np.ndarray) and .roi(dict))
        ctrl : object with set_roi(w,h,x,y) and set_timing(fps, exposure_ms)
        spot : target spot
        cfg  : output + behavior
        """
        super().__init__()
        self.cam  = cam
        self.ctrl = ctrl
        self.spot = spot
        self.cfg  = cfg

        self._applied_roi: Tuple[int,int,int,int] = (0,0,0,0)  # x,y,w,h

        self._chunk_len = int(max(1000, cfg.chunk_len))
        self._buf = np.empty((self._chunk_len, 5), dtype=np.float32)  # t, I0, I45, I90, I135
        self._buf_i = 0
        self._total = 0

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._prefix = f"{cfg.base_name}_{ts}"
        self._writer = _ShardWriter(cfg.out_dir, self._prefix)
        self._writer_thr = QThread(self)
        self._writer.moveToThread(self._writer_thr)
        self._writer_thr.started.connect(self._writer.run)
        self._writer.finished.connect(self._writer_thr.quit)
        self._writer.error.connect(self.error)

        self.cam.roi.connect(self._on_roi, Qt.QueuedConnection)
        self.cam.frame.connect(self._on_frame, Qt.QueuedConnection)

        self._saved_roi = None  # best-effort restore
        self._saved_fps = None
        self._running = False
        self._t0 = 0.0

    # ---- lifecycle ----
    def start(self) -> None:
        try: os.makedirs(self.cfg.out_dir, exist_ok=True)
        except Exception: pass

        # Small HW ROI centered on spot
        cx, cy, r = self.spot.cx, self.spot.cy, max(4.0, float(self.spot.r))
        want_side = int(round(2.2 * r))
        hw = _round_up(max(MIN_W, want_side), W_STEP)
        hh = _round_up(max(MIN_H, want_side), H_STEP)
        x0 = int(round(cx - hw/2)); y0 = int(round(cy - hh/2))
        self.ctrl.set_roi(hw, hh, x0, y0)

        if self.cfg.maximize_camera_fps:
            try: self.ctrl.set_timing(float("inf"), None)
            except Exception: pass

        # metadata
        meta = {
            "spot": {"cx": cx, "cy": cy, "r": self.spot.r, "area": self.spot.area, "inten": self.spot.inten},
            "layout": {"(0,0)": "90", "(0,1)": "45", "(1,0)": "135", "(1,1)": "0"},
            "roi_hw_requested": {"w": hw, "h": hh, "x": x0, "y": y0},
            "chunk_len": self._chunk_len,
        }
        try:
            with open(os.path.join(self.cfg.out_dir, f"{self._prefix}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

        self._writer_thr.start()
        self._running = True
        self._t0 = time.perf_counter()
        self.started.emit()

    def stop(self) -> None:
        if not self._running: return
        self._running = False
        # flush
        try:
            if self._buf_i > 0:
                self._writer.enqueue(self._buf[:self._buf_i].copy())
                self._buf_i = 0
        except Exception: pass
        try: self._writer.close()
        except Exception: pass
        self.stopped.emit()

    # ---- camera reports ----
    @Slot(dict)
    def _on_roi(self, d: dict) -> None:
        try:
            w = int(round(float(d.get("Width", 0))))
            h = int(round(float(d.get("Height", 0))))
            x = int(round(float(d.get("OffsetX", 0))))
            y = int(round(float(d.get("OffsetY", 0))))
            if w and h: self._applied_roi = (x, y, w, h)
        except Exception:
            pass

    @Slot(object)
    def _on_frame(self, frame_obj: object) -> None:
        if not self._running: return
        a16 = np.asarray(frame_obj)
        if a16.ndim != 2: return

        ax, ay, aw, ah = self._applied_roi
        if aw <= 0 or ah <= 0:
            ah, aw = a16.shape; ax = ay = 0

        rcx = float(self.spot.cx) - float(ax)
        rcy = float(self.spot.cy) - float(ay)

        diameter = max(2.0, 2.0 * float(max(4.0, self.spot.r)))
        want = int(math.ceil(diameter + 2 * PAD_SW))
        side = _snap_even(max(MIN_CROP, min(aw, ah, want)))

        ix = _clamp(int(round(rcx)) - side // 2, 0, max(0, aw - side))
        iy = _clamp(int(round(rcy)) - side // 2, 0, max(0, ah - side))
        jx = _clamp(ix + side, ix + 1, aw)
        jy = _clamp(iy + side, iy + 1, ah)

        # keep within received frame
        ix = _clamp(ix, 0, a16.shape[1] - 1); jx = _clamp(jx, ix + 1, a16.shape[1])
        iy = _clamp(iy, 0, a16.shape[0] - 1); jy = _clamp(jy, iy + 1, a16.shape[0])

        crop = a16[iy:jy, ix:jx]

        row0 = (ay + iy) & 1
        col0 = (ax + ix) & 1

        s90  = crop[row0::2,        col0::2]
        s45  = crop[row0::2,        (col0 ^ 1)::2]
        s135 = crop[(row0 ^ 1)::2,  col0::2]
        s0   = crop[(row0 ^ 1)::2,  (col0 ^ 1)::2]

        def m(x): return float(np.mean(x, dtype=np.float64)) if x.size else 0.0
        I0, I45, I90, I135 = m(s0), m(s45), m(s90), m(s135)

        t = time.perf_counter() - self._t0
        i = self._buf_i
        self._buf[i, 0] = t
        self._buf[i, 1] = I0
        self._buf[i, 2] = I45
        self._buf[i, 3] = I90
        self._buf[i, 4] = I135
        self._buf_i += 1

        if self._buf_i >= self._chunk_len:
            try: self._writer.enqueue(self._buf.copy())
            except Exception as e: self.error.emit(str(e))
            self._buf_i = 0
            self._total += self._chunk_len
            self.progress.emit(self._total)
