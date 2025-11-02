# ------------------------------------------
# IDS backend: robust worker (ROI + Timing + Gains), buffer windowing,
# pixel format forcing, safe snapping & float clamp.
# ------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from time import sleep
from threading import Event, Lock
from typing import Any, Dict, List, Optional, Tuple, Union

from PySide6.QtCore import QObject, QThread, Signal, Slot, QMetaObject, Qt

# ---- IDS imports (guarded) ----
try:
    import ids_peak.ids_peak as ids
    from ids_peak import ids_peak_ipl_extension as ipl_ext
    _IDS_IMPORT_ERROR: Optional[str] = None
except Exception as e:  # pragma: no cover
    ids = None  # type: ignore
    ipl_ext = None  # type: ignore
    _IDS_IMPORT_ERROR = str(e)


@dataclass
class _State:
    open: bool = False
    acquiring: bool = False


# ---- Buffer policy (tunable) ----
BUFFER_WINDOW_SEC = 0.20   # aim to buffer ~200 ms worth of frames
BUFFER_MAX_COUNT = 128     # hard cap on count
BUFFER_MAX_BYTES = 512 * 1024 * 1024  # ~512 MiB cap


class _StreamWorker(QObject):
    """Worker that owns the IDS datastream and camera node map."""

    # Signals back to GUI
    started = Signal()
    stopped = Signal()
    closed = Signal()
    frame = Signal(object)   # numpy array (mono)
    error = Signal(str)
    roi = Signal(dict)       # {Width, Height, OffsetX, OffsetY}
    timing = Signal(dict)    # {fps, exposure_us, resulting_fps, ...}
    gains = Signal(dict)     # {'analog':{val,min,max,inc}, 'digital':{...}}

    def __init__(self, node_map, datastream) -> None:
        super().__init__()
        self._nm = node_map
        self._ds = datastream
        self._announced: List[object] = []
        self._run = Event()
        self._closed = False
        # Mailboxes
        self._roi_lock = Lock()
        self._pending_roi: Optional[Dict[str, Any]] = None
        self._tim_lock = Lock()
        self._pending_timing: Optional[Dict[str, Any]] = None
        self._gain_lock = Lock()
        self._pending_gains: Optional[Dict[str, Any]] = None

    # ---------- helpers ----------
    def _img_dims(self, img) -> Tuple[int, int]:
        try:
            return int(img.Width()), int(img.Height())
        except Exception:
            pass
        try:
            sz = img.Size()
            w = getattr(sz, "width", None)
            h = getattr(sz, "height", None)
            if w is not None and h is not None:
                return int(w), int(h)
            W = getattr(sz, "Width", None)
            H = getattr(sz, "Height", None)
            if callable(W) and callable(H):
                return int(W()), int(H())
            if W is not None and H is not None:
                return int(W), int(H)
        except Exception:
            pass
        raise RuntimeError("Cannot obtain image size from IDS IPL image")

    def _clear_pool(self) -> None:
        try:
            self._ds.Flush(ids.DataStreamFlushMode_DiscardAll)
        except Exception:
            pass
        try:
            for b in self._announced:
                try:
                    self._ds.RevokeBuffer(b)
                except Exception:
                    pass
        except Exception:
            pass
        self._announced = []

    def _current_payload_bytes(self) -> int:
        try:
            return int(self._nm.FindNode("PayloadSize").Value())
        except Exception:
            return 0

    def _estimate_target_buffers(self, desired_fps: Optional[float] = None) -> Tuple[int, int, int]:
        """Return (target_count, required_min, payload_bytes)."""
        # required min from driver
        try:
            required = int(self._ds.NumBuffersAnnouncedMinRequired())
        except Exception:
            required = 3
        required = max(required, 1)
        # fps hint
        fps = None
        if desired_fps is not None:
            try:
                fps = float(desired_fps)
            except Exception:
                fps = None
        if fps is None:
            t = self._read_timing()
            fps = (t.get("resulting_fps") or t.get("fps") or 30.0)
            try:
                fps = float(fps)  # type: ignore[arg-type]
            except Exception:
                fps = 30.0
        # payload bytes
        payload = self._current_payload_bytes()
        # window-based target
        window_count = int(max(1, round(float(fps) * BUFFER_WINDOW_SEC)))
        base = max(8, 2 * required)
        target = max(base, window_count)
        # memory cap
        if payload > 0:
            max_by_mem = max(1, int(BUFFER_MAX_BYTES // payload))
            target = min(target, BUFFER_MAX_COUNT, max_by_mem)
        else:
            target = min(target, BUFFER_MAX_COUNT)
        return target, required, payload

    def _announce_and_queue(self, desired_fps: Optional[float] = None) -> bool:
        try:
            self._clear_pool()
            target, required, payload = self._estimate_target_buffers(desired_fps)
            if payload <= 0:
                self.error.emit("PayloadSize is 0")
                return False
            for _ in range(target):
                buf = self._ds.AllocAndAnnounceBuffer(payload)
                self._announced.append(buf)
            for b in self._announced:
                self._ds.QueueBuffer(b)
            # tiny yield to let driver see queued buffers
            sleep(0.005)
            try:
                mb = payload * len(self._announced)
                print(f"[BufferPool] announced={len(self._announced)} required>={required} payload={payload} (~{mb/1048576:.1f} MiB)")
            except Exception:
                pass
            return True
        except Exception as e:
            self.error.emit(f"Pool prepare failed: {e}")
            try:
                print("[Worker] pool prepare failed:", e)
            except Exception:
                pass
            return False

    def _force_mono_format(self) -> None:
        """Try to force a simple grayscale output (Mono12 -> Mono16 -> Mono8)."""
        try:
            pf = self._nm.FindNode("PixelFormat")
            if pf and hasattr(pf, "SetCurrentEntry"):
                for candidate in ("Mono12", "Mono16", "Mono8"):
                    try:
                        pf.SetCurrentEntry(candidate)
                        print(f"[Worker] PixelFormat -> {candidate}")
                        return
                    except Exception:
                        continue
        except Exception:
            pass

    def _get_node(self, name: str):
        try:
            return self._nm.FindNode(name)
        except Exception:
            return None

    def _node_value(self, name: str) -> Optional[Union[int, float]]:
        n = self._get_node(name)
        if n is None:
            return None
        for attr in ("Value", "GetValue"):
            try:
                return float(getattr(n, attr)())
            except Exception:
                pass
        # Some enums expose numeric-like Value on their current entry
        for attr in ("CurrentEntry", "GetCurrentEntry"):
            try:
                e = getattr(n, attr)()
                if hasattr(e, "Value"):
                    return float(e.Value())
            except Exception:
                pass
        return None

    def _node_min(self, name: str) -> Optional[float]:
        n = self._get_node(name)
        if n is None:
            return None
        for attr in ("Minimum", "GetMinimum", "Min", "GetMin"):
            try:
                return float(getattr(n, attr)())
            except Exception:
                pass
        return None

    def _node_max(self, name: str) -> Optional[float]:
        n = self._get_node(name)
        if n is None:
            return None
        for attr in ("Maximum", "GetMaximum", "Max", "GetMax"):
            try:
                return float(getattr(n, attr)())
            except Exception:
                pass
        return None

    def _node_inc(self, name: str) -> Optional[float]:
        n = self._get_node(name)
        if n is None:
            return None
        for attr in ("Increment", "GetIncrement", "Step", "GetStep"):
            try:
                return float(getattr(n, attr)())
            except Exception:
                pass
        return None

    def _snap(self, name: str, value: float) -> float:
        mn = self._node_min(name)
        mx = self._node_max(name)
        inc = self._node_inc(name)
        # clamp
        if mn is not None:
            value = max(value, mn)
        if mx is not None:
            value = min(value, mx)
        # floor to step to avoid rounding just above max
        if inc and inc > 0:
            base = mn if mn is not None else 0.0
            steps = math.floor((value - base) / inc + 1e-9)
            value = base + steps * inc
        # epsilon clamp at top edge
        if mx is not None and value > mx:
            value = mx
        if mx is not None and abs(value - mx) < 1e-7:
            value = mx - (inc or 1e-3) * 1e-3
        return float(value)

    # ---------- ROI mailbox ----------
    def _pop_pending_roi(self) -> Optional[Dict[str, Any]]:
        with self._roi_lock:
            d = self._pending_roi
            self._pending_roi = None
            return d

    def _apply_roi_payload(self, d: Dict[str, Any], while_running: bool) -> None:
        try:
            print(f"[Worker] apply_roi(payload) while_running={while_running}: {d}")
            width = float(d.get("Width"))
            height = float(d.get("Height"))
            offx = float(d.get("OffsetX"))
            offy = float(d.get("OffsetY"))
        except Exception as e:
            self.error.emit(f"ROI args invalid: {e}")
            return
        # quick pause if running
        if while_running:
            try:
                self._nm.FindNode("AcquisitionStop").Execute()
            except Exception:
                pass
            try:
                self._ds.StopAcquisition()
            except Exception:
                pass
            self._clear_pool()
        # ensure unlock so ROI nodes are writable
        try:
            tln = self._nm.FindNode("TLParamsLocked")
            if tln:
                tln.SetValue(0)
                print("[Worker] TLParamsLocked=0 for ROI")
        except Exception:
            pass
        # optional: some cameras require setting region selector
        try:
            rs = self._nm.FindNode("RegionSelector")
            if rs and hasattr(rs, "SetCurrentEntry"):
                try:
                    rs.SetCurrentEntry("Region0")
                    print("[Worker] RegionSelector=Region0")
                except Exception:
                    pass
        except Exception:
            pass
        # offsets -> size -> offsets
        try:
            mnx = self._node_min("OffsetX")
            mny = self._node_min("OffsetY")
            if mnx is not None:
                self._nm.FindNode("OffsetX").SetValue(int(round(mnx)))
            if mny is not None:
                self._nm.FindNode("OffsetY").SetValue(int(round(mny)))
        except Exception:
            pass
        try:
            wv = int(round(self._snap("Width", width)))
            hv = int(round(self._snap("Height", height)))
            print(f"[Worker] ROI size snapped -> Width={wv}, Height={hv}")
            self._nm.FindNode("Width").SetValue(wv)
            self._nm.FindNode("Height").SetValue(hv)
        except Exception as e:
            self.error.emit(f"ROI width/height failed: {e}")
            return
        try:
            xv = int(round(self._snap("OffsetX", offx)))
            yv = int(round(self._snap("OffsetY", offy)))
            mx_x = self._node_max("OffsetX")
            mx_y = self._node_max("OffsetY")
            if mx_x is not None:
                xv = min(xv, int(round(mx_x)))
            if mx_y is not None:
                yv = min(yv, int(round(mx_y)))
            print(f"[Worker] ROI offsets snapped -> OffsetX={xv}, OffsetY={yv}")
            self._nm.FindNode("OffsetX").SetValue(xv)
            self._nm.FindNode("OffsetY").SetValue(yv)
        except Exception as e:
            self.error.emit(f"ROI offsets failed: {e}")
            return
        # rebuild pool (force pixel format, then announce)
        self._force_mono_format()
        self._announce_and_queue(desired_fps=None)
        # relock
        try:
            tln = self._nm.FindNode("TLParamsLocked")
            if tln:
                tln.SetValue(1)
                print("[Worker] TLParamsLocked=1 after ROI")
        except Exception:
            pass
        # resume if running
        if while_running:
            try:
                self._ds.StartAcquisition()
                self._nm.FindNode("AcquisitionStart").Execute()
                print("[Worker] resumed acquisition after ROI")
            except Exception as e:
                self.error.emit(f"ROI resume failed: {e}")
        # publish
        self.query_roi()
        self.query_timing()

    # ---------- Timing mailbox ----------
    def _read_timing(self) -> Dict[str, Union[float, bool, None]]:
        d: Dict[str, Union[float, bool, None]] = {}
        fps_node = self._get_node("AcquisitionFrameRate")
        d["fps_writable"] = fps_node is not None
        if fps_node is not None:
            d["fps"] = self._node_value("AcquisitionFrameRate")
            d["fps_min"] = self._node_min("AcquisitionFrameRate")
            d["fps_max"] = self._node_max("AcquisitionFrameRate")
            d["fps_inc"] = self._node_inc("AcquisitionFrameRate")
        else:
            d["fps"] = None
            d["fps_min"] = None
            d["fps_max"] = None
            d["fps_inc"] = None
        rf = self._node_value("ResultingFrameRate")
        if rf is not None:
            d["resulting_fps"] = rf
        d["exposure_us"] = self._node_value("ExposureTime")
        d["exposure_min"] = self._node_min("ExposureTime")
        d["exposure_max"] = self._node_max("ExposureTime")
        d["exposure_inc"] = self._node_inc("ExposureTime")
        return d

    def _apply_timing_payload(self, d: Dict[str, Any], while_running: bool) -> None:
        """Apply FPS / Exposure with safe ordering and pool handling."""
        print(f"[Worker] apply_timing while_running={while_running}: {d}")
        fps_req = d.get("fps")
        exp_ms_req = d.get("exposure_ms")

        def _ensure_unlocked() -> None:
            try:
                tln = self._nm.FindNode("TLParamsLocked")
                if tln:
                    tln.SetValue(0)
                    print("[Worker] TLParamsLocked=0 for timing")
            except Exception:
                pass

        def _soft_pause() -> None:
            try:
                self._nm.FindNode("AcquisitionStop").Execute()
            except Exception:
                pass
            try:
                self._ds.StopAcquisition()
            except Exception:
                pass

        def _resume() -> None:
            try:
                self._ds.StartAcquisition()
                self._nm.FindNode("AcquisitionStart").Execute()
                print("[Worker] resumed acquisition after timing")
            except Exception as e:
                self.error.emit(f"Timing resume failed: {e}")

        def _rebuild_pool(desired_fps: Optional[float] = None) -> None:
            if self._announce_and_queue(desired_fps=desired_fps):
                print("[Worker] buffer pool rebuilt for timing")

        _ensure_unlocked()
        paused = False

        def _set_exposure(exp_ms: float) -> None:
            nonlocal paused
            exp_us = float(exp_ms) * 1000.0
            exp_us = self._snap("ExposureTime", exp_us)
            # final guard against max epsilon
            mx = self._node_max("ExposureTime")
            if mx is not None and exp_us > mx:
                exp_us = mx - 1e-3
            try:
                auto = self._get_node("ExposureAuto")
                if auto is not None and hasattr(auto, "SetCurrentEntry"):
                    try:
                        auto.SetCurrentEntry("Off")
                    except Exception:
                        pass
                self._get_node("ExposureTime").SetValue(exp_us)
                print(f"[Worker] set Exposure -> {exp_us} us")
            except Exception as e:
                print(f"[Worker] set Exposure failed (will try paused): {e}")
                if while_running and not paused:
                    _soft_pause(); paused = True
                    try:
                        self._get_node("ExposureTime").SetValue(exp_us)
                        print(f"[Worker] set Exposure (paused) -> {exp_us} us")
                    except Exception as e2:
                        self.error.emit(f"Set Exposure failed: {e2}")

        def _set_fps(fps_val: float) -> None:
            nonlocal paused
            fps_snapped = self._snap("AcquisitionFrameRate", float(fps_val))
            try:
                en = self._get_node("AcquisitionFrameRateEnable")
                if en is not None:
                    try:
                        en.SetValue(True)
                    except Exception:
                        pass
                self._get_node("AcquisitionFrameRate").SetValue(fps_snapped)
                print(f"[Worker] set FPS -> {fps_snapped}")
            except Exception as e:
                print(f"[Worker] set FPS failed (will try paused): {e}")
                if while_running and not paused:
                    _soft_pause(); paused = True
                    try:
                        self._get_node("AcquisitionFrameRate").SetValue(fps_snapped)
                        print(f"[Worker] set FPS (paused) -> {fps_snapped}")
                    except Exception as e2:
                        self.error.emit(f"Set FPS failed: {e2}")

        # If both supplied, set exposure first (it can change FPS max)
        if exp_ms_req is not None and fps_req is not None:
            _set_exposure(exp_ms_req)
            _set_fps(fps_req)
        else:
            if fps_req is not None:
                _set_fps(fps_req)
            if exp_ms_req is not None:
                _set_exposure(exp_ms_req)

        if paused and while_running:
            # Pixel format can flip on some models after timing changes; enforce
            self._force_mono_format()
            _rebuild_pool(desired_fps=fps_req if fps_req is not None else None)
            # Relock before resume
            try:
                tln = self._nm.FindNode("TLParamsLocked")
                if tln:
                    tln.SetValue(1)
                    print("[Worker] TLParamsLocked=1 after timing")
            except Exception:
                pass
            _resume()
        else:
            try:
                tln = self._nm.FindNode("TLParamsLocked")
                if tln:
                    tln.SetValue(1)
                    print("[Worker] TLParamsLocked=1 after timing")
            except Exception:
                pass

        self.query_timing()

    # ---------- Gains mailbox ----------
    def _read_gains(self) -> Dict[str, Dict[str, Optional[float]]]:
        out: Dict[str, Dict[str, Optional[float]]] = {"analog": {}, "digital": {}}
        try:
            # Analog
            try:
                sel = self._get_node("GainSelector")
                if sel and hasattr(sel, "SetCurrentEntry"):
                    try:
                        sel.SetCurrentEntry("AnalogAll")
                    except Exception:
                        pass
                g = self._get_node("Gain")
                out["analog"]["val"] = self._node_value("Gain")
                out["analog"]["min"] = self._node_min("Gain")
                out["analog"]["max"] = self._node_max("Gain")
                out["analog"]["inc"] = self._node_inc("Gain")
            except Exception:
                out["analog"] = {"val": None, "min": None, "max": None, "inc": None}
            # Digital
            try:
                sel = self._get_node("GainSelector")
                if sel and hasattr(sel, "SetCurrentEntry"):
                    try:
                        sel.SetCurrentEntry("DigitalAll")
                    except Exception:
                        pass
                out["digital"]["val"] = self._node_value("Gain")
                out["digital"]["min"] = self._node_min("Gain")
                out["digital"]["max"] = self._node_max("Gain")
                out["digital"]["inc"] = self._node_inc("Gain")
            except Exception:
                out["digital"] = {"val": None, "min": None, "max": None, "inc": None}
        except Exception:
            out = {"analog": {"val": None, "min": None, "max": None, "inc": None},
                   "digital": {"val": None, "min": None, "max": None, "inc": None}}
        return out

    def _apply_gains_payload(self, d: Dict[str, Any], while_running: bool) -> None:
        print(f"[Worker] apply_gains while_running={while_running}: {d}")
        def _soft_pause():
            try:
                self._nm.FindNode("AcquisitionStop").Execute()
            except Exception:
                pass
            try:
                self._ds.StopAcquisition()
            except Exception:
                pass

        def _resume():
            try:
                self._ds.StartAcquisition()
                self._nm.FindNode("AcquisitionStart").Execute()
                print("[Worker] resumed acquisition after gains")
            except Exception as e:
                self.error.emit(f"Gains resume failed: {e}")

        paused = False
        # Safer on this camera: always pause when changing both/any gains during run
        if while_running:
            _soft_pause()
            paused = True

        # apply analog
        if "analog" in d and d["analog"] is not None:
            try:
                sel = self._get_node("GainSelector")
                if sel and hasattr(sel, "SetCurrentEntry"):
                    try:
                        sel.SetCurrentEntry("AnalogAll")
                    except Exception:
                        pass
                val = float(d["analog"])
                mn, mx, inc = self._node_min("Gain"), self._node_max("Gain"), self._node_inc("Gain")
                if mn is not None:
                    val = max(val, mn)
                if mx is not None:
                    val = min(val, mx)
                if inc and inc > 0:
                    base = mn if mn is not None else 0.0
                    steps = math.floor((val - base) / inc + 1e-9)
                    val = base + steps * inc
                self._get_node("Gain").SetValue(val)
                print(f"[Worker] set Analog Gain -> {val}")
            except Exception as e:
                self.error.emit(f"Set Analog Gain failed: {e}")

        # apply digital
        if "digital" in d and d["digital"] is not None:
            try:
                sel = self._get_node("GainSelector")
                if sel and hasattr(sel, "SetCurrentEntry"):
                    try:
                        sel.SetCurrentEntry("DigitalAll")
                    except Exception:
                        pass
                val = float(d["digital"])
                mn, mx, inc = self._node_min("Gain"), self._node_max("Gain"), self._node_inc("Gain")
                if mn is not None:
                    val = max(val, mn)
                if mx is not None:
                    val = min(val, mx)
                if inc and inc > 0:
                    base = mn if mn is not None else 0.0
                    steps = math.floor((val - base) / inc + 1e-9)
                    val = base + steps * inc
                self._get_node("Gain").SetValue(val)
                print(f"[Worker] set Digital Gain -> {val}")
            except Exception as e:
                self.error.emit(f"Set Digital Gain failed: {e}")

        if paused and while_running:
            self._force_mono_format()
            # No need to rebuild pool for gains, but some drivers like a re-queue
            if self._announce_and_queue(desired_fps=None):
                print("[Worker] buffer pool rebuilt for gains")
            _resume()

        # publish
        self.gains.emit(self._read_gains())

    # ---------- slots (run in worker thread) ----------
    @Slot()
    def start(self) -> None:
        if self._closed or self._run.is_set():
            return
        try:
            print("[Worker] start requested")
            self._force_mono_format()
            if not self._announce_and_queue(desired_fps=None):
                self.error.emit("Start failed: buffer pool not ready")
                return
            try:
                n = self._nm.FindNode("TLParamsLocked")
                n and n.SetValue(1)
            except Exception:
                pass
            self._ds.StartAcquisition()
            self._nm.FindNode("AcquisitionStart").Execute()
            self._run.set()
            self.started.emit()
            print("[Worker] started acquisition")

            while self._run.is_set():
                # ROI first (affects payload)
                pending_roi = self._pop_pending_roi()
                if pending_roi is not None:
                    self._apply_roi_payload(pending_roi, while_running=True)
                    continue
                # Timing next
                with self._tim_lock:
                    t = self._pending_timing
                    self._pending_timing = None
                if t is not None:
                    self._apply_timing_payload(t, while_running=True)
                    continue
                # Gains next
                with self._gain_lock:
                    g = self._pending_gains
                    self._pending_gains = None
                if g is not None:
                    self._apply_gains_payload(g, while_running=True)
                    continue

                buf = None
                try:
                    buf = self._ds.WaitForFinishedBuffer(10)
                except Exception:
                    continue
                try:
                    img = ipl_ext.BufferToImage(buf)
                    if hasattr(img, "get_numpy_1D"):
                        import numpy as _np
                        w, h = self._img_dims(img)
                        flat = img.get_numpy_1D()
                        bufarr = _np.asarray(flat)
                        if bufarr.ndim != 1:
                            bufarr = bufarr.reshape(-1)
                        expected = w * h
                        if bufarr.size == 2 * expected:
                            if not bufarr.flags.c_contiguous:
                                bufarr = _np.ascontiguousarray(bufarr)
                            arr = bufarr.view('<u2').reshape(h, w).copy()
                        else:
                            arr = bufarr.reshape(h, w).copy()
                    else:
                        raise RuntimeError("IDS IPL image missing numpy accessor")
                    self.frame.emit(arr)
                except Exception as e:
                    self.error.emit(f"Convert failed: {e}")
                finally:
                    if buf is not None:
                        try:
                            self._ds.QueueBuffer(buf)
                        except Exception:
                            pass

            # finalize stop
            print("[Worker] finalize stop begin")
            try:
                self._nm.FindNode("AcquisitionStop").Execute()
            except Exception:
                pass
            try:
                self._ds.StopAcquisition()
            except Exception:
                pass
            self._clear_pool()
            try:
                n = self._nm.FindNode("TLParamsLocked")
                n and n.SetValue(0)
            except Exception:
                pass
            print("[Worker] finalize stop done")
            self.stopped.emit()
        except Exception as e:
            self.error.emit(f"Start failed: {e}")
            try:
                print("[Worker] start exception:", e)
            except Exception:
                pass

    @Slot()
    def set_stop_flag(self) -> None:
        print("[Worker] set_stop_flag (DirectConnection) — clearing run flag")
        self._run.clear()

    @Slot()
    def request_stop(self) -> None:
        print("[Worker] request_stop (Queued) — clearing run flag and trying CancelWait")
        self._run.clear()
        try:
            cancel = getattr(self._ds, "CancelWait", None)
            if callable(cancel):
                cancel()
        except Exception:
            pass

    # ---- ROI API ----
    @Slot()
    def query_roi(self) -> None:
        print("[Worker] query_roi requested")
        names = ("Width", "Height", "OffsetX", "OffsetY")
        out: Dict[str, Optional[float]] = {}
        for n in names:
            out[n] = self._node_value(n)
        print(f"[Worker] query_roi -> {out}")
        self.roi.emit(out)

    @Slot(object)
    def enqueue_roi(self, roi: object) -> None:
        print(f"[Worker] enqueue_roi <- {roi}")
        if not isinstance(roi, dict):
            return
        with self._roi_lock:
            self._pending_roi = roi

    @Slot()
    def process_pending_roi(self) -> None:
        d = self._pop_pending_roi()
        if d is not None:
            self._apply_roi_payload(d, while_running=False)

    # ---- Timing API ----
    @Slot()
    def query_timing(self) -> None:
        d = self._read_timing()
        print(f"[Worker] timing -> {d}")
        self.timing.emit(d)

    @Slot(object)
    def enqueue_timing(self, timing: object) -> None:
        print(f"[Worker] enqueue_timing <- {timing}")
        if not isinstance(timing, dict):
            return
        with self._tim_lock:
            self._pending_timing = timing

    @Slot()
    def process_pending_timing(self) -> None:
        with self._tim_lock:
            d = self._pending_timing
            self._pending_timing = None
        if d is not None:
            self._apply_timing_payload(d, while_running=False)

    # ---- Gains API ----
    @Slot()
    def query_gains(self) -> None:
        g = self._read_gains()
        print(f"[Worker] gains -> {g}")
        self.gains.emit(g)

    @Slot(object)
    def enqueue_gains(self, gains: object) -> None:
        print(f"[Worker] enqueue_gains <- {gains}")
        if not isinstance(gains, dict):
            return
        with self._gain_lock:
            self._pending_gains = gains

    @Slot()
    def process_pending_gains(self) -> None:
        with self._gain_lock:
            g = self._pending_gains
            self._pending_gains = None
        if g is not None:
            self._apply_gains_payload(g, while_running=False)

    @Slot()
    def close(self) -> None:
        if self._closed:
            return
        print("[Worker] close requested")
        self._run.clear()
        try:
            self._nm.FindNode("AcquisitionStop").Execute()
        except Exception:
            pass
        try:
            self._ds.StopAcquisition()
        except Exception:
            pass
        self._clear_pool()
        self._closed = True
        print("[Worker] closed")
        self.closed.emit()


class IDSCamera(QObject):
    """GUI-thread facade. All heavy lifting is delegated to _StreamWorker."""

    opened = Signal(str)
    started = Signal()
    stopped = Signal()
    closed = Signal()
    frame = Signal(object)
    error = Signal(str)
    roi = Signal(dict)
    timing = Signal(dict)
    gains = Signal(dict)

    # Direct cross-thread mailboxes
    send_roi = Signal(object)
    send_timing = Signal(object)
    send_gains = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._st = _State()
        self._device = None
        self._remote = None
        self._datastream = None
        self._node_map = None
        self._worker: Optional[_StreamWorker] = None
        self._thread: Optional[QThread] = None

    def _on_started(self) -> None:
        print("[UI] started() signal")
        self._st.acquiring = True
        self.started.emit()

    def _on_stopped(self) -> None:
        print("[UI] stopped() signal")
        self._st.acquiring = False
        self.stopped.emit()

    @Slot()
    def open(self) -> None:
        print("[UI] open() called")
        if self._st.open:
            print("[UI] open ignored: already open")
            return
        if _IDS_IMPORT_ERROR is not None:
            self.error.emit(f"ids_peak import failed: {_IDS_IMPORT_ERROR}")
            return
        try:
            ids.Library.Initialize()
            dm = ids.DeviceManager.Instance()
            dm.Update()
            devs = dm.Devices()
            if devs.empty():
                self.error.emit("No device found!")
                self._library_close_safe()
                return
            chosen = next((d for d in devs if d.IsOpenable()), None)
            if chosen is None:
                self.error.emit("Device could not be opened!")
                self._library_close_safe()
                return
            try:
                self._device = chosen.OpenDevice(ids.DeviceAccessType_Exclusive)
            except Exception:
                self._device = chosen.OpenDevice(ids.DeviceAccessType_Control)
            self._remote = self._device.RemoteDevice()
            self._node_map = self._remote.NodeMaps()[0]
            streams = self._device.DataStreams()
            if streams.empty():
                self.error.emit("Device has no DataStream!")
                self._device = None
                self._library_close_safe()
                return
            self._datastream = streams[0].OpenDataStream()

            self._thread = QThread(self)
            self._worker = _StreamWorker(self._node_map, self._datastream)
            self._worker.moveToThread(self._thread)
            # bridges
            self._worker.started.connect(self._on_started)
            self._worker.stopped.connect(self._on_stopped)
            self._worker.closed.connect(self.closed)
            self._worker.frame.connect(self.frame)
            self._worker.error.connect(self.error)
            self._worker.roi.connect(self.roi)
            self._worker.timing.connect(self.timing)
            self._worker.gains.connect(self.gains)
            self.send_roi.connect(self._worker.enqueue_roi, Qt.DirectConnection)
            self.send_timing.connect(self._worker.enqueue_timing, Qt.DirectConnection)
            self.send_gains.connect(self._worker.enqueue_gains, Qt.DirectConnection)
            self._thread.start()

            self._st.open = True
            name = chosen.DisplayName() if hasattr(chosen, "DisplayName") else "IDS camera"
            print("[UI] opened")
            self.opened.emit(str(name))
            # initial snapshots
            QMetaObject.invokeMethod(self._worker, "query_roi", Qt.QueuedConnection)
            QMetaObject.invokeMethod(self._worker, "query_timing", Qt.QueuedConnection)
            QMetaObject.invokeMethod(self._worker, "query_gains", Qt.QueuedConnection)
        except Exception as e:
            self.error.emit(f"Failed to open device: {e}")
            self._cleanup_all()

    @Slot()
    def start(self) -> None:
        print(f"[UI] start() called; acquiring={self._st.acquiring}")
        if not self._st.open or not self._worker:
            print("[UI] start ignored: not open or no worker")
            return
        if self._st.acquiring:
            print("[UI] start ignored: already acquiring")
            return
        QMetaObject.invokeMethod(self._worker, "start", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self._worker, "query_timing", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self._worker, "query_gains", Qt.QueuedConnection)

    @Slot()
    def stop(self) -> None:
        print(f"[UI] stop() called; acquiring={self._st.acquiring}")
        if not self._worker:
            print("[UI] stop ignored: no worker")
            return
        QMetaObject.invokeMethod(self._worker, "set_stop_flag", Qt.DirectConnection)
        QMetaObject.invokeMethod(self._worker, "request_stop", Qt.QueuedConnection)

    @Slot()
    def set_roi(self, w: float, h: float, x: float, y: float) -> None:
        if not self._worker:
            print("[UI] set_roi ignored: no worker")
            return
        print(f"[UI] set_roi(w={w}, h={h}, x={x}, y={y}); acquiring={self._st.acquiring}")
        payload = {"Width": w, "Height": h, "OffsetX": x, "OffsetY": y}
        self.send_roi.emit(payload)
        if not self._st.acquiring:
            QMetaObject.invokeMethod(self._worker, "process_pending_roi", Qt.QueuedConnection)
        else:
            QMetaObject.invokeMethod(self._worker, "query_timing", Qt.QueuedConnection)

    @Slot()
    def set_timing(self, fps: Optional[float], exposure_ms: Optional[float]) -> None:
        if not self._worker:
            print("[UI] set_timing ignored: no worker")
            return
        print(f"[UI] set_timing(fps={fps}, exposure_ms={exposure_ms}); acquiring={self._st.acquiring}")
        payload: Dict[str, float] = {}
        if fps is not None:
            try:
                payload["fps"] = float(fps)
            except Exception:
                pass
        if exposure_ms is not None:
            try:
                payload["exposure_ms"] = float(exposure_ms)
            except Exception:
                pass
        self.send_timing.emit(payload)
        if not self._st.acquiring:
            QMetaObject.invokeMethod(self._worker, "process_pending_timing", Qt.QueuedConnection)

    @Slot()
    def set_gains(self, analog: Optional[float], digital: Optional[float]) -> None:
        if not self._worker:
            print("[UI] set_gains ignored: no worker")
            return
        print(f"[UI] set_gains(analog={analog}, digital={digital}); acquiring={self._st.acquiring}")
        payload: Dict[str, Optional[float]] = {}
        if analog is not None:
            payload["analog"] = float(analog)
        if digital is not None:
            payload["digital"] = float(digital)
        self.send_gains.emit(payload)
        if not self._st.acquiring:
            QMetaObject.invokeMethod(self._worker, "process_pending_gains", Qt.QueuedConnection)

    @Slot()
    def refresh_timing(self) -> None:
        if not self._worker:
            print("[UI] refresh_timing ignored: no worker")
            return
        QMetaObject.invokeMethod(self._worker, "query_timing", Qt.QueuedConnection)

    @Slot()
    def refresh_gains(self) -> None:
        if not self._worker:
            print("[UI] refresh_gains ignored: no worker")
            return
        QMetaObject.invokeMethod(self._worker, "query_gains", Qt.QueuedConnection)

    @Slot()
    def close(self) -> None:
        print(f"[UI] close() called; acquiring={self._st.acquiring}")
        if not self._st.open:
            print("[UI] close ignored: not open")
            return
        if self._st.acquiring:
            self.error.emit("Stop before Close.")
            print("[UI] close refused: acquiring")
            return
        if self._worker:
            QMetaObject.invokeMethod(self._worker, "close", Qt.BlockingQueuedConnection)
        self._cleanup_all()

    def _cleanup_all(self) -> None:
        try:
            if self._thread:
                self._thread.quit()
                self._thread.wait(1000)
        except Exception:
            pass
        self._worker = None
        self._thread = None
        self._datastream = None
        self._remote = None
        self._device = None
        self._library_close_safe()
        self._st = _State(open=False, acquiring=False)
        print("[UI] cleanup complete")

    def _library_close_safe(self) -> None:
        try:
            if ids is not None:
                ids.Library.Close()
        except Exception:
            pass
