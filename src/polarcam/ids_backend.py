# ------------------------------------------
# File: src/polarcam/ids_backend.py
# (Minimal + ROI v2 — non‑blocking ROI via mailbox processed inside grab loop)
# ------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock
from typing import List, Optional, Tuple, Dict, Union, Any

from PySide6.QtCore import QObject, QThread, Signal, Slot, QMetaObject, Qt

# ---- IDS imports (guarded) ----
try:
    import ids_peak.ids_peak as ids
    from ids_peak import ids_peak_ipl_extension as ipl_ext
    _IDS_IMPORT_ERROR: Optional[str] = None
except Exception as e:
    ids = None  # type: ignore
    ipl_ext = None  # type: ignore
    _IDS_IMPORT_ERROR = str(e)


@dataclass
class _State:
    open: bool = False
    acquiring: bool = False


# -----------------------------
# Worker: owns the datastream
# -----------------------------
class _StreamWorker(QObject):
    started = Signal()
    stopped = Signal()
    closed = Signal()
    frame = Signal(object)   # numpy array (mono)
    error = Signal(str)
    roi = Signal(dict)       # {Width, Height, OffsetX, OffsetY}

    def __init__(self, node_map, datastream) -> None:
        super().__init__()
        self._nm = node_map
        self._ds = datastream
        self._announced: List[object] = []
        self._run = Event()
        self._closed = False
        # ROI mailbox (thread‑safe)
        self._roi_lock = Lock()
        self._pending_roi: Optional[Dict[str, Any]] = None

    # ---------- helpers ----------
    def _img_dims(self, img) -> Tuple[int,int]:
        try: return int(img.Width()), int(img.Height())
        except Exception: pass
        try:
            sz = img.Size()
            w = getattr(sz, "width", None); h = getattr(sz, "height", None)
            if w is not None and h is not None: return int(w), int(h)
            W = getattr(sz, "Width", None); H = getattr(sz, "Height", None)
            if callable(W) and callable(H): return int(W()), int(H())
            if W is not None and H is not None: return int(W), int(H)
        except Exception: pass
        raise RuntimeError("Cannot obtain image size from IDS IPL image")

    def _clear_pool(self) -> None:
        try:
            self._ds.Flush(ids.DataStreamFlushMode_DiscardAll)
        except Exception:
            pass
        try:
            for b in self._announced:
                try: self._ds.RevokeBuffer(b)
                except Exception: pass
        except Exception:
            pass
        self._announced = []

    def _announce_and_queue(self) -> bool:
        try:
            self._clear_pool()
            # required buffer count
            try:
                required = int(self._ds.NumBuffersAnnouncedMinRequired())
            except Exception:
                required = 6
            required = max(required, 1)
            # payload size
            try:
                payload = int(self._nm.FindNode("PayloadSize").Value())
            except Exception:
                payload = 0
            if payload <= 0:
                self.error.emit("PayloadSize is 0")
                return False
            # announce >= max(8, 2*required)
            target = max(8, 2 * required)
            for _ in range(target):
                buf = self._ds.AllocAndAnnounceBuffer(payload)
                self._announced.append(buf)
            for b in self._announced:
                self._ds.QueueBuffer(b)
            try:
                print(f"[BufferPool] announced={len(self._announced)} required>={required} payload={payload}")
            except Exception:
                pass
            return True
        except Exception as e:
            self.error.emit(f"Pool prepare failed: {e}")
            try: print("[Worker] pool prepare failed:", e)
            except Exception: pass
            return False

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
        if mn is not None:
            value = max(value, mn)
        if mx is not None:
            value = min(value, mx)
        if inc and inc > 0:
            base = mn if mn is not None else 0.0
            steps = round((value - base) / inc)
            value = base + steps * inc
        return float(value)

    def _pop_pending_roi(self) -> Optional[Dict[str, Any]]:
        with self._roi_lock:
            d = self._pending_roi
            self._pending_roi = None
            return d

    def _apply_roi_payload(self, d: Dict[str, Any], while_running: bool) -> None:
        try:
            print(f"[Worker] apply_roi(payload) while_running={while_running}: {d}")
            width = float(d.get("Width")); height = float(d.get("Height"))
            offx = float(d.get("OffsetX")); offy = float(d.get("OffsetY"))
        except Exception as e:
            self.error.emit(f"ROI args invalid: {e}")
            return
        # quick pause if running
        if while_running:
            try: self._nm.FindNode("AcquisitionStop").Execute()
            except Exception: pass
            try: self._ds.StopAcquisition()
            except Exception: pass
            self._clear_pool()
        # ensure TLParamsUnlocked so Width/Height/Offsets are writable
        try:
            tln = self._nm.FindNode("TLParamsLocked")
            if tln:
                tln.SetValue(0)
                print("[Worker] TLParamsLocked=0 for ROI")
        except Exception:
            pass
        # optional: select Region0 if present (some cameras expose ROI via region selector)
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
            mnx = self._node_min("OffsetX"); mny = self._node_min("OffsetY")
            if mnx is not None: self._nm.FindNode("OffsetX").SetValue(int(round(mnx)))
            if mny is not None: self._nm.FindNode("OffsetY").SetValue(int(round(mny)))
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
            mx_x = self._node_max("OffsetX"); mx_y = self._node_max("OffsetY")
            if mx_x is not None: xv = min(xv, int(round(mx_x)))
            if mx_y is not None: yv = min(yv, int(round(mx_y)))
            print(f"[Worker] ROI offsets snapped -> OffsetX={xv}, OffsetY={yv}")
            self._nm.FindNode("OffsetX").SetValue(xv)
            self._nm.FindNode("OffsetY").SetValue(yv)
        except Exception as e:
            self.error.emit(f"ROI offsets failed: {e}")
            return
        # rebuild pool
        self._announce_and_queue()
        # lock transport layer params again if we had unlocked
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

    # ---------- slots (run in worker thread) ----------
    @Slot()
    def start(self) -> None:
        if self._closed or self._run.is_set():
            return
        try:
            print("[Worker] start requested")
            if not self._announce_and_queue():
                self.error.emit("Start failed: buffer pool not ready")
                return
            # Start camera & host
            try:
                n = self._nm.FindNode("TLParamsLocked"); n and n.SetValue(1)
            except Exception:
                pass
            self._ds.StartAcquisition()
            self._nm.FindNode("AcquisitionStart").Execute()
            self._run.set(); self.started.emit()
            print("[Worker] started acquisition")

            while self._run.is_set():
                # process any pending ROI first
                pending = self._pop_pending_roi()
                if pending is not None:
                    self._apply_roi_payload(pending, while_running=True)
                    continue
                buf = None
                try:
                    buf = self._ds.WaitForFinishedBuffer(10)  # short timeout for responsiveness
                except Exception:
                    continue
                try:
                    img = ipl_ext.BufferToImage(buf)
                    # Mono12-only fast path: 16-bit container or fallback to 8-bit
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
                        try: self._ds.QueueBuffer(buf)
                        except Exception: pass

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
                n = self._nm.FindNode("TLParamsLocked"); n and n.SetValue(0)
            except Exception:
                pass
            print("[Worker] finalize stop done")
            self.stopped.emit()
        except Exception as e:
            self.error.emit(f"Start failed: {e}")
            try: print("[Worker] start exception:", e)
            except Exception: pass

    @Slot()
    def set_stop_flag(self) -> None:
        print("[Worker] set_stop_flag (DirectConnection) — clearing run flag")
        self._run.clear()

    @Slot()
    def request_stop(self) -> None:
        # Queued: runs only when the worker's event loop is free
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
        # Lightweight mailbox setter; safe to call via DirectConnection from GUI thread
        print(f"[Worker] enqueue_roi <- {roi}")
        if not isinstance(roi, dict):
            return
        with self._roi_lock:
            self._pending_roi = roi

    @Slot()
    def process_pending_roi(self) -> None:
        # For the non-running case: apply immediately in worker thread
        d = self._pop_pending_roi()
        if d is not None:
            self._apply_roi_payload(d, while_running=False)

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


# -----------------------------
# Facade (GUI thread)
# -----------------------------
class IDSCamera(QObject):
    opened = Signal(str)
    started = Signal()
    stopped = Signal()
    closed = Signal()
    frame = Signal(object)
    error = Signal(str)
    roi = Signal(dict)
    # Direct cross-thread ROI mailbox sender (carries a Python dict)
    send_roi = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._st = _State()
        self._device = None
        self._remote = None
        self._datastream = None
        self._node_map = None
        self._worker: Optional[_StreamWorker] = None
        self._thread: Optional[QThread] = None

    def _on_started(self):
        print("[UI] started() signal")
        self._st.acquiring = True
        self.started.emit()

    def _on_stopped(self):
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
            dm = ids.DeviceManager.Instance(); dm.Update()
            devs = dm.Devices()
            if devs.empty():
                self.error.emit("No device found!"); self._library_close_safe(); return
            chosen = next((d for d in devs if d.IsOpenable()), None)
            if chosen is None:
                self.error.emit("Device could not be opened!"); self._library_close_safe(); return
            try:
                self._device = chosen.OpenDevice(ids.DeviceAccessType_Exclusive)
            except Exception:
                self._device = chosen.OpenDevice(ids.DeviceAccessType_Control)
            self._remote = self._device.RemoteDevice(); self._node_map = self._remote.NodeMaps()[0]
            streams = self._device.DataStreams()
            if streams.empty():
                self.error.emit("Device has no DataStream!"); self._device=None; self._library_close_safe(); return
            self._datastream = streams[0].OpenDataStream()

            self._thread = QThread(self)
            self._worker = _StreamWorker(self._node_map, self._datastream)
            self._worker.moveToThread(self._thread)
            # bridge
            self._worker.started.connect(self._on_started)
            self._worker.stopped.connect(self._on_stopped)
            self._worker.closed.connect(self.closed)
            self._worker.frame.connect(self.frame)
            self._worker.error.connect(self.error)
            self._worker.roi.connect(self.roi)
            # Wire direct ROI mailbox signal (runs slot in GUI thread via DirectConnection)
            self.send_roi.connect(self._worker.enqueue_roi, Qt.DirectConnection)
            self._thread.start()

            self._st.open = True
            name = chosen.DisplayName() if hasattr(chosen, "DisplayName") else "IDS camera"
            print("[UI] opened")
            self.opened.emit(str(name))
            # fetch initial ROI
            QMetaObject.invokeMethod(self._worker, "query_roi", Qt.QueuedConnection)
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

    @Slot()
    def stop(self) -> None:
        print(f"[UI] stop() called; acquiring={self._st.acquiring}")
        if not self._worker:
            print("[UI] stop ignored: no worker")
            return
        # 1) Clear the run flag immediately in the GUI thread (DirectConnection)
        QMetaObject.invokeMethod(self._worker, "set_stop_flag", Qt.DirectConnection)
        # 2) Also ask the worker (Queued) to call CancelWait()/Stop if available
        QMetaObject.invokeMethod(self._worker, "request_stop", Qt.QueuedConnection)

    @Slot()
    def set_roi(self, w: float, h: float, x: float, y: float) -> None:
        if not self._worker:
            print("[UI] set_roi ignored: no worker")
            return
        print(f"[UI] set_roi(w={w}, h={h}, x={x}, y={y}); acquiring={self._st.acquiring}")
        payload = {"Width": w, "Height": h, "OffsetX": x, "OffsetY": y}
        # Drop into worker mailbox immediately via direct signal (safe: slot just locks and stores)
        self.send_roi.emit(payload)
        if not self._st.acquiring:
            # If not running, ask worker to process it now in worker thread
            QMetaObject.invokeMethod(self._worker, "process_pending_roi", Qt.QueuedConnection)

    @Slot()
    def refresh_roi(self) -> None:
        # Non‑disruptive: just read from camera without stopping
        if not self._worker:
            print("[UI] refresh_roi ignored: no worker")
            return
        print(f"[UI] refresh_roi() called (non‑disruptive); acquiring={self._st.acquiring}")
        QMetaObject.invokeMethod(self._worker, "query_roi", Qt.QueuedConnection)

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

    # ---- internals ----
    def _cleanup_all(self) -> None:
        try:
            if self._thread:
                self._thread.quit(); self._thread.wait(1000)
        except Exception:
            pass
        self._worker = None; self._thread = None
        self._datastream = None; self._remote = None; self._device = None
        self._library_close_safe(); self._st = _State(open=False, acquiring=False)
        print("[UI] cleanup complete")

    def _library_close_safe(self) -> None:
        try:
            if ids is not None: ids.Library.Close()
        except Exception:
            pass
