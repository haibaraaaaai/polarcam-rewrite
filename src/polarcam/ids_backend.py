# ------------------------------------------
# File: src/polarcam/ids_backend.py
# (Step 2 — hardened buffers; Mono12 fast path; PF switch)
# ------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Dict, List, Optional, Tuple, Union

from PySide6.QtCore import QObject, QThread, Signal, Slot, QMetaObject, Qt

# ---- Guarded IDS imports ----
try:
    import ids_peak.ids_peak as ids
    from ids_peak import ids_peak_ipl_extension as ipl_ext
    _IDS_IMPORT_ERROR: Optional[str] = None
except Exception as e:  # ImportError / DLL load error
    ids = None  # type: ignore
    ipl_ext = None  # type: ignore
    _IDS_IMPORT_ERROR = str(e)

try:
    import ids_peak_ipl.ids_peak_ipl as ipl
except Exception:
    try:
        import ids_peak_ipl as ipl  # type: ignore
    except Exception:
        ipl = None  # type: ignore


@dataclass
class _State:
    open: bool = False
    acquiring: bool = False
    payload: int = 0
    min_bufs: int = 6


class _StreamWorker(QObject):
    started = Signal()
    stopped = Signal()
    closed = Signal()
    frame = Signal(object)            # numpy array (mono)
    error = Signal(str)
    parameters_updated = Signal(dict) # {name: {min,max,current,increment}, PixelFormat: {current:str}}
    pixel_format_changed = Signal(str)

    def __init__(self, node_map, datastream, payload: int, min_bufs: int) -> None:
        super().__init__()
        self._nm = node_map
        self._ds = datastream
        self._payload = int(payload)
        self._min_bufs = max(6, int(min_bufs))
        self._announced: List[object] = []
        self._run = Event()          # loop control
        self._stopped_evt = Event()  # signals stop complete
        self._closed = False

    # ---------- helpers ----------
    @Slot()
    def request_stop(self) -> None:
        self._run.clear()

    @Slot()
    def stop(self) -> None:
        """Force-stop acquisition quickly so WaitForFinishedBuffer unblocks, then wait."""
        # Signal the loop to exit
        self._run.clear()
        # Proactively stop the stream to break any pending waits fast
        try:
            n = self._nm.FindNode("AcquisitionStop"); n and n.Execute()
        except Exception:
            pass
        try:
            # Some stacks have CancelWait to interrupt a blocking WaitForFinishedBuffer
            cancel = getattr(self._ds, "CancelWait", None)
            if callable(cancel):
                try: cancel()
                except Exception: pass
        except Exception:
            pass
        try:
            self._ds.StopAcquisition()
        except Exception:
            pass
        # Give the finalize-stop path up to 1.5s to complete
        self._stopped_evt.wait(1.5)

    def _img_dims(self, img) -> Tuple[int,int]:
        try: return int(img.Width()), int(img.Height())
        except Exception: pass
        try:
            sz = img.Size(); w = getattr(sz, "width", None); h = getattr(sz, "height", None)
            if w is not None and h is not None: return int(w), int(h)
            W = getattr(sz, "Width", None); H = getattr(sz, "Height", None)
            if callable(W) and callable(H): return int(W()), int(H())
            if W is not None and H is not None: return int(W), int(H)
        except Exception: pass
        raise RuntimeError("Cannot obtain image size from IDS IPL image")

    def _announce_and_queue(self) -> bool:
        """(Re)build the buffer pool and queue it. Returns True on success.
        Steps:
          - Flush & revoke any previous buffers
          - Re-read required buffer count and payload size
          - Try a healthy pool; fall back to the minimum if needed
        """
        try:
            # 1) Clear any previous queue/pool completely
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

            # 2) Read current requirements & payload
            try:
                required = int(self._ds.NumBuffersAnnouncedMinRequired())
            except Exception:
                required = 6
            required = max(required, 1)

            try:
                self._payload = int(self._nm.FindNode("PayloadSize").Value())
            except Exception:
                self._payload = 0
            if self._payload <= 0:
                self.error.emit("Prepare failed: PayloadSize is 0")
                return False

            # 3) Announce buffers (try bigger, fall back to minimum)
            target = max(required * 2, 8)
            announced = 0
            last_err = None
            for nbuf in (target, required + 2, required):
                announced = 0; last_err = None
                try:
                    for _ in range(nbuf):
                        buf = self._ds.AllocAndAnnounceBuffer(self._payload)
                        self._announced.append(buf); announced += 1
                    break
                except Exception as e:
                    last_err = e
                    # Clean partial attempts
                    try:
                        for b in self._announced:
                            try: self._ds.RevokeBuffer(b)
                            except Exception: pass
                    except Exception:
                        pass
                    self._announced = []
                    continue

            if announced < required:
                self.error.emit(f"Prepare failed: only announced {announced} < required {required}. Last error: {last_err}")
                try: print(f"[BufferPool] announce failed: {announced}<{required} payload={self._payload} err={last_err}")
                except Exception: pass
                return False

            # 4) Queue everything
            try:
                for b in self._announced:
                    self._ds.QueueBuffer(b)
            except Exception as e:
                self.error.emit(f"Prepare failed while queuing: {e}")
                try: print("[BufferPool] queue exception:", e)
                except Exception: pass
                return False

            # 5) Diagnostics
            try:
                print(f"[BufferPool] announced={len(self._announced)} required>={required} payload={self._payload}")
            except Exception:
                pass
            return True
        except Exception as e:
            self.error.emit(f"Prepare failed: {e}")
            try: print("[BufferPool] prepare exception:", e)
            except Exception: pass
            return False

    def _get_node(self, name: str):
        try: return self._nm.FindNode(name)
        except Exception: return None

    def _node_value(self, name: str) -> Optional[Union[int,float,str]]:
        n = self._get_node(name)
        if n is None: return None
        # Enum-like nodes may have .CurrentEntry().SymbolicValue()
        for attr in ("CurrentEntry", "GetCurrentEntry"):
            try:
                e = getattr(n, attr)()
                for a2 in ("SymbolicValue", "GetSymbolicValue", "Value", "GetValue"):
                    try: return getattr(e, a2)()
                    except Exception: pass
            except Exception:
                pass
        # Numeric value
        for attr in ("Value", "GetValue", "value"):
            try:
                v = getattr(n, attr)()
                return float(v) if isinstance(v, (int,float)) else v
            except Exception:
                pass
        return None

    def _node_minmaxinc(self, name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        n = self._get_node(name)
        if n is None: return None, None, None
        def _try(names):
            for a in names:
                try:
                    v = getattr(n, a)()
                    return float(v)
                except Exception:
                    pass
            return None
        mn = _try(["Minimum", "GetMinimum", "Min", "GetMin", "GetLower"])
        mx = _try(["Maximum", "GetMaximum", "Max", "GetMax", "GetUpper"])
        inc = _try(["Increment", "GetIncrement", "GetInc", "Step", "GetStep"])
        return mn, mx, inc

    def _snap(self, name: str, value: Union[int,float]) -> float:
        mn, mx, inc = self._node_minmaxinc(name)
        if mn is not None: value = max(value, mn)
        if mx is not None: value = min(value, mx)
        if inc and inc > 0:
            base = mn if mn is not None else 0.0
            steps = round((value - base) / inc)
            value = base + steps * inc
        return float(value)

    def _fetch_parameters(self) -> Dict[str,Dict[str,Optional[float]]]:
        names = [
            "AcquisitionFrameRate",
            "ExposureTime",
            "Width", "Height",
            "OffsetX", "OffsetY",
            "AnalogGain", "DigitalGain",
        ]
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for name in names:
            cur = self._node_value(name)
            mn, mx, inc = self._node_minmaxinc(name)
            out[name] = {"min": mn, "max": mx, "current": cur, "increment": inc}
        # add pixel format
        pf = self._node_value("PixelFormat")
        out["PixelFormat"] = {"current": pf}
        return out

    # ---------- slots (worker thread) ----------
    @Slot()
    def prepare(self) -> None:
        if self._closed:
            return
        # Do NOT pre-announce buffers on open; just report parameters.
        self.parameters_updated.emit(self._fetch_parameters())

    @Slot()
    def start(self) -> None:
        if self._closed or self._run.is_set():
            return
        try:
            # Always (re)build buffer pool; bail if it fails
            if not self._announce_and_queue():
                self.error.emit("Start failed: buffer pool could not be prepared")
                return

            try:
                n = self._nm.FindNode("TLParamsLocked"); n and n.SetValue(1)
            except Exception:
                pass

            self._ds.StartAcquisition()
            self._nm.FindNode("AcquisitionStart").Execute()
            self._stopped_evt.clear()
            self._run.set()
            self.started.emit()

            while bool(self._run.is_set()):
                buf = None
                try:
                    buf = self._ds.WaitForFinishedBuffer(50)
                except Exception:
                    continue
                try:
                    img = ipl_ext.BufferToImage(buf)
                    # --- Mono12-only fast path (no ConvertTo, no RGB) ---
                    if hasattr(img, "get_numpy_1D"):
                        w, h = self._img_dims(img)
                        flat = img.get_numpy_1D()
                        import numpy as _np
                        bufarr = _np.asarray(flat)
                        if bufarr.ndim != 1:
                            bufarr = bufarr.reshape(-1)
                        expected = w * h
                        if bufarr.size == 2 * expected:
                            # 12-bit stored in 16-bit container -> reinterpret bytes as <u2
                            if not bufarr.flags.c_contiguous:
                                bufarr = _np.ascontiguousarray(bufarr)
                            arr = bufarr.view('<u2').reshape(h, w).copy()
                        else:
                            # Fallback (e.g., Mono8)
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
            try:
                self._nm.FindNode("AcquisitionStop").Execute()
            except Exception:
                pass
            try:
                self._ds.StopAcquisition()
            except Exception:
                pass
            # Flush & revoke so next Start begins clean
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
            self._announced.clear()
            try:
                n = self._nm.FindNode("TLParamsLocked"); n and n.SetValue(0)
            except Exception:
                pass
            self._stopped_evt.set()
            self.stopped.emit()
        except Exception as e:
            self.error.emit(f"Start failed: {e}")

    @Slot(dict)
    def apply_parameters(self, updates: Dict[str, Dict[str, Union[int,float]]]) -> None:
        try:
            was_running = self._run.is_set()
            if was_running:
                self.request_stop(); self._stopped_evt.wait(1.0)
            roi_keys = {"Width", "Height", "OffsetX", "OffsetY"}
            if any(k in updates for k in roi_keys):
                # Step 0: optionally set offsets to min to widen ROI if needed
                mn_x, _, _ = self._node_minmaxinc("OffsetX"); mn_y, _, _ = self._node_minmaxinc("OffsetY")
                if mn_x is not None: self._node_set("OffsetX", mn_x)
                if mn_y is not None: self._node_set("OffsetY", mn_y)
                # Step 1: width/height first
                for name in ("Width", "Height"):
                    if name in updates:
                        v = float(updates[name]["current"])  # type: ignore[index]
                        self._node_set(name, self._snap(name, v))
                # Step 2: offsets
                for name in ("OffsetX", "OffsetY"):
                    if name in updates:
                        v = float(updates[name]["current"])  # type: ignore[index]
                        self._node_set(name, self._snap(name, v))
                # Step 3: ROI changes alter payload -> refresh and rebuild pool
                try:
                    self._payload = int(self._nm.FindNode("PayloadSize").Value())
                except Exception:
                    pass
                self._announce_and_queue()
            for name in ("ExposureTime", "AcquisitionFrameRate", "AnalogGain", "DigitalGain"):
                if name in updates:
                    v = float(updates[name]["current"])  # type: ignore[index]
                    if name in ("AnalogGain", "DigitalGain"):
                        try:
                            sel = self._get_node("GainSelector")
                            if sel is not None:
                                sel_name = "AnalogAll" if name == "AnalogGain" else "DigitalAll"
                                for setter in ("SetCurrentEntry", "SetValue"):
                                    try:
                                        getattr(sel, setter)(sel_name); break
                                    except Exception: pass
                        except Exception: pass
                    self._node_set(name, self._snap(name, v))
            self.parameters_updated.emit(self._fetch_parameters())
            if was_running:
                self.start()
        except Exception as e:
            self.error.emit(f"Parameter update failed: {e}")

    @Slot(str)
    def apply_pixel_format(self, pf_name: str) -> None:
        """Safely switch camera PixelFormat (Mono8/10/12/10p/12p).
        Stops if needed, sets node, updates PayloadSize, re-announces buffers,
        refreshes bounds, restarts if previously running.
        """
        try:
            was_running = self._run.is_set()
            if was_running:
                self.request_stop(); self._stopped_evt.wait(1.0)
            # Set PixelFormat
            if not self._node_set("PixelFormat", pf_name):
                self.error.emit(f"Setting PixelFormat to '{pf_name}' failed.")
                return
            # Update payload size
            try:
                self._payload = int(self._nm.FindNode("PayloadSize").Value())
            except Exception:
                pass
            # Rebuild buffer pool (check success)
            if not self._announce_and_queue():
                self.error.emit("PixelFormat switch failed: buffer pool could not be prepared")
                return
            self.pixel_format_changed.emit(str(pf_name))
            self.parameters_updated.emit(self._fetch_parameters())
            if was_running:
                self.start()
        except Exception as e:
            self.error.emit(f"PixelFormat switch failed: {e}")

    @Slot()
    def refresh_parameters(self) -> None:
        self.parameters_updated.emit(self._fetch_parameters())

    @Slot()
    def close(self) -> None:
        if self._closed: return
        self._run.clear()
        if not self._stopped_evt.is_set():
            try: self._nm.FindNode("AcquisitionStop").Execute()
            except Exception: pass
            try: self._ds.StopAcquisition()
            except Exception: pass
            self._stopped_evt.wait(1.0)
        try:
            try: self._ds.Flush(ids.DataStreamFlushMode_DiscardAll)
            except Exception: pass
            for b in self._announced:
                try: self._ds.RevokeBuffer(b)
                except Exception: pass
            self._announced.clear(); self._closed = True; self.closed.emit()
        except Exception as e:
            self.error.emit(f"Close failed: {e}")


class IDSCamera(QObject):
    opened = Signal(str)
    started = Signal()
    stopped = Signal()
    closed = Signal()
    frame = Signal(object)
    error = Signal(str)
    parameters_updated = Signal(dict)
    pixel_format_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._st = _State()
        self._device = None
        self._remote = None
        self._datastream = None
        self._node_map = None
        self._worker: Optional[_StreamWorker] = None
        self._thread: Optional[QThread] = None

    @Slot()
    def open(self) -> None:
        if self._st.open: return
        if _IDS_IMPORT_ERROR is not None:
            self.error.emit(f"ids_peak import failed: {_IDS_IMPORT_ERROR}")
            return
        try:
            ids.Library.Initialize(); dm = ids.DeviceManager.Instance(); dm.Update()
            devs = dm.Devices()
            if devs.empty(): self.error.emit("No device found!"); self._library_close_safe(); return
            chosen = next((d for d in devs if d.IsOpenable()), None)
            if chosen is None: self.error.emit("Device could not be opened!"); self._library_close_safe(); return
            try: self._device = chosen.OpenDevice(ids.DeviceAccessType_Exclusive)
            except Exception: self._device = chosen.OpenDevice(ids.DeviceAccessType_Control)
            self._remote = self._device.RemoteDevice(); self._node_map = self._remote.NodeMaps()[0]
            streams = self._device.DataStreams()
            if streams.empty(): self.error.emit("Device has no DataStream!"); self._device=None; self._library_close_safe(); return
            self._datastream = streams[0].OpenDataStream()
            try: self._st.payload = int(self._node_map.FindNode("PayloadSize").Value())
            except Exception: self._st.payload = 0
            try: self._st.min_bufs = max(6, int(self._datastream.NumBuffersAnnouncedMinRequired()))
            except Exception: self._st.min_bufs = 6
            self._thread = QThread(self)
            self._worker = _StreamWorker(self._node_map, self._datastream, self._st.payload, self._st.min_bufs)
            self._worker.moveToThread(self._thread)
            self._worker.started.connect(self.started)
            self._worker.stopped.connect(self.stopped)
            self._worker.closed.connect(self.closed)
            self._worker.frame.connect(self.frame)
            self._worker.error.connect(self.error)
            self._worker.parameters_updated.connect(self.parameters_updated)
            self._worker.pixel_format_changed.connect(self.pixel_format_changed)
            self._thread.start()
            # Only emit parameters on open; buffers are built on Start
            QMetaObject.invokeMethod(self._worker, "prepare", Qt.QueuedConnection)
            self._st.open = True
            name = chosen.DisplayName() if hasattr(chosen, "DisplayName") else "IDS camera"
            self.opened.emit(str(name))
        except Exception as e:
            self.error.emit(f"Failed to open device: {e}")
            self._cleanup_all()

    @Slot()
    def start(self) -> None:
        if not self._st.open or self._st.acquiring or not self._worker: return
        self._st.acquiring = True
        QMetaObject.invokeMethod(self._worker, "start", Qt.QueuedConnection)

    @Slot()
    def stop(self) -> None:
        if not self._st.acquiring or not self._worker:
            return
        # Force-stop on the worker and block briefly so the stream actually halts
        QMetaObject.invokeMethod(self._worker, "stop", Qt.BlockingQueuedConnection)
        self._st.acquiring = False

    @Slot()
    def close(self) -> None:
        if not self._st.open: return
        if self._st.acquiring:
            self.error.emit("Close is disabled during acquisition. Press Stop first.")
            return
        if self._worker:
            QMetaObject.invokeMethod(self._worker, "close", Qt.BlockingQueuedConnection)
        self._cleanup_all()

    # ---- parameter & PF wrappers ----
    @Slot(dict)
    def set_parameters(self, updates: Dict[str, Dict[str, Union[int,float]]]) -> None:
        if not self._worker: return
        self._worker.apply_parameters(updates)  # marshalled by Qt (slot)

    @Slot()
    def refresh_parameters(self) -> None:
        if not self._worker: return
        QMetaObject.invokeMethod(self._worker, "refresh_parameters", Qt.QueuedConnection)

    @Slot(str)
    def set_pixel_format(self, pf_name: str) -> None:
        if not self._worker: return
        self._worker.apply_pixel_format(pf_name)  # marshalled by Qt (slot)

    # ---- internals ----
    def _cleanup_all(self) -> None:
        try:
            if self._thread:
                self._thread.quit(); self._thread.wait(1000)
        except Exception: pass
        self._worker = None; self._thread = None
        self._datastream = None; self._remote = None; self._device = None
        self._library_close_safe(); self._st = _State()

    def _library_close_safe(self) -> None:
        try:
            if ids is not None: ids.Library.Close()
        except Exception: pass


