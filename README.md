# PolarCam (rewrite) — lab README


## 0) TL;DR — run it

```bash
# 1) create & activate venv
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install -U pip

# 2) editable install
python -m pip install -e .

# 3) IDS SDK
# Install the IDS peak Cockpit.
# Then the Python wheels:
python -m pip install --no-deps ids-peak ids-peak-ipl

# 4) launch (choose one)
polarcam          # console visible (logs)
polarcam-gui      # no console window on Windows
# or
python -m polarcam.cli
```

**First things to check in the UI**

* **Exposure is in milliseconds**
* **ROI**: "Full sensor" to set max ROI; then use small ROIs for speed.
* **Gains**: hit **Refresh gains** to populate min/max if not already shown.

---

## 1) What this app does today

* Live preview of the polarization camera (12→8‑bit highlight LUT with floor/cap/gamma).
* ROI + timing controls (FPS + **exposure in ms**), desaturate helper.
* Analog/digital gains.
* **Spot detection** on a single frame (DoG‑based, with two‑stage φ‑coverage / r‑uniformity classification).
* **Spot viewer**: shows a zoomed crop with circular ROI mask, live 4‑channel anisotropy scatter plot, optional **recording at max camera FPS** (saves raw pixel `.npy` shards + JSON metadata with mosaic layout info).
* **Manual add‑spot** with live preview circle (cyan dashed) shown before committing.
* **Multi‑spot cycler/recorder**: hops a tight HW‑ROI around selected spots with configurable dwell time, records 4×pol mean signals per dwell to compressed `.npz` shards. Optional **raw pixel saving** (one uncompressed `.npz` + companion `.json` per spot per dwell with full mosaic layout metadata).
* **Test (AVI)** button: loads an AVI file as a mock camera for offline testing.

**Directory conventions**

* Cycle outputs land in `./cycles/cycle_spotXX/…npz` (and `…_raw.npz` + `…_raw.json` if raw pixel saving is enabled).
* Spot viewer recordings land in `./captures/…npy` + `…_meta.json`.

---

## 2) Typical workflows

### A) Live + detect + view

1. **Open → Start**.
2. **Full sensor**, then tweak exposure/gains or run **Desaturate**.
3. **Detect spots** (threshold mode: *absolute* DN or *percentile*; defaults are fine).
4. Select one or more spots → **View spot…** (separate window with scatter plot). This pauses the main live preview while the viewer runs.

   * **Start Rec @ max FPS** inside the viewer to save raw pixel crops to `.npy` shards + a `_meta.json` with crop position, channel layout, and ROI details.

### B) Cycle multiple spots

1. Detect/select spots (or add manually with cx/cy/r fields — preview circle shows before adding).
2. Set **dwell time** (default 1.0 s), check **Save cycle data** and/or **Save raw pixels** as needed.
3. Click **Start Cycle**.

   * Worker reduces HW‑ROI around each spot aggressively (respecting IDS step/min limits).
   * After each ROI change, worker calls **`set_timing(inf, None)`** to hit the maximum FPS permitted by ROI + exposure.
   * Circular mask is applied — only pixels within the spot radius contribute to channel means.
   * UI preview is throttled (≤20 Hz), while recording uses the max camera rate.
4. Outputs go to `./cycles/cycle_spotXX/`:
   * `*_sNN_cNNNN.npz` — per‑chunk 4‑channel mean signals (if "Save cycle data" is checked).
   * `*_sNN_dNNNN_raw.npz` + `*_sNN_dNNNN_raw.json` — raw pixel stacks per dwell (if "Save raw pixels" is checked).

---

## 3) Data formats

### Cycle `.npz` shards (channel means)

Each file contains arrays: `t` (seconds from local `t0`), `c0`, `c45`, `c90`, `c135` (float64 per‑frame means), and a `meta` JSON blob (bytes) with:

* `spot = {cx, cy, r, label}`
* `applied_roi = {x, y, w, h}` (HW‑ROI during dwell)
* `crop_abs = {x, y, w, h}` (software crop used to compute means)
* `t0_perf_counter`

### Cycle raw pixel `.npz` + `.json` (per spot per dwell)

The `.npz` contains `frames` `(N, h, w)` uint16 stack, `t` (relative timestamps), and `meta` (JSON as byte array). The companion `.json` has the same metadata in a human‑readable file:

* `spot = {cx, cy, r, label}`
* `layout = {"(0,0)": "90", "(0,1)": "45", "(1,0)": "135", "(1,1)": "0"}`
* `crop_top_left_sensor_yx = [y, x]` — position of the crop's top‑left pixel in full‑sensor coordinates
* `crop_top_left_channel` — polarization angle of that top‑left pixel (e.g. `"90"`)
* `applied_roi`, `crop_abs`, `t0_perf_counter`, `n_frames`, `n_discarded`

### Spot viewer `.npy` shards + `_meta.json`

`.npy` shards contain raw pixel crops `(chunk_len, h, w)` uint16. The `_meta.json` records:

* `spot`, `layout`, `crop_top_left_sensor_yx`, `crop_top_left_channel`
* `roi_hw_requested`, `crop_hw`, `crop_offset_in_roi`

---

## 4) Offline helpers

### `offline/merge_recording.py`

Merges chunked full‑frame `.npy` recordings into a single file.

### `offline/merge_spot_capture.py`

Merges spot‑viewer `.npy` shards for a given capture into a single contiguous array.

---

## 5) Polarization mosaic layout

The sensor uses a 2×2 super‑pixel layout indexed by `(row % 2, col % 2)`:

| (row%2, col%2) | Angle |
|---|---|
| (0, 0) | 90° |
| (0, 1) | 45° |
| (1, 0) | 135° |
| (1, 1) | 0° |

All saved metadata references (`crop_top_left_channel`, `layout` dicts) follow this convention.

---

## 6) Project structure

```
src/polarcam/
├── __init__.py          # Package bootstrap, logging setup
├── __main__.py          # python -m polarcam entry point
├── cli.py               # QApplication launcher, arg parsing
├── hardware.py          # Shared sensor constants, mosaic layout, snap helpers
├── analysis/
│   ├── detect.py        # DoG spot detection
│   ├── pol_reconstruction.py  # (X,Y) anisotropy + (Q,U) reconstructors
│   └── smap.py          # S-map accumulator (per-pixel min/max of Q/U)
├── app/
│   ├── main_window.py   # Main GUI window
│   ├── lut_widget.py    # 12-bit highlight LUT widget
│   ├── spot_detect.py   # Detection + classification orchestrator
│   ├── spot_viewer.py   # Per-spot viewer with scatter plot + recorder
│   ├── spot_cycler.py   # Multi-spot cycling recorder
│   └── spot_recorder.py # Per-spot raw pixel shard recorder
├── backend/
│   ├── base.py          # Abstract camera interface (ICamera)
│   ├── ids_backend.py   # IDS peak camera backend
│   └── mock_backend.py  # AVI-based mock camera for testing
├── capture/
│   └── frame_writer.py  # Full-frame chunked writer with background I/O
└── controller/
    └── controller.py    # Thin controller over IDSCamera
```

---

## 7) Known gotchas

* **ROI snapping**: hardware enforces step/min/max; snapping to a spot can over/under cut it.
* **Dwell/save settings are snapshot**: toggling checkboxes mid‑cycle has no effect; values are read when you click Start Cycle.
* **Transition frames**: when the HW‑ROI hops between spots the first 2–3 frames may have the wrong crop dimensions. The cycler discards these automatically (reported as `n_discarded` in raw metadata).

---

## 8) Notes

* Alias guard: fps/4.
* Hardware: Sensor 2464×2056, ROI width step 4, height step 2, min width 256, min height 2.
* Polarization layout: `(0,0)=90°`, `(0,1)=45°`, `(1,0)=135°`, `(1,1)=0°`.
