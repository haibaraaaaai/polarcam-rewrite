# PolarCam (rewrite) — lab README


## 0) TL;DR — run it

```bash
# 1) create & activate venv (Windows PowerShell shown)
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install -U pip

# 2) editable install
python -m pip install -e .

# 3) IDS SDK (if using IDS camera)
# Install the IDS Peak runtime/driver (GUI installer from IDS).
# Then the Python wheels:
python -m pip install --no-deps ids-peak ids-peak-ipl

# 4) launch (choose one)
polarcam          # console visible (logs)
polarcam-gui      # no console window on Windows
# or
python -m polarcam.cli
```

**First things to check in the UI**

* **Exposure is in milliseconds** (the textbox label says *Exposure (ms)*).
* **ROI**: “Full sensor” first; then use small ROIs for speed.
* **Gains**: hit **Refresh gains** to populate min/max; apply only if you actually set values.

---

## 1) What this app does today

* Live preview of the polarization camera (12→8‑bit highlight LUT with floor/cap/gamma, + histogram).
* ROI + timing controls (FPS + **exposure in ms**), desaturate helper.
* Analog/digital gains.
* **Spot detection** on a single frame (SciPy morphology; overlays + numbered labels).
* **Spot viewer**: shows a zoomed crop (UI capped ≤20 Hz), optional **recording at max camera FPS** (saves `.npy` shards).
* **Multi‑spot cycler/recorder**: hops a tight HW‑ROI around selected spots, records 4×‐pol mean signals per dwell to compressed `.npz` shards.
* **Variance map** tool: capture N frames → compute per‑pixel **intensity range** or **stddev** → preview + save `.npy` (and optional `.png`).

**Directory conventions**

* Cycle outputs land in `./cycles/cycle_spotXX/…npz`.
* Spot viewer recordings land in `./captures/…npy`.
* Variance‑map captures: `stack_*.npy` and `varmap_*.npy` beside each other (dialog prints paths).

---

## 2) IDS camera specifics we currently assume

Hardware constraints baked into the spot‑cycler (edit in code if your sensor differs):

* **Width**: min **256**, max **2464**, **step 4**
* **Height**: min **2**, max **2056**, **step 2**
* **OffsetX**: min **0**, max **2208**, **step 4**
* **OffsetY**: min **0**, max **2054**, **step 2**

The cycler picks the **smallest legal HW‑ROI** centered on the spot, based on spot radius, and re‑applies **`set_timing(inf, None)` after each ROI change** so FPS re‑maxes under the new ROI/exposure. Preview stays capped to reduce UI load; recording runs at full camera rate.

---

## 3) Typical workflows

### A) Live + detect + view

1. **Open → Start**.
2. **Full sensor**, then tweak exposure/gains or run **Desaturate**.
3. **Detect spots** (threshold mode: *absolute* DN or *percentile*; defaults are fine).
4. Select one or more spots → **View spot…** (separate window). This pauses the main live preview while the viewer runs.

   * **Start Rec @ max FPS** inside the viewer to save mean 0°/45°/90°/135° signals to `.npy` shards.

### B) Cycle multiple spots (record 4×pol means)

1. Detect/select spots.
2. **Start Cycle (1 s / spot)**.

   * Worker reduces HW‑ROI around each spot aggressively (respecting IDS step/min limits).
   * After each ROI change, worker calls **`set_timing(inf, None)`** to hit the maximum FPS permitted by ROI + exposure.
   * UI preview is throttled (≤20 Hz), while recording uses the max camera rate.
3. Outputs go to `./cycles/cycle_spotXX/*.npz`.

### C) Variance map (per‑pixel activity)

1. Click **Variance…**.
2. Choose number of frames and mode (**intensity_range** or **stddev**).
3. Optionally use memmap capture to keep RAM usage low.
4. Preview and save `.npy` (and `.png`).

---

## 4) Data formats

### Cycle `.npz` shards

Each file contains arrays: `t` (seconds from local `t0`), `c0`, `c45`, `c90`, `c135` (float64 per‑frame means), and a `meta` JSON blob (bytes) with:

* `spot = {cx, cy, r, area, inten}`
* `applied_roi = {x, y, w, h}` (HW‑ROI during dwell)
* `crop_abs = {x, y, w, h}` (software crop used to compute means)
* `t0_perf_counter`

### Spot viewer `.npy` shards

`
N×5 float32` arrays: columns = `[t, I0, I45, I90, I135]`.

### Variance outputs

* `stack_*.npy`: raw `(T,H,W)` `uint16` frames captured for the varmap.
* `varmap_*.npy`: `(H,W)` `uint16` per‑pixel map (range or stddev). Optional `varmap_*.png` preview.

---

## 5) Offline helpers (scripts)

Location: `polarcam/offline/`

### 5.1 `analyze_cycle_npz.py` — FPS sanity + dwell summary

Quickly summarizes cycle NPZ shards by dwell and spot. Prints per‑dwell sample counts, durations, mean FPS, and writes a CSV.

**Usage**

```bash
python -m polarcam.offline.analyze_cycle_npz cycles/cycle_spot01/*.npz
# or a folder
python -m polarcam.offline.analyze_cycle_npz cycles/cycle_spot01
```

### 5.2 `inspect_captures_npz.py` — explore viewer `.npy` shards

Loads spot‑viewer shards, checks gaps, basic stats, quick plots if you want them.

**Usage**

```bash
python -m polarcam.offline.inspect_captures_npz captures
```

### 5.3 `spot_from_stack.py` — test varmap & compute XY traces

Takes a saved stack `(T,H,W) uint16`, builds a highlight LUT view of frame 0, finds spots on the view, computes per‑spot normalized XY traces using the correct 2×2 polarization parity, and writes small per‑spot reports (PNGs + CSVs).

**Usage**

```bash
python -m polarcam.offline.spot_from_stack \
  path/to/stack_2025-11-xx.npy \
  --floor 1200 --cap 4095 --gamma 0.6 \
  --thr_pct 99.7 --min_area 8 --max_area 5000 --grow 1
```

Outputs live under `spot_from_stack_YYYYMMDD-HHMMSS/`.

---

## 6) What’s next / near‑term plan

* **UI polish**: clearer state (started/paused), progress bars, cancellation that never blocks the UI.
* **Spot viewer analysis**: add **FFT/peak‑finder** and **anisotropy (x,y)** readouts live; optional export of XY traces.
* **Variance map**: tune and benchmark on gold beads; add a tiny quantile‑range mode if useful.
* **Detection**: expose dedup radius; better debug overlay (exact frame used, mask edges).
* **Cycler**: smarter dwell timing; optional per‑spot exposure set; robust shut‑down; better logs.
* **Threading**: keep UI timers in the GUI thread only; all camera/frame work stays in workers with `Qt.DirectConnection` or explicit queues.
* **Packaging**: PyInstaller spec for a one‑folder app with IDS DLLs bundled.

---

## 7) Known gotchas

* If you see *“QObject::startTimer / killTimer from another thread”*, it’s almost always a timer created or stopped outside the GUI thread. We avoid `QApplication.processEvents()` in worker loops and keep timer usage strictly on the main thread.
* Exposure textbox is **milliseconds**. `set_timing(inf, None)` lets the camera choose the **max FPS** for the current ROI + exposure; changing ROI or exposure changes that ceiling.
* ROI snapping: hardware enforces the step/min/max above; what you ask for isn’t always what you get (watch the status bar or ROI fields for the applied values).

---

## 8) Contact / notes to self

* Core analysis defaults: alias guard (fps/4), phase‑binned A26 as occupancy cue, HMM/change‑points for N(t), Allan variance for timescales.
* Polarization layout reminder: (0,0)=90°, (0,1)=45°, (1,0)=135°, (1,1)=0°.

(End)
