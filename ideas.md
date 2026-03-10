Here's a clean checklist based on everything we've discussed. I've grouped them into logical phases so you can work top-to-bottom.

---

## Polarcam Rewrite — Work Checklist

### Phase 1 — Spot Detection
- [ ] **1.1** Replace `spot_detect.py` (single-frame intensity threshold) with S-map based detection from Hugh's approach
- [ ] **1.2** Add `pol_reconstruction.py` — the intersection-grid `make_xy_reconstructor` / `make_qu_reconstructor` (only needed for detection, lives in `analysis/`)
- [ ] **1.3** Add `analysis/smap.py` — S-map computation: per-frame smooth x,y → track min/max → `S = range(x)² + range(y)²`
- [ ] **1.4** Add `analysis/detect.py` — DoG + connected components on S-map (extracted from Hugh's `detect_spinners.py`)
- [ ] **1.5** Wire detection into the GUI — expose tunable parameters (k_std, min/max area, N frames) as GUI controls

### Phase 2 — Data Saving
- [ ] **2.1** Remove `capture/varmap_capture.py` and `analysis/varmap.py` — not needed
- [ ] **2.2** Investigate IDS camera native save commands (check `ids_peak` API for hardware-triggered recording)
- [ ] **2.3** If no native save: implement a simple frame-to-disk writer (memmap or chunked `.npy`) — stripped down version of what `fetch_frames.py` does but as a proper `QObject` subscriber like `_ShardWriter` in `spot_recorder.py`

### Phase 3 — Per-Spot Analysis Display
- [ ] **3.1** Per-spot XY time series extraction — window mean per confirmed spot, per frame (as Hugh does in `_append_xy_from_frame`)
- [ ] **3.2** Compute φ(t) = ½ arctan2(y, x) and r(t) = √(x²+y²) per spot from time series
- [ ] **3.3** Build compact spot analysis panel — small inline display (not full-screen takeover), shows XY scatter + φ(t) trace at minimum
- [ ] **3.4** Hook panel into spot cycler — update display as cycler dwells on each spot

### Phase 4 — Spot Cycler Improvements
- [ ] **4.1** Manual spot add — click on live view to add a spot by hand
- [ ] **4.2** Manual spot remove — select and delete a spot from the list
- [ ] **4.3** Auto-inspect trigger — after S-map detection, automatically run a short high-fps burst per confirmed spot (like Hugh's `_auto_inspect_top_spots`)
- [ ] **4.4** XY range filter — post-dwell filter to drop confirmed-dead spots (no anisotropy movement) from the cycle list

### Phase 5 — Deferred / Optional
- [ ] **5.1** Background / flat-field correction — skip for now, revisit later
- [ ] **5.2** θ reconstruction from r — add Fourkas formula once φ pipeline is validated
- [ ] **5.3** PSD-based directionality (Hugh's `DIR_*` parameters) — revisit after basic φ works

---

Things to note before starting:

- **Phase 1 is a prerequisite for everything else** — until you have the S-map and DoG working, the spot list that the cycler and analysis panel depend on doesn't exist.
- **Phase 2 can be done independently** at any point, it doesn't block other phases.
- **Phase 3 and 4 are interleaved** — the display panel and the cycler improvements should be built together since they share the per-spot time series data.
