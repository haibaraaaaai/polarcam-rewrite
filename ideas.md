# PolarCam — TODO / roadmap

> Phases 1–4 complete. Below is what's left to revisit.

---

## Analysis / reconstruction (offline first, then live)

- [ ] **Background / flat-field correction** — subtract dark frame and divide by flat-field before channel extraction. Acquire calibration frames (lens-cap dark, uniform illumination flat) and store as `.npy`. Apply correction in `pol_reconstruction` or a new `calibration.py` module.
- [ ] **θ and φ reconstruction (Fourkas)** — implement the Fourkas formulae to extract polar angle θ and azimuthal angle φ from the four polarization channels. Start as an offline script on saved raw-pixel data, then integrate into the live spot viewer once validated.
- [ ] **PSD-based θ estimation** — compute per-spot power spectral density of the anisotropy signal; extract θ from the modulation depth / DC ratio. Useful as a cross-check against Fourkas.

## Code quality

- [ ] **Split `main_window.py` into mixins** — at ~1300 lines it handles video, detection, recording, cycling, and spot management. Extract into focused mixins or helper classes (e.g. `_CycleMixin`, `_DetectMixin`, `_RecordMixin`) that the main window composes.
- [ ] **Deduplicate ROI-for-spot logic** — `spot_cycler._hw_roi_request_for_spot`, `spot_recorder._roi_for_spot`, and `spot_viewer._set_spot_roi` are near-identical. Extract a shared `roi_for_spot(cx, cy, r)` into `hardware.py`.
- [ ] **Unify `Spot` vs `DetectedSpot`** — `spot_recorder.Spot(cx, cy, r, area, inten)` and `spot_detect.DetectedSpot(cx, cy, r, label, phi_cov, std_median_r)` represent the same concept with different fields. Consider a single dataclass.

## UX / live features

- [ ] **XY-range auto-prune** — after each dwell, check if a spot shows zero anisotropy movement and optionally drop it from the cycle list.
- [ ] **Live FFT / peak-finder** — add a frequency-domain readout to the spot viewer scatter plot.
- [ ] **Per-spot exposure in cycler** — allow different exposure times per spot during cycling (useful when spots have very different brightness).
- [ ] **Clearer state indicators** — started/paused/recording badges in the toolbar; progress bar for long operations.

## Infrastructure

- [ ] **PyInstaller packaging** — one-folder app bundle with IDS DLLs included.
- [ ] **Condition.wait() timeouts** — the shard-writer threads in `frame_writer.py` and `spot_recorder.py` use `Condition.wait()` with no timeout; add a timeout so threads don't hang on crash.
- [ ] **Mock backend ROI signals** — `AviMockCamera` silently no-ops ROI/timing calls; the cycler's `_applied_roi` never updates. Emit synthetic `roi` signals so cycling works in test mode.
