python -m pip install --upgrade pip wheel setuptools

most camera apps default to fit-to-window with aspect ratio preserved, offer a 1:1 (pixel-perfect) toggle, and allow wheel zoom + drag-to-pan. We can add those once the feed works.

Check Component Selector is Raw
Component Enable True
Source pixel format Mono12


adding back gain
remove intensity floor
adjust morphology behavior


src/polarcam/
│
├─ app/                      # only Qt widgets + glue to controller
│   ├─ main_window.py        # docks, video view, menus, status
│   ├─ video_view.py         # QImage/QPixmap rendering, overlays
│   ├─ spot_viewer.py        # zoomed ROI + live spectra widget
│   └─ styles.py             # fonts/colors/icons (optional)
│
├─ controller/               # orchestration, no heavy compute
│   └─ controller.py         # the one object GUI talks to
│
├─ backend/                  # camera implementations
│   ├─ base.py               # ICamera interface + dataclasses  ##
│   └─ ids_backend.py        # real IDS (worker thread)
│
├─ services/                 # long-running workers using frames
│   ├─ auto_desat.py         # exposure control state machine
│   ├─ detector.py           # runs chosen detection pipeline
│   ├─ recorder.py           # per-spot recorder
│   └─ round_robin.py        # scheduler: hop ROI across spots
│
├─ pipelines/                # pure numpy/scikit funcs, stateless
│   ├─ demosaic.py           # split 2×2 pol layout → I0/I45/I90/I135
│   ├─ variance_map.py       # Ix/Iy range magnitude over window
│   ├─ motion_diff.py        # simple frame-diff detector
│   ├─ morphology.py         # open/close/remove_small, params
│   ├─ labeling.py           # label + props → Spot list
│   ├─ spectra.py            # FFT, peak finding, power plots
│   └─ colormaps.py          # AoP/DoLP visualization (optional)
│
├─ models/                   # dataclasses shared everywhere
│   └─ types.py              # Frame, ROI, Timing, Spot, Params…
│
├─ utils/
│   ├─ roi_snap.py           # enforce sensor increments
│   ├─ ringbuffer.py         # lock-free fixed buffers
│   ├─ timing.py             # monotonic clocks, fps estimates
│   └─ log.py                # structured logger
│
└─ cli.py                    # entrypoint (creates MainWindow)  ##



need for varmap, some sort of background substraction since it just seem to mirror hte base intensity of the image a bit (so something bright will just be somewhat bright in the varmap compare to dimmer stuff, when both not rotating)