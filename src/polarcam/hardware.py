"""
Shared hardware constants and small helpers for the IDS polarization camera.

Polar mosaic layout (row%2, col%2):
    (0,0) = 90°    (0,1) = 45°
    (1,0) = 135°   (1,1) = 0°
"""
from __future__ import annotations

# ---- Sensor geometry ----
SENSOR_W, SENSOR_H = 2464, 2056

# ---- ROI step / min / max ----
STEP_W, STEP_H     = 4, 2
MIN_W,  MIN_H      = 256, 2
MAX_W,  MAX_H      = SENSOR_W, SENSOR_H

# ---- Offset constraints ----
OFFX_MIN, OFFX_MAX = 0, 2208       # SENSOR_W - MIN_W
OFFY_MIN, OFFY_MAX = 0, 2054       # SENSOR_H - MIN_H
OFFX_STEP, OFFY_STEP = 4, 2

# ---- Polarization mosaic layout ----
MOSAIC_LAYOUT: dict[tuple[int, int], str] = {
    (0, 0): "90",
    (0, 1): "45",
    (1, 0): "135",
    (1, 1): "0",
}

MOSAIC_LAYOUT_STR: dict[str, str] = {
    "(0,0)": "90",
    "(0,1)": "45",
    "(1,0)": "135",
    "(1,1)": "0",
}


# ---- Small helpers ----
def snap_down(v: int, step: int) -> int:
    """Round *v* down to the nearest multiple of *step*."""
    s = int(step) if step > 0 else 1
    return int((int(v) // s) * s)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def snap_even(v: int) -> int:
    """Round up to the nearest even integer."""
    return int(v if (int(v) % 2 == 0) else int(v) + 1)
