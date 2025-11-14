# src/polarcam/app/spot_detect.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np

try:
    from scipy import ndimage as ndi
except Exception as e:
    raise RuntimeError("spot_detect requires SciPy (scipy.ndimage).") from e

# Spot tuple (what main_window expects):
# (cx, cy, area_px, approx_w, approx_h) — all in FRAME coordinates
Spot = Tuple[float, float, float, int, int]


def _disk(r: int) -> np.ndarray:
    if r <= 0:
        return np.array([[1]], dtype=bool)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def detect_spots_oneframe(
    img16: np.ndarray,
    *,
    thr_mode: str = "absolute",      # "absolute" (DN) or "percentile"
    thr_value: float = 1200.0,       # DN or [0..100] percentile
    min_area: int = 6,               # px
    max_area: int = 5000,            # px
    open_radius: int = 1,            # px (morphology)
    close_radius: int = 1,           # px (morphology)
    remove_border: bool = True,
    max_spots: int = 500,
    dedup_radius: float = 6.0,       # px, suppress very-close duplicates
    # extras that help tame CFA texture while keeping semantics:
    fill_holes: bool = True,         # fill small holes after morphology
    peak_pick: bool = True,          # choose the strongest pixel inside each blob
    peak_sigma: float = 0.7,         # Gaussian smoothing before peak picking
) -> List[Spot]:
    """
    One-frame bright-spot detector on raw 12-bit data.

    Returns a list of (cx, cy, area_px, approx_w, approx_h), sorted by area desc.
    """
    if img16.ndim != 2:
        return []

    a_u16 = img16.astype(np.uint16, copy=False)
    H, W = a_u16.shape

    # ----- threshold mask on the ORIGINAL DN image (keeps "absolute" semantics)
    if thr_mode.lower().startswith("perc"):
        t = float(np.percentile(a_u16, np.clip(thr_value, 0, 100)))
    else:
        t = float(thr_value)
    mask = a_u16 >= t

    # morphology to remove speckle and connect near pixels
    if open_radius > 0:
        mask = ndi.binary_opening(mask, structure=_disk(open_radius))
    if close_radius > 0:
        mask = ndi.binary_closing(mask, structure=_disk(close_radius))
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    # strip 1-px frame border if requested
    if remove_border:
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False

    lab, n = ndi.label(mask)
    if n == 0:
        return []

    # ----- optional peak image for stable "one spot per blob"
    peak_img = a_u16.astype(np.float32, copy=False)
    if peak_pick and peak_sigma > 0:
        peak_img = ndi.gaussian_filter(peak_img, float(peak_sigma), mode="nearest")

    boxes = ndi.find_objects(lab)
    spots: List[Spot] = []

    for i in range(n):
        sl = boxes[i]
        if sl is None:
            continue

        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        if remove_border and (r0 <= 0 or c0 <= 0 or r1 >= H or c1 >= W):
            # touches border
            continue

        reg = (lab[sl] == (i + 1))
        area = int(reg.sum())
        if area < int(min_area) or area > int(max_area):
            continue

        # centroid (uniform) as a fallback
        yy, xx = np.nonzero(reg)
        cy = float(r0 + (yy.mean() if yy.size else 0.0))
        cx = float(c0 + (xx.mean() if xx.size else 0.0))

        # pick the strongest pixel inside this region for a single, stable center
        if peak_pick and reg.any():
            vals = peak_img[sl][reg]
            j = int(vals.argmax())
            flat_idx = np.flatnonzero(reg)[j]
            ry, rx = np.unravel_index(flat_idx, reg.shape)
            cy = float(r0 + ry)
            cx = float(c0 + rx)

        approx_h = int(r1 - r0)
        approx_w = int(c1 - c0)

        spots.append((cx, cy, float(area), max(1, approx_w), max(1, approx_h)))

    # sort by area desc
    spots.sort(key=lambda s: s[2], reverse=True)

    # ----- de-duplicate very close centers (greedy NMS in coordinate space)
    if len(spots) > 1 and dedup_radius > 0:
        keep: List[Spot] = []
        r2 = float(dedup_radius) * float(dedup_radius)
        for s in spots:
            x0, y0 = s[0], s[1]
            if any((x0 - t[0]) ** 2 + (y0 - t[1]) ** 2 <= r2 for t in keep):
                continue
            keep.append(s)
            if len(keep) >= max_spots:
                break
        spots = keep
    else:
        spots = spots[:max_spots]

    return spots
