# src/polarcam/app/spot_detect.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np

try:
    from scipy import ndimage as ndi
except Exception as e:
    raise RuntimeError("spot_detect requires SciPy (scipy.ndimage).") from e

# Old/desired spot tuple:
#   (cx, cy, r, area, inten)
Spot = Tuple[float, float, float, int, int]


def _disk(r: int) -> np.ndarray:
    if r <= 0:
        return np.array([[1]], dtype=bool)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def _remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    """SciPy-only version of skimage.remove_small_objects."""
    if min_area <= 1:
        return mask
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    areas = ndi.sum(mask, labels=lab, index=np.arange(1, n + 1)).astype(np.int64)
    kill = set(int(i + 1) for i, a in enumerate(areas) if a < int(min_area))
    if not kill:
        return mask
    out = mask.copy()
    for k in kill:
        out[lab == k] = False
    return out


def detect_spots_oneframe(
    img16: np.ndarray,
    *,
    thr_mode: str = "absolute",   # "absolute" (DN) or "percentile"
    thr_value: float = 1200.0,    # DN or [0..100] percentile
    min_area: int = 6,            # px
    max_area: int = 5000,         # px
    open_radius: int = 2,         # px
    close_radius: int = 1,        # px
    remove_border: bool = True,
    debug: bool = False,
) -> List[Spot]:
    """
    Single-frame detection:

      1) threshold on intensity (absolute or percentile)
      2) binary opening (de-speckle)
      3) remove small objects
      4) binary closing (mend tiny gaps)
      5) label (4-connectivity), compute centroid + area
      6) radius r = sqrt(area / pi), intensity from original frame at rounded centroid
    """
    a = img16.astype(np.uint16, copy=False)
    H, W = a.shape

    # --- 1) threshold ---
    if thr_mode.lower().startswith("perc"):
        t = float(np.percentile(a, np.clip(thr_value, 0, 100)))
        mode_str = f"percentile {thr_value:.1f}"
    else:
        t = float(thr_value)
        mode_str = f"absolute DN {thr_value:.1f}"
    mask = a >= t

    # strip 1px border to prevent edge ribbons
    if remove_border:
        mask[0, :] = mask[-1, :] = False
        mask[:, 0] = mask[:, -1] = False

    # --- 2) opening ---
    if open_radius > 0:
        mask = ndi.binary_opening(mask, structure=_disk(open_radius))

    # --- 3) remove tiny blobs early ---
    mask = _remove_small_objects(mask, min_area=max(1, int(min_area)))

    # --- 4) closing ---
    if close_radius > 0:
        mask = ndi.binary_closing(mask, structure=_disk(close_radius))

    # --- 5) label + props ---
    lab, n = ndi.label(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8))
    if n == 0:
        if debug:
            print(f"[Detect] {H}x{W}  mode={mode_str}  thr={t:.1f}  pixels>thr=0")
        return []

    areas = ndi.sum(mask, labels=lab, index=np.arange(1, n + 1)).astype(np.int64)
    centers = ndi.center_of_mass(mask, lab, index=np.arange(1, n + 1))
    boxes = ndi.find_objects(lab)

    out: List[Spot] = []
    for i in range(n):
        area = int(areas[i])
        if area < int(min_area) or area > int(max_area):
            continue
        sl = boxes[i]
        if sl is None:
            continue
        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        if remove_border and (r0 <= 0 or c0 <= 0 or r1 >= H or c1 >= W):
            continue

        cy, cx = centers[i]  # center_of_mass returns (row, col)
        iy, ix = int(round(cy)), int(round(cx))
        iy = max(0, min(H - 1, iy))
        ix = max(0, min(W - 1, ix))
        inten = int(a[iy, ix])

        r = float(np.sqrt(max(1.0, float(area)) / np.pi))
        out.append((float(ix), float(iy), r, int(area), int(inten)))

    if debug:
        print(f"[Detect] spots={len(out)}  mode={mode_str}  thr={t:.1f}  px>thr={int(mask.sum())}")

    # bright-first
    out.sort(key=lambda z: -z[4])
    return out
