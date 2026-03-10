# src/polarcam/app/spot_detect.py
from __future__ import annotations
import math
from typing import List, Optional, Tuple
import numpy as np

# Spot tuple: (cx, cy, r, area, inten)  — all coords in raw-frame pixel space
Spot = Tuple[float, float, float, int, int]


def detect_spots_smap(
    s_map: np.ndarray,
    *,
    sigma_small: float = 1.5,
    sigma_large: float = 3.0,
    k_std: float = 3.0,
    min_area: int = 5,
    max_area: Optional[int] = None,
    connectivity: int = 2,
) -> List[Spot]:
    """
    Run DoG + connected-components on an S-map and return Spot tuples scaled
    to raw-frame coordinates.

    The S-map lives on the intersection grid (H//2, W//2).  Each S-map pixel
    covers a 2×2 block in the raw frame, so returned (cx, cy) are multiplied
    by 2 so that overlays drawn on the raw frame align correctly.

    Parameters
    ----------
    s_map : np.ndarray
        2-D float array from ``SMapAccumulator.compute()``.
    sigma_small, sigma_large : float
        DoG Gaussian scales (in S-map pixels).
    k_std : float
        Detection threshold = mean + k_std * std of DoG image.
    min_area, max_area : int
        Connected-component area bounds (S-map pixels).
    connectivity : int
        1 = 4-connected, 2 = 8-connected.

    Returns
    -------
    spots : List[Spot]
        Each entry is (cx, cy, r, area, inten) in raw-frame pixel space.
        ``inten`` is the S-map value at the centroid scaled to an integer.
    """
    from polarcam.analysis.detect import find_spot_centers_dog

    centers = find_spot_centers_dog(
        s_map,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        k_std=k_std,
        min_area=min_area,
        max_area=max_area,
        connectivity=connectivity,
    )
    if not centers:
        return []

    ih, iw = s_map.shape
    # Radius proxy: blob footprint ≈ sigma_large in S-map pixels → ×2 for raw frame
    r_raw = float(sigma_large) * 2.0
    area_raw = max(1, int(math.pi * r_raw ** 2))

    out: List[Spot] = []
    for cx_s, cy_s in centers:
        cx_raw = cx_s * 2.0 + 0.5
        cy_raw = cy_s * 2.0 + 0.5
        ci = max(0, min(ih - 1, int(round(cy_s))))
        cj = max(0, min(iw - 1, int(round(cx_s))))
        # Scale S-map value to a readable integer (S is 0–8 for normalised anisotropy)
        inten = int(float(s_map[ci, cj]) * 1000.0)
        out.append((cx_raw, cy_raw, r_raw, area_raw, inten))

    return out
