"""
Spot detection on the S-map for PolarCam.

Uses a Difference-of-Gaussians (DoG) blob enhancer followed by thresholding
and connected-component labelling to locate candidate spinning-fluorophore
spots.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, label


def find_spot_centers_dog(
    s_map: np.ndarray,
    sigma_small: float = 1.5,
    sigma_large: float = 3.0,
    k_std: float = 3.0,
    min_area: int = 5,
    max_area: Optional[int] = None,
    connectivity: int = 2,
) -> list[tuple[float, float]]:
    """
    Detect bright spots on an S-map using Difference-of-Gaussians filtering.

    The DoG response enhances compact bright blobs while suppressing slow
    background gradients.  Pixels above ``mean + k_std * std`` of the DoG
    image are thresholded, and connected regions are labelled.  The centroid
    of each region that passes the area filter is returned.

    Parameters
    ----------
    s_map : np.ndarray
        2-D float array (H, W) — e.g. from ``SMapAccumulator.compute()``.
    sigma_small : float
        Inner Gaussian sigma (fine scale).
    sigma_large : float
        Outer Gaussian sigma (coarse scale).  Must be > sigma_small.
    k_std : float
        Threshold = mean(DoG) + k_std * std(DoG).  Higher values → fewer,
        more confident detections.
    min_area : int
        Minimum connected-region size in pixels.
    max_area : int or None
        Maximum connected-region size in pixels.  None disables the upper
        limit.
    connectivity : int
        1 → 4-connected,  2 → 8-connected (default).

    Returns
    -------
    centers : list of (cx, cy)
        Sub-pixel centroid coordinates (x = column, y = row) for each
        detected spot, sorted by descending mean DoG response.
    """
    if s_map.ndim != 2:
        raise ValueError(f"Expected 2-D s_map, got shape {s_map.shape}")
    if sigma_large <= sigma_small:
        raise ValueError(
            f"sigma_large ({sigma_large}) must be greater than sigma_small ({sigma_small})"
        )

    finite_mask = np.isfinite(s_map)
    if not finite_mask.any():
        return []

    # --- Difference-of-Gaussians blob enhancement ---
    s32 = s_map.astype(np.float32, copy=False)
    dog = gaussian_filter(s32, sigma=sigma_small) - gaussian_filter(s32, sigma=sigma_large)

    # Threshold on the DoG image
    dog_vals = dog[finite_mask]
    mu = float(dog_vals.mean())
    sigma = float(dog_vals.std())
    if sigma == 0.0:
        return []

    thr = mu + k_std * sigma
    binary = (dog >= thr) & finite_mask
    if not binary.any():
        return []

    # --- Connected-component labelling ---
    struct = _connectivity_struct(connectivity)
    labelled, n_labels = label(binary, structure=struct)
    if n_labels == 0:
        return []

    # Compute centroid + mean dog response per label in one pass
    ys, xs = np.nonzero(binary)
    labs = labelled[ys, xs]

    counts   = np.bincount(labs, minlength=n_labels + 1)
    sum_x    = np.bincount(labs, weights=xs.astype(np.float64), minlength=n_labels + 1)
    sum_y    = np.bincount(labs, weights=ys.astype(np.float64), minlength=n_labels + 1)
    sum_resp = np.bincount(
        labs, weights=dog[ys, xs].astype(np.float64), minlength=n_labels + 1
    )

    max_area_i = int(max_area) if max_area is not None else None

    results: list[tuple[float, float, float]] = []  # (cx, cy, mean_response)
    for lbl in range(1, n_labels + 1):
        cnt = int(counts[lbl])
        if cnt < min_area:
            continue
        if max_area_i is not None and cnt > max_area_i:
            continue
        cx = float(sum_x[lbl] / cnt)
        cy = float(sum_y[lbl] / cnt)
        mean_resp = float(sum_resp[lbl] / cnt)
        results.append((cx, cy, mean_resp))

    # Sort strongest response first
    results.sort(key=lambda t: t[2], reverse=True)
    return [(cx, cy) for cx, cy, _ in results]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connectivity_struct(connectivity: int) -> np.ndarray:
    """Return a binary structuring element for scipy.ndimage.label."""
    if connectivity == 1:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    if connectivity == 2:
        return np.ones((3, 3), dtype=np.int32)
    raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")
