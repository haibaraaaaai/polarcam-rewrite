from __future__ import annotations
from typing import Literal
import numpy as np

VarMode = Literal["intensity_range", "stddev", "max_pairwise_dist"]

def compute_varmap(stack: np.ndarray, mode: VarMode = "intensity_range") -> np.ndarray:
    """
    Compute a per-pixel activity/variance map from a stack of frames.

    Parameters
    ----------
    stack : (N, H, W) ndarray
        Time-first video stack. Any numeric dtype is accepted; values are treated as intensities.
    mode : {"intensity_range", "stddev", "max_pairwise_dist"}
        - "intensity_range": max(stack) - min(stack) per pixel (fast, robust to outliers).
        - "stddev": population standard deviation across time (float32 compute, uint16 out).
        - "max_pairwise_dist": mathematically equivalent to max - min; provided as an alias
          so a future O(N^2) algorithm (true max |xi - xj|) can be plugged in if ever needed.

    Returns
    -------
    varmap : (H, W) uint16 ndarray
        For "stddev", values are rounded and clipped to 0..65535. For other modes, computation
        is performed in a widened integer domain and clipped to uint16 range.
    """
    if stack.ndim != 3:
        raise ValueError(f"stack must be (N,H,W); got shape {getattr(stack, 'shape', None)}")

    if mode in ("intensity_range", "max_pairwise_dist"):
        # Work in widened integer to avoid wrap; then clip to uint16
        vmax = stack.max(axis=0)
        vmin = stack.min(axis=0)
        diff = vmax.astype(np.int64) - vmin.astype(np.int64)
        return np.clip(diff, 0, 65535).astype(np.uint16, copy=False)

    if mode == "stddev":
        s = stack.astype(np.float32, copy=False).std(axis=0, ddof=0)
        return np.clip(np.rint(s), 0, 65535).astype(np.uint16, copy=False)

    raise ValueError(f"Unknown varmap mode: {mode}")
