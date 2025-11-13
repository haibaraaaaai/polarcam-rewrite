# src/polarcam/analysis/varmap.py
from __future__ import annotations
import numpy as np
from typing import Literal

VarMode = Literal["intensity_range", "stddev"]

def compute_varmap(stack: np.ndarray, mode: VarMode = "intensity_range") -> np.ndarray:
    """
    Compute a simple per-pixel activity/variance map from a stack of frames.

    Parameters
    ----------
    stack : (N, H, W) uint16/uint8/float ndarray
        Video stack in time-first order. Values are treated as intensities.
    mode : {"intensity_range", "stddev"}
        - "intensity_range": max(stack)-min(stack) per pixel (uint16-safe).
        - "stddev": population standard deviation across time.

    Returns
    -------
    varmap : (H, W) uint16 ndarray
        A 16-bit map for easy saving/consumption. For "stddev" mode, values
        are clipped to 0..65535 after computing in float32.
    """
    if stack.ndim != 3:
        raise ValueError(f"stack must be (N,H,W); got shape {stack.shape}")

    # Work in float32 minimally for stddev, otherwise keep cheap ops.
    if mode == "intensity_range":
        vmax = stack.max(axis=0)
        vmin = stack.min(axis=0)
        diff = vmax.astype(np.int64) - vmin.astype(np.int64)
        diff = np.clip(diff, 0, 65535).astype(np.uint16)
        return diff

    elif mode == "stddev":
        s = stack.astype(np.float32, copy=False).std(axis=0, ddof=0)
        s = np.clip(np.rint(s), 0, 65535).astype(np.uint16)
        return s

    else:
        raise ValueError(f"Unknown varmap mode: {mode}")
