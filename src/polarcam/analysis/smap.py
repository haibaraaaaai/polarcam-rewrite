"""
S-map accumulator for PolarCam.

Tracks the per-pixel range of the normalised anisotropy components (X, Y)
over a rolling window of frames.  The resulting S-map is:

    S(r, c) = (X_max − X_min)² + (Y_max − Y_min)²

Pixels with large S values are candidates for spinning fluorophores.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from .pol_reconstruction import make_xy_reconstructor


class SMapAccumulator:
    """
    Accumulate per-pixel (X, Y) range statistics and produce an S-map.

    Parameters
    ----------
    frame_shape : (H, W)
        Raw mosaic frame dimensions (must be even).
    smooth_k : int
        Side length of the uniform (box) filter applied to X and Y before
        accumulation.  Use 1 to disable smoothing.
    """

    def __init__(self, frame_shape: tuple[int, int], smooth_k: int = 5) -> None:
        H, W = frame_shape
        if H % 2 or W % 2:
            raise ValueError(f"frame_shape must be even, got {frame_shape}")
        if smooth_k < 1:
            raise ValueError(f"smooth_k must be >= 1, got {smooth_k}")

        self._reconstruct = make_xy_reconstructor(frame_shape, normalize=True)
        self._smooth_k = int(smooth_k)

        ih, iw = H // 2, W // 2
        # Min/max buffers — initialised to ±inf so the first frame sets them.
        self._x_min = np.full((ih, iw), np.inf,  dtype=np.float32)
        self._x_max = np.full((ih, iw), -np.inf, dtype=np.float32)
        self._y_min = np.full((ih, iw), np.inf,  dtype=np.float32)
        self._y_max = np.full((ih, iw), -np.inf, dtype=np.float32)
        self._n: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of frames accumulated since last reset."""
        return self._n

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Shape of the output S-map (H//2, W//2)."""
        return self._x_min.shape

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._x_min[:] =  np.inf
        self._x_max[:] = -np.inf
        self._y_min[:] =  np.inf
        self._y_max[:] = -np.inf
        self._n = 0

    def update(self, frame: np.ndarray) -> None:
        """
        Process one raw mosaic frame and update the running min/max.

        Parameters
        ----------
        frame : np.ndarray
            Raw polarisation mosaic, shape (H, W), any integer/float dtype.
        """
        X, Y = self._reconstruct(frame)

        if self._smooth_k > 1:
            # uniform_filter writes into new arrays; copy back to float32
            Xs = uniform_filter(X, size=self._smooth_k, mode="reflect").astype(
                np.float32, copy=False
            )
            Ys = uniform_filter(Y, size=self._smooth_k, mode="reflect").astype(
                np.float32, copy=False
            )
        else:
            # X and Y are views into internal reconstructor buffers — copy so
            # the next call to _reconstruct doesn't overwrite them.
            Xs = X.copy()
            Ys = Y.copy()

        np.minimum(self._x_min, Xs, out=self._x_min)
        np.maximum(self._x_max, Xs, out=self._x_max)
        np.minimum(self._y_min, Ys, out=self._y_min)
        np.maximum(self._y_max, Ys, out=self._y_max)
        self._n += 1

    def compute(self) -> np.ndarray | None:
        """
        Return the S-map: ``(X_max − X_min)² + (Y_max − Y_min)²``.

        Returns
        -------
        s_map : np.ndarray, shape (H//2, W//2), float32
            None if no frames have been accumulated yet.
        """
        if self._n == 0:
            return None

        rx = self._x_max - self._x_min  # range of X component
        ry = self._y_max - self._y_min  # range of Y component
        s_map: np.ndarray = rx * rx + ry * ry
        return s_map.astype(np.float32, copy=False)
