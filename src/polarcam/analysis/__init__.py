"""
Analysis utilities for PolarCam.

Exports:
- compute_varmap: per-pixel activity/variance maps from a frame stack.
"""
from .varmap import compute_varmap
__all__ = ["compute_varmap"]
