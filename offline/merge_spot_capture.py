"""
merge_spot_capture.py — PolarCam offline spot-capture post-processor
====================================================================

Merges a chunked spot capture (produced by SpotSignalRecorder) into:

    captures_processed/<prefix>.npy    — (N, cropH, cropW) uint16 raw masked pixels
    captures_processed/<prefix>.json   — clean metadata
    captures_processed/<prefix>.avi    — 8-bit MJPEG video of the masked spot at correct fps

Usage
-----
    # Interactive (will ask for path):
    python offline/merge_spot_capture.py

    # Pass the meta JSON directly:
    python offline/merge_spot_capture.py captures/spot1_20260310_134957_meta.json

    # Or pass the captures directory (processes all found):
    python offline/merge_spot_capture.py captures/

Dependencies
------------
    numpy, opencv-python (cv2)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_prefix(path: str) -> tuple[str, str | None]:
    """
    Return (source_dir, prefix) from any of:
      - a _meta.json file
      - a directory (auto-select or 'all')
      - a raw prefix
    """
    p = Path(path)

    if p.suffix == ".json" and "_meta" in p.stem:
        prefix = p.stem[: -len("_meta")]
        return str(p.parent), prefix

    if not p.is_dir():
        return str(p.parent), p.name

    metas = sorted(glob.glob(str(p / "*_meta.json")))
    if not metas:
        raise FileNotFoundError(f"No *_meta.json found in {p}")
    if len(metas) > 1:
        print("Multiple captures found:")
        for i, m in enumerate(metas):
            print(f"  [{i}] {m}")
        choice = input("Enter index to process (or 'all'): ").strip()
        if choice.lower() == "all":
            return str(p), None
        metas = [metas[int(choice)]]
    meta_path = Path(metas[0])
    prefix = meta_path.stem[: -len("_meta")]
    return str(meta_path.parent), prefix


def _find_chunks(src_dir: str, prefix: str) -> list[str]:
    pattern = os.path.join(src_dir, f"{prefix}_chunk*.npy")
    chunks = sorted(glob.glob(pattern))
    if not chunks:
        raise FileNotFoundError(f"No chunk files matching {pattern}")
    return chunks


# ---------------------------------------------------------------------------
# Helpers for uint16→uint8 scaling (same approach as merge_recording.py)
# ---------------------------------------------------------------------------

def _sample_scale_range(
    chunks: list[str], low_pct: float = 0.5, high_pct: float = 99.5, sample_every: int = 10
) -> tuple[float, float]:
    """Compute robust uint16→uint8 scale bounds by sampling frames."""
    samples: list[np.ndarray] = []
    for path in chunks:
        arr = np.load(path, mmap_mode="r")
        samples.append(arr[::sample_every].reshape(-1).astype(np.float32))
    flat = np.concatenate(samples)
    # Ignore exact-zero (masked out) pixels for range computation
    nonzero = flat[flat > 0]
    if nonzero.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(nonzero, low_pct))
    hi = float(np.percentile(nonzero, high_pct))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _to_uint8(frame_u16: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale uint16 (H,W) → uint8 (H,W) using [lo, hi] → [0, 255]."""
    f = frame_u16.astype(np.float32)
    f = np.clip((f - lo) / (hi - lo) * 255.0, 0, 255)
    return f.astype(np.uint8)


def _load_timestamps(src_dir: str, prefix: str) -> Optional[np.ndarray]:
    ts_path = os.path.join(src_dir, f"{prefix}_timestamps.npy")
    if os.path.isfile(ts_path):
        return np.load(ts_path).astype(np.float64)
    return None


def _compute_fps(timestamps: Optional[np.ndarray]) -> float:
    """Return measured fps from timestamps, or 0 if unavailable."""
    if timestamps is not None and len(timestamps) > 1:
        diffs = np.diff(timestamps)
        median_dt = float(np.median(diffs))
        if median_dt > 0:
            return 1.0 / median_dt
    return 0.0


# ---------------------------------------------------------------------------
# Main merge routine
# ---------------------------------------------------------------------------

def merge(src_dir: str, prefix: str, out_dir: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Capture   : {prefix}")
    print(f"Source    : {src_dir}")
    print(f"Output    : {out_dir}")
    print(f"{'=' * 60}")

    # ---- meta -------------------------------------------------------------
    meta_path = os.path.join(src_dir, f"{prefix}_meta.json")
    meta: dict = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # ---- chunks -----------------------------------------------------------
    chunks = _find_chunks(src_dir, prefix)
    print(f"Chunks    : {len(chunks)}")

    # count total frames via mmap
    frame_counts = [np.load(c, mmap_mode="r").shape[0] for c in chunks]
    total_frames = sum(frame_counts)

    # infer crop shape from first chunk
    first = np.load(chunks[0], mmap_mode="r")
    if first.ndim == 3:
        cropH, cropW = first.shape[1], first.shape[2]
    elif first.ndim == 2:
        # legacy format: (N, 5) float32 signal data
        cropH, cropW = 0, first.shape[1]
    else:
        raise ValueError(f"Unexpected chunk shape: {first.shape}")

    print(f"Frames    : {total_frames}  ({cropH}×{cropW} uint16)")

    # ---- timestamps -------------------------------------------------------
    timestamps = _load_timestamps(src_dir, prefix)
    if timestamps is not None:
        timestamps = timestamps[:total_frames]
    fps = _compute_fps(timestamps)
    duration_s = float(timestamps[-1] - timestamps[0]) if timestamps is not None and len(timestamps) > 1 else (total_frames / max(1.0, fps) if fps > 0 else 0.0)
    if fps <= 0:
        fps = 30.0  # fallback for AVI
    print(f"FPS       : {fps:.1f} Hz")
    print(f"Duration  : {duration_s:.3f} s")

    os.makedirs(out_dir, exist_ok=True)

    # ---- merged .npy (memmap for large datasets) --------------------------
    out_npy = os.path.join(out_dir, f"{prefix}.npy")
    print(f"Writing   : {out_npy}  ...", end="", flush=True)
    if cropH > 0:
        out_arr = np.lib.format.open_memmap(
            out_npy, mode="w+", dtype=np.uint16, shape=(total_frames, cropH, cropW)
        )
        cursor = 0
        for path in chunks:
            chunk = np.load(path)
            n = chunk.shape[0]
            out_arr[cursor : cursor + n] = chunk
            cursor += n
        del out_arr
    else:
        # legacy: just concatenate
        arrays = [np.load(c) for c in chunks]
        np.save(out_npy, np.concatenate(arrays, axis=0))
    print(" done")

    # ---- metadata JSON ----------------------------------------------------
    spot_info = meta.get("spot", {})
    roi_info = meta.get("roi_hw_requested", {})
    # Compute reference pixel info (from source meta or derive from ROI + crop offset)
    tl_yx = meta.get("crop_top_left_sensor_yx")
    tl_ch = meta.get("crop_top_left_channel")
    if tl_yx is None and roi_info:
        ry = int(roi_info.get("y", 0))
        rx = int(roi_info.get("x", 0))
        c_off = meta.get("crop_offset_in_roi", [0, 0])
        tl_yx = [ry + c_off[0], rx + c_off[1]]
    if tl_yx and tl_ch is None:
        _L = {(0,0): "90", (0,1): "45", (1,0): "135", (1,1): "0"}
        tl_ch = _L.get((tl_yx[0] % 2, tl_yx[1] % 2), "90")

    out_meta = {
        "prefix": prefix,
        "n_frames": total_frames,
        "crop_shape_hw": [cropH, cropW],
        "dtype": "uint16",
        "duration_s": round(duration_s, 4),
        "fps_measured": round(fps, 2),
        "spot": spot_info,
        "roi_hw_requested": roi_info,
        "crop_offset_in_roi": meta.get("crop_offset_in_roi"),
        "crop_top_left_sensor_yx": tl_yx,
        "crop_top_left_channel": tl_ch,
        "mosaic_layout": meta.get("layout", {
            "(0,0)": "90", "(0,1)": "45", "(1,0)": "135", "(1,1)": "0"
        }),
        "source_dir": str(Path(src_dir).resolve()),
        "n_source_chunks": len(chunks),
    }
    out_json = os.path.join(out_dir, f"{prefix}.json")
    with open(out_json, "w") as f:
        json.dump(out_meta, f, indent=2)
    print(f"Metadata  : {out_json}")

    # ---- AVI output -------------------------------------------------------
    if cropH <= 0:
        print("  (legacy signal format — no AVI generated)")
        print("Done.\n")
        return

    # Scale small crops up so the video is viewable
    MIN_VID_SIDE = 256
    scale = max(1, MIN_VID_SIDE // max(cropH, cropW))
    vid_w = cropW * scale
    vid_h = cropH * scale

    print(f"Computing scale range ...", end="", flush=True)
    lo, hi = _sample_scale_range(chunks)
    print(f" [{lo:.0f} – {hi:.0f}]")

    out_avi = os.path.join(out_dir, f"{prefix}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(out_avi, fourcc, min(fps, 60.0), (vid_w, vid_h), isColor=False)
    if not writer.isOpened():
        print(f"  WARNING: could not open VideoWriter for {out_avi}")
    else:
        print(f"Writing   : {out_avi}  (scale={scale}x) ...", end="", flush=True)
        written = 0
        for path in chunks:
            chunk = np.load(path)
            for i in range(chunk.shape[0]):
                u8 = _to_uint8(chunk[i], lo, hi)
                if scale > 1:
                    u8 = cv2.resize(u8, (vid_w, vid_h), interpolation=cv2.INTER_NEAREST)
                writer.write(u8)
                written += 1
                if written % 500 == 0:
                    print(f"\r  {written}/{total_frames} frames", end="", flush=True)
        writer.release()
        print(f"\r  {written}/{total_frames} frames — done")

    print("Done.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge PolarCam spot capture chunks.")
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to _meta.json, captures directory, or prefix",
    )
    parser.add_argument(
        "--out",
        default="captures_processed",
        help="Output directory (default: captures_processed/)",
    )
    args = parser.parse_args()

    path = args.path
    if not path:
        path = input("Enter path to capture (meta JSON, directory, or prefix): ").strip()
        if not path:
            print("No path given — exiting.")
            sys.exit(1)

    src_dir, prefix = _resolve_prefix(path)

    if prefix is None:
        metas = sorted(glob.glob(os.path.join(src_dir, "*_meta.json")))
        for m in metas:
            pfx = Path(m).stem[: -len("_meta")]
            try:
                merge(src_dir, pfx, args.out)
            except Exception as e:
                print(f"ERROR processing {pfx}: {e}")
    else:
        merge(src_dir, prefix, args.out)


if __name__ == "__main__":
    main()
