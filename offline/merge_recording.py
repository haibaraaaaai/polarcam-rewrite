"""
merge_recording.py — PolarCam offline post-processor
=====================================================

Merges a chunked recording (produced by FrameWriter) into:

    recordings_processed/<prefix>.npy        — (N, H, W) uint16, full 16-bit lossless
    recordings_processed/<prefix>.json       — clean metadata (shape, fps, duration, ...)
    recordings_processed/<prefix>.avi        — 8-bit MJPEG video at correct fps

Usage
-----
    # Interactive (will ask for path):
    python offline/merge_recording.py

    # Pass the meta JSON directly:
    python offline/merge_recording.py recordings/frames_20260309_115906_meta.json

    # Or pass the recording directory (processes all recordings found there):
    python offline/merge_recording.py recordings/

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

def _resolve_prefix(path: str) -> tuple[str, str]:
    """
    Return (out_dir, prefix) from *any* of:
      - a _meta.json file
      - a directory that contains exactly one recording (auto-select)
      - a raw prefix like 'recordings/frames_20260309_115906'
    """
    p = Path(path)

    # meta JSON?
    if p.suffix == ".json" and "_meta" in p.stem:
        # e.g. recordings/frames_20260309_115906_meta.json
        prefix = p.stem[: -len("_meta")]       # strip '_meta'
        return str(p.parent), prefix

    # bare prefix (directory + base)?
    if not p.is_dir():
        return str(p.parent), p.name

    # directory — find all _meta.json files under it
    metas = sorted(glob.glob(str(p / "*_meta.json")))
    if not metas:
        raise FileNotFoundError(f"No *_meta.json found in {p}")
    if len(metas) > 1:
        print("Multiple recordings found:")
        for i, m in enumerate(metas):
            print(f"  [{i}] {m}")
        choice = input("Enter index to process (or 'all'): ").strip()
        if choice.lower() == "all":
            return str(p), None          # caller handles list
        metas = [metas[int(choice)]]
    meta_path = Path(metas[0])
    prefix = meta_path.stem[: -len("_meta")]
    return str(meta_path.parent), prefix


def _find_chunks(rec_dir: str, prefix: str) -> list[str]:
    pattern = os.path.join(rec_dir, f"{prefix}_chunk*.npy")
    chunks = sorted(glob.glob(pattern))
    if not chunks:
        raise FileNotFoundError(f"No chunk files found matching {pattern}")
    return chunks


def _load_timestamps(rec_dir: str, prefix: str) -> Optional[np.ndarray]:
    ts_path = os.path.join(rec_dir, f"{prefix}_timestamps.npy")
    if os.path.isfile(ts_path):
        return np.load(ts_path).astype(np.float64)
    return None


def _compute_fps(timestamps: Optional[np.ndarray], fps_configured: float) -> float:
    """Return measured fps (from timestamps) or fall back to configured fps."""
    if timestamps is not None and len(timestamps) > 1:
        diffs = np.diff(timestamps)
        # Ignore outliers (e.g. first frame may stall briefly)
        median_dt = float(np.median(diffs))
        if median_dt > 0:
            return 1.0 / median_dt
    if fps_configured > 0:
        return fps_configured
    return 30.0  # last resort default


def _sample_scale_range(
    chunks: list[str], low_pct: float = 0.5, high_pct: float = 99.5, sample_every: int = 10
) -> tuple[float, float]:
    """
    Compute robust uint16→uint8 scale bounds by sampling ~1 frame per chunk.
    Returns (lo, hi) in uint16 counts.
    """
    samples: list[np.ndarray] = []
    for path in chunks:
        arr = np.load(path, mmap_mode="r")          # (K, H, W)
        idx = min(sample_every - 1, arr.shape[0] - 1)
        samples.append(arr[::sample_every].reshape(-1).astype(np.float32))
    flat = np.concatenate(samples)
    lo = float(np.percentile(flat, low_pct))
    hi = float(np.percentile(flat, high_pct))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _to_uint8(frame_u16: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale a uint16 (H,W) frame to uint8 (H,W) using [lo, hi] → [0, 255]."""
    f = frame_u16.astype(np.float32)
    f = np.clip((f - lo) / (hi - lo) * 255.0, 0, 255)
    return f.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main conversion routine
# ---------------------------------------------------------------------------

def merge(rec_dir: str, prefix: str, out_dir: str) -> None:
    print(f"\n{'='*60}")
    print(f"Recording : {prefix}")
    print(f"Source    : {rec_dir}")
    print(f"Output    : {out_dir}")
    print(f"{'='*60}")

    # ---- load meta --------------------------------------------------------
    meta_path = os.path.join(rec_dir, f"{prefix}_meta.json")
    meta: dict = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    fps_configured = float(meta.get("fps_configured", 0.0))

    # ---- find chunks -------------------------------------------------------
    chunks = _find_chunks(rec_dir, prefix)
    print(f"Chunks    : {len(chunks)}")

    # ---- count total frames (via mmap — no RAM cost) ----------------------
    frame_counts = [np.load(c, mmap_mode="r").shape[0] for c in chunks]
    total_frames = sum(frame_counts)

    # infer shape from first chunk
    first = np.load(chunks[0], mmap_mode="r")
    H, W = first.shape[1], first.shape[2]
    print(f"Frames    : {total_frames}  ({H}×{W} uint16)")

    # ---- timestamps --------------------------------------------------------
    timestamps = _load_timestamps(rec_dir, prefix)
    if timestamps is not None:
        timestamps = timestamps[:total_frames]   # guard against mismatch
    fps = _compute_fps(timestamps, fps_configured)
    duration_s = total_frames / fps if fps > 0 else 0.0
    print(f"FPS       : {fps:.3f} (configured={fps_configured:.3f})")
    print(f"Duration  : {duration_s:.2f} s")

    os.makedirs(out_dir, exist_ok=True)

    # ---- merged .npy (memmap write — stays off RAM) ----------------------
    out_npy = os.path.join(out_dir, f"{prefix}.npy")
    print(f"\nWriting   : {out_npy}  ...", end="", flush=True)
    out_arr = np.lib.format.open_memmap(
        out_npy, mode="w+", dtype=np.uint16, shape=(total_frames, H, W)
    )
    cursor = 0
    for path in chunks:
        chunk = np.load(path)
        n = chunk.shape[0]
        out_arr[cursor : cursor + n] = chunk
        cursor += n
    del out_arr   # flush memmap
    print(" done")

    # ---- compute 8-bit scale range (sample-based, low RAM) ---------------
    print(f"Computing scale range from samples ...", end="", flush=True)
    lo, hi = _sample_scale_range(chunks)
    print(f" [{lo:.0f} – {hi:.0f}]")

    # ---- AVI output -------------------------------------------------------
    out_avi = os.path.join(out_dir, f"{prefix}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(out_avi, fourcc, fps, (W, H), isColor=False)
    if not writer.isOpened():
        print(f"WARNING: could not open VideoWriter for {out_avi}")
    else:
        print(f"Writing   : {out_avi}  ...", end="", flush=True)
        written = 0
        for path in chunks:
            chunk = np.load(path)          # (K, H, W) uint16
            for i in range(chunk.shape[0]):
                writer.write(_to_uint8(chunk[i], lo, hi))
                written += 1
                if written % 100 == 0:
                    print(f"\r  {written}/{total_frames} frames", end="", flush=True)
        writer.release()
        print(f"\r  {written}/{total_frames} frames — done")

    # ---- output metadata JSON ---------------------------------------------
    out_json = os.path.join(out_dir, f"{prefix}.json")
    out_meta = {
        "base_name"       : meta.get("base_name", prefix),
        "started_utc"     : meta.get("started_utc", ""),
        "n_frames"        : total_frames,
        "frame_shape_hw"  : [H, W],
        "dtype"           : "uint16",
        "fps_configured"  : fps_configured,
        "fps_measured"    : round(fps, 4),
        "duration_s"      : round(duration_s, 4),
        "scale_u16_low"   : round(lo, 2),
        "scale_u16_high"  : round(hi, 2),
        "mosaic_layout"   : meta.get("mosaic_layout", {
            "(row%2, col%2)": {"(0,0)": "90°", "(0,1)": "45°", "(1,0)": "135°", "(1,1)": "0°"}
        }),
        "source_prefix"   : prefix,
        "source_dir"      : str(Path(rec_dir).resolve()),
        "n_source_chunks" : len(chunks),
    }
    with open(out_json, "w") as f:
        json.dump(out_meta, f, indent=2)
    print(f"Metadata  : {out_json}")
    print("Done.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge PolarCam chunked recording.")
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to _meta.json, recording directory, or prefix "
             "(e.g. recordings/frames_20260309_115906)",
    )
    parser.add_argument(
        "--out",
        default="recordings_processed",
        help="Output directory (default: recordings_processed/)",
    )
    args = parser.parse_args()

    path = args.path
    if not path:
        path = input("Enter path to recording (meta JSON, directory, or prefix): ").strip()
        if not path:
            print("No path given — exiting.")
            sys.exit(1)

    rec_dir, prefix = _resolve_prefix(path)

    if prefix is None:
        # 'all' mode — collect all prefixes in the directory
        metas = sorted(glob.glob(os.path.join(rec_dir, "*_meta.json")))
        for m in metas:
            pfx = Path(m).stem[: -len("_meta")]
            try:
                merge(rec_dir, pfx, args.out)
            except Exception as e:
                print(f"ERROR processing {pfx}: {e}")
    else:
        merge(rec_dir, prefix, args.out)


if __name__ == "__main__":
    main()
