#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


def find_chunks(path: Path) -> List[Path]:
    """
    Find chunk files under `path`. Supports:
      - SpotSignalRecorder: *_chunkXXXX.npy (shape (N,5))
      - MultiSpotCycler:   *.npz with keys t,c0,c45,c90,c135
    """
    if path.is_file():
        return [path]
    files = []
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() == ".npy" and "_chunk" in p.stem:
            files.append(p)
        elif p.suffix.lower() == ".npz":
            files.append(p)
    # Stable sort chunks by any trailing number in the stem, else lexicographic
    def keyfun(p: Path):
        m = re.search(r"(\d+)$", p.stem)
        return (p.parent.as_posix(), int(m.group(1)) if m else -1, p.name)
    files.sort(key=keyfun)
    return files


def load_chunk(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single chunk and return (t, I0, I45, I90, I135).
    Supports:
      - .npy shaped (N,5): [t, I0, I45, I90, I135]
      - .npz with named arrays {t, c0, c45, c90, c135}
      - .npz with unnamed arr_0 shaped (N,5)
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError(f"{path.name}: expected (N,5) npy; got {arr.shape}")
        t, I0, I45, I90, I135 = (arr[:, i].astype(np.float64, copy=False) for i in range(5))
        return t, I0, I45, I90, I135

    # npz
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.keys())
        if {"t", "c0", "c45", "c90", "c135"}.issubset(keys):
            t = z["t"].astype(np.float64, copy=False)
            I0 = z["c0"].astype(np.float64, copy=False)
            I45 = z["c45"].astype(np.float64, copy=False)
            I90 = z["c90"].astype(np.float64, copy=False)
            I135 = z["c135"].astype(np.float64, copy=False)
            return t, I0, I45, I90, I135
        # unnamed
        if "arr_0" in keys:
            arr = z["arr_0"]
            if arr.ndim != 2 or arr.shape[1] < 5:
                raise ValueError(f"{path.name}: expected arr_0 as (N,5); got {arr.shape}")
            t, I0, I45, I90, I135 = (arr[:, i].astype(np.float64, copy=False) for i in range(5))
            return t, I0, I45, I90, I135
        raise ValueError(f"{path.name}: unsupported npz structure; keys={sorted(keys)}")


def try_read_meta(dirpath: Path) -> Optional[dict]:
    # Look for *_meta.json in the directory
    metas = sorted(dirpath.glob("*_meta.json"))
    if not metas:
        return None
    try:
        return json.loads(metas[0].read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize_times(t: np.ndarray) -> dict:
    if t.size < 2:
        return {"n": int(t.size), "dur_s": 0.0, "fps_med": float("nan"), "fps_mean": float("nan")}
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return {"n": int(t.size), "dur_s": float(t[-1] - t[0]), "fps_med": float("nan"), "fps_mean": float("nan")}
    fps = 1.0 / dt
    return {
        "n": int(t.size),
        "dur_s": float(t[-1] - t[0]),
        "fps_med": float(np.median(fps)),
        "fps_mean": float(np.mean(fps)),
        "fps_min": float(np.min(fps)),
        "fps_max": float(np.max(fps)),
        "fps_p05": float(np.percentile(fps, 5)),
        "fps_p95": float(np.percentile(fps, 95)),
    }


def anisotropy(I0, I45, I90, I135):
    # Per your project notes: x=(I0−I90)/(I0+I90), y=(I45−I135)/(I45+I135)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (I0 - I90) / (I0 + I90)
        y = (I45 - I135) / (I45 + I135)
    x = np.where(np.isfinite(x), x, 0.0)
    y = np.where(np.isfinite(y), y, 0.0)
    return x, y


def dump_csv(path: Path, t, I0, I45, I90, I135, out_csv: Path) -> None:
    x, y = anisotropy(I0, I45, I90, I135)
    arr = np.column_stack([t, I0, I45, I90, I135, x, y])
    hdr = "t,I0,I45,I90,I135,x,y"
    np.savetxt(out_csv, arr, delimiter=",", header=hdr, comments="", fmt="%.9f")
    print(f"[write] {out_csv}  (rows={arr.shape[0]})")


def quick_plots(base_out: Path, t, I0, I45, I90, I135) -> None:
    if not HAVE_PLT:
        print("[warn] matplotlib not available; skipping plots.")
        return
    # Timeseries
    plt.figure()
    plt.plot(t, I0, label="I0")
    plt.plot(t, I45, label="I45")
    plt.plot(t, I90, label="I90")
    plt.plot(t, I135, label="I135")
    plt.xlabel("t (s)"); plt.ylabel("intensity"); plt.legend()
    f1 = base_out.with_suffix(".signals.png")
    plt.tight_layout(); plt.savefig(f1, dpi=120); plt.close()
    print(f"[plot] {f1}")

    # Anisotropy cloud
    x, y = anisotropy(I0, I45, I90, I135)
    plt.figure()
    plt.scatter(x, y, s=4, alpha=0.6)
    plt.xlabel("x = (I0−I90)/(I0+I90)")
    plt.ylabel("y = (I45−I135)/(I45+I135)")
    plt.xlim(-1, 1); plt.ylim(-1, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    f2 = base_out.with_suffix(".anisotropy.png")
    plt.tight_layout(); plt.savefig(f2, dpi=120); plt.close()
    print(f"[plot] {f2}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect capture/cycle chunks (.npy/.npz).")
    ap.add_argument("path", type=Path, help="Chunk file or directory containing chunks")
    ap.add_argument("--csv", type=Path, default=None, help="Write concatenated CSV to this path")
    ap.add_argument("--plot", action="store_true", help="Emit simple PNG plots next to --csv (requires matplotlib)")
    args = ap.parse_args(argv)

    path = args.path
    if not path.exists():
        print(f"error: path not found: {path}", file=sys.stderr)
        return 2

    chunks = find_chunks(path)
    if not chunks:
        print("No chunk files found (.npy or .npz).")
        return 1

    print(f"[info] found {len(chunks)} chunk(s) under {path}")
    # Try meta for context
    meta = try_read_meta(path if path.is_dir() else path.parent)
    if meta:
        print("[meta]", json.dumps(meta, indent=2))

    all_t = []; all0 = []; all45 = []; all90 = []; all135 = []
    for i, p in enumerate(chunks, 1):
        try:
            t, I0, I45, I90, I135 = load_chunk(p)
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
            continue
        s = summarize_times(t)
        print(f"[{i:02d}] {p.name:40s}  n={s['n']:6d}  dur={s['dur_s']:.3f}s  "
              f"fps_med={s['fps_med']:.2f}  mean={s['fps_mean']:.2f}  "
              f"p05={s['fps_p05']:.2f}  p95={s['fps_p95']:.2f}")
        all_t.append(t if not all_t else (t + (all_t[-1][-1] - t[0] + 1e-6)))  # make globally increasing
        all0.append(I0); all45.append(I45); all90.append(I90); all135.append(I135)

    if not all_t:
        print("No readable chunks.")
        return 1

    t = np.concatenate(all_t)
    I0 = np.concatenate(all0)
    I45 = np.concatenate(all45)
    I90 = np.concatenate(all90)
    I135 = np.concatenate(all135)

    s = summarize_times(t)
    print("\n=== OVERALL ===")
    print(f"rows={s['n']}  duration={s['dur_s']:.3f}s  fps_med={s['fps_med']:.2f}  "
          f"mean={s['fps_mean']:.2f}  min={s['fps_min']:.2f}  max={s['fps_max']:.2f}")

    if args.csv is not None:
        out_csv = args.csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        dump_csv(path, t, I0, I45, I90, I135, out_csv)
        if args.plot:
            base = out_csv.with_suffix("")
            quick_plots(base, t, I0, I45, I90, I135)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
