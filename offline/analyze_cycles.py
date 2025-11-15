
#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import numpy as np

CHUNK_RE = re.compile(r"_s(\d+)_d(\d+)_p(\d+)\.npz$")

def load_npz_info(path: Path):
    try:
        with np.load(path, allow_pickle=False) as z:
            t = np.asarray(z["t"], dtype=np.float64)
            c0 = np.asarray(z["c0"], dtype=np.float64)
            c45 = np.asarray(z["c45"], dtype=np.float64)
            c90 = np.asarray(z["c90"], dtype=np.float64)
            c135 = np.asarray(z["c135"], dtype=np.float64)
            meta_bytes = np.asarray(z["meta"], dtype=np.uint8).tobytes()
            meta = json.loads(meta_bytes.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read {path.name}: {e}")
    import re as _re
    m = CHUNK_RE.search(path.name)
    if not m:
        raise RuntimeError(f"Filename does not match pattern '*_sXX_dYYYY_pZZZZ.npz': {path.name}")
    s_idx, d_idx, p_idx = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # Basic validations
    n = len(t)
    if not (len(c0) == len(c45) == len(c90) == len(c135) == n):
        raise RuntimeError(f"{path.name}: channel lengths mismatch (t={n}, c0={len(c0)}, c45={len(c45)}, c90={len(c90)}, c135={len(c135)})")

    # dt & fps stats (ignore non-positive diffs if any)
    if n >= 2:
        dt = np.diff(t)
        dt_pos = dt[dt > 0]
        if dt_pos.size == 0:
            fps_med = float("nan")
            fps_mean = float("nan")
            dt_ms_med = float("nan")
            dt_ms_p5 = float("nan")
            dt_ms_p95 = float("nan")
        else:
            dt_ms = dt_pos * 1000.0
            dt_ms_med = float(np.median(dt_ms))
            dt_ms_p5  = float(np.percentile(dt_ms, 5))
            dt_ms_p95 = float(np.percentile(dt_ms, 95))
            fps_med = 1000.0 / dt_ms_med if dt_ms_med > 0 else float("nan")
            fps_mean = 1000.0 / float(np.mean(dt_ms)) if float(np.mean(dt_ms)) > 0 else float("nan")
    else:
        dt_ms_med = dt_ms_p5 = dt_ms_p95 = float("nan")
        fps_med = fps_mean = float("nan")

    # Absolute time window using t0_perf_counter (seconds)
    t0_abs = float(meta.get("t0_perf_counter", 0.0))
    t1_abs = t0_abs + (float(t[-1]) if n else 0.0)

    # Crop & ROI for context
    crop_abs = meta.get("crop_abs") or {}
    applied_roi = meta.get("applied_roi") or {}
    crop_w = int(crop_abs.get("w") or 0)
    crop_h = int(crop_abs.get("h") or 0)
    roi_w = int(applied_roi.get("w") or 0)
    roi_h = int(applied_roi.get("h") or 0)

    return {
        "file": path.name,
        "spot": s_idx,
        "dwell": d_idx,
        "part": p_idx,
        "n": n,
        "t0_abs": t0_abs,
        "t1_abs": t1_abs,
        "fps_med": fps_med,
        "fps_mean": fps_mean,
        "dt_ms_med": dt_ms_med,
        "dt_ms_p5": dt_ms_p5,
        "dt_ms_p95": dt_ms_p95,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "roi_w": roi_w,
        "roi_h": roi_h,
    }

def summarize_group(chunks):
    # chunks: list of dicts for same (spot, dwell)
    if not chunks:
        return None
    total_samples = sum(c["n"] for c in chunks)
    start = min(c["t0_abs"] for c in chunks)
    end   = max(c["t1_abs"] for c in chunks)
    duration = max(0.0, end - start)
    fps_overall = (total_samples / duration) if duration > 0 else float("nan")
    # simple medians for ROI/crop
    xs = [c["roi_w"] for c in chunks if c["roi_w"]>0]
    ys = [c["roi_h"] for c in chunks if c["roi_h"]>0]
    cx = [c["crop_w"] for c in chunks if c["crop_w"]>0]
    cy = [c["crop_h"] for c in chunks if c["crop_h"]>0]
    med = lambda arr: float(np.median(arr)) if arr else 0.0
    return {
        "spot": chunks[0]["spot"],
        "dwell": chunks[0]["dwell"],
        "parts": len(chunks),
        "samples": total_samples,
        "duration_s": duration,
        "fps_overall": fps_overall,
        "roi_med": (med(xs), med(ys)),
        "crop_med": (med(cx), med(cy)),
    }

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Analyze PolarCam cycler NPZ chunks and estimate FPS.")
    ap.add_argument("cycles_dir", help="Path to the cycles directory (will recurse).")
    ap.add_argument("--per-chunk", action="store_true", help="Print per-chunk stats (not only group summaries).")
    args = ap.parse_args()

    base = Path(args.cycles_dir).expanduser().resolve()
    files = sorted(base.rglob("*.npz"))
    if not files:
        print(f"No .npz found under {base}")
        return

    rows = []
    for f in files:
        try:
            rows.append(load_npz_info(f))
        except Exception as e:
            print(str(e))

    if args.per_chunk:
        print("\nPer-chunk stats:")
        print("file, spot, dwell, part, n, fps_med, fps_mean, dt_ms_med, dt_ms_p5, dt_ms_p95, roi_w, roi_h, crop_w, crop_h")
        for r in sorted(rows, key=lambda z: (z["spot"], z["dwell"], z["part"])):
            print(f"{r['file']}, {r['spot']}, {r['dwell']}, {r['part']}, {r['n']}, "
                  f"{r['fps_med']:.2f}, {r['fps_mean']:.2f}, "
                  f"{r['dt_ms_med']:.3f}, {r['dt_ms_p5']:.3f}, {r['dt_ms_p95']:.3f}, "
                  f"{r['roi_w']}, {r['roi_h']}, {r['crop_w']}, {r['crop_h']}")

    # Group by (spot, dwell)
    groups = {}
    for r in rows:
        key = (r["spot"], r["dwell"])
        groups.setdefault(key, []).append(r)

    print("\nPer-spot/dwell summary:")
    print("spot, dwell, parts, samples, duration_s, fps_overall, roi_med_w, roi_med_h, crop_med_w, crop_med_h")
    for key in sorted(groups.keys()):
        g = sorted(groups[key], key=lambda z: (z["t0_abs"], z["part"]))
        s = summarize_group(g)
        if s:
            print(f"{s['spot']}, {s['dwell']}, {s['parts']}, {s['samples']}, "
                  f"{s['duration_s']:.3f}, {s['fps_overall']:.2f}, "
                  f"{int(s['roi_med'][0])}, {int(s['roi_med'][1])}, "
                  f"{int(s['crop_med'][0])}, {int(s['crop_med'][1])}")

    # Quick hints
    print("\nNotes:")
    print("• fps_med uses median(1/dt). fps_overall = total_samples / total_time across all chunks in the dwell.")
    print("• ROI med is the median applied HW ROI size seen in meta; crop med is the median software crop size.")
    print("• If fps_overall is far below 'fps_med', exposure (ms) or ROI snapping may be limiting.")
    print("• Any non-positive dt in a chunk is ignored when computing medians.")
    print("• Filenames must match '*_sXX_dYYYY_pZZZZ.npz' for grouping.")

if __name__ == '__main__':
    main()
