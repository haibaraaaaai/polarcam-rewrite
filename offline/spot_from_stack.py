# polarcam/offline/spot_from_stack.py
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


# --------- LUT (12-bit -> 8-bit) ----------
def build_highlight_lut(floor: int, cap: int, gamma: float) -> np.ndarray:
    cap = max(floor + 1, min(int(cap), 4095))
    floor = max(0, min(int(floor), 4094))
    gamma = float(max(0.05, min(gamma, 10.0)))
    x = np.arange(4096, dtype=np.float32)
    t = (x - floor) / float(cap - floor)
    t = np.clip(t, 0.0, 1.0)
    y = np.power(t, gamma) * 255.0
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


# Polarization layout (match the app):
# (row%2,col%2): (0,0)->90°, (0,1)->45°, (1,0)->135°, (1,1)->0°
def channel_parity_masks(h: int, w: int):
    r = (np.arange(h)[:, None] & 1)
    c = (np.arange(w)[None, :] & 1)
    m0   = (r == 1) & (c == 1)  # 0°
    m45  = (r == 0) & (c == 1)  # 45°
    m90  = (r == 0) & (c == 0)  # 90°
    m135 = (r == 1) & (c == 0)  # 135°
    return m0, m45, m90, m135


# --------- Spot finding on the 8-bit view ----------
def find_spots_on_view(view8: np.ndarray,
                       thr_percentile: float = 99.7,
                       min_area: int = 8,
                       max_area: int = 5000,
                       grow_px: int = 1) -> list[np.ndarray]:
    """
    Returns a list of boolean masks (H,W) — one per spot.
    """
    H, W = view8.shape
    thr = float(np.percentile(view8, thr_percentile))
    mask = view8 >= thr

    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), bool))
    lbl, n = ndi.label(mask)
    if n == 0:
        return []

    masks: list[np.ndarray] = []
    for lab in range(1, n + 1):
        comp = (lbl == lab)
        area = int(comp.sum())
        if area < min_area or area > max_area:
            continue
        if grow_px > 0:
            comp = ndi.binary_dilation(comp, structure=np.ones((2 * grow_px + 1, 2 * grow_px + 1), bool))
        masks.append(comp)

    return masks


# --------- Compute normalized XY per spot ----------
def compute_xy_traces_normalized(stack: np.ndarray,
                                 spot_masks: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    stack: (T,H,W) uint16
    spot_masks: list of (H,W) bool masks
    Returns: list of (x_t, y_t) arrays per spot (length T). NaNs if a channel is missing.
    """
    assert stack.ndim == 3
    T, H, W = stack.shape
    m0, m45, m90, m135 = channel_parity_masks(H, W)
    traces: list[tuple[np.ndarray, np.ndarray]] = []
    eps = 1e-9

    for mask in spot_masks:
        mask = mask.astype(bool, copy=False)
        m0_s   = m0   & mask
        m45_s  = m45  & mask
        m90_s  = m90  & mask
        m135_s = m135 & mask

        # Enforce channel safety: if any channel absent, mark NaNs
        if not (m0_s.any() and m45_s.any() and m90_s.any() and m135_s.any()):
            traces.append((np.full(T, np.nan), np.full(T, np.nan)))
            continue

        x = np.empty(T, dtype=np.float64)
        y = np.empty(T, dtype=np.float64)

        for t in range(T):
            f = stack[t]
            c0   = float(f[m0_s].mean())
            c45  = float(f[m45_s].mean())
            c90  = float(f[m90_s].mean())
            c135 = float(f[m135_s].mean())

            x[t] = (c0  - c90 ) / (c0  + c90  + eps)
            y[t] = (c45 - c135) / (c45 + c135 + eps)

        traces.append((x, y))

    return traces


# --------- Plot helpers ----------
def save_overlay(view8: np.ndarray,
                 spot_masks: list[np.ndarray],
                 out_path: Path) -> None:
    H, W = view8.shape
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(view8, cmap="gray", vmin=0, vmax=255)
    for i, m in enumerate(spot_masks):
        edge = m ^ ndi.binary_erosion(m, structure=np.ones((3, 3)))
        ys, xs = np.nonzero(edge)
        if ys.size:
            plt.scatter(xs, ys, s=2, alpha=0.9)
        cy, cx = ndi.center_of_mass(m)
        if np.isfinite(cx) and np.isfinite(cy):
            plt.text(cx, cy, f"{i}", color="yellow", fontsize=9, ha="center", va="center")
    plt.title("First frame with detected spots")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _bbox_from_mask(mask: np.ndarray, pad: int, H: int, W: int):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, 0, H, W
    y0 = max(0, ys.min() - pad)
    y1 = min(H, ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(W, xs.max() + pad + 1)
    return y0, y1, x0, x1


def save_xy_per_spot_with_crop(traces: list[tuple[np.ndarray, np.ndarray]],
                               view8: np.ndarray,
                               spot_masks: list[np.ndarray],
                               outdir: Path) -> None:
    """
    For each spot: make a 2-panel figure:
      Left: crop of the first frame with mask outline and colored per-channel pixels used.
      Right: XY scatter with axes fixed to [-1,1].
    Also writes xy_sN.csv per spot.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    H, W = view8.shape
    m0, m45, m90, m135 = channel_parity_masks(H, W)

    # Colors for channels
    # 0°   -> red
    # 45°  -> blue
    # 90°  -> green
    # 135° -> orange
    ch_colors = {
        "0°":   "#e41a1c",
        "45°":  "#377eb8",
        "90°":  "#4daf4a",
        "135°": "#ff7f00",
    }

    for i, (xs, ys) in enumerate(traces):
        mask = spot_masks[i].astype(bool, copy=False)

        # CSV (x,y per frame)
        csv_path = outdir / f"xy_s{i}.csv"
        arr = np.column_stack([xs.astype(float), ys.astype(float)])
        np.savetxt(csv_path, arr, delimiter=",", header="x,y", comments="", fmt="%.6f")

        # Build crop bounds
        y0, y1, x0, x1 = _bbox_from_mask(mask, pad=4, H=H, W=W)
        crop = view8[y0:y1, x0:x1]

        # Channel-specific selections within mask
        m0_s   = (m0   & mask)[y0:y1, x0:x1]
        m45_s  = (m45  & mask)[y0:y1, x0:x1]
        m90_s  = (m90  & mask)[y0:y1, x0:x1]
        m135_s = (m135 & mask)[y0:y1, x0:x1]
        edge = (mask ^ ndi.binary_erosion(mask, structure=np.ones((3, 3))))[y0:y1, x0:x1]

        # Compose figure
        fig = plt.figure(figsize=(10, 4.8), dpi=130)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.28)

        # Left: crop with overlays
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(crop, cmap="gray", vmin=0, vmax=255)
        # mask edge
        ey, ex = np.nonzero(edge)
        if ey.size:
            ax0.scatter(ex, ey, s=6, c="yellow", alpha=0.9, label="mask edge")
        # channel pixels (inside mask)
        for m_sel, label, color in [
            (m0_s,   "0°",   ch_colors["0°"]),
            (m45_s,  "45°",  ch_colors["45°"]),
            (m90_s,  "90°",  ch_colors["90°"]),
            (m135_s, "135°", ch_colors["135°"]),
        ]:
            yy, xx = np.nonzero(m_sel)
            if yy.size:
                ax0.scatter(xx, yy, s=8, c=color, alpha=0.85, label=label)

        ax0.set_title(f"Spot s{i} — pixels used per channel")
        ax0.set_xlim([0, crop.shape[1]])
        ax0.set_ylim([crop.shape[0], 0])  # imshow coords
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.legend(loc="upper right", frameon=True, fontsize=8)

        # Right: XY scatter (no connections)
        ax1 = fig.add_subplot(gs[0, 1])
        m_valid = np.isfinite(xs) & np.isfinite(ys)
        if m_valid.any():
            ax1.scatter(xs[m_valid], ys[m_valid], s=14, alpha=0.9)
        ax1.axhline(0, lw=0.8, alpha=0.6, color="k")
        ax1.axvline(0, lw=0.8, alpha=0.6, color="k")
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_aspect("equal", adjustable="box")
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.set_xlabel("(I0 − I90) / (I0 + I90)")
        ax1.set_ylabel("(I45 − I135) / (I45 + I135)")
        ax1.set_title(f"Spot s{i} — XY scatter")

        fig.tight_layout()
        png_path = outdir / f"xy_s{i}.png"
        fig.savefig(png_path)
        plt.close(fig)


# --------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Spot analysis from saved stack (.npy).")
    ap.add_argument("stack_path", type=Path, help="Path to stack_*.npy (T,H,W) uint16")
    ap.add_argument("--floor", type=int, default=1200, help="LUT floor (12-bit DN)")
    ap.add_argument("--cap", type=int, default=4095, help="LUT cap (12-bit DN)")
    ap.add_argument("--gamma", type=float, default=0.6, help="LUT gamma")
    ap.add_argument("--thr_pct", type=float, default=99.7, help="Threshold percentile for spot finding")
    ap.add_argument("--min_area", type=int, default=8, help="Min spot area (px)")
    ap.add_argument("--max_area", type=int, default=5000, help="Max spot area (px)")
    ap.add_argument("--grow", type=int, default=1, help="Grow each spot mask by N pixels")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory (default: alongside stack)")
    args = ap.parse_args()

    stack_path: Path = args.stack_path
    if not stack_path.exists():
        raise SystemExit(f"Stack not found: {stack_path}")

    outdir = args.outdir or (stack_path.parent / f"spot_from_stack_{time.strftime('%Y%m%d-%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Load stack (memmap if possible)
    stack = np.load(stack_path, mmap_mode="r")
    if stack.ndim != 3:
        raise SystemExit(f"Expected (T,H,W) array; got {stack.shape}")
    T, H, W = stack.shape
    if stack.dtype != np.uint16:
        stack = stack.astype(np.uint16, copy=False)

    print(f"[info] stack: T={T}, H={H}, W={W}, dtype={stack.dtype}")

    # Build LUT & view of first frame
    lut = build_highlight_lut(args.floor, args.cap, args.gamma)
    f0 = stack[0]
    view8 = lut[f0]

    # Detect spots
    masks = find_spots_on_view(view8,
                               thr_percentile=args.thr_pct,
                               min_area=args.min_area,
                               max_area=args.max_area,
                               grow_px=args.grow)
    print(f"[info] detected {len(masks)} spot(s)")

    # Save overlay of frame0
    overlay_png = outdir / "frame0_spots.png"
    save_overlay(view8, masks, overlay_png)
    print(f"[ok] overlay -> {overlay_png}")

    # Compute normalized XY traces and save per-spot outputs with crops
    traces = compute_xy_traces_normalized(stack, masks)
    per_spot_dir = outdir / "per_spot"
    save_xy_per_spot_with_crop(traces, view8, masks, per_spot_dir)
    print(f"[ok] per-spot outputs -> {per_spot_dir}")


if __name__ == "__main__":
    main()
