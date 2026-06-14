"""Paper figures for Nature Comm submission.

Generates 4 figures into output/paper_figures/:
  1. tjii_fp_intensity.png — 3-panel TP/FP/TN distributions (TJII, DCLDE,
     RadDet HW001) with frac-above-TN-p95 + Wasserstein ratio annotations.
  2. raddet_example.png — RadDet spectrogram + GT box overlay (top) and
     predicted pixel mask (bottom), 128HW009 FMCW case.
  3. raddet_per_class_ap.png — horizontal bar chart of per-class AP at
     IoU=0.5, sorted descending, for best variant (128HW009).
  4. pr_curves.png — PR curves for TJII / DCLDE / RadDet with default
     and F1-optimal operating points marked.

All data is read from existing CSVs except (2), which needs one tar sample.
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path("/scratch/gpfs/nc1514/tokeye")
RESULTS = ROOT / "data" / "eval" / "results"
OUT = ROOT / "output" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1: 3-panel intensity distributions (TJ-II / DCLDE / RadDet HW001)
# ---------------------------------------------------------------------------

def load_hist_csv(path: Path, group_key: str):
    """Returns {(group, thr, cat): (bin_centers, counts)}."""
    rows = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            k = (r[group_key], float(r["threshold"]), r["category"])
            rows[k].append((float(r["bin_left"]), float(r["bin_right"]),
                            int(r["count"])))
    out = {}
    for k, lst in rows.items():
        lst.sort()
        centers = np.array([0.5 * (a + b) for a, b, _ in lst])
        counts = np.array([c for _, _, c in lst], dtype=np.int64)
        out[k] = (centers, counts)
    return out


def normalize(counts):
    s = counts.sum()
    return counts.astype(np.float64) / s if s > 0 else counts.astype(np.float64)


def wasserstein(c1, c2, centers):
    n1 = normalize(c1); n2 = normalize(c2)
    cdf1 = np.cumsum(n1); cdf2 = np.cumsum(n2)
    dx = float(centers[1] - centers[0])
    return float(np.sum(np.abs(cdf1 - cdf2)) * dx)


def frac_above(counts, centers, value):
    mask = centers > value
    total = counts.sum()
    return float(counts[mask].sum() / total) if total > 0 else 0.0


def find_p(counts, centers, p):
    total = counts.sum()
    if total == 0:
        return 0.0
    cdf = np.cumsum(counts) / total
    idx = np.searchsorted(cdf, p)
    return float(centers[min(idx, len(centers) - 1)])


def panel_dist(ax, centers, hists, thr, title, color_map=None):
    """hists: dict {cat -> counts}. Plot TN, TP, FP densities (log-y)."""
    cmap = color_map or {"TN": "0.5", "TP": "#1b9e77", "FP": "#d95f02"}
    dx = float(centers[1] - centers[0])
    for cat in ("TN", "TP", "FP"):
        c = hists.get(cat)
        if c is None or c.sum() == 0:
            continue
        density = normalize(c) / dx
        n = int(c.sum())
        ax.plot(centers, density, color=cmap[cat], lw=1.5,
                label=f"{cat} (n={n:,})")
    tn_p95 = find_p(hists["TN"], centers, 0.95)
    ax.axvline(tn_p95, color="0.3", ls="--", lw=0.8, alpha=0.7)
    ax.text(tn_p95, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 1,
            "TN p95", rotation=90, va="top", ha="right",
            fontsize=7, color="0.3")

    frac_fp = frac_above(hists["FP"], centers, tn_p95)
    frac_tp = frac_above(hists["TP"], centers, tn_p95)
    w_fp_tp = wasserstein(hists["FP"], hists["TP"], centers)
    w_fp_tn = wasserstein(hists["FP"], hists["TN"], centers)
    ratio = w_fp_tn / w_fp_tp if w_fp_tp > 0 else float("inf")

    txt = (
        f"FP > TN p95: {frac_fp*100:.0f}%\n"
        f"TP > TN p95: {frac_tp*100:.0f}%\n"
        f"W(FP,TN) / W(FP,TP): {ratio:.1f}×"
    )
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=8,
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.6", "alpha": 0.95})

    ax.set_yscale("log")
    ax.set_xlabel("standardised spectrogram intensity")
    ax.set_ylabel("pixel density (log)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def figure_intensity():
    print("Building figure 1: tjii_fp_intensity.png")

    dclde = load_hist_csv(RESULTS / "DCLDE2011_energy_hist.csv", "species")
    raddet = load_hist_csv(RESULTS / "RadDet_energy_hist.csv", "variant")

    # TJII hist CSV has columns: threshold,category,bin_left,bin_right,count
    # (no dataset field). Load separately.
    def load_tjii_hist():
        rows_by_key = defaultdict(list)
        with (RESULTS / "TJII2021_energy_hist.csv").open() as f:
            for r in csv.DictReader(f):
                k = (float(r["threshold"]), r["category"])
                rows_by_key[k].append((float(r["bin_left"]),
                                       float(r["bin_right"]),
                                       int(r["count"])))
        out = {}
        for k, lst in rows_by_key.items():
            lst.sort()
            centers = np.array([0.5 * (a + b) for a, b, _ in lst])
            counts = np.array([c for _, _, c in lst], dtype=np.int64)
            out[k] = (centers, counts)
        return out

    tjii = load_tjii_hist()

    tjii_thr = 0.8
    dclde_thr = 0.85
    raddet_variant = "RadDet40k128HW001Tv2"
    raddet_thr = 0.65

    tjii_hists = {cat: tjii[(tjii_thr, cat)][1] for cat in ("TN", "TP", "FP")}
    tjii_centers = tjii[(tjii_thr, "TN")][0]

    dclde_hists = {
        cat: dclde[("_aggregate", dclde_thr, cat)][1]
        for cat in ("TN", "TP", "FP")
    }
    dclde_centers = dclde[("_aggregate", dclde_thr, "TN")][0]

    raddet_hists = {
        cat: raddet[(raddet_variant, raddet_thr, cat)][1]
        for cat in ("TN", "TP", "FP")
    }
    raddet_centers = raddet[(raddet_variant, raddet_thr, "TN")][0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panel_dist(axes[0], tjii_centers, tjii_hists, tjii_thr,
               f"TJ-II plasma — pixel intensity by category\n"
               f"(F1-opt thr={tjii_thr})")
    panel_dist(axes[1], dclde_centers, dclde_hists, dclde_thr,
               f"DCLDE (4 species, aggregate)\n"
               f"(F1-opt thr={dclde_thr})")
    panel_dist(axes[2], raddet_centers, raddet_hists, raddet_thr,
               f"RadDet 128HW001\n"
               f"(F1-opt thr={raddet_thr})")

    fig.suptitle("Pixel-level FP intensity tracks TP, not background — "
                 "replicated across three domains", fontsize=11, y=1.00)
    fig.tight_layout()
    out_path = OUT / "tjii_fp_intensity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: RadDet example — spec + GT box overlay, predicted mask
# ---------------------------------------------------------------------------

def figure_raddet_example():
    print("Building figure 2: raddet_example.png")
    import RadDet  # local import to avoid heavy load if just figure 1/3/4

    model = RadDet.build_model(RadDet.device)
    variant = "RadDet40k128HW009Tv2"
    parts = sorted(RadDet.data_path.glob(f"{variant}.tar.part-*"))
    if not parts:
        raise RuntimeError(f"no tar parts for {variant}")

    # Want one strong FMCW (class 10) case. Stream the tar and pick the
    # first sample whose label has >=1 FMCW box and >=2 boxes total.
    proc = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    tf = tarfile.open(fileobj=proc.stdout, mode="r|")
    pi, pl = {}, {}
    chosen = None
    try:
        for m in tf:
            if not m.isfile():
                continue
            if "/images/test/" in m.name and m.name.endswith(".png"):
                s = Path(m.name).stem
                pi[s] = tf.extractfile(m).read()
            elif "/labels/test/" in m.name and m.name.endswith(".txt"):
                s = Path(m.name).stem
                pl[s] = tf.extractfile(m).read()
            common = pi.keys() & pl.keys()
            for s in list(common):
                lbl = pl.pop(s); png = pi.pop(s)
                lines = [l for l in lbl.decode().splitlines() if l.strip()]
                if len(lines) < 1:
                    continue
                classes = [int(l.split()[0]) for l in lines]
                # FMCW = class 10
                if 10 in classes and len(lines) >= 2:
                    chosen = (s, png, lbl, lines)
                    break
            if chosen:
                break
    finally:
        proc.stdout.close(); proc.wait()
    if chosen is None:
        raise RuntimeError("no FMCW sample found")

    stem, png, lbl_bytes, lines = chosen
    print(f"  using sample {stem} with {len(lines)} boxes")

    # RadDet.preprocess returns boxes already in the eval coordinate frame
    # (after FLIP_FREQ_AXIS + UN_FFT_SHIFT). Use those directly so the overlay
    # matches the displayed spec.
    spec, mask, boxes_eval = RadDet.preprocess(png, lbl_bytes)
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(RadDet.device)
    sig = RadDet._forward_sig(model, spec_t)
    if RadDet.TTA_TIME_FLIP:
        sig_f = RadDet._forward_sig(model, spec_t.flip(-1)).flip(-1)
        sig = 0.5 * (sig + sig_f)
    sig_np = sig.cpu().numpy()[0, 0]

    H, W = spec.shape
    class_names = {
        0: "Rect", 1: "Barker", 2: "Frank", 3: "P1", 4: "P2", 5: "P3",
        6: "P4", 7: "Px", 8: "ZadoffChu", 9: "LFM", 10: "FMCW",
    }
    # boxes_eval is list of (class, x0, y0, x1, y1) in pixel coords. Convert
    # to (cls, x0, y0_lower, w, h) for matplotlib origin="lower" rendering:
    # imshow with origin="lower" flips the row axis visually, so a box with
    # row index y0 in the array appears at vertical position y0 from bottom.
    # Boxes are already in array-coords; no further flip needed.
    boxes = [(cid, x0, y0, x1 - x0, y1 - y0) for cid, x0, y0, x1, y1 in boxes_eval]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    lo = float(np.percentile(spec, 10))
    hi = float(spec.max())
    axes[0].imshow(spec, origin="lower", aspect="auto", cmap="gist_heat",
                   vmin=lo, vmax=hi)
    for cid, x0, y0, w, h in boxes:
        c = "lime" if cid == 10 else "cyan"
        rect = mpatches.Rectangle((x0, y0), w, h, linewidth=1.4,
                                  edgecolor=c, facecolor="none")
        axes[0].add_patch(rect)
        axes[0].text(x0, y0 + h + 1.5, class_names.get(cid, str(cid)),
                     color=c, fontsize=7, weight="bold",
                     bbox={"facecolor": "black", "edgecolor": "none",
                               "alpha": 0.6, "pad": 1})
    axes[0].set_title("RadDet 128HW009 — spectrogram + GT boxes "
                      "(FMCW in lime, others cyan)", fontsize=10)
    axes[0].set_xlabel("time bin"); axes[0].set_ylabel("freq bin")

    axes[1].imshow(sig_np, origin="lower", aspect="auto", cmap="magma",
                   vmin=0, vmax=1)
    axes[1].set_title("Predicted pixel mask (sigmoid). Pixel F1 ≈ 0.58 "
                      "on this variant; mAP₅₀ ≈ 0.11", fontsize=10)
    axes[1].set_xlabel("time bin"); axes[1].set_ylabel("freq bin")

    fig.suptitle("Same prediction → strong pixel agreement, weak box-level "
                 "mAP from mask→box conversion", fontsize=11, y=1.00)
    fig.tight_layout()
    out_path = OUT / "raddet_example.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: per-class AP horizontal bar chart (best variant 128HW009)
# ---------------------------------------------------------------------------

def figure_per_class_ap():
    print("Building figure 3: raddet_per_class_ap.png")
    variant = "RadDet40k128HW009Tv2"
    rows = []
    with (RESULTS / "RadDet_per_class_ap.csv").open() as f:
        for r in csv.DictReader(f):
            if r["variant"] != variant:
                continue
            rows.append((r["class_name"], float(r["ap"])))
    rows.sort(key=lambda r: r[1])  # ascending; barh draws bottom-up

    names = [r[0] for r in rows]
    aps = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, aps, color="#3182bd")
    for bar, name in zip(bars, names):
        if name == "FMCW":
            bar.set_color("#e6550d")
        elif name == "LFM":
            bar.set_color("#fdae6b")
    for bar, v in zip(bars, aps):
        ax.text(v + max(aps) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("AP at IoU=0.5")
    ax.set_title(f"RadDet per-class AP (zero-shot, best variant "
                 f"{variant.replace('RadDet40k', '')})\n"
                 f"FMCW = 14× short-pulse-code AP: longer coherent signals "
                 f"transfer better", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, max(aps) * 1.15)

    fig.tight_layout()
    out_path = OUT / "raddet_per_class_ap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: PR curves with default + F1-opt operating points marked
# ---------------------------------------------------------------------------

def load_sweep(path: Path, key_field: str):
    out = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            out[r[key_field]].append((
                float(r["threshold"]),
                float(r["precision"]),
                float(r["recall"]),
                float(r["f1"]),
            ))
    for k in out:
        out[k].sort()
    return out


def figure_pr_curves():
    print("Building figure 4: pr_curves.png")
    tjii = load_sweep(RESULTS / "TJII2021_pr_sweep.csv", "dataset")
    dclde = load_sweep(RESULTS / "DCLDE2011_pr_sweep.csv", "species")
    raddet = load_sweep(RESULTS / "RadDet_pr_sweep.csv", "variant")

    # Read F1-opt thresholds from the f1_optimal CSVs
    def load_f1opt(path, key):
        out = {}
        with path.open() as f:
            for r in csv.DictReader(f):
                out[r[key]] = (
                    float(r["f1_optimal_threshold"]),
                    float(r["precision_at_opt"]),
                    float(r["recall_at_opt"]),
                )
        return out

    tjii_opt = load_f1opt(RESULTS / "TJII2021_f1_optimal.csv", "dataset")
    dclde_opt = load_f1opt(RESULTS / "DCLDE2011_f1_optimal.csv", "species")
    raddet_opt = load_f1opt(RESULTS / "RadDet_f1_optimal.csv", "variant")

    fig, ax = plt.subplots(figsize=(8, 6))

    def plot_curve(rows, label, color, marker, default_thr=0.5, f1opt=None):
        ts = [r[0] for r in rows]
        ps = [r[1] for r in rows]
        rs = [r[2] for r in rows]
        ax.plot(rs, ps, color=color, lw=1.2, alpha=0.7,
                marker=marker, ms=3, label=label)
        # Mark default threshold point
        if default_thr in ts:
            i = ts.index(default_thr)
            ax.scatter(rs[i], ps[i], color=color, marker="o", s=60,
                       edgecolor="black", lw=0.8, zorder=5)
        # Mark F1-opt point
        if f1opt is not None:
            opt_thr, p_opt, r_opt = f1opt
            ax.scatter(r_opt, p_opt, color=color, marker="*", s=140,
                       edgecolor="black", lw=0.8, zorder=6)

    # TJII
    for k, rows in tjii.items():
        plot_curve(rows, "TJ-II", "#d62728", "o", 0.5, tjii_opt.get(k))

    # DCLDE — skip P. electra
    dclde_skip = {"Peponocephala electra"}
    palette_d = ["#2ca02c", "#98df8a", "#1f77b4", "#aec7e8"]
    keys = [k for k in dclde if k not in dclde_skip]
    for i, k in enumerate(keys):
        short = k.split()[0][:8] + " " + k.split()[-1][:8]
        plot_curve(dclde[k], f"DCLDE {short}", palette_d[i % len(palette_d)],
                   "^", 0.5, dclde_opt.get(k))

    # RadDet — show all 6 variants
    palette_r = ["#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
                 "#e377c2", "#f7b6d2"]
    for i, (k, rows) in enumerate(raddet.items()):
        plot_curve(rows, f"RadDet {k.replace('RadDet40k', '').replace('Tv2', '')}",
                   palette_r[i % len(palette_r)], "s", 0.5, raddet_opt.get(k))

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("PR curves — pixel-level\n"
                 "○ default threshold (0.5)   ★ F1-optimal threshold",
                 fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUT / "pr_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    figure_intensity()
    figure_per_class_ap()
    figure_pr_curves()
    figure_raddet_example()
    print(f"\nAll figures saved to {OUT}")


if __name__ == "__main__":
    main()
