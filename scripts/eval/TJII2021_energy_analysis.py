"""Energy-distribution analysis: do TokEye's TJII FPs land on signal or noise?

For each pixel in the TJII test set, classify as TN/TP/FN/FP using TokEye
predictions and Bustos ground truth, then accumulate the spectrogram-intensity
distribution per category. If FP distribution looks like TP (not TN),
TokEye is finding real spectral structure that the ground-truth annotation
missed -- a quantitative argument that the precision ceiling is set by label
incompleteness, not model error.

Outputs:
  TJII2021_energy_dist.csv -- per-category stats (mean, median, p5, p95)
  TJII2021_energy_hist.csv -- normalised histograms
  TJII2021_energy_dist.png -- side-by-side distribution figure
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import TJII2021 as T

PRIMARY_THRESHOLD = 0.5
F1_OPT_THRESHOLD = 0.8  # from prior PR sweep
HIST_BINS = 200
HIST_RANGE = (-3.0, 12.0)  # standard-deviation-normalised spectrogram range


def main():
    print(f"device: {T.device}")
    model = T.build_model()
    print("model loaded")

    results_dir = T.results_dir
    out_csv_stats = results_dir / "TJII2021_energy_dist.csv"
    out_csv_hist = results_dir / "TJII2021_energy_hist.csv"
    out_fig = results_dir / "figures" / "TJII2021_energy_dist.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    bins = np.linspace(*HIST_RANGE, HIST_BINS + 1)
    cats = ["TN", "TP", "FN", "FP"]
    # Two threshold settings: default 0.5 and F1-opt 0.8
    histograms = {
        thr: {cat: np.zeros(HIST_BINS, dtype=np.int64) for cat in cats}
        for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD)
    }
    # Also accumulate sums and counts for mean / median estimates
    sums = {
        thr: dict.fromkeys(cats, 0.0)
        for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD)
    }
    counts = {
        thr: dict.fromkeys(cats, 0)
        for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD)
    }

    shotns = [name.stem.split("_")[1]
              for name in T.data_path.glob("input/*.png")]
    print(f"{len(shotns)} shots")

    for shotn in tqdm(shotns):
        try:
            ip = T.data_path / "input" / f"spectrogram_{shotn}.png"
            gtp = T.data_path / "gt" / f"spectrogram_{shotn}.png"
            spec = np.array(Image.open(ip).convert("L"))
            ann = np.array(Image.open(gtp).convert("L"))
            spec = np.flip(spec, axis=0).copy()
            ann = np.flip(ann, axis=0).copy()
            spec_norm = (spec - T.MEAN) / T.STD  # standardized
            ann_bin = (ann // 255).astype(np.bool_)

            spec_t = torch.from_numpy(spec_norm).float().unsqueeze(0).unsqueeze(0).to(T.device)
            with torch.no_grad():
                out = model(spec_t)[0]
            sig = torch.sigmoid(out[:, 0:1]).cpu().numpy()[0, 0]

            for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD):
                pred = sig > thr
                tn = (~pred) & (~ann_bin)
                tp = pred & ann_bin
                fn = (~pred) & ann_bin
                fp = pred & (~ann_bin)
                masks = {"TN": tn, "TP": tp, "FN": fn, "FP": fp}
                for cat, mask in masks.items():
                    if mask.any():
                        vals = spec_norm[mask]
                        h, _ = np.histogram(vals, bins=bins)
                        histograms[thr][cat] += h
                        sums[thr][cat] += float(vals.sum())
                        counts[thr][cat] += int(mask.sum())
        except Exception as e:
            print(f"Error processing shot {shotn}: {e}")
            continue

    # Compute statistics: mean, median (from histogram), p5, p95, percentile-above-95
    def stats_from_hist(hist):
        total = int(hist.sum())
        if total == 0:
            return {"mean": 0.0, "median": 0.0, "p5": 0.0, "p95": 0.0,
                    "n_pixels": 0}
        centres = 0.5 * (bins[:-1] + bins[1:])
        cdf = np.cumsum(hist) / total
        p5 = float(centres[np.searchsorted(cdf, 0.05)])
        p50 = float(centres[np.searchsorted(cdf, 0.50)])
        p95 = float(centres[np.searchsorted(cdf, 0.95)])
        mean = float((centres * hist).sum() / total)
        return {"mean": mean, "median": p50, "p5": p5, "p95": p95,
                "n_pixels": total}

    # CSV stats
    stats_rows = []
    for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD):
        tn_hist = histograms[thr]["TN"]
        tn_total = int(tn_hist.sum())
        if tn_total > 0:
            centres = 0.5 * (bins[:-1] + bins[1:])
            cdf = np.cumsum(tn_hist) / tn_total
            tn_p95 = float(centres[np.searchsorted(cdf, 0.95)])
        else:
            tn_p95 = 0.0
        for cat in cats:
            h = histograms[thr][cat]
            st = stats_from_hist(h)
            # fraction of pixels above TN p95
            if h.sum() > 0:
                idx_p95 = np.searchsorted(bins, tn_p95)
                frac_above = float(h[idx_p95:].sum() / h.sum())
            else:
                frac_above = 0.0
            stats_rows.append({
                "threshold": thr,
                "category": cat,
                **st,
                "frac_above_TN_p95": frac_above,
                "TN_p95_intensity": tn_p95,
            })

    with out_csv_stats.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)
    print(f"saved {out_csv_stats}")

    # CSV histograms (long form)
    with out_csv_hist.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "category", "bin_left", "bin_right", "count"])
        for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD):
            for cat in cats:
                h = histograms[thr][cat]
                for i in range(HIST_BINS):
                    w.writerow([thr, cat, float(bins[i]), float(bins[i + 1]),
                                int(h[i])])
    print(f"saved {out_csv_hist}")

    # Distance metrics: how close is FP distribution to TP vs TN?
    def normalize(h):
        s = h.sum()
        return h.astype(np.float64) / s if s > 0 else h.astype(np.float64)

    def wasserstein_1d(h1, h2):
        # 1D earth mover distance via CDF difference, in units of bin centres
        c1 = np.cumsum(normalize(h1))
        c2 = np.cumsum(normalize(h2))
        dx = bins[1] - bins[0]
        return float(np.sum(np.abs(c1 - c2)) * dx)

    print("\n=== Distance summary ===")
    for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD):
        h = histograms[thr]
        d_fp_tp = wasserstein_1d(h["FP"], h["TP"])
        d_fp_tn = wasserstein_1d(h["FP"], h["TN"])
        d_tp_tn = wasserstein_1d(h["TP"], h["TN"])
        print(f"thr={thr}: W(FP,TP)={d_fp_tp:.3f} W(FP,TN)={d_fp_tn:.3f} "
              f"W(TP,TN)={d_tp_tn:.3f} -> FP is "
              f"{'closer to TP' if d_fp_tp < d_fp_tn else 'closer to TN'}")

    # Figure: 2 columns (thr=0.5 default, thr=0.8 F1-opt), 4 distributions overlaid
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    for ax, thr in zip(axes, (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD)):
        centres = 0.5 * (bins[:-1] + bins[1:])
        for cat, color in zip(cats, ("0.5", "C2", "C1", "C3")):
            h = histograms[thr][cat]
            if h.sum() == 0:
                continue
            density = normalize(h) / (bins[1] - bins[0])  # PDF estimate
            ax.plot(centres, density, label=f"{cat} (n={int(h.sum()):,})",
                    color=color, lw=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("standardised spectrogram intensity")
        ax.set_ylabel("pixel density (log)")
        ax.set_title(f"TJII pixel-intensity by category @ threshold={thr}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=130)
    plt.close(fig)
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
