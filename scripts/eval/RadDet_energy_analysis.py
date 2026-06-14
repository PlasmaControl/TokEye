"""Energy-distribution analysis for RadDet (all 6 variants).

For each pixel in the RadDet test set, classify as TN/TP/FN/FP using TokEye
predictions and the rasterised YOLO-bbox ground truth, then accumulate the
preprocessed-spectrogram intensity distribution per category. Per-variant.

Outputs CSV stats + histograms + figure.
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
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import RadDet as R

PRIMARY_THRESHOLD = R.PRIMARY_THRESHOLD
F1_OPT_THRESHOLDS = {  # from RadDet_f1_optimal.csv
    "RadDet40k128HW001Tv2": 0.65,
    "RadDet40k128HW009Tv2": 0.50,
    "RadDet40k256HW001Tv2": 0.55,
    "RadDet40k256HW009Tv2": 0.45,
    "RadDet40k512HW001Tv2": 0.50,
    "RadDet40k512HW009Tv2": 0.30,
}
HIST_BINS = 200
HIST_RANGE = (-3.0, 12.0)


def main():
    print(f"device: {R.device}")
    model = R.build_model(R.device)
    print("model loaded")

    results_dir = R.results_dir
    out_stats = results_dir / "RadDet_energy_dist.csv"
    out_hist = results_dir / "RadDet_energy_hist.csv"
    out_fig = results_dir / "figures" / "RadDet_energy_dist.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    bins = np.linspace(*HIST_RANGE, HIST_BINS + 1)
    cats = ["TN", "TP", "FN", "FP"]

    histograms = {}
    for variant in R.VARIANTS:
        print(f"\n=== {variant} ===")
        thrs = (PRIMARY_THRESHOLD, F1_OPT_THRESHOLDS[variant])
        histograms[variant] = {
            thr: {cat: np.zeros(HIST_BINS, dtype=np.int64) for cat in cats}
            for thr in thrs
        }

        n = 0
        for stem, png_bytes, lbl_bytes in tqdm(
            R.iter_test_pairs(variant), desc=variant, unit="img",
            mininterval=2.0,
        ):
            try:
                spec, mask, _boxes = R.preprocess(png_bytes, lbl_bytes)
                mask_bin = mask > 0.5
                spec_t = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(R.device)
                # Apply same TTA + both-channels post-processing as eval
                def _fwd(x):
                    with torch.no_grad():
                        o = model(x)[0]
                    s0 = torch.sigmoid(o[:, 0:1])
                    if R.USE_BOTH_CHANNELS:
                        s1 = torch.sigmoid(o[:, 1:2])
                        return torch.maximum(s0, s1)
                    return s0
                sig = _fwd(spec_t)
                if R.TTA_TIME_FLIP:
                    sig_f = _fwd(spec_t.flip(-1)).flip(-1)
                    sig = 0.5 * (sig + sig_f)
                sig = sig.cpu().numpy()[0, 0]

                for thr in thrs:
                    pred = sig > thr
                    masks = {
                        "TN": (~pred) & (~mask_bin),
                        "TP": pred & mask_bin,
                        "FN": (~pred) & mask_bin,
                        "FP": pred & (~mask_bin),
                    }
                    for cat, m in masks.items():
                        if m.any():
                            vals = spec[m]
                            h, _ = np.histogram(vals, bins=bins)
                            histograms[variant][thr][cat] += h
                n += 1
            except Exception as e:
                print(f"err {stem}: {e}")
                continue
        print(f"{variant}: {n} images")

    # Stats / aggregate / Wasserstein
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

    def normalize(h):
        s = h.sum()
        return h.astype(np.float64) / s if s > 0 else h.astype(np.float64)

    def wasserstein_1d(h1, h2):
        c1 = np.cumsum(normalize(h1))
        c2 = np.cumsum(normalize(h2))
        dx = bins[1] - bins[0]
        return float(np.sum(np.abs(c1 - c2)) * dx)

    stats_rows = []
    print("\n=== Distance summary per variant @ F1-opt ===")
    for variant in R.VARIANTS:
        thr_opt = F1_OPT_THRESHOLDS[variant]
        h_opt = histograms[variant][thr_opt]
        d_fp_tp = wasserstein_1d(h_opt["FP"], h_opt["TP"])
        d_fp_tn = wasserstein_1d(h_opt["FP"], h_opt["TN"])
        d_tp_tn = wasserstein_1d(h_opt["TP"], h_opt["TN"])
        print(f"{variant} thr={thr_opt}: W(FP,TP)={d_fp_tp:.3f} "
              f"W(FP,TN)={d_fp_tn:.3f} W(TP,TN)={d_tp_tn:.3f} "
              f"-> {'TP' if d_fp_tp < d_fp_tn else 'TN'}-like FPs")

        for thr in (PRIMARY_THRESHOLD, thr_opt):
            h = histograms[variant][thr]
            tn_total = int(h["TN"].sum())
            if tn_total > 0:
                centres = 0.5 * (bins[:-1] + bins[1:])
                cdf = np.cumsum(h["TN"]) / tn_total
                tn_p95 = float(centres[np.searchsorted(cdf, 0.95)])
            else:
                tn_p95 = 0.0
            for cat in cats:
                st = stats_from_hist(h[cat])
                if h[cat].sum() > 0:
                    idx_p95 = np.searchsorted(bins, tn_p95)
                    frac_above = float(h[cat][idx_p95:].sum() / h[cat].sum())
                else:
                    frac_above = 0.0
                stats_rows.append({
                    "variant": variant, "threshold": thr, "category": cat,
                    **st, "frac_above_TN_p95": frac_above,
                    "TN_p95_intensity": tn_p95,
                })

    with out_stats.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)
    print(f"saved {out_stats}")

    with out_hist.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "threshold", "category", "bin_left",
                    "bin_right", "count"])
        for variant in R.VARIANTS:
            for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLDS[variant]):
                for cat in cats:
                    h = histograms[variant][thr][cat]
                    for i in range(HIST_BINS):
                        w.writerow([variant, thr, cat, float(bins[i]),
                                    float(bins[i + 1]), int(h[i])])
    print(f"saved {out_hist}")

    # Figure: 6 panels, one per variant, at F1-opt threshold
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for ax, variant in zip(axes.flatten(), R.VARIANTS):
        thr = F1_OPT_THRESHOLDS[variant]
        h = histograms[variant][thr]
        centres = 0.5 * (bins[:-1] + bins[1:])
        for cat, color in zip(cats, ("0.5", "C2", "C1", "C3")):
            if h[cat].sum() == 0:
                continue
            density = normalize(h[cat]) / (bins[1] - bins[0])
            ax.plot(centres, density,
                    label=f"{cat} (n={int(h[cat].sum()):,})",
                    color=color, lw=1.3)
        ax.set_yscale("log")
        ax.set_xlabel("standardised spectrogram intensity")
        ax.set_ylabel("pixel density")
        ax.set_title(f"{variant.replace('RadDet40k', '')} @ thr={thr}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=130)
    plt.close(fig)
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
