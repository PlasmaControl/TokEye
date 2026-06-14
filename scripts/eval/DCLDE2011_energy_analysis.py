"""Energy-distribution analysis for DCLDE2011.

For each pixel in the DCLDE test set, classify as TN/TP/FN/FP using TokEye
predictions and the ROCCA-contour ground truth, then accumulate the
spectrogram-intensity distribution per category. Same logic as
TJII2021_energy_analysis.py.

Outputs CSV stats + histograms + figure (per species + aggregate).
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
from scipy.ndimage import median_filter
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import DCLDE2011 as D

PRIMARY_THRESHOLD = 0.5
F1_OPT_THRESHOLD = 0.85  # typical for DCLDE
HIST_BINS = 200
HIST_RANGE = (-3.0, 12.0)
MEDIAN_FILTER_SIZE = 3
EXCLUDE_FROM_AGGREGATE = {"Peponocephala electra"}  # broken dataset


def main():
    print(f"device: {D.device}")
    model = D.build_model()
    print("model loaded")

    results_dir = D.results_dir
    out_stats = results_dir / "DCLDE2011_energy_dist.csv"
    out_hist = results_dir / "DCLDE2011_energy_hist.csv"
    out_fig = results_dir / "figures" / "DCLDE2011_energy_dist.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    bins = np.linspace(*HIST_RANGE, HIST_BINS + 1)
    cats = ["TN", "TP", "FN", "FP"]
    species_list = list(D.STATS.keys())

    # histograms[species][thr][cat] = np.zeros(HIST_BINS)
    histograms = {
        sp: {
            thr: {cat: np.zeros(HIST_BINS, dtype=np.int64) for cat in cats}
            for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD)
        }
        for sp in species_list
    }

    for species_name, stats in D.STATS.items():
        print(f"\n=== {species_name} ===")
        try:
            data_dir = D.data_base / species_name

            class P:
                def __init__(self, m, s):
                    self.m, self.s = m, s
                def __call__(self, x):
                    return (x - self.m) / self.s

            from tokeye.extra.eval.silbidopy.data import AudioTonalDataset
            dataset = AudioTonalDataset(
                data_dir, data_dir,
                annotation_extension="ann",
                time_patch_frames=250, freq_patch_frames=250,
                post_processing_function=P(stats["mean"], stats["std"]),
            )

            for i in tqdm(range(len(dataset))):
                spec, ann = dataset[i]
                spec = np.flip(spec, axis=0).copy()
                ann = np.flip(ann, axis=0).copy()
                ann_bin = ann > 0.5
                spec_t = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(D.device)
                with torch.no_grad():
                    out = model(spec_t)[0]
                sig = torch.sigmoid(out[:, 0:1]).cpu().numpy()[0, 0]
                if MEDIAN_FILTER_SIZE > 0:
                    sig = median_filter(sig, size=(MEDIAN_FILTER_SIZE, MEDIAN_FILTER_SIZE))

                for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD):
                    pred = sig > thr
                    masks = {
                        "TN": (~pred) & (~ann_bin),
                        "TP": pred & ann_bin,
                        "FN": (~pred) & ann_bin,
                        "FP": pred & (~ann_bin),
                    }
                    for cat, mask in masks.items():
                        if mask.any():
                            vals = spec[mask]
                            h, _ = np.histogram(vals, bins=bins)
                            histograms[species_name][thr][cat] += h
        except Exception as e:
            print(f"error: {e}")

    # Stats per species + aggregate (sum over species)
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

    # Aggregate across species (excluding broken-dataset species)
    agg_species = [sp for sp in species_list if sp not in EXCLUDE_FROM_AGGREGATE]
    print(f"\nAggregate built from {len(agg_species)} species "
          f"(excluded: {sorted(EXCLUDE_FROM_AGGREGATE)})")
    agg = {
        thr: {cat: sum((histograms[sp][thr][cat] for sp in agg_species),
                       np.zeros(HIST_BINS, dtype=np.int64))
              for cat in cats}
        for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD)
    }

    stats_rows = []
    print("\n=== Distance summary (aggregate across all species) ===")
    for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD):
        h = agg[thr]
        d_fp_tp = wasserstein_1d(h["FP"], h["TP"])
        d_fp_tn = wasserstein_1d(h["FP"], h["TN"])
        d_tp_tn = wasserstein_1d(h["TP"], h["TN"])
        print(f"thr={thr}: W(FP,TP)={d_fp_tp:.3f} W(FP,TN)={d_fp_tn:.3f} "
              f"W(TP,TN)={d_tp_tn:.3f} -> FP closer to "
              f"{'TP' if d_fp_tp < d_fp_tn else 'TN'}")
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
                "species": "_aggregate",
                "threshold": thr, "category": cat, **st,
                "frac_above_TN_p95": frac_above,
                "TN_p95_intensity": tn_p95,
            })

    # Per-species rows
    for sp in species_list:
        for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD):
            h_sp = histograms[sp][thr]
            tn_total = int(h_sp["TN"].sum())
            if tn_total > 0:
                centres = 0.5 * (bins[:-1] + bins[1:])
                cdf = np.cumsum(h_sp["TN"]) / tn_total
                tn_p95 = float(centres[np.searchsorted(cdf, 0.95)])
            else:
                tn_p95 = 0.0
            for cat in cats:
                st = stats_from_hist(h_sp[cat])
                if h_sp[cat].sum() > 0:
                    idx_p95 = np.searchsorted(bins, tn_p95)
                    frac_above = float(h_sp[cat][idx_p95:].sum() / h_sp[cat].sum())
                else:
                    frac_above = 0.0
                stats_rows.append({
                    "species": sp, "threshold": thr, "category": cat, **st,
                    "frac_above_TN_p95": frac_above,
                    "TN_p95_intensity": tn_p95,
                })

    with out_stats.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)
    print(f"saved {out_stats}")

    with out_hist.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "threshold", "category", "bin_left",
                    "bin_right", "count"])
        for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD):
            for cat in cats:
                h = agg[thr][cat]
                for i in range(HIST_BINS):
                    w.writerow(["_aggregate", thr, cat, float(bins[i]),
                                float(bins[i + 1]), int(h[i])])
        # Per-species histograms too: the pooled _aggregate mixes species with
        # incompatible intensity scales (e.g. Tursiops-SoCal's bright
        # background), which distorts the shared-axis distribution and inflates
        # the pooled TN-p95. Per-species rows let figures use a representative
        # species with a self-consistent TN-p95 / FP-fraction.
        for sp in species_list:
            for thr in (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD):
                for cat in cats:
                    h = histograms[sp][thr][cat]
                    for i in range(HIST_BINS):
                        w.writerow([sp, thr, cat, float(bins[i]),
                                    float(bins[i + 1]), int(h[i])])
    print(f"saved {out_hist}")

    # Figure: aggregate distributions
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    for ax, thr in zip(axes, (PRIMARY_THRESHOLD, F1_OPT_THRESHOLD)):
        centres = 0.5 * (bins[:-1] + bins[1:])
        for cat, color in zip(cats, ("0.5", "C2", "C1", "C3")):
            h = agg[thr][cat]
            if h.sum() == 0:
                continue
            density = normalize(h) / (bins[1] - bins[0])
            ax.plot(centres, density, label=f"{cat} (n={int(h.sum()):,})",
                    color=color, lw=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("standardised spectrogram intensity")
        ax.set_ylabel("pixel density (log)")
        ax.set_title(f"DCLDE2011 aggregate pixel-intensity by category "
                     f"@ threshold={thr}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=130)
    plt.close(fig)
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
