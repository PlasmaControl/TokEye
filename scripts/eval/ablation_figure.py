"""Bar chart of ablation results."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("/scratch/gpfs/nc1514/tokeye/data/eval/results")
fig_dir = root / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    with (root / "RadDet_ablation.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    names = [r["config"].replace("+flip+invert+unshift", "+f+i+u") for r in rows]
    p = [float(r["pixel_p"]) for r in rows]
    rr = [float(r["pixel_r"]) for r in rows]
    ap10 = [float(r["ap_iou0.1"]) for r in rows]
    ap10_lo = [float(r["ap_iou0.1"]) - float(r["ap_iou0.1_lo"]) for r in rows]
    ap10_hi = [float(r["ap_iou0.1_hi"]) - float(r["ap_iou0.1"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(names))

    axes[0].bar(x - 0.2, p, 0.4, label="precision", color="C0")
    axes[0].bar(x + 0.2, rr, 0.4, label="recall", color="C1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("pixel-level metric")
    axes[0].set_title("Pixel P / R by ablation config (RadDet 128HW009)")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, ap10, 0.6, color="C2",
                yerr=[ap10_lo, ap10_hi], capsize=3,
                label="AP @ IoU=0.1 (bootstrap 95% CI)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("detection AP @ IoU=0.1")
    axes[1].set_title("Detection AP by ablation config")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out = fig_dir / "RadDet_ablation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
