"""Per-class recall bar chart across RadDet variants."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("/scratch/gpfs/nc1514/tokeye/data/eval/results")
fig_dir = root / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = [
    "Rect", "Barker", "Frank", "P1", "P2", "P3", "P4",
    "Px", "ZadoffChu", "LFM", "FMCW",
]


def main():
    csv_path = root / "RadDet_per_class.csv"
    data = defaultdict(lambda: defaultdict(float))
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            data[row["variant"]][row["class_name"]] = float(row["recall"])

    variants = list(data.keys())
    n_var = len(variants)
    width = 0.85 / n_var

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(CLASS_ORDER))
    for i, variant in enumerate(variants):
        ys = [data[variant][c] for c in CLASS_ORDER]
        short = variant.replace("RadDet40k", "")
        ax.bar(x + i * width - 0.4, ys, width, label=short)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER, rotation=30, ha="right")
    ax.set_ylabel("recall @ IoU=0.5")
    ax.set_title("Per-class detection recall on RadDet (zero-shot, plasma-trained model)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = fig_dir / "RadDet_per_class.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
