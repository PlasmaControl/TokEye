"""TJ-II ablation figure (error bars = 95% CI across folds) + LaTeX table.

Reads ``data/eval/results/TJII2021_ablation.csv`` produced by
``scripts/eval/TJII2021_ablation.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/scratch/gpfs/nc1514/tokeye")
RESULTS = ROOT / "data" / "eval" / "results"
CSV = RESULTS / "TJII2021_ablation.csv"
FIG_DIR = RESULTS / "figures"

VARIANT_LABEL = {
    "full": "Full recipe",
    "mag": "Magnitude\n(no complex)",
    "nobaseline": "No baseline\nremoval",
    "nodenoise": "No multichannel\ndenoising",
}
METRICS = [("recall", "Recall @0.5"), ("f1", "F1 @F1-opt"), ("iou", "per-image IoU @F1-opt")]


def _read_rows() -> list[dict]:
    with CSV.open() as f:
        return list(csv.DictReader(f))


def make_figure(rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    variants = [r["variant"] for r in rows]
    x = np.arange(len(variants))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(variants)), 5))
    for i, (key, label) in enumerate(METRICS):
        means = np.array([float(r[f"{key}_mean"]) for r in rows])
        lo = np.array([float(r[f"{key}_ci_lo"]) for r in rows])
        hi = np.array([float(r[f"{key}_ci_hi"]) for r in rows])
        yerr = np.clip(np.vstack([means - lo, hi - means]), 0, None)
        ax.bar(x + (i - 1) * width, means, width, yerr=yerr, capsize=4, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABEL.get(v, v) for v in variants], fontsize=9)
    ax.set_ylabel("score")
    n_folds = rows[0].get("n_folds", "?") if rows else "?"
    n_images = rows[0].get("n_images", "?") if rows else "?"
    ax.set_title(
        f"TJ-II ablation (mean $\\pm$ 95% CI across {n_folds} folds; n_images={n_images})"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "tjii_ablation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def make_table(rows: list[dict]) -> None:
    pretty = {
        "full": "Full recipe",
        "mag": "Magnitude-only",
        "nobaseline": "$-$baseline removal",
        "nodenoise": "$-$denoising",
    }
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variant & Recall@0.5 & F1@opt & per-image IoU@opt \\",
        r"\midrule",
    ]
    for r in rows:
        cells = []
        for key in ("recall", "f1", "iou"):
            m, s = float(r[f"{key}_mean"]), float(r[f"{key}_std"])
            lo, hi = float(r[f"{key}_ci_lo"]), float(r[f"{key}_ci_hi"])
            cells.append(f"{m:.3f}$\\pm${s:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"{pretty.get(r['variant'], r['variant'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = RESULTS / "tjii_ablation_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}; run scripts/eval/TJII2021_ablation.py first")
    rows = _read_rows()
    make_figure(rows)
    make_table(rows)


if __name__ == "__main__":
    main()
