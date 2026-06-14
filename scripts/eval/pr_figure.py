"""Plot PR curves across all four datasets from saved sweep CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

root = Path("/scratch/gpfs/nc1514/tokeye/data/eval/results")
fig_dir = root / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)


def load_sweep(path: Path, key_field: str):
    """Return dict: key -> list of (threshold, P, R, F1)."""
    out: dict[str, list] = {}
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            k = row[key_field]
            out.setdefault(k, []).append((
                float(row["threshold"]),
                float(row["precision"]),
                float(row["recall"]),
                float(row["f1"]),
            ))
    for k in out:
        out[k].sort()
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_pr, ax_f1 = axes

    p_raddet = root / "RadDet_pr_sweep.csv"
    p_dclde = root / "DCLDE2011_pr_sweep.csv"
    p_tjii = root / "TJII2021_pr_sweep.csv"

    if p_tjii.exists():
        d = load_sweep(p_tjii, "dataset")
        for k, rows in d.items():
            ts = [r[0] for r in rows]
            ps = [r[1] for r in rows]
            rs = [r[2] for r in rows]
            f1s = [r[3] for r in rows]
            ax_pr.plot(rs, ps, marker="o", ms=3, label=f"TJII2021 ({k})")
            ax_f1.plot(ts, f1s, marker="o", ms=3, label="TJII2021")

    if p_dclde.exists():
        d = load_sweep(p_dclde, "species")
        for k, rows in d.items():
            ts = [r[0] for r in rows]
            ps = [r[1] for r in rows]
            rs = [r[2] for r in rows]
            f1s = [r[3] for r in rows]
            ax_pr.plot(rs, ps, marker="^", ms=3, label=f"DCLDE {k[:18]}")
            ax_f1.plot(ts, f1s, marker="^", ms=3, label=f"DCLDE {k[:18]}")

    if p_raddet.exists():
        d = load_sweep(p_raddet, "variant")
        for k, rows in d.items():
            ts = [r[0] for r in rows]
            ps = [r[1] for r in rows]
            rs = [r[2] for r in rows]
            f1s = [r[3] for r in rows]
            ax_pr.plot(rs, ps, marker="s", ms=3, label=f"RadDet {k.replace('RadDet40k', '')}")
            ax_f1.plot(ts, f1s, marker="s", ms=3, label=f"RadDet {k.replace('RadDet40k', '')}")

    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title("PR curves (pixel-level)")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1)
    ax_pr.legend(fontsize=7, loc="upper right")
    ax_pr.grid(True, alpha=0.3)

    ax_f1.set_xlabel("sigmoid threshold")
    ax_f1.set_ylabel("F1 score")
    ax_f1.set_title("F1 vs threshold")
    ax_f1.set_xlim(0, 1)
    ax_f1.legend(fontsize=7, loc="upper right")
    ax_f1.grid(True, alpha=0.3)

    fig.tight_layout()
    out = fig_dir / "pr_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
