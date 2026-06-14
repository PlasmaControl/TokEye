"""Across-fold aggregation: mean +/- std and t-based 95% CI."""

from __future__ import annotations

import numpy as np


def aggregate_across_folds(rows: list[dict], metrics: list[str], ci: float = 95.0) -> dict:
    """Aggregate per-fold metric rows into mean/std/CI/n per metric.

    Uses Student-t CI on the across-fold mean (small n). With a single fold the
    CI collapses to the point estimate.
    """
    from scipy import stats as st

    out: dict = {}
    for m in metrics:
        vals = np.array([r[m] for r in rows], dtype=float)
        n = len(vals)
        mean = float(vals.mean()) if n else float("nan")
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            sem = std / np.sqrt(n)
            t = st.t.ppf(1 - (1 - ci / 100) / 2, df=n - 1)
            lo, hi = mean - t * sem, mean + t * sem
        else:
            lo = hi = mean
        out[m] = {"mean": mean, "std": std, "ci_lo": float(lo), "ci_hi": float(hi), "n": n}
    return out
