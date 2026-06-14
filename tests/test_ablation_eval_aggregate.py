from __future__ import annotations

from tokeye.extra.eval.fold_stats import aggregate_across_folds


def test_aggregate_mean_std_ci():
    rows = [{"recall": r, "f1": r, "iou": r} for r in [0.50, 0.52, 0.48, 0.54, 0.46]]
    agg = aggregate_across_folds(rows, ["recall", "f1", "iou"], ci=95.0)
    assert abs(agg["recall"]["mean"] - 0.50) < 1e-9
    assert agg["recall"]["std"] > 0
    assert agg["recall"]["ci_lo"] < agg["recall"]["mean"] < agg["recall"]["ci_hi"]
    assert agg["recall"]["n"] == 5


def test_single_fold_collapses_ci():
    agg = aggregate_across_folds([{"recall": 0.7}], ["recall"], ci=95.0)
    assert agg["recall"]["mean"] == 0.7
    assert agg["recall"]["ci_lo"] == 0.7 == agg["recall"]["ci_hi"]
    assert agg["recall"]["n"] == 1
