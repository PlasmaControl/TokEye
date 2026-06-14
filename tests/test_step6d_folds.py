from __future__ import annotations

from tokeye.training.big_tf_unet_ablation.step_6d_final import make_fold_indices


def test_make_fold_indices_partition():
    folds = make_fold_indices(n_samples=10, n_folds=5, seed=42)
    assert len(folds) == 5
    # each fold: (train_idx, val_idx); val sets partition 0..9
    val_all = sorted(i for _, val in folds for i in val)
    assert val_all == list(range(10))
    for train, val in folds:
        assert set(train).isdisjoint(val)
        assert len(train) + len(val) == 10
