from __future__ import annotations

import numpy as np

from tokeye.training.big_tf_unet_ablation.window_filter import (
    activity_score_from_sigmoid,
    select_window_indices,
)


def test_activity_score_counts_active_fraction():
    sig = np.zeros((2, 10, 10), dtype=np.float32)  # (2ch, H, W)
    sig[0, :5, :] = 0.9  # half active in coherent ch
    score = activity_score_from_sigmoid(sig, threshold=0.5)
    assert abs(score - 0.5) < 1e-6  # max over channels of active fraction


def test_select_window_indices_caps_and_drops_empties():
    scores = {0: 0.9, 1: 0.0001, 2: 0.4, 3: 0.7, 4: 0.0}
    kept = select_window_indices(scores, max_windows=2, min_activity=0.001)
    assert kept == [0, 3]  # top-2 above floor, sorted by score desc
