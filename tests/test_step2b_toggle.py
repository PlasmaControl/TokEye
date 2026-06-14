from __future__ import annotations

import numpy as np

from tokeye.training.big_tf_unet_ablation.step_2b_filter_spectrogram import (
    _process_rotation,
)


def test_baseline_off_is_logmag_without_subtraction():
    rng = np.random.default_rng(0)
    data = rng.random((64, 32)).astype(np.float32) + 0.1
    out_off, _bl = _process_rotation(
        data, 2, 2, "fabc", {"lam": 1e5}, baseline_enabled=False
    )
    expected = np.log1p(np.abs(data))
    # edges are masked, but interior should equal log1p(|.|) (no division by baseline)
    assert np.allclose(out_off[5:-5], expected[5:-5], atol=1e-5)


def test_baseline_on_differs_from_off():
    rng = np.random.default_rng(1)
    data = rng.random((64, 32)).astype(np.float32) + 0.1
    on, _ = _process_rotation(data, 2, 2, "fabc", {"lam": 1e5}, baseline_enabled=True)
    off, _ = _process_rotation(data, 2, 2, "fabc", {"lam": 1e5}, baseline_enabled=False)
    assert not np.allclose(on, off)
