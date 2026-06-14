from __future__ import annotations

import numpy as np

from tokeye.training.big_tf_unet_ablation.step_6a_convert_tif import (
    process_data_img,
    process_data_mask_dual,
)


def test_dual_mask_stacks_two_channels():
    coh = np.ones((20, 30, 1), dtype=np.float32)
    tra = np.zeros((20, 30, 1), dtype=np.float32)
    out = process_data_mask_dual(coh, tra)
    assert out.shape == (2, 20, 30)
    assert out[0].sum() > 0 and out[1].sum() == 0  # ch0 coherent, ch1 transient


def test_process_data_img_standardizes():
    # complex (F, T, 2) input for one channel
    rng = np.random.default_rng(0)
    data = rng.random((16, 16, 2)).astype(np.float32)
    stats = {"means": [0.3], "stds": [0.2]}
    out = process_data_img(data, channel_idx=0, stats=stats, zscore_clip=3)
    assert out.shape == (16, 16)
    assert out.dtype == np.float32
