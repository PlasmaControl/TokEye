from __future__ import annotations

import torch

from tokeye.training.big_tf_unet_ablation.step_3a_correlation_analysis import (
    BTN,
    BTNMag,
)


def test_complex_btn_shapes():
    m = BTN(in_channels=4, num_layers=3, first_layer_size=8)
    x = torch.randn(2, 4, 64, 32, 2)  # (B,C,F,T,2)
    y = m(x)
    assert y.shape == (2, 4, 64, 32, 2)


def test_mag_btn_shapes():
    m = BTNMag(in_channels=4, num_layers=3, first_layer_size=8)
    x = torch.randn(2, 4, 64, 32)  # (B,C,F,T) magnitude
    y = m(x)
    assert y.shape == (2, 4, 64, 32)
