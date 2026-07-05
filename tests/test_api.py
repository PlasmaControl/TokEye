from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn

import tokeye
from tokeye.api import TokEye


@pytest.fixture
def eye(monkeypatch):
    monkeypatch.setattr(
        "tokeye.hub.load_model", lambda source, device="auto": nn.Conv2d(1, 2, 1)
    )
    return TokEye(n_fft=64, hop=16)


class TestLazyExport:
    def test_package_getattr_resolves_class(self):
        assert tokeye.TokEye is TokEye

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError, match="no_such_thing"):
            _ = tokeye.no_such_thing


class TestPredict:
    def test_1d_signal_returns_two_channel_mask(self, eye):
        rng = np.random.default_rng(0)
        mask = eye(rng.standard_normal(2048))

        assert mask.ndim == 3
        assert mask.shape[0] == 2
        assert np.all((mask >= 0) & (mask <= 1))  # sigmoid scores

    def test_2d_spectrogram_preserves_shape(self, eye):
        mask = eye(np.random.default_rng(0).random((48, 40)))

        assert mask.shape == (2, 48, 40)

    def test_3d_input_raises(self, eye):
        with pytest.raises(ValueError, match="ndim=3"):
            eye(np.zeros((2, 8, 8)))

    def test_call_matches_predict(self, eye):
        arr = np.random.default_rng(1).random((16, 16))

        np.testing.assert_allclose(eye(arr), eye.predict(arr))


class TestLogOption:
    def test_off_by_default_2d_passthrough(self, eye):
        arr = np.random.default_rng(0).random((16, 16))

        np.testing.assert_allclose(eye.spectrogram(arr), arr)

    def test_instance_level_log_applies_log1p(self, monkeypatch):
        monkeypatch.setattr(
            "tokeye.hub.load_model", lambda source, device="auto": nn.Conv2d(1, 2, 1)
        )
        arr = np.random.default_rng(0).random((16, 16))

        eye = TokEye(log=True)

        np.testing.assert_allclose(eye.spectrogram(arr), np.log1p(arr))

    def test_per_call_override_wins(self, eye):
        arr = np.random.default_rng(0).random((16, 16))

        np.testing.assert_allclose(eye.spectrogram(arr, log=True), np.log1p(arr))
        assert eye.log is False  # instance setting untouched

    def test_negative_values_with_log_raise(self, eye):
        with pytest.raises(ValueError, match="negative"):
            eye.spectrogram(np.full((8, 8), -3.0), log=True)

    def test_log_ignored_for_1d_signal(self, eye):
        signal = np.random.default_rng(0).standard_normal(2048)

        # STFT already log-scales; a signed signal must not trip the
        # linear-scale guard.
        spec = eye.spectrogram(signal, log=True)

        assert spec.ndim == 2
