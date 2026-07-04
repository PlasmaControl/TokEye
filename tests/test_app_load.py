from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn

from tokeye.app.analyze import load


def test_model_load_delegates_to_hub_and_warmup(monkeypatch):
    """model_load should just be hub.load_model(...) + inference.warmup(...)."""
    stub_model = nn.Conv2d(1, 2, 1)
    calls = {}

    def fake_load_model(source, device):
        calls["source"] = source
        calls["device"] = device
        return stub_model

    def fake_warmup(model):
        calls["warmed_up"] = model

    monkeypatch.setattr("tokeye.hub.load_model", fake_load_model)
    monkeypatch.setattr("tokeye.inference.warmup", fake_warmup)

    result = load.model_load("big_tf_unet", device="cpu")

    assert result is stub_model
    assert calls["source"] == "big_tf_unet"
    assert calls["device"] == "cpu"
    assert calls["warmed_up"] is stub_model


def test_find_models_lists_fake_pt_files(tmp_path, monkeypatch):
    monkeypatch.setattr(load, "MODEL_DIR", tmp_path)
    (tmp_path / "model_a.pt").touch()
    (tmp_path / "model_b.pt2").touch()
    (tmp_path / "not_a_model.txt").touch()

    models = load.find_models()

    assert sorted(models) == sorted(
        [str(tmp_path / "model_a.pt"), str(tmp_path / "model_b.pt2")]
    )


def test_load_single_1d_signal_returns_spectrogram(tmp_path):
    signal_path = tmp_path / "signal.npy"
    np.save(signal_path, np.random.default_rng(0).normal(size=4096))

    transform_args = {
        "n_fft": 256,
        "hop_length": 64,
        "clip_dc": True,
        "percentile_low": 1.0,
        "percentile_high": 99.0,
    }
    spectrogram = load.load_single(signal_path, transform_args)

    assert spectrogram is not None
    assert spectrogram.ndim == 2


def test_load_single_2d_signal_returns_none(tmp_path):
    signal_path = tmp_path / "signal_2d.npy"
    np.save(signal_path, np.zeros((4, 4)))

    transform_args = {"n_fft": 256, "hop_length": 64}
    assert load.load_single(signal_path, transform_args) is None


@pytest.mark.parametrize("cached_value", [None, "/path/to/cached.pt"])
def test_is_model_cached(monkeypatch, cached_value):
    monkeypatch.setattr(
        load, "try_to_load_from_cache", lambda repo_id, filename: cached_value
    )

    assert load.is_model_cached("big_tf_unet") is (cached_value is not None)


def test_create_app_constructs_without_error():
    """Blocks construction should be pure UI wiring: no model load, no network."""
    from tokeye.app.__main__ import create_app

    app = create_app()

    assert app is not None
