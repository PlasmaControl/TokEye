from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from tokeye.hub import DEFAULT_MODEL, MODEL_REGISTRY, load_model


def test_default_model_is_registered():
    assert DEFAULT_MODEL in MODEL_REGISTRY


def test_load_model_from_registry_downloads_and_loads(tmp_path, monkeypatch):
    spec = MODEL_REGISTRY["big_tf_unet"]
    weights_path = tmp_path / spec.filename
    torch.save(spec.builder().state_dict(), weights_path)

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        assert filename == spec.filename
        return str(weights_path)

    monkeypatch.setattr("tokeye.hub.hf_hub_download", fake_hf_hub_download)

    model = load_model("big_tf_unet", device="cpu")

    assert not model.training  # eval mode
    with torch.no_grad():
        out = model(torch.randn(1, 1, 64, 64))
    assert out[0].shape == (1, 2, 64, 64)


def test_load_model_from_local_path_with_matching_state_dict(tmp_path):
    spec = MODEL_REGISTRY["big_tf_unet"]
    weights_path = tmp_path / "checkpoint.pt"
    torch.save(spec.builder().state_dict(), weights_path)

    model = load_model(weights_path, device="cpu")

    assert not model.training
    with torch.no_grad():
        out = model(torch.randn(1, 1, 64, 64))
    assert out[0].shape == (1, 2, 64, 64)


def test_load_model_legacy_pickled_module_falls_back(tmp_path, caplog):
    legacy_path = tmp_path / "legacy.pt"
    torch.save(nn.Linear(2, 2), legacy_path)

    with caplog.at_level(logging.WARNING, logger="tokeye.hub"):
        model = load_model(legacy_path, device="cpu")

    assert isinstance(model, nn.Linear)
    assert not model.training
    assert "trust" in caplog.text.lower()


def test_load_model_unknown_name_raises_value_error_listing_registry():
    with pytest.raises(ValueError, match="big_tf_unet"):
        load_model("not_a_real_model_name")


def test_load_model_nonexistent_path_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_model("/nonexistent/directory/does_not_exist.pt")
