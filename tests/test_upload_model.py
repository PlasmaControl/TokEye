from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tokeye.hub import MODEL_REGISTRY

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "upload_model.py"


def _load_upload_model_module():
    spec = importlib.util.spec_from_file_location("upload_model", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


upload_model = _load_upload_model_module()


def test_verify_checkpoint_accepts_good_state_dict(tmp_path):
    spec = MODEL_REGISTRY["big_tf_unet"]
    path = tmp_path / "good.pt"
    torch.save(spec.builder().state_dict(), path)

    upload_model.verify_checkpoint(path, spec.builder)  # must not raise


def test_verify_checkpoint_rejects_renamed_keys(tmp_path):
    spec = MODEL_REGISTRY["big_tf_unet"]
    path = tmp_path / "renamed.pt"
    state_dict = spec.builder().state_dict()
    renamed = {f"old.{key}": value for key, value in state_dict.items()}
    torch.save(renamed, path)

    with pytest.raises(SystemExit):
        upload_model.verify_checkpoint(path, spec.builder)


def test_verify_checkpoint_rejects_pickled_full_module(tmp_path):
    spec = MODEL_REGISTRY["big_tf_unet"]
    path = tmp_path / "legacy.pt"
    torch.save(nn.Linear(2, 2), path)

    with pytest.raises(SystemExit):
        upload_model.verify_checkpoint(path, spec.builder)
