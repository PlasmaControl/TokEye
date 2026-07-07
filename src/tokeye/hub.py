"""Model loading: HuggingFace Hub downloads plus local file resolution.

Gradio-free so it can be shared between the app and a future CLI.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .models.ae_tf_maskrcnn.config_ae_tf_maskrcnn import AETFMaskConfig
from .models.ae_tf_maskrcnn.model_ae_tf_maskrcnn import AETFMaskModel
from .models.big_tf_unet.config_big_tf_unet import BigTFUNetConfig
from .models.big_tf_unet.model_big_tf_unet import BigTFUNetModel

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = os.environ.get("TOKEYE_HF_REPO", "nc1/big_tf_unet")
DEFAULT_MODEL = "big_tf_unet"

_PATH_SUFFIXES = {".pt", ".pt2"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str  # file in the HF repo
    builder: Callable[[], nn.Module]
    repo_id: str | None = None  # None -> DEFAULT_REPO_ID (TOKEYE_HF_REPO override)


# Insertion order matters: _build_from_state_dict tries specs in order, so the
# default segmentation model must stay first — U-Net checkpoints should never
# construct the (much slower) R-CNN builder.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "big_tf_unet": ModelSpec(
        "big_tf_unet",
        "big_tf_unet_251210.pt",
        lambda: BigTFUNetModel(BigTFUNetConfig()),
    ),
    "ae_tf_maskrcnn": ModelSpec(
        "ae_tf_maskrcnn",
        "ae_tf_maskrcnn_251223.pt",
        lambda: AETFMaskModel(AETFMaskConfig(weights=None)),
        repo_id="nc1/ae_tf_maskrcnn",
    ),
}


def repo_for(name: str) -> str:
    """Hugging Face repo a model name resolves to (for error messages)."""
    spec = MODEL_REGISTRY.get(str(name))
    if spec is not None and spec.repo_id is not None:
        return spec.repo_id
    return DEFAULT_REPO_ID


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def download_model(name: str = DEFAULT_MODEL, repo_id: str | None = None) -> Path:
    try:
        spec = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {name!r}; valid names: {sorted(MODEL_REGISTRY)}"
        ) from exc
    resolved_repo_id = repo_id or spec.repo_id or DEFAULT_REPO_ID
    return Path(hf_hub_download(resolved_repo_id, spec.filename))


def _build_from_state_dict(state_dict: Mapping, device: str) -> nn.Module:
    mismatches: list[str] = []
    for spec in MODEL_REGISTRY.values():
        model = spec.builder()
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            mismatches.append(f"{spec.name}: {exc}")
            continue
        return model.to(device).eval()

    details = "\n".join(mismatches)
    raise ValueError(
        "State dict does not match any known TokEye architecture "
        f"({', '.join(sorted(MODEL_REGISTRY))}).\n{details}"
    )


def _load_from_registry(name: str, device: str) -> nn.Module:
    spec = MODEL_REGISTRY[name]
    path = download_model(name)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model = spec.builder()
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _load_pt2(path: Path, device: str) -> nn.Module:
    module = torch.export.load(str(path)).module()
    return module.to(device)


def _load_pt(path: Path, device: str) -> nn.Module:
    try:
        loaded = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # Legacy checkpoint pickled as a full module (not just a state dict).
        # Only ever done for local files: the registry/download path above
        # always loads with weights_only=True.
        logger.warning(
            "%s could not be loaded safely (weights_only=True); falling back "
            "to unpickling the full file. Only do this for local files you "
            "trust.",
            path,
        )
        model = torch.load(path, map_location=device, weights_only=False)
        return model.to(device).eval()

    if isinstance(loaded, Mapping):
        return _build_from_state_dict(loaded, device)

    return loaded.to(device).eval()


def load_model(source: str | Path = DEFAULT_MODEL, device: str = "auto") -> nn.Module:
    resolved_device = resolve_device(device)
    name = str(source)

    if name in MODEL_REGISTRY:
        return _load_from_registry(name, resolved_device)

    path = Path(source)
    if not path.exists():
        if path.suffix in _PATH_SUFFIXES:
            raise FileNotFoundError(f"Model file not found: {path}")
        raise ValueError(
            f"Unknown model {name!r}; valid registry names: {sorted(MODEL_REGISTRY)}"
        )

    if path.suffix == ".pt2":
        return _load_pt2(path, resolved_device)
    if path.suffix == ".pt":
        return _load_pt(path, resolved_device)

    raise ValueError(f"Unsupported model file suffix: {path.suffix!r}")
