from __future__ import annotations

import logging
import shutil
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Original helpers (unchanged API)
# ---------------------------------------------------------------------------


def load_settings(
    config_path: Path | str | None,
    default_settings: dict | None = None,
) -> dict:
    """Load settings from YAML file or use defaults."""
    if default_settings is None:
        default_settings = {}
    if config_path is None:
        cfg = default_settings
    else:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.error("Config file not found")

        cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.to_container(cfg, resolve=True)

        for key, value in cfg.items():
            if (key.endswith("_dir") or key.endswith("_path")) and value is not None:
                cfg[key] = Path(value)

    return cfg


def load_input_paths(
    input_dir: Path,
) -> list[Path]:
    """Load input paths from input directory."""
    input_paths = list(input_dir.glob("*.joblib"))
    input_paths.sort(key=lambda x: int(x.stem))
    return input_paths


def setup_directory(
    path: Path | str,
    overwrite: bool = True,
) -> Path:
    """Setup output directory."""
    path = Path(path)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Pipeline config helpers (new)
# ---------------------------------------------------------------------------

_PIPELINE_YAML = Path(__file__).resolve().parent.parent / "config" / "ablation.yaml"


def load_pipeline_config(config_path: Path | str | None = None) -> dict:
    """Load the central ``pipeline.yaml`` configuration.

    Falls back to the default bundled config when *config_path* is ``None``.
    """
    path = Path(config_path) if config_path else _PIPELINE_YAML
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def get_combo_settings(
    config: dict,
    combo,  # ComboConfig — avoid circular import
    step_name: str,
) -> dict:
    """Build a fully-resolved settings dict for one step of one combo.

    Merges the relevant section of *config* with derived values from
    *combo* (modality channels, STFT params, etc.) and resolves paths.
    """
    from tokeye.training.big_tf_unet_ablation.config.modality import (
        ComboConfig,  # local import
    )

    assert isinstance(combo, ComboConfig)

    cache_dir = Path(config["paths"]["cache_dir"])
    mod = combo.modality
    stft = combo.stft

    # Base settings shared by all steps
    base: dict = {
        "modality_name": mod.name,
        "input_key": mod.input_key,
        "input_channels": list(mod.channels),
        "num_channels": mod.num_channels,
        "total_channels": mod.total_channels,
        "adjacent_channels": mod.adjacent_channels,
        "nfft": stft.nfft,
        "hop_length": stft.hop_length,
        "freq_bins": stft.freq_bins,
        "combo_id": combo.combo_id,
        "overwrite": True,
    }

    # Shared step dirs (0a, 0b)
    base["shared_step_0a_dir"] = ComboConfig.shared_dir(cache_dir, "step_0a")
    base["shared_step_0b_dir"] = ComboConfig.shared_dir(cache_dir, "step_0b")

    # Per-modality dirs (0c, 1a)
    base["modality_step_0c_dir"] = combo.modality_dir(cache_dir, "step_0c")
    base["modality_step_1a_dir"] = combo.modality_dir(cache_dir, "step_1a")

    # Per-combo dirs (2a onward)
    combo_base = cache_dir / mod.name / f"{stft.nfft}_{stft.hop_length}"
    for step in [
        "step_2a", "step_2b", "step_2b_baseline",
        "step_3a", "step_3b",
        "step_4a", "step_4a_baseline",
        "step_5a",
        "step_6a", "step_6b", "step_6c", "step_6d",
    ]:
        base[f"{step}_dir"] = combo_base / step

    base["frame_info_path"] = combo.modality_dir(cache_dir, "frame_info.csv")
    base["threshold_output_path"] = combo_base / "thresholds.csv"

    # Merge step-specific config sections
    for section in ["extraction", "baseline", "correlation", "threshold", "refiner", "final"]:
        if section in config:
            base[section] = config[section]

    # Paths from top-level config
    for key in ["shots_path", "faith_cfg_path", "cache_dir", "training_dir", "task_matrix_path"]:
        if key in config.get("paths", {}):
            base[key] = Path(config["paths"][key])

    return base
