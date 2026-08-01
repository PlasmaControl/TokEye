"""Typed, validated run configuration.

``load_run_config`` merges a workspace's config files ON TOP of the bundled
``config/defaults.yaml``, in this order:

    config/defaults.yaml  <-  run.yml  <-  step_N_*.yml

so each file only carries what it overrides (the replace-not-merge behavior of
the legacy ``load_settings`` is retired). Splitting the *files* per step does not
split the *config object*: the merge is deep and the result is validated as one
whole ``RunConfig``, so ``step_3`` still reads ``run.nfft`` and ``modalities``.

Bad knob values die here, before any compute, with one-line messages naming the
field and the allowed range. Unknown keys (typos) are rejected too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .paths import DEFAULTS_YAML, SCALE_CONFIGS

Auto = Literal["auto"]


class ConfigError(ValueError):
    """A run.yaml problem, formatted for humans (one line per issue)."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSection(_Section):
    nfft: int = Field(gt=0)
    hop: int = Field(gt=0)
    allow_custom_scale: bool = False

    @model_validator(mode="after")
    def _check_scale(self) -> RunSection:
        if self.hop > self.nfft:
            raise ValueError(f"hop ({self.hop}) must be <= nfft ({self.nfft})")
        if not self.allow_custom_scale and (self.nfft, self.hop) not in SCALE_CONFIGS:
            grid = ", ".join(f"{n}/{h}" for n, h in SCALE_CONFIGS)
            raise ValueError(
                f"(nfft={self.nfft}, hop={self.hop}) is not in the scale grid "
                f"[{grid}]. Set run.allow_custom_scale: true to use it anyway."
            )
        return self


class ModalityConfig(_Section):
    input_key: str
    channels: list[int] = Field(min_length=1)


class ExtractionSection(_Section):
    subseq_len: int = Field(gt=0)
    preemphasis_coeff: float = Field(ge=0.0, le=1.0)
    fs_khz: float = Field(gt=0)
    target_rate_khz: float = Field(gt=0)
    ip_threshold: float = Field(ge=0.0)
    max_windows_per_shot_precap: int = Field(gt=0)


class WindowFilterSection(_Section):
    enabled: bool
    weights: str
    max_windows_per_shot: int = Field(gt=0)
    activity_threshold: float = Field(gt=0.0, lt=1.0)
    min_activity: float = Field(ge=0.0, lt=1.0)
    mean: float | Auto
    std: float | Auto


class BaselineSection(_Section):
    method: str
    lam: float | Auto
    edge_method: Literal["energy", "gradient"]
    edge_k: float = Field(gt=1.0)
    edge_max_fraction: float = Field(gt=0.0, le=0.5)
    gradient_threshold: float = Field(gt=0.0)


class DenoiseSection(_Section):
    representation: Literal["complex", "magnitude"]
    normalization: Literal["robust_asinh", "zscore"]
    a: float = Field(gt=0.0)
    first_layer_size: int = Field(gt=0)
    num_layers: int | Auto
    batch_size: int | Auto
    base_batch_size: int = Field(gt=0)
    precision: str
    max_epochs: int = Field(gt=0)
    tv_patience: int = Field(ge=0)
    num_workers: int = Field(ge=0)


class LabelsSection(_Section):
    knee_sensitivity: float = Field(gt=0.0)
    delta: float
    fallback_frac: float = Field(gt=0.0, lt=0.5)
    min_size: int | Auto
    min_size_fraction: float = Field(gt=0.0, lt=0.1)
    remove_bottom_rows: int | Auto
    remove_top_rows: int | Auto
    row_removal_fraction_bottom: float = Field(ge=0.0, lt=0.2)
    row_removal_fraction_top: float = Field(ge=0.0, lt=0.2)


class DatasetSection(_Section):
    a: float = Field(gt=0.0)
    stats_windows: int = Field(gt=0)


class RefineSection(_Section):
    model_trust: float = Field(ge=0.0, le=1.0)
    n_folds: int = Field(ge=2)
    first_layer_size: int = Field(gt=0)
    num_layers: int | Auto
    batch_size: int | Auto
    base_batch_size: int = Field(gt=0)
    precision: str
    max_epochs: int = Field(gt=0)
    loss_type: str
    num_workers: int = Field(ge=0)


class FinalSection(_Section):
    first_layer_size: int = Field(gt=0)
    num_layers: int | Auto
    batch_size: int | Auto
    base_batch_size: int = Field(gt=0)
    precision: str
    max_epochs: int = Field(gt=0)
    loss_type: str
    gamma: float = Field(gt=0.0)
    num_workers: int = Field(ge=0)


class EvalSection(_Section):
    dataset_dir: str
    n_thresholds: int = Field(ge=2)


class PathsSection(_Section):
    shots_path: str
    foundation_dir: str


class SmokeSection(_Section):
    enabled: bool
    n_shots: int = Field(gt=0)
    max_windows_per_shot: int = Field(gt=0)
    n_folds: int = Field(ge=2)
    max_epochs: int = Field(gt=0)
    refine_max_epochs: int = Field(gt=0)
    final_max_epochs: int = Field(gt=0)


class RunConfig(_Section):
    run: RunSection
    overwrite: bool = True
    modalities: dict[str, ModalityConfig] = Field(min_length=1)
    extraction: ExtractionSection
    window_filter: WindowFilterSection
    baseline: BaselineSection
    denoise: DenoiseSection
    labels: LabelsSection
    dataset: DatasetSection
    refine: RefineSection
    final: FinalSection
    eval: EvalSection
    paths: PathsSection
    smoke: SmokeSection

    @property
    def modality_names(self) -> list[str]:
        return list(self.modalities)

    @property
    def n_freq(self) -> int:
        return self.run.nfft // 2 + 1

    @property
    def n_time(self) -> int:
        return self.extraction.subseq_len // self.run.hop + 1


def _format_validation_error(err: ValidationError, sources: list[Path]) -> str:
    lines = []
    for issue in err.errors():
        loc = ".".join(str(p) for p in issue["loc"])
        lines.append(f"  {loc}: {issue['msg']}")
    n = len(lines)
    plural = "s" if n != 1 else ""
    where = " + ".join(p.name for p in sources)
    return f"{n} problem{plural} in {where}:\n" + "\n".join(lines)


def config_sources(config_path: str | Path) -> list[Path]:
    """The files merged for ``config_path``, in increasing precedence.

    ``run.yml`` is always included: a step's own ``.yml`` carries only its knob
    section, so the shared scale/modalities/paths come from its sibling. Passing
    ``run.yml`` itself yields just that file (no self-duplication).
    """
    config_path = Path(config_path)
    run_yml = config_path.parent / "run.yml"
    if config_path.name == "run.yml" or not run_yml.exists():
        return [config_path]
    return [run_yml, config_path]


def load_run_config(config_path: str | Path) -> RunConfig:
    """Merge ``run.yml`` then the given step config over the defaults; validate."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(f"config not found: {config_path}")
    sources = config_sources(config_path)
    merged = OmegaConf.load(DEFAULTS_YAML)
    for source in sources:
        merged = OmegaConf.merge(merged, OmegaConf.load(source))
    raw = OmegaConf.to_container(merged, resolve=True)
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as err:
        raise ConfigError(_format_validation_error(err, sources)) from None


def check_scale_lock(cfg: RunConfig, run_meta_path: Path) -> None:
    """The configured scale must match the scale this run first ran at.

    A different scale is a different run (different cache root and workspace) —
    changing nfft/hop partway through would silently mix artifacts, so it is
    refused here. The lock file is written by the runner on the first step, and
    this check no-ops until then.
    """
    if not run_meta_path.exists():
        return
    meta = json.loads(run_meta_path.read_text())
    locked = (meta.get("nfft"), meta.get("hop"))
    current = (cfg.run.nfft, cfg.run.hop)
    if locked != current:
        raise ConfigError(
            f"run.yml scale nfft={current[0]}/hop={current[1]} does not match "
            f"this run's locked scale nfft={locked[0]}/hop={locked[1]}. "
            f"To train a different scale, copy the template to a new folder."
        )
