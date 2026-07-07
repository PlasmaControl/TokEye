"""Typed, validated run configuration.

``load_run_config`` merges the intern's ``run.yaml`` ON TOP of the bundled
``config/defaults.yaml`` (so run.yaml only carries overrides — the
replace-not-merge behavior of the legacy ``load_settings`` is retired) and
validates the result into a pydantic model. Bad knob values die here, before
any compute, with one-line messages naming the field and the allowed range.
Unknown keys (typos) are rejected too.
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


class SlurmSection(_Section):
    account: str | None
    cpu_partition: str | None
    gpu_partition: str | None
    cpu_cpus: int = Field(gt=0)
    cpu_mem: str
    cpu_time: str
    gpu_cpus: int = Field(gt=0)
    gpu_mem: str
    gpu_time: str


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
    slurm: SlurmSection
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


def _format_validation_error(err: ValidationError) -> str:
    lines = []
    for issue in err.errors():
        loc = ".".join(str(p) for p in issue["loc"])
        lines.append(f"  {loc}: {issue['msg']}")
    n = len(lines)
    plural = "s" if n != 1 else ""
    return f"{n} problem{plural} in run.yaml:\n" + "\n".join(lines)


def load_run_config(run_yaml: str | Path) -> RunConfig:
    """Merge run.yaml over the bundled defaults and validate."""
    run_yaml = Path(run_yaml)
    if not run_yaml.exists():
        raise ConfigError(f"run.yaml not found: {run_yaml}")
    merged = OmegaConf.merge(OmegaConf.load(DEFAULTS_YAML), OmegaConf.load(run_yaml))
    raw = OmegaConf.to_container(merged, resolve=True)
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as err:
        raise ConfigError(_format_validation_error(err)) from None


def check_scale_lock(cfg: RunConfig, run_meta_path: Path) -> None:
    """The scale in run.yaml must match the scale this run was created with.

    A different scale is a different run (different run_id, cache root, and
    workspace) — changing nfft/hop in an existing run.yaml would silently mix
    artifacts, so it is refused here.
    """
    if not run_meta_path.exists():
        return
    meta = json.loads(run_meta_path.read_text())
    locked = (meta.get("nfft"), meta.get("hop"))
    current = (cfg.run.nfft, cfg.run.hop)
    if locked != current:
        raise ConfigError(
            f"run.yaml scale nfft={current[0]}/hop={current[1]} does not match "
            f"this run's locked scale nfft={locked[0]}/hop={locked[1]}. "
            f"To train a different scale, scaffold a new run instead."
        )
