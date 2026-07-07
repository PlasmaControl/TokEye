"""Single source of truth for run identity, step registry, and artifact paths.

One pipeline run = one (nfft, hop) scale. Every artifact of a run lives under
``data/cache/big_tf_unet_2/<run_id>/`` (pipeline caches) and
``model/big_tf_unet_2/<run_id>/`` (trained weights). The intern workspace
(run.yaml + notebooks) lives in ``dev/training/<run_id>/`` (gitignored).

All paths are repo-root relative — run every entry point from the repo root,
matching the convention of the other training pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# The canonical scale grid (migrated from big_tf_unet/step_7b_train_multiscale).
SCALE_CONFIGS: list[tuple[int, int]] = [
    (128, 64),
    (256, 64),
    (256, 128),
    (512, 128),
    (512, 256),
    (1024, 128),
    (1024, 256),
    (1024, 512),
    (2048, 256),
    (2048, 512),
]


def run_id_for(nfft: int, hop: int, suffix: str = "") -> str:
    base = f"nfft{nfft}_hop{hop}"
    return f"{base}_{suffix}" if suffix else base


def repo_root() -> Path:
    """Locate the repo root robustly, independent of the current directory.

    Notebooks run with their own directory as cwd, so the root cannot be
    ``Path.cwd()``. Walk up from cwd for a ``pyproject.toml`` (handles running
    from anywhere inside a checkout); fall back to this module's fixed location
    (``<root>/src/tokeye/training/big_tf_unet_2/paths.py``).
    """
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class StepSpec:
    """Static description of one pipeline step."""

    name: str  # e.g. "step_2"
    title: str  # short human label for status tables / notebooks
    module: str  # module basename inside this package
    per_modality: bool  # True: runs once per modality, artifacts under <mod>/
    exec_mode: str  # "inline" | "sbatch_cpu" | "sbatch_gpu"
    knob_section: str  # run.yaml section holding this step's knobs


STEPS: list[StepSpec] = [
    StepSpec("step_0", "intake", "step_0_intake", True, "inline", "extraction"),
    StepSpec(
        "step_1", "spectrogram", "step_1_spectrogram", True, "inline", "window_filter"
    ),
    StepSpec("step_2", "baseline", "step_2_baseline", True, "sbatch_cpu", "baseline"),
    StepSpec("step_3", "denoise", "step_3_denoise", True, "sbatch_gpu", "denoise"),
    StepSpec("step_4", "labels", "step_4_labels", True, "inline", "labels"),
    StepSpec("step_5", "dataset", "step_5_dataset", False, "inline", "dataset"),
    StepSpec("step_6", "refine", "step_6_refine", False, "sbatch_gpu", "refine"),
    StepSpec("step_7", "final", "step_7_final", False, "sbatch_gpu", "final"),
    StepSpec("step_8", "eval", "step_8_eval", False, "inline", "eval"),
]

STEP_ORDER: list[str] = [s.name for s in STEPS]
_STEP_BY_NAME: dict[str, StepSpec] = {s.name: s for s in STEPS}


def get_step(name: str) -> StepSpec:
    try:
        return _STEP_BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"Unknown step {name!r}. Valid steps: {', '.join(STEP_ORDER)}"
        ) from None


def steps_after(name: str) -> list[StepSpec]:
    """Steps strictly downstream of ``name`` in pipeline order."""
    idx = STEP_ORDER.index(get_step(name).name)
    return STEPS[idx + 1 :]


@dataclass(frozen=True)
class RunPaths:
    """Every artifact path of one run, derived from its run_id."""

    run_id: str
    root: Path = field(default_factory=repo_root)

    # ------------------------------------------------------------------
    # Roots
    # ------------------------------------------------------------------

    @property
    def cache_root(self) -> Path:
        return self.root / "data" / "cache" / "big_tf_unet_2" / self.run_id

    @property
    def model_dir(self) -> Path:
        return self.root / "model" / "big_tf_unet_2" / self.run_id

    @property
    def workspace(self) -> Path:
        return self.root / "dev" / "training" / self.run_id

    # ------------------------------------------------------------------
    # Run-level files
    # ------------------------------------------------------------------

    @property
    def run_yaml(self) -> Path:
        return self.workspace / "run.yaml"

    @property
    def run_meta(self) -> Path:
        return self.cache_root / "run_meta.json"

    @property
    def task_matrix_path(self) -> Path:
        return self.cache_root / "task_matrix.json"

    @property
    def resolved_params_path(self) -> Path:
        return self.cache_root / "resolved_params.yaml"

    @property
    def slurm_dir(self) -> Path:
        return self.cache_root / "slurm"

    # ------------------------------------------------------------------
    # Step artifacts
    # ------------------------------------------------------------------

    def mod_dir(self, modality: str) -> Path:
        return self.cache_root / modality

    def step_h5(self, step: str, modality: str | None = None) -> Path:
        spec = get_step(step)
        if spec.per_modality:
            if modality is None:
                raise ValueError(f"{step} is per-modality; a modality is required")
            return self.mod_dir(modality) / f"{step}.h5"
        return self.cache_root / f"{step}.h5"

    def baseline_h5(self, modality: str) -> Path:
        return self.mod_dir(modality) / "step_2_baseline.h5"

    def frame_info(self, modality: str, raw: bool = False) -> Path:
        name = "frame_info_raw.csv" if raw else "frame_info.csv"
        return self.mod_dir(modality) / name

    def thresholds_csv(self, modality: str) -> Path:
        return self.mod_dir(modality) / "thresholds.csv"

    @property
    def eval_csv(self) -> Path:
        return self.cache_root / "eval_tjii.csv"

    @property
    def deploy_manifest(self) -> Path:
        return self.model_dir / "deploy_manifest.yaml"

    def artifacts(self, step: str, modalities: list[str]) -> list[Path]:
        """Every file/dir a step writes — the exact clear list for that step."""
        spec = get_step(step)
        out: list[Path] = []
        if spec.per_modality:
            for mod in modalities:
                out.append(self.step_h5(step, mod))
                if step == "step_0":
                    out.append(self.frame_info(mod, raw=True))
                elif step == "step_1":
                    out.append(self.frame_info(mod))
                elif step == "step_2":
                    out.append(self.baseline_h5(mod))
                elif step == "step_4":
                    out.append(self.thresholds_csv(mod))
        elif step == "step_7":
            out.append(self.model_dir)
        elif step == "step_8":
            out.append(self.eval_csv)
        else:
            out.append(self.step_h5(step))
        return out


NOTEBOOK_TEMPLATE_DIR = Path(__file__).parent / "notebooks"
CONFIG_DIR = Path(__file__).parent / "config"
DEFAULTS_YAML = CONFIG_DIR / "defaults.yaml"
RUN_TEMPLATE_YAML = CONFIG_DIR / "run_template.yaml"
