"""Step orchestration: settings builders, gating, and the CLI entry point.

The same entry point serves notebook inline cells and sbatch payloads:

    python -m tokeye.training.big_tf_unet_2.runner \
        --run-config dev/training/nfft512_hop128/run.yaml --steps step_0,step_1

Contract with step modules: each ``step_N_*.py`` exposes
``main(settings: dict) -> dict | None``. The runner builds the settings dict
(config values + resolved autos + artifact paths), clears the step's previous
artifacts, tracks status in the task matrix, and records any in-step resolved
values the step returns (e.g. per-modality stats) into the ledger.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from typing import Any

from . import auto_resolve
from .clearing import clear_step
from .paths import STEP_ORDER, RunPaths, get_step
from .run_config import RunConfig, check_scale_lock, load_run_config
from .task_matrix import RunTaskMatrix, params_hash

logger = logging.getLogger(__name__)

# Keys excluded from the params hash: they change resources, never outputs.
_VOLATILE_KEYS = {"n_workers", "num_workers"}


# ---------------------------------------------------------------------------
# Settings builders
# ---------------------------------------------------------------------------

def build_step_settings(
    cfg: RunConfig, step: str, modality: str | None, paths: RunPaths
) -> dict[str, Any]:
    """Assemble one step's settings dict (config + autos + paths)."""
    smoke = cfg.smoke.enabled
    resolved = auto_resolve.resolve_step_autos(cfg, step, modality, paths)

    def auto(section_value: Any, key: str) -> Any:
        return resolved[key] if section_value == "auto" else section_value

    s: dict[str, Any] = {"run_id": paths.run_id, "smoke": smoke}

    def rooted(rel: str) -> Path:
        """Resolve a config path against the repo root (cwd-independent)."""
        p = Path(rel)
        return p if p.is_absolute() else paths.root / p

    if step == "step_0":
        mod = cfg.modalities[modality]
        ext = cfg.extraction
        s.update(
            shots_path=rooted(cfg.paths.shots_path),
            foundation_dir=rooted(cfg.paths.foundation_dir),
            modality=modality,
            input_key=mod.input_key,
            channels=list(mod.channels),
            subseq_len=ext.subseq_len,
            preemphasis_coeff=ext.preemphasis_coeff,
            fs_khz=ext.fs_khz,
            target_rate_khz=ext.target_rate_khz,
            ip_threshold=ext.ip_threshold,
            max_windows_per_shot=(
                cfg.smoke.max_windows_per_shot
                if smoke
                else ext.max_windows_per_shot_precap
            ),
            n_shots=cfg.smoke.n_shots if smoke else None,
            out_h5=paths.step_h5("step_0", modality),
            frame_info_csv=paths.frame_info(modality, raw=True),
        )

    elif step == "step_1":
        wf = cfg.window_filter
        s.update(
            in_h5=paths.step_h5("step_0", modality),
            out_h5=paths.step_h5("step_1", modality),
            frame_info_in=paths.frame_info(modality, raw=True),
            frame_info_out=paths.frame_info(modality),
            nfft=cfg.run.nfft,
            hop=cfg.run.hop,
            filter_enabled=wf.enabled,
            weights=rooted(wf.weights),
            max_windows_per_shot=(
                cfg.smoke.max_windows_per_shot if smoke else wf.max_windows_per_shot
            ),
            activity_threshold=wf.activity_threshold,
            min_activity=wf.min_activity,
            mean=wf.mean,  # "auto" -> step computes per-modality logmag stats
            std=wf.std,
        )

    elif step == "step_2":
        b = cfg.baseline
        s.update(
            in_h5=paths.step_h5("step_1", modality),
            out_h5=paths.step_h5("step_2", modality),
            baseline_h5=paths.baseline_h5(modality),
            method=b.method,
            lam=auto(b.lam, "lam"),
            edge_bins_lower=resolved.get("edge_bins_lower", 1),
            edge_bins_upper=resolved.get("edge_bins_upper", 1),
            n_workers=None,  # step resolves: setting > SLURM_CPUS_PER_TASK > cpus
        )

    elif step == "step_3":
        d = cfg.denoise
        n_channels = len(cfg.modalities[modality].channels)
        s.update(
            in_h5=paths.step_h5("step_2", modality),
            out_h5=paths.step_h5("step_3", modality),
            representation=d.representation,
            normalization=d.normalization,
            a=d.a,
            total_channels=n_channels,
            adjacent_channels=max(1, n_channels // 2),
            first_layer_size=d.first_layer_size,
            num_layers=auto(d.num_layers, "num_layers"),
            batch_size=auto(d.batch_size, "batch_size"),
            precision=d.precision,
            max_epochs=cfg.smoke.max_epochs if smoke else d.max_epochs,
            tv_patience=d.tv_patience,
            num_workers=d.num_workers,
            edge_bins_lower=resolved.get("edge_bins_lower", 1),
            edge_bins_upper=resolved.get("edge_bins_upper", 1),
        )

    elif step == "step_4":
        la = cfg.labels
        s.update(
            denoised_h5=paths.step_h5("step_3", modality),
            baseline_h5=paths.baseline_h5(modality),
            out_h5=paths.step_h5("step_4", modality),
            frame_info=paths.frame_info(modality),
            thresholds_csv=paths.thresholds_csv(modality),
            knee_sensitivity=la.knee_sensitivity,
            delta=la.delta,
            fallback_frac=la.fallback_frac,
            min_size=auto(la.min_size, "min_size"),
            remove_bottom_rows=auto(la.remove_bottom_rows, "remove_bottom_rows"),
            remove_top_rows=auto(la.remove_top_rows, "remove_top_rows"),
        )

    elif step == "step_5":
        s.update(
            inputs={
                mod: {
                    "img_h5": paths.step_h5("step_1", mod),
                    "mask_h5": paths.step_h5("step_4", mod),
                    "frame_info": paths.frame_info(mod),
                }
                for mod in cfg.modality_names
            },
            out_h5=paths.step_h5("step_5"),
            a=cfg.dataset.a,
            stats_windows=cfg.dataset.stats_windows,
        )

    elif step == "step_6":
        r = cfg.refine
        s.update(
            in_h5=paths.step_h5("step_5"),
            out_h5=paths.step_h5("step_6"),
            ckpt_dir=paths.cache_root / "step_6_ckpts",
            n_folds=cfg.smoke.n_folds if smoke else r.n_folds,
            first_layer_size=r.first_layer_size,
            num_layers=auto(r.num_layers, "num_layers"),
            batch_size=auto(r.batch_size, "batch_size"),
            precision=r.precision,
            max_epochs=cfg.smoke.refine_max_epochs if smoke else r.max_epochs,
            loss_type=r.loss_type,
            num_workers=r.num_workers,
        )

    elif step == "step_7":
        f = cfg.final
        s.update(
            dataset_h5=paths.step_h5("step_5"),
            refine_h5=paths.step_h5("step_6"),
            # model_trust applies HERE (not in step_6): tuning the lambda knob
            # re-trains one final model instead of five folds.
            model_trust=cfg.refine.model_trust,
            model_dir=paths.model_dir,
            deploy_manifest=paths.deploy_manifest,
            resolved_params=paths.resolved_params_path,
            nfft=cfg.run.nfft,
            hop=cfg.run.hop,
            first_layer_size=f.first_layer_size,
            num_layers=auto(f.num_layers, "num_layers"),
            batch_size=auto(f.batch_size, "batch_size"),
            precision=f.precision,
            max_epochs=cfg.smoke.final_max_epochs if smoke else f.max_epochs,
            loss_type=f.loss_type,
            gamma=f.gamma,
            num_workers=f.num_workers,
        )

    elif step == "step_8":
        s.update(
            model_dir=paths.model_dir,
            dataset_dir=rooted(cfg.eval.dataset_dir),
            n_thresholds=cfg.eval.n_thresholds,
            out_csv=paths.eval_csv,
        )

    else:
        raise KeyError(f"No settings builder for {step}")

    return s


# ---------------------------------------------------------------------------
# Gating + execution
# ---------------------------------------------------------------------------

def _check_upstream(
    matrix: RunTaskMatrix, cfg: RunConfig, step: str, modality: str | None
) -> None:
    target = get_step(step)
    for name in STEP_ORDER[: STEP_ORDER.index(step)]:
        spec = get_step(name)
        if spec.per_modality:
            mods = [modality] if (target.per_modality and modality) else (
                cfg.modality_names
            )
            missing = [m for m in mods if not matrix.is_complete(name, m)]
            if missing:
                raise RuntimeError(
                    f"{step} needs {name} complete for {', '.join(missing)} "
                    f"first (rerun with --force to override)"
                )
        elif not matrix.is_complete(name):
            raise RuntimeError(
                f"{step} needs {name} complete first "
                f"(rerun with --force to override)"
            )


def run_step(
    run_yaml: str | Path,
    step: str,
    modality: str | None = None,
    force: bool = False,
) -> None:
    """Run one step (one modality for per-modality steps) end to end."""
    run_yaml = Path(run_yaml).resolve()
    cfg = load_run_config(run_yaml)
    paths = RunPaths(run_yaml.parent.name)
    check_scale_lock(cfg, paths.run_meta)

    spec = get_step(step)
    if spec.per_modality and modality is None:
        for mod in cfg.modality_names:
            run_step(run_yaml, step, mod, force)
        return

    matrix = RunTaskMatrix(paths.task_matrix_path)
    if not force:
        _check_upstream(matrix, cfg, step, modality)

    settings = build_step_settings(cfg, step, modality, paths)
    pre_resolved = auto_resolve.resolve_step_autos(cfg, step, modality, paths)
    auto_resolve.record(paths, step, modality, pre_resolved, source="auto")

    # Idempotent rerun: previous artifacts go away before the step starts, and
    # the matrix (not file existence) is the source of truth for completion.
    clear_step(paths, cfg, step, modality)
    for target in paths.artifacts(step, [modality] if modality else cfg.modality_names):
        target.parent.mkdir(parents=True, exist_ok=True)

    hashable = {k: v for k, v in settings.items() if k not in _VOLATILE_KEYS}
    matrix.mark_running(step, modality)
    label = f"{step}" + (f":{modality}" if modality else "")
    logger.info(f"[{paths.run_id}] running {label}")
    try:
        module = importlib.import_module(f".{spec.module}", package=__package__)
        in_step = module.main(settings)
    except Exception:
        matrix.mark_failed(step, modality)
        raise
    if in_step:
        auto_resolve.record(paths, step, modality, in_step, source="in-step")
    matrix.mark_complete(step, modality, params_hash(hashable), cfg.modality_names)
    logger.info(f"[{paths.run_id}] {label} complete")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument(
        "--steps", required=True, help="comma-separated, e.g. step_0,step_1"
    )
    parser.add_argument(
        "--modalities", default=None, help="comma-separated subset (default: all)"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mods = args.modalities.split(",") if args.modalities else [None]
    for step in args.steps.split(","):
        step = step.strip()
        if get_step(step).per_modality and args.modalities:
            for mod in mods:
                run_step(args.run_config, step, mod, args.force)
        else:
            run_step(args.run_config, step, None, args.force)


if __name__ == "__main__":
    main()
