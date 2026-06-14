"""Ablation pipeline orchestrator.

Reconstructs the paper-correct dual-mask dataflow (coherent = threshold(denoised
step_3b); transient = threshold(step_2b_baseline)) over the multiscale automation
infrastructure, for the 4 leave-one-out ablation variants.

Modes::

    # shared, once (0c/2a/2f loop over modalities; 0a/0b are modality-agnostic)
    python -m tokeye.training.big_tf_unet_ablation.orchestrator \
        --shared-steps step_0c,step_2a,step_2f

    # one variant end-to-end (SLURM array 0..3)
    python -m tokeye.training.big_tf_unet_ablation.orchestrator --variant-index 0

    python -m tokeye.training.big_tf_unet_ablation.orchestrator --status
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .ablation_matrix import build_variants, n_variants, variant_from_index
from .config.modalities import build_modalities
from .task_matrix import TaskMatrix
from .utils.configuration import load_pipeline_config

if TYPE_CHECKING:
    from .config.modalities import Modality
    from .config.variants import AblationVariant

logger = logging.getLogger(__name__)

PKG = "tokeye.training.big_tf_unet_ablation"

_STEP_MODULES = {
    "step_0a": ".step_0a_extract_faithdata",
    "step_0b": ".step_0b_filter_faithdata",
    "step_0c": ".step_0c_convert_faithdata",
    "step_0f": ".step_0f_foundation",
    "step_0g": ".step_0g_raw_fast",
    "step_2a": ".step_2a_make_spectrogram",
    "step_2f": ".step_2f_filter_windows",
    "step_2b": ".step_2b_filter_spectrogram",
    "step_3a": ".step_3a_correlation_analysis",
    "step_3b": ".step_3b_extract_correlation",
    "step_4a_coh": ".step_4a_threshold",
    "step_4a_tra": ".step_4a_threshold",
    "step_6a": ".step_6a_convert_tif",
    "step_6b": ".step_6b_refiner",
    "step_6c": ".step_6c_convert_predictions",
    "step_6d": ".step_6d_final",
}

SHARED_MOD_STEPS = ["step_0c", "step_0f", "step_0g", "step_2a", "step_2f"]  # per-modality shared steps
PER_MOD_STEPS = ["step_2b", "step_3a", "step_3b", "step_4a_coh", "step_4a_tra"]
COMBINED_STEPS = ["step_6a", "step_6b", "step_6c", "step_6d"]
VARIANT_STEPS = PER_MOD_STEPS + COMBINED_STEPS


# --------------------------------------------------------------------------- #
# path helpers
# --------------------------------------------------------------------------- #
def _cache(cfg) -> Path:
    return Path(cfg["paths"]["cache_dir"])


def _shared_dir(cfg) -> Path:
    return _cache(cfg) / "shared"


def _shared_mod_dir(cfg, m: Modality) -> Path:
    return _shared_dir(cfg) / m.name


def _variant_mod_dir(cfg, v: AblationVariant, m: Modality) -> Path:
    return _cache(cfg) / v.id / m.name


def _run(step: str, settings: dict) -> None:
    mod = importlib.import_module(_STEP_MODULES[step], package=PKG)
    mod.main(settings=settings)


# --------------------------------------------------------------------------- #
# settings builders
# --------------------------------------------------------------------------- #
def build_shared_step_settings(cfg, step, modality: Modality | None = None) -> dict:
    shared = _shared_dir(cfg)
    s: dict = {"overwrite": True}
    if "extraction" in cfg:
        s.update(dict(cfg["extraction"]))
    for k in ("shots_path", "faith_cfg_path"):
        if k in cfg.get("paths", {}):
            s[k] = Path(cfg["paths"][k])

    if step == "step_0a":
        s["output_dir"] = shared / "step_0a"
    elif step == "step_0b":
        s["input_dir"] = shared / "step_0a"
        s["output_dir"] = shared / "step_0b"
    elif step in SHARED_MOD_STEPS:
        assert modality is not None
        md = _shared_mod_dir(cfg, modality)
        if step == "step_0c":
            # Prefer a freshly-extracted shared/step_0b (full shot set from a
            # local step_0a/step_0b run); fall back to the configured source
            # (e.g. the 2-shot multiscale cache) only when none was extracted.
            shared_0b = shared / "step_0b"
            src = cfg["paths"].get("source_step_0b_dir")
            if shared_0b.exists():
                in_dir = shared_0b
            elif src and Path(src).exists():
                in_dir = Path(src)
            else:
                in_dir = shared_0b
            s["input_dir"] = in_dir
            s["output_dir"] = md
            s["input_key"] = modality.input_key
            s["input_channels"] = list(modality.channels)
            s["frame_info_path"] = md / "frame_info.csv"
            s["max_windows_per_shot_precap"] = cfg.get("extraction", {}).get(
                "max_windows_per_shot_precap", None
            )
        elif step == "step_0f":
            # foundation_model loader (replaces 0a/0b/0c); reads group/ydata (C,N)
            s["foundation_dir"] = cfg["paths"].get(
                "foundation_dir", "/scratch/gpfs/EKOLEMEN/foundation_model"
            )
            s["shots_path"] = Path(cfg["paths"]["shots_path"])
            s["input_key"] = modality.input_key
            s["input_channels"] = list(modality.channels)
            s["subseq_len"] = cfg.get("extraction", {}).get("subseq_len", 66000)
            s["preemphasis_coeff"] = cfg.get("extraction", {}).get("preemphasis_coeff", 0.99)
            s["output_dir"] = md
            s["frame_info_path"] = md / "frame_info.csv"
        elif step == "step_0g":
            # raw_fast loader: single-shot files (all 4 modalities), resample to target rate
            s["raw_fast_dir"] = cfg["paths"].get("raw_fast_dir", "data/autoprocess/raw_fast")
            s["shots_path"] = Path(cfg["paths"]["shots_path"])
            s["input_key"] = modality.input_key
            s["input_channels"] = list(modality.channels)
            s["subseq_len"] = cfg.get("extraction", {}).get("subseq_len", 66000)
            s["preemphasis_coeff"] = cfg.get("extraction", {}).get("preemphasis_coeff", 0.99)
            s["target_rate_khz"] = cfg.get("extraction", {}).get("target_rate_khz", 500)
            s["output_dir"] = md
            s["frame_info_path"] = md / "frame_info.csv"
        elif step == "step_2a":
            s["input_h5"] = md / "step_0c.h5"
            s["output_h5"] = md / "step_2a.h5"
            s["nfft"] = cfg["stft"]["nfft"]
            s["hop_length"] = cfg["stft"]["hop_length"]
        elif step == "step_2f":
            wf = cfg.get("window_filter", {})
            s.update(wf)
            s["input_h5"] = md / "step_2a.h5"
            s["output_h5"] = md / "step_2a_filtered.h5"
            s["frame_info_csv"] = md / "frame_info.csv"
            s["frame_info_out"] = md / "frame_info_filtered.csv"
    return s


def build_variant_step_settings(
    cfg, v: AblationVariant, step, modality: Modality | None = None
) -> dict:
    s: dict = {"overwrite": True, "combo_id": v.id}
    for sec in ("baseline", "correlation", "threshold", "refiner", "final"):
        if sec in cfg:
            s[sec] = cfg[sec]

    if step in PER_MOD_STEPS:
        assert modality is not None
        vmd = _variant_mod_dir(cfg, v, modality)
        smd = _shared_mod_dir(cfg, modality)
        s["frame_info_csv"] = smd / "frame_info_filtered.csv"
        s["total_channels"] = modality.total_channels
        s["adjacent_channels"] = modality.adjacent_channels
        if step == "step_2b":
            s["input_h5"] = smd / "step_2a_filtered.h5"
            s["output_h5"] = vmd / "step_2b.h5"
            s["output_baseline_h5"] = vmd / "step_2b_baseline.h5"
            s["baseline_enabled"] = v.baseline
        elif step == "step_3a":
            s["input_h5"] = vmd / "step_2b.h5"
            s["output_h5"] = vmd / "step_3a.h5"
            if "correlation" in cfg:
                s.update(cfg["correlation"])
            # modality-derived channel counts + variant representation MUST win
            # over any correlation-section defaults.
            s["representation"] = v.representation
            s["total_channels"] = modality.total_channels
            s["adjacent_channels"] = modality.adjacent_channels
        elif step == "step_3b":
            s["input_h5"] = vmd / "step_3a.h5"
            s["reference_h5"] = vmd / "step_2b.h5"
            s["output_h5"] = vmd / "step_3b.h5"
        elif step == "step_4a_coh":
            s["input_h5"] = (vmd / "step_3b.h5") if v.denoise else (vmd / "step_2b.h5")
            s["output_h5"] = vmd / "step_4a_threshold.h5"
            s["threshold_output_path"] = vmd / "thresholds_coh.csv"
            if "threshold" in cfg:
                s.update(cfg["threshold"])
        elif step == "step_4a_tra":
            s["input_h5"] = vmd / "step_2b_baseline.h5"
            s["output_h5"] = vmd / "step_4a_threshold_baseline.h5"
            s["threshold_output_path"] = vmd / "thresholds_tra.csv"
            if "threshold" in cfg:
                s.update(cfg["threshold"])
        return s

    # combined steps
    vdir = _cache(cfg) / v.id
    if step == "step_6a":
        mods = build_modalities(cfg)
        s["modality_inputs"] = [
            {
                "name": m.name,
                "img_h5": str(_shared_mod_dir(cfg, m) / "step_2a_filtered.h5"),
                "coh_h5": str(_variant_mod_dir(cfg, v, m) / "step_4a_threshold.h5"),
                "tra_h5": str(_variant_mod_dir(cfg, v, m) / "step_4a_threshold_baseline.h5"),
            }
            for m in mods
        ]
        s["output_dir"] = vdir / "step_6a"
        s["zscore_clip"] = 3
    elif step == "step_6b":
        s["input_dir"] = vdir / "step_6a"
        s["output_dir"] = vdir / "step_6b"
        s["model_dir"] = vdir / "step_6b" / "models"
        if "refiner" in cfg:
            s.update(cfg["refiner"])
    elif step == "step_6c":
        s["input_dir"] = vdir / "step_6a"
        s["predictions_file"] = vdir / "step_6b" / "all_folds_predictions.h5"
        s["output_dir"] = vdir / "step_6c"
        if "refiner" in cfg:
            s["n_folds"] = cfg["refiner"].get("n_folds", 5)
    elif step == "step_6d":
        s["input_dir"] = vdir / "step_6c"
        s["model_dir"] = Path(cfg["paths"]["model_dir"]) / v.id
        if "final" in cfg:
            s.update(cfg["final"])
    return s


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #
def run_shared_steps(cfg, tm: TaskMatrix, steps: list[str]) -> None:
    _shared_dir(cfg).mkdir(parents=True, exist_ok=True)
    mods = build_modalities(cfg)
    for step in steps:
        if step in SHARED_MOD_STEPS:
            for m in mods:
                key = f"{step}:{m.name}"
                if tm.is_shared_step_complete(key):
                    logger.info(f"skip shared {key}")
                    continue
                _run(step, build_shared_step_settings(cfg, step, m))
                tm.mark_shared_step_complete(key)
        else:
            if tm.is_shared_step_complete(step):
                logger.info(f"skip shared {step}")
                continue
            _run(step, build_shared_step_settings(cfg, step))
            tm.mark_shared_step_complete(step)


def run_variant_steps(
    cfg, tm: TaskMatrix, vidx: int, steps: list[str], cleanup: bool = True
) -> None:
    v = variant_from_index(cfg, vidx)
    mods = build_modalities(cfg)
    for step in steps:
        if step in PER_MOD_STEPS:
            if step in ("step_3a", "step_3b") and not v.denoise:
                continue
            for m in mods:
                key = f"{step}:{m.name}"
                if tm.is_step_complete(v.id, key):
                    logger.info(f"skip {v.id}/{key}")
                    continue
                _run(step, build_variant_step_settings(cfg, v, step, m))
                tm.mark_step_complete(v.id, key)
        else:
            if tm.is_step_complete(v.id, step):
                logger.info(f"skip {v.id}/{step}")
                continue
            _run(step, build_variant_step_settings(cfg, v, step))
            tm.mark_step_complete(v.id, step)

    if cleanup and cfg.get("cleanup", {}).get("enabled", False):
        _cleanup_variant(cfg, v)


def _cleanup_variant(cfg, v: AblationVariant) -> None:
    vdir = _cache(cfg) / v.id
    delete = set(cfg.get("cleanup", {}).get("delete_steps", []))
    for p in vdir.rglob("*.h5"):
        # Match by filename stem (e.g. step_2b.h5 -> "step_2b") OR by the parent
        # step-directory name (e.g. step_6b/all_folds_predictions.h5 -> "step_6b").
        # The latter reclaims the ~280GB refiner predictions, which carry a generic
        # filename but live under step_6b/; keeping all 4 variants' copies (~1.1TB)
        # overflows the 3TB scratch quota and crashes step_6c mid-write. eval reads
        # step_6d's final.torchscript.pt, never this h5, so deletion is safe.
        if any(p.stem.startswith(d) or p.parent.name == d for d in delete):
            p.unlink()
            logger.info(f"cleaned {p}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation pipeline orchestrator")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--shared-steps", type=str, default=None)
    parser.add_argument("--variant-index", type=int, default=None)
    parser.add_argument("--steps", type=str, default=None)
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip the post-run intermediate cleanup. Use when this is a partial "
        "run whose outputs (e.g. step_2b) a later job still needs.",
    )
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    cfg = load_pipeline_config(args.config)
    tm = TaskMatrix(cfg["paths"]["task_matrix_path"])

    if args.status:
        tm.print_status([v.id for v in build_variants(cfg)])
        return
    if args.shared_steps:
        run_shared_steps(cfg, tm, [s.strip() for s in args.shared_steps.split(",")])
        return
    if args.variant_index is not None:
        if not (0 <= args.variant_index < n_variants(cfg)):
            raise SystemExit(f"variant-index out of range 0..{n_variants(cfg) - 1}")
        steps = [s.strip() for s in args.steps.split(",")] if args.steps else VARIANT_STEPS
        run_variant_steps(
            cfg, tm, args.variant_index, steps, cleanup=not args.no_cleanup
        )
        return
    parser.print_help()


if __name__ == "__main__":
    main()
