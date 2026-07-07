"""Resolution of ``"auto"`` knob values + the resolved-parameter ledger.

Every knob that supports ``"auto"`` is resolved once per run (from the run's
own data or geometry) and recorded in ``resolved_params.yaml`` together with
its source (``auto`` / ``user`` / ``in-step``). That file is the experiment
record the notebooks display, and the source of the deployment manifest.

Two kinds of resolution:

- **pre-run** (this module): values computable from the config geometry or an
  upstream artifact before the step starts (lam, edge bins, num_layers, ...).
- **in-step**: values that only exist mid-step (per-modality stats computed
  from data the step itself produces). Step ``main()`` returns them and the
  runner records them here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from .utils import auto_params

if TYPE_CHECKING:
    from .paths import RunPaths
    from .run_config import RunConfig


def _pad_geometry(cfg: RunConfig, num_layers: int) -> tuple[int, int]:
    mult = 2**num_layers
    return (
        auto_params.pad_to_multiple(cfg.n_freq, mult),
        auto_params.pad_to_multiple(cfg.n_time, mult),
    )


def _resolve_model_geometry(cfg: RunConfig, section: Any) -> dict[str, Any]:
    """num_layers + batch_size for one model section (denoise/refine/final)."""
    out: dict[str, Any] = {}
    num_layers = section.num_layers
    if num_layers == "auto":
        num_layers = auto_params.compute_num_layers(cfg.n_freq, cfg.n_time)
        out["num_layers"] = num_layers
    if section.batch_size == "auto":
        out["batch_size"] = auto_params.compute_batch_size(
            section.base_batch_size, cfg.n_freq, cfg.n_time, num_layers
        )
    return out


def resolve_step_autos(
    cfg: RunConfig, step: str, modality: str | None, paths: RunPaths
) -> dict[str, Any]:
    """Pre-run auto resolutions for one step (empty dict if none apply)."""
    resolved: dict[str, Any] = {}

    if step == "step_2":
        if cfg.baseline.lam == "auto":
            resolved["lam"] = auto_params.compute_lam(cfg.n_freq)
        in_h5 = paths.step_h5("step_1", modality)
        if in_h5.exists():
            if cfg.baseline.edge_method == "energy":
                lower, upper = auto_params.detect_edge_bins_energy(
                    in_h5,
                    k=cfg.baseline.edge_k,
                    max_fraction=cfg.baseline.edge_max_fraction,
                )
            else:
                lower, upper = auto_params.detect_edge_bins(
                    in_h5, gradient_threshold=cfg.baseline.gradient_threshold
                )
            resolved["edge_bins_lower"] = lower
            resolved["edge_bins_upper"] = upper

    elif step == "step_3":
        resolved.update(_resolve_model_geometry(cfg, cfg.denoise))
        # Reuse step_2's edge bins for input masking (consistency over
        # recomputation) — fall back to recomputing only if absent.
        ledger = read_ledger(paths)
        prior = ledger.get("step_2", {}).get(modality or "-", {})
        for key in ("edge_bins_lower", "edge_bins_upper"):
            if key in prior:
                resolved[key] = prior[key]["value"]

    elif step == "step_4":
        labels = cfg.labels
        if labels.min_size == "auto":
            resolved["min_size"] = auto_params.compute_min_size(
                cfg.n_freq, cfg.n_time, labels.min_size_fraction
            )
        if labels.remove_bottom_rows == "auto" or labels.remove_top_rows == "auto":
            bottom, top = auto_params.compute_row_removal(
                cfg.n_freq,
                labels.row_removal_fraction_bottom,
                labels.row_removal_fraction_top,
            )
            if labels.remove_bottom_rows == "auto":
                resolved["remove_bottom_rows"] = bottom
            if labels.remove_top_rows == "auto":
                resolved["remove_top_rows"] = top

    elif step in ("step_6", "step_7"):
        section = cfg.refine if step == "step_6" else cfg.final
        resolved.update(_resolve_model_geometry(cfg, section))

    return resolved


def suggest(cfg: RunConfig, step: str, modality: str | None, paths: RunPaths) -> dict:
    """Dry-run resolution for notebook display; never writes the ledger."""
    return resolve_step_autos(cfg, step, modality, paths)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def read_ledger(paths: RunPaths) -> dict:
    if paths.resolved_params_path.exists():
        return yaml.safe_load(paths.resolved_params_path.read_text()) or {}
    return {}


def record(
    paths: RunPaths,
    step: str,
    modality: str | None,
    values: dict[str, Any],
    source: str,
) -> None:
    """Merge resolved values for (step, modality) into resolved_params.yaml."""
    if not values:
        return
    ledger = read_ledger(paths)
    slot = ledger.setdefault(step, {}).setdefault(modality or "-", {})
    for key, value in values.items():
        slot[key] = {"value": value, "source": source}
    paths.resolved_params_path.parent.mkdir(parents=True, exist_ok=True)
    paths.resolved_params_path.write_text(
        yaml.safe_dump(ledger, sort_keys=False, default_flow_style=False)
    )
