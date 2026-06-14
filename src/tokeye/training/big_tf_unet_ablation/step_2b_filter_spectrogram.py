"""Step 2b: Baseline-correct spectrograms with auto edge detection.

Reads ``(C, F, T, 2)`` from step_2a HDF5, applies 2D baseline fitting
(pybaselines FABC), and writes the corrected spectrogram + baseline to
two separate HDF5 files.

Edge bins are detected automatically (once per combo) rather than
hard-coded.  Masked bin values are set to the data mean.
"""

from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from pybaselines import Baseline2D

from .utils.auto_params import detect_edge_bins
from .utils.configuration import load_settings
from .utils.hdf5_io import (
    create_step_file,
    get_sample_count,
    iter_samples,
    write_sample,
)

logger = logging.getLogger(__name__)

default_settings = {
    "baseline_method": "fabc",
    "baseline_method_kwargs": {"lam": 1e5},
    "bin_cutting": "auto",
    "gradient_threshold": 0.5,
    "input_h5": Path("data/cache/step_2a.h5"),
    "output_h5": Path("data/cache/step_2b.h5"),
    "output_baseline_h5": Path("data/cache/step_2b_baseline.h5"),
    "overwrite": True,
}


def _fit_baseline(
    data: np.ndarray,
    method: str = "fabc",
    method_kwargs: dict | None = None,
) -> np.ndarray:
    if method_kwargs is None:
        method_kwargs = {"lam": 1e5}
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    fitter = Baseline2D(x, y)
    baseline, _ = fitter.individual_axes(
        data, axes=0, method=method, method_kwargs=method_kwargs
    )
    return baseline


def _process_rotation(
    data: np.ndarray,
    lower_idx: int,
    upper_idx: int,
    method: str,
    method_kwargs: dict,
    baseline_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Process one real/imag component of one channel.

    When ``baseline_enabled`` is False (the ``nobaseline`` ablation), the ALS
    baseline is still *computed* (so the transient branch is identical across
    variants) but NOT subtracted from the coherent-path output -- isolating the
    effect of broadband-coherent separation on the coherent path alone.
    """
    sxx = np.abs(data)
    sxx = np.log1p(sxx)

    # Mask edges with data mean (auto-detected bin cutting)
    edge_val_lo = sxx[lower_idx].mean() if lower_idx > 0 else sxx.mean()
    edge_val_hi = sxx[-upper_idx].mean() if upper_idx > 0 else sxx.mean()
    if lower_idx > 0:
        sxx[:lower_idx] = edge_val_lo
    if upper_idx > 0:
        sxx[-upper_idx:] = edge_val_hi

    baseline = _fit_baseline(sxx, method, method_kwargs)
    if baseline_enabled:
        sxx = (sxx - baseline) / (baseline + 1e-6)
    return sxx, baseline


def _process_channel(
    data: np.ndarray,
    lower_idx: int,
    upper_idx: int,
    method: str,
    method_kwargs: dict,
    baseline_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Process one channel ``(F, T, 2)``."""
    real_out, bl_real = _process_rotation(
        data[..., 0], lower_idx, upper_idx, method, method_kwargs, baseline_enabled
    )
    imag_out, bl_imag = _process_rotation(
        data[..., 1], lower_idx, upper_idx, method, method_kwargs, baseline_enabled
    )
    return np.stack([real_out, imag_out], axis=-1), np.stack([bl_real, bl_imag], axis=-1)


def _process_sample(
    data: np.ndarray,
    lower_idx: int,
    upper_idx: int,
    method: str,
    method_kwargs: dict,
    baseline_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline-correct one full sample ``(C, F, T, 2)`` across all channels."""
    out = np.zeros_like(data)
    bl = np.zeros_like(data)
    for c in range(data.shape[0]):
        out[c], bl[c] = _process_channel(
            data[c], lower_idx, upper_idx, method, method_kwargs, baseline_enabled
        )
    return out, bl


# Kept alive for the worker's lifetime so the thread limit isn't GC'd away.
_THREAD_LIMITER = None


def _init_worker() -> None:
    """Cap each worker's native thread pools to 1 thread.

    Baseline fitting is run across many worker processes; without this each
    process's BLAS/OpenMP pool would try to use every core, oversubscribing them
    and thrashing.  ``threadpoolctl`` limits at runtime (after numpy/scipy import),
    so it works regardless of start method.
    """
    global _THREAD_LIMITER
    try:
        from threadpoolctl import threadpool_limits

        _THREAD_LIMITER = threadpool_limits(limits=1)
    except Exception:  # optional dependency; fall back to inherited env vars
        _THREAD_LIMITER = None


def _worker(
    args: tuple[int, np.ndarray, int, int, str, dict, bool],
) -> tuple[int, np.ndarray, np.ndarray]:
    """ProcessPool task: baseline-correct one sample -> ``(idx, out, baseline)``."""
    idx, data, lower_idx, upper_idx, method, method_kwargs, baseline_enabled = args
    out, bl = _process_sample(
        data, lower_idx, upper_idx, method, method_kwargs, baseline_enabled
    )
    return idx, out, bl


def _resolve_n_workers(settings: dict, bl_cfg: dict) -> int:
    """Worker count: explicit setting > SLURM allocation > CPU count, min 1."""
    n = settings.get("n_workers", bl_cfg.get("n_workers"))
    if n is None:
        n = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)
    return max(1, int(n))


def main(
    config_path: Path | str | None = None,
    settings: dict | None = None,
) -> None:
    if settings is None:
        settings = load_settings(config_path, default_settings)

    input_h5 = Path(settings.get("input_h5", default_settings["input_h5"]))
    output_h5 = Path(settings.get("output_h5", default_settings["output_h5"]))
    output_baseline_h5 = Path(settings.get("output_baseline_h5", default_settings["output_baseline_h5"]))

    # Support both flat keys and a nested ``baseline`` config section.
    bl_cfg = settings.get("baseline", {})
    method = bl_cfg.get("method", settings.get("baseline_method", default_settings["baseline_method"]))
    method_kwargs = bl_cfg.get(
        "method_kwargs",
        settings.get("baseline_method_kwargs", default_settings["baseline_method_kwargs"]),
    )
    baseline_enabled = settings.get("baseline_enabled", bl_cfg.get("enabled", True))

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    output_baseline_h5.parent.mkdir(parents=True, exist_ok=True)

    # Auto edge detection (once per combo)
    bin_cutting = bl_cfg.get("bin_cutting", settings.get("bin_cutting", "auto"))
    if bin_cutting == "auto":
        grad_thresh = bl_cfg.get("gradient_threshold", settings.get("gradient_threshold", 0.5))
        lower_idx, upper_idx = detect_edge_bins(input_h5, gradient_threshold=grad_thresh)
    else:
        lower_idx = int(bin_cutting)
        upper_idx = int(bin_cutting)

    n_samples = get_sample_count(input_h5)
    n_workers = _resolve_n_workers(settings, bl_cfg)
    logger.info(
        f"Filtering {n_samples} spectrograms "
        f"(edges: lower={lower_idx}, upper={upper_idx}; {n_workers} workers)"
    )

    h5_out = create_step_file(output_h5, metadata={
        "lower_idx": lower_idx, "upper_idx": upper_idx,
        "method": method, "num_samples": n_samples,
        "baseline_enabled": bool(baseline_enabled),
    })
    h5_bl = create_step_file(output_baseline_h5, metadata={"num_samples": n_samples})

    log_every = max(1, n_samples // 20)
    n_done = 0

    def _store(idx: int, out: np.ndarray, bl: np.ndarray) -> None:
        nonlocal n_done
        write_sample(h5_out, idx, out)
        write_sample(h5_bl, idx, bl)
        n_done += 1
        if n_done % log_every == 0 or n_done == n_samples:
            logger.info(f"  baseline-corrected {n_done}/{n_samples}")

    def _arg(idx: int, data: np.ndarray) -> tuple:
        return (idx, data, lower_idx, upper_idx, method, method_kwargs, baseline_enabled)

    try:
        if n_workers <= 1:
            for idx, data in iter_samples(input_h5):
                _store(idx, *_process_sample(
                    data, lower_idx, upper_idx, method, method_kwargs, baseline_enabled
                ))
        else:
            # Per-sample baseline fitting is the pipeline bottleneck and is pure CPU
            # (each sample = C channels x 2 components x ~T 1D FABC fits, no GPU work),
            # so a serial loop pegs one core for hours while the GPU sits idle. Fan out
            # over the allocated cores with a bounded in-flight set so the fitting
            # saturates every core while peak memory stays flat. A forkserver context
            # starts workers from a clean server process: no per-worker re-import of the
            # (torch-importing) entrypoint that spawn incurs, and no inherited open HDF5
            # handle/CUDA state that plain fork would carry. Results may arrive out of
            # order, but write_sample keys by index so ordering does not matter.
            samples = iter_samples(input_h5)
            max_inflight = max(1, n_workers * 2)
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=get_context("forkserver"),
                initializer=_init_worker,
            ) as ex:
                pending = set()
                for _ in range(max_inflight):
                    nxt = next(samples, None)
                    if nxt is None:
                        break
                    pending.add(ex.submit(_worker, _arg(*nxt)))
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        _store(*fut.result())
                        nxt = next(samples, None)
                        if nxt is not None:
                            pending.add(ex.submit(_worker, _arg(*nxt)))
    except Exception as e:
        logger.error(f"Error during filtering: {e}")
        raise
    finally:
        h5_out.close()
        h5_bl.close()

    logger.info(f"Wrote filtered spectrograms to {output_h5}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
