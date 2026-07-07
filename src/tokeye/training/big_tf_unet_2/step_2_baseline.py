"""Step 2: baseline-correct spectrograms (FABC residual).

Reads ``(C, F, T, 2)`` real/imag samples from the step_1 HDF5, applies 2D
baseline fitting (pybaselines FABC along the frequency axis) to the log1p
magnitude of each component, and writes the relative residual to ``out_h5``
and the fitted baseline to ``baseline_h5``.

Edge frequency bins (``edge_bins_lower`` bottom rows, ``edge_bins_upper`` top
rows) are filled with the sample's post-log1p median before the fit and zeroed
in the residual after correction.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from pybaselines import Baseline2D

from .utils.hdf5_io import (
    create_step_file,
    get_sample_count,
    iter_samples,
    write_sample,
)

logger = logging.getLogger(__name__)


def _fit_baseline(data: np.ndarray, method: str, lam: float) -> np.ndarray:
    """Fit a 2D baseline along the frequency axis of one ``(F, T)`` field."""
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    fitter = Baseline2D(x, y)
    baseline, _ = fitter.individual_axes(
        data, axes=0, method=method, method_kwargs={"lam": lam}
    )
    return baseline


def _process_component(
    sxx: np.ndarray,
    edge_lower: int,
    edge_upper: int,
    method: str,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline-correct one post-log1p ``(F, T)`` component field.

    The masked edge rows are EXCLUDED from the fit rather than filled: FABC
    classifies baseline points by noise statistics, and a block of constant
    synthetic rows starves the classifier ("not enough baseline points") and
    makes the Whittaker system singular. The baseline is fit on the interior
    rows only; edge rows get the nearest interior baseline value in the
    baseline output and 0 in the residual.
    """
    n_freq = sxx.shape[0]
    lo = edge_lower
    hi = n_freq - edge_upper if edge_upper > 0 else n_freq

    interior_baseline = _fit_baseline(sxx[lo:hi], method, lam)
    baseline = np.empty_like(sxx)
    baseline[lo:hi] = interior_baseline
    if lo > 0:
        baseline[:lo] = interior_baseline[0]
    if edge_upper > 0:
        baseline[hi:] = interior_baseline[-1]

    residual = np.zeros_like(sxx)
    residual[lo:hi] = (sxx[lo:hi] - interior_baseline) / (interior_baseline + 1e-6)
    return residual, baseline


def _process_sample(
    data: np.ndarray,
    edge_lower: int,
    edge_upper: int,
    method: str,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline-correct one full sample ``(C, F, T, 2)`` across all channels."""
    out = np.zeros_like(data)
    bl = np.zeros_like(data)
    log_field = np.log1p(np.abs(data))
    for c in range(data.shape[0]):
        for r in range(data.shape[-1]):
            out[c, ..., r], bl[c, ..., r] = _process_component(
                log_field[c, ..., r], edge_lower, edge_upper, method, lam
            )
    return out, bl


# Kept alive for the worker's lifetime so the thread limit isn't GC'd away.
_THREAD_LIMITER = None


def _init_worker() -> None:
    """Cap each worker's native thread pools to 1 thread.

    Baseline fitting is run across many worker processes; without this each
    process's BLAS/OpenMP pool would try to use every core, oversubscribing them
    and thrashing.  ``threadpoolctl`` limits at runtime (after numpy/scipy
    import), so it works regardless of start method.
    """
    global _THREAD_LIMITER
    try:
        from threadpoolctl import threadpool_limits

        _THREAD_LIMITER = threadpool_limits(limits=1)
    except Exception:  # optional dependency; fall back to inherited env vars
        _THREAD_LIMITER = None


def _worker(
    args: tuple[int, np.ndarray, int, int, str, float],
) -> tuple[int, np.ndarray, np.ndarray]:
    """ProcessPool task: baseline-correct one sample -> ``(idx, out, baseline)``."""
    idx, data, edge_lower, edge_upper, method, lam = args
    out, bl = _process_sample(data, edge_lower, edge_upper, method, lam)
    return idx, out, bl


def _resolve_n_workers(settings: dict) -> int:
    """Worker count: explicit setting > SLURM allocation > CPU count, min 1."""
    n = settings.get("n_workers")
    if n is None:
        n = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)
    return max(1, int(n))


def main(settings: dict) -> dict | None:
    in_h5 = Path(settings["in_h5"])
    out_h5 = Path(settings["out_h5"])
    baseline_h5 = Path(settings["baseline_h5"])
    method = str(settings["method"])
    lam = float(settings["lam"])
    edge_lower = int(settings["edge_bins_lower"])
    edge_upper = int(settings["edge_bins_upper"])

    n_samples = get_sample_count(in_h5)
    n_workers = _resolve_n_workers(settings)
    logger.info(
        f"Baseline-correcting {n_samples} spectrograms "
        f"(method={method}, lam={lam:g}, "
        f"edges: lower={edge_lower}, upper={edge_upper}; {n_workers} workers)"
    )

    h5_out = create_step_file(
        out_h5,
        metadata={
            "run_id": settings["run_id"],
            "method": method,
            "lam": lam,
            "edge_bins_lower": edge_lower,
            "edge_bins_upper": edge_upper,
            "num_samples": n_samples,
        },
    )
    h5_bl = create_step_file(
        baseline_h5,
        metadata={"run_id": settings["run_id"], "num_samples": n_samples},
    )

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
        return (idx, data, edge_lower, edge_upper, method, lam)

    try:
        if n_workers <= 1:
            for idx, data in iter_samples(in_h5):
                _store(
                    idx,
                    *_process_sample(data, edge_lower, edge_upper, method, lam),
                )
        else:
            # Per-sample baseline fitting is the pipeline bottleneck and is pure CPU
            # (each sample = C channels x 2 components x ~T 1D FABC fits, no GPU
            # work), so a serial loop pegs one core for hours. Fan out over the
            # allocated cores with a bounded in-flight set so the fitting saturates
            # every core while peak memory stays flat. A forkserver context starts
            # workers from a clean server process: no per-worker re-import of the
            # (torch-importing) entrypoint that spawn incurs, and no inherited open
            # HDF5 handle/CUDA state that plain fork would carry. Results may arrive
            # out of order, but write_sample keys by index so ordering does not
            # matter.
            samples = iter_samples(in_h5)
            max_inflight = max(1, n_workers * 2)
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context("forkserver"),
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
    finally:
        h5_out.close()
        h5_bl.close()

    logger.info(f"Wrote residuals to {out_h5} and baselines to {baseline_h5}")
    return None
