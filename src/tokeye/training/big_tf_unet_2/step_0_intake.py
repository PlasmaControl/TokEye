"""Step 0: intake — load, resample, pre-emphasize, and window shot data.

Ported from ``big_tf_unet_ablation/step_0f_foundation.py`` to the
``big_tf_unet_2`` runner contract: ``main(settings)`` receives a fully
resolved settings dict (paths, channels, rates already picked by the
runner) and returns ``None`` — there are no in-step resolved params for
this step.

Each shot's ``{shot}_processed.h5`` holds one HDF5 group per modality, with
``xdata (N,)`` (time, in seconds) and ``ydata (C, N)`` (channels x samples).
Native sample rates differ per modality -- and even per shot, since they
come from real hardware timestamps rather than a nominal rate -- so the
native rate is derived from the ``xdata`` spacing and each modality is
resampled to ``target_rate_khz`` with an anti-aliased polyphase filter
before windowing (mirrors the pattern in
``big_tf_unet_ablation/step_0g_raw_fast.py``).

Processing per shot: select channels -> resample to the target rate ->
pre-emphasis -> consecutive non-overlapping windows of ``subseq_len``
samples, capped at ``max_windows_per_shot``.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

from .utils.hdf5_io import create_step_file, write_sample
from .utils.signal_filters import Preemphasis

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Number of leading samples read from ``xdata`` to estimate the native rate --
# spacing is uniform enough that a small prefix is plenty, and avoids reading
# a multi-million-sample time axis just to compute one number.
_RATE_PROBE_SAMPLES = 1024


def _read_shots(shots_path: Path, n_shots: int | None) -> list[int]:
    shots = [int(s) for s in shots_path.read_text().split()]
    return shots[:n_shots] if n_shots is not None else shots


def _native_rate_khz(xdata: np.ndarray) -> float:
    """Native sample rate (kHz) implied by ``xdata`` spacing (seconds)."""
    dt_s = float(np.median(np.diff(xdata)))
    return 1.0 / (dt_s * 1000.0)


def _resample_to(sig: np.ndarray, src_khz: float, target_khz: float) -> np.ndarray:
    """Resample ``(C, N)`` from ``src_khz`` to ``target_khz`` (anti-aliased)."""
    frac = Fraction(int(round(target_khz)), int(round(src_khz))).limit_denominator(
        1000
    )
    up, down = frac.numerator, frac.denominator
    if (up, down) == (1, 1):
        return sig
    return resample_poly(sig, up, down, axis=1)


def main(settings: dict) -> dict | None:
    shots_path: Path = settings["shots_path"]
    foundation_dir: Path = settings["foundation_dir"]
    modality: str = settings["modality"]
    input_key: str = settings["input_key"]
    channels: list[int] = list(settings["channels"])
    subseq_len: int = int(settings["subseq_len"])
    coeff: float = float(settings["preemphasis_coeff"])
    target_khz: float = float(settings["target_rate_khz"])
    max_windows_per_shot: int = int(settings["max_windows_per_shot"])
    n_shots: int | None = settings["n_shots"]
    out_h5: Path = settings["out_h5"]
    frame_info_csv: Path = settings["frame_info_csv"]
    run_id: str = settings["run_id"]
    smoke: bool = bool(settings["smoke"])

    shots = _read_shots(shots_path, n_shots)
    preemph = Preemphasis(coeff)

    h5 = create_step_file(
        out_h5,
        metadata={
            "modality": modality,
            "input_key": input_key,
            "num_channels": len(channels),
            "subseq_len": subseq_len,
            "target_rate_khz": target_khz,
            "run_id": run_id,
        },
    )

    rows: list[dict[str, int]] = []
    index = 0
    n_shots_used = 0
    try:
        for shot in shots:
            path = foundation_dir / f"{shot}_processed.h5"
            if not path.exists():
                logger.warning(f"{modality}: shot {shot} absent in foundation_dir")
                continue

            with h5py.File(path, "r") as fh:
                if input_key not in fh or "ydata" not in fh[input_key]:
                    logger.warning(
                        f"{modality}: shot {shot} has no {input_key!r} group"
                    )
                    continue
                group = fh[input_key]
                ydata = group["ydata"]
                xdata = group["xdata"]
                n = ydata.shape[1]
                if n < subseq_len:
                    logger.warning(
                        f"{modality}: shot {shot} degenerate (N={n} < "
                        f"subseq_len={subseq_len}); skip"
                    )
                    continue
                if max(channels) >= ydata.shape[0]:
                    logger.warning(
                        f"{modality}: shot {shot} has {ydata.shape[0]} channels "
                        f"< requested max {max(channels)}; skip"
                    )
                    continue
                if xdata.shape[0] != n:
                    logger.warning(
                        f"{modality}: shot {shot} xdata({xdata.shape[0]}) != "
                        f"ydata N({n}); skip"
                    )
                    continue
                sig = np.asarray(ydata[channels, :], dtype=np.float64)
                x_probe = np.asarray(
                    xdata[: min(_RATE_PROBE_SAMPLES, n)], dtype=np.float64
                )

            if not np.any(sig):
                logger.warning(f"{modality}: shot {shot} is all-zero; skip")
                continue

            src_khz = _native_rate_khz(x_probe)
            sig = _resample_to(sig, src_khz, target_khz)
            sig_t = preemph(torch.from_numpy(sig).float()).numpy()

            n_windows = min(sig_t.shape[1] // subseq_len, max_windows_per_shot)
            for w in range(n_windows):
                s0, s1 = w * subseq_len, (w + 1) * subseq_len
                write_sample(h5, index, sig_t[:, s0:s1])
                rows.append({"index": index, "shotn": shot, "window_start": s0})
                index += 1
            n_shots_used += 1
            logger.info(
                f"{modality}: shot {shot} -> {n_windows} windows "
                f"({src_khz:.2f} kHz -> {target_khz:.0f} kHz)"
            )
    finally:
        h5.close()

    pd.DataFrame(rows, columns=["index", "shotn", "window_start"]).to_csv(
        frame_info_csv, index=False
    )

    smoke_tag = " (smoke)" if smoke else ""
    logger.info(
        f"[{run_id}] step_0_intake [{modality}]{smoke_tag}: {index} windows "
        f"from {n_shots_used}/{len(shots)} shots -> {out_h5}"
    )
    return None
