"""Step 0f: load + preemphasis + window from foundation-format shot files.

Replaces the dead ``step_0a``/``step_0b``/``step_0c`` chain, which needed the
external ``fusionaihub`` and ``faith`` packages and the old FAITH ``{shot}.h5``
layout (both gone). Reads the project-local foundation copy
``data/autoprocess/foundation/{shot}_processed.h5`` (built by
``big_tf_unet_ablation/preprocess/raw_fast_to_foundation.py``), where each modality
is an HDF5 group with ``xdata (N,)`` (time, seconds) and ``ydata (C, N)``
(channels x samples) -- the same layout as ``/scratch/gpfs/EKOLEMEN/foundation_model``.

Invoked once per modality (like the old step_0c); writes per-window joblib files
``{index}.joblib`` -- each a numpy ``(C, subseq_len)`` array -- plus
``frame_info.csv``, in the SAME format ``step_1a``/``step_2a`` expect, so the rest of
the pipeline is unchanged. Shots that are absent or whose modality is degenerate
(``N < min_samples`` or too few channels) are logged and skipped.

If ``target_rate_khz`` is set, each shot is resampled to that rate at read time
(the rate is inferred from the ``xdata`` spacing). With the recommended converter
defaults the files are already 500 kHz, so leave it ``None`` (no-op).
"""

from __future__ import annotations

import csv
import logging
import sys
from fractions import Fraction
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from scipy.signal import resample_poly

from .step_0b_filter_faithdata import Preemphasis
from .utils.configuration import load_settings, setup_directory

logger = logging.getLogger(__name__)

default_settings = {
    "shots_path": Path("data/autoprocess/settings/shots.txt"),
    "foundation_dir": Path("data/autoprocess/foundation"),
    "input_key": "bes",
    "input_channels": [26, 28, 30, 32, 34, 36, 38, 40],
    "subseq_len": 66000,
    "preemphasis_coeff": 0.99,
    "min_samples": 100000,
    "target_rate_khz": None,  # None -> use stored rate as-is (converter is already 500)
    "output_dir": Path("data/cache/step_0c_convert_faithdata"),
    "frame_info_path": Path("data/cache/step_0c_convert_faithdata/frame_info.csv"),
    "overwrite": True,
}


def _read_shots(shots_path: Path) -> list[int]:
    return [int(s) for s in Path(shots_path).read_text().split()]


def _resample_to(sig: np.ndarray, src_khz: float, target_khz: float) -> np.ndarray:
    """Resample ``(C, N)`` from ``src_khz`` to ``target_khz`` (anti-aliased)."""
    frac = Fraction(int(round(target_khz)), int(round(src_khz))).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    if (up, down) == (1, 1):
        return sig
    return resample_poly(sig, up, down, axis=1)


def main(config_path=None, settings=None):
    if settings is None:
        settings = load_settings(config_path, default_settings)

    foundation_dir = Path(
        settings.get("foundation_dir", default_settings["foundation_dir"])
    )
    modality = settings.get("input_key", default_settings["input_key"])
    channels = list(settings.get("input_channels", default_settings["input_channels"]))
    subseq_len = int(settings.get("subseq_len", default_settings["subseq_len"]))
    min_samples = int(settings.get("min_samples", default_settings["min_samples"]))
    coeff = float(settings.get("preemphasis_coeff", default_settings["preemphasis_coeff"]))
    target_khz = settings.get("target_rate_khz", default_settings["target_rate_khz"])

    output_dir = Path(settings.get("output_dir", default_settings["output_dir"]))
    setup_directory(output_dir, overwrite=settings.get("overwrite", True))
    frame_info_path = Path(
        settings.get("frame_info_path", default_settings["frame_info_path"])
    )
    frame_info_path.parent.mkdir(parents=True, exist_ok=True)

    shots = _read_shots(settings.get("shots_path", default_settings["shots_path"]))
    preemph = Preemphasis(coeff)

    index = 0
    n_shots_used = 0
    with frame_info_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shotn", "time_start", "time_end"])
        for shot in shots:
            p = foundation_dir / f"{shot}_processed.h5"
            if not p.exists():
                logger.warning(f"{modality}: shot {shot} absent in {foundation_dir}")
                continue
            with h5py.File(p, "r") as fh:
                if modality not in fh or "ydata" not in fh[modality]:
                    logger.warning(f"{modality}: shot {shot} has no {modality} group")
                    continue
                ydata = fh[modality]["ydata"]
                xdata = fh[modality]["xdata"]
                n = ydata.shape[1]
                if n < min_samples:
                    logger.warning(f"{modality}: shot {shot} degenerate (N={n}); skip")
                    continue
                if max(channels) >= ydata.shape[0]:
                    logger.warning(
                        f"{modality}: shot {shot} has {ydata.shape[0]} channels "
                        f"< requested max index {max(channels)}; skip"
                    )
                    continue
                if xdata.shape[0] != n:
                    logger.warning(
                        f"{modality}: shot {shot} xdata({xdata.shape[0]}) != "
                        f"ydata N({n}); skip"
                    )
                    continue
                sig = np.asarray(ydata[channels, :])  # (C_sel, N)
                time = np.asarray(xdata[:]).reshape(-1)

            # optional resample to a common rate (no-op when target_khz is None)
            if target_khz is not None and len(time) > 1:
                src_khz = 1.0 / (float(time[1] - time[0]) * 1000.0)  # seconds -> kHz
                sig = _resample_to(sig, src_khz, float(target_khz))
                dt_ms = 1.0 / float(target_khz)
                t0_ms = float(time[0]) * 1000.0
                time = (t0_ms + np.arange(sig.shape[1]) * dt_ms) / 1000.0

            sig_t = preemph(torch.from_numpy(sig).float()).numpy()
            n_eff = sig_t.shape[1]
            for w in range(n_eff // subseq_len):
                s0, s1 = w * subseq_len, (w + 1) * subseq_len
                window = sig_t[:, s0:s1]  # (C_sel, subseq_len)
                t0 = float(time[s0]) if s0 < len(time) else 0.0
                t1 = float(time[s1 - 1]) if (s1 - 1) < len(time) else 0.0
                writer.writerow([shot, f"{t0:.6f}", f"{t1:.6f}"])
                joblib.dump(window, output_dir / f"{index}.joblib", compress=True)
                index += 1
            n_shots_used += 1

    logger.info(
        f"step_0f [{modality}]: {index} windows from {n_shots_used} shots -> {output_dir}"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
