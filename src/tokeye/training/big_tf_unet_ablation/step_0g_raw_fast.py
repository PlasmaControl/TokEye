"""Step 0g: load + resample + preemphasis + window from the preserved single-shot
``raw_fast`` files (``data/autoprocess/raw_fast/{shot}.h5``).

Each file holds all four modalities as groups (``/ece /mhr /bes /co2``); channels are
named datasets sharing a uniform time base in ``(t0_ms, dt_ms, n, rate_khz)`` group
attrs. This loader replaces step_0a/0b/0c (and step_0f) for the preserved data: invoked
once per modality, it

  1. selects ``input_channels`` (0-based indices into the sorted channel list),
  2. resamples each to ``target_rate_khz`` (default 500 -- the uniform rate the
     foundation pipeline used; ece is already 500, mhr 2000->500, bes 1000->500,
     co2 ~1667->500) with an anti-aliasing polyphase filter,
  3. applies preemphasis, windows into ``subseq_len`` chunks,

and writes ``step_0c.h5`` + ``frame_info.csv`` in the format step_2a expects, so the
rest of the pipeline is unchanged.
"""

from __future__ import annotations

import csv
import logging
import sys
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.signal import resample_poly

from .step_0b_filter_faithdata import Preemphasis
from .utils.configuration import load_settings
from .utils.hdf5_io import create_step_file, write_sample

logger = logging.getLogger(__name__)

default_settings = {
    "shots_path": Path("data/autoprocess/settings/shots.txt"),
    "raw_fast_dir": Path("data/autoprocess/raw_fast"),
    "input_key": "ece",
    "input_channels": [8, 12, 16, 20, 24, 28, 32, 36],
    "subseq_len": 66000,
    "preemphasis_coeff": 0.99,
    "target_rate_khz": 500,
    "output_dir": Path("data/cache/ablation/shared/ece"),
    "frame_info_path": Path("data/cache/ablation/shared/ece/frame_info.csv"),
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

    raw_dir = Path(settings.get("raw_fast_dir", default_settings["raw_fast_dir"]))
    modality = settings.get("input_key", default_settings["input_key"])
    channels = list(settings.get("input_channels", default_settings["input_channels"]))
    subseq_len = int(settings.get("subseq_len", default_settings["subseq_len"]))
    coeff = float(settings.get("preemphasis_coeff", default_settings["preemphasis_coeff"]))
    target_khz = float(settings.get("target_rate_khz", default_settings["target_rate_khz"]))

    output_dir = Path(settings.get("output_dir", default_settings["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_info_path = Path(settings.get("frame_info_path", default_settings["frame_info_path"]))
    frame_info_path.parent.mkdir(parents=True, exist_ok=True)

    shots = _read_shots(settings["shots_path"])
    preemph = Preemphasis(coeff)

    h5_path = output_dir / "step_0c.h5"
    h5 = create_step_file(h5_path, metadata={
        "input_key": modality, "num_channels": len(channels),
        "subseq_len": subseq_len, "target_rate_khz": target_khz,
    })

    index = 0
    n_shots_used = 0
    try:
        with frame_info_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["shotn", "time_start", "time_end"])
            for shot in shots:
                p = raw_dir / f"{shot}.h5"
                if not p.exists():
                    logger.warning(f"{modality}: shot {shot} absent in raw_fast")
                    continue
                with h5py.File(p, "r") as fh:
                    if modality not in fh:
                        logger.warning(f"{modality}: shot {shot} has no /{modality} group; skip")
                        continue
                    g = fh[modality]
                    chan_names = sorted(g.keys())
                    if max(channels) >= len(chan_names):
                        logger.warning(
                            f"{modality}: shot {shot} has {len(chan_names)} channels "
                            f"< requested max index {max(channels)}; skip"
                        )
                        continue
                    src_khz = float(g.attrs["rate_khz"])
                    t0 = float(g.attrs["t0_ms"])
                    sig = np.stack([g[chan_names[c]][:] for c in channels]).astype(np.float64)

                sig = _resample_to(sig, src_khz, target_khz)  # (C_sel, N') @ target rate
                dt_new = 1.0 / target_khz  # ms per sample
                sig_t = preemph(torch.from_numpy(sig).float()).numpy()

                n = sig_t.shape[1]
                for w in range(n // subseq_len):
                    s0, s1 = w * subseq_len, (w + 1) * subseq_len
                    writer.writerow([shot, f"{t0 + s0 * dt_new:.4f}", f"{t0 + (s1 - 1) * dt_new:.4f}"])
                    write_sample(h5, index, sig_t[:, s0:s1])
                    index += 1
                n_shots_used += 1
    finally:
        h5.close()

    logger.info(
        f"step_0g [{modality}]: {index} windows from {n_shots_used} shots "
        f"@ {target_khz}kHz -> {h5_path}"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
