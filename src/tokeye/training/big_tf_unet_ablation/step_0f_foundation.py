"""Step 0f: load + preemphasis + window directly from the foundation_model set.

Replaces step_0a/0b/0c for shots stored as ``{shot}_processed.h5`` in
``/scratch/gpfs/EKOLEMEN/foundation_model``, where each modality is an HDF5
group with ``xdata (N,)`` (time) and ``ydata (C, N)`` (channels x samples) --
a different layout from the FAITH ``d3d_fusion_data`` files that step_0a reads.

Invoked once per modality (like step_0c); writes ``step_0c.h5`` + ``frame_info.csv``
in the SAME format step_2a expects, so the rest of the pipeline is unchanged.
Shots whose modality is degenerate (``N < min_samples``, e.g. co2/bes missing)
are skipped for that modality.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from .step_0b_filter_faithdata import Preemphasis
from .utils.configuration import load_settings
from .utils.hdf5_io import create_step_file, write_sample

logger = logging.getLogger(__name__)

default_settings = {
    "shots_path": Path("data/autoprocess/settings/shots_ablation.txt"),
    "foundation_dir": Path("/scratch/gpfs/EKOLEMEN/foundation_model"),
    "input_key": "bes",
    "input_channels": [26, 28, 30, 32, 34, 36, 38, 40],
    "subseq_len": 66000,
    "preemphasis_coeff": 0.99,
    "min_samples": 100000,
    "output_dir": Path("data/cache/ablation/shared/bes"),
    "frame_info_path": Path("data/cache/ablation/shared/bes/frame_info.csv"),
    "overwrite": True,
}


def _read_shots(shots_path: Path) -> list[int]:
    return [int(s) for s in Path(shots_path).read_text().split()]


def main(config_path=None, settings=None):
    if settings is None:
        settings = load_settings(config_path, default_settings)

    foundation_dir = Path(settings.get("foundation_dir", default_settings["foundation_dir"]))
    modality = settings.get("input_key", default_settings["input_key"])
    channels = list(settings.get("input_channels", default_settings["input_channels"]))
    subseq_len = int(settings.get("subseq_len", default_settings["subseq_len"]))
    min_samples = int(settings.get("min_samples", default_settings["min_samples"]))
    coeff = float(settings.get("preemphasis_coeff", default_settings["preemphasis_coeff"]))

    output_dir = Path(settings.get("output_dir", default_settings["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_info_path = Path(settings.get("frame_info_path", default_settings["frame_info_path"]))
    frame_info_path.parent.mkdir(parents=True, exist_ok=True)

    shots = _read_shots(settings["shots_path"])
    preemph = Preemphasis(coeff)

    h5_path = output_dir / "step_0c.h5"
    h5 = create_step_file(
        h5_path,
        metadata={"input_key": modality, "num_channels": len(channels), "subseq_len": subseq_len},
    )

    index = 0
    n_shots_used = 0
    try:
        with frame_info_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["shotn", "time_start", "time_end"])
            for shot in shots:
                p = foundation_dir / f"{shot}_processed.h5"
                if not p.exists():
                    logger.warning(f"{modality}: shot {shot} absent in foundation_model")
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
                            f"< requested max {max(channels)}; skip"
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
                # preemphasis per channel
                sig_t = preemph(torch.from_numpy(sig).float()).numpy()
                n_windows = n // subseq_len
                for w in range(n_windows):
                    s0, s1 = w * subseq_len, (w + 1) * subseq_len
                    window = sig_t[:, s0:s1]  # (C_sel, subseq_len)
                    t0 = float(time[s0]) if s0 < len(time) else 0.0
                    t1 = float(time[s1 - 1]) if (s1 - 1) < len(time) else 0.0
                    writer.writerow([shot, f"{t0:.4f}", f"{t1:.4f}"])
                    write_sample(h5, index, window)
                    index += 1
                n_shots_used += 1
    finally:
        h5.close()

    logger.info(
        f"step_0f [{modality}]: {index} windows from {n_shots_used} shots -> {h5_path}"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
