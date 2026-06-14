"""Step 3b: Extract individual samples from batched step_3a predictions.

Reads batched ``(B, C, H, W, Z)`` arrays from step_3a HDF5, un-batches
them, and writes individual ``(C, H, W, Z)`` samples to step_3b HDF5.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .utils.configuration import load_settings
from .utils.hdf5_io import (
    create_step_file,
    iter_samples,
    write_sample,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_settings = {
    "input_h5": Path("data/cache/step_3a.h5"),
    "output_h5": Path("data/cache/step_3b.h5"),
    "overwrite": True,
}


def main(
    config_path: Path | str | None = None,
    settings: dict | None = None,
) -> None:
    if settings is None:
        settings = load_settings(config_path, default_settings)

    input_h5 = Path(settings.get("input_h5", default_settings["input_h5"]))
    output_h5 = Path(settings.get("output_h5", default_settings["output_h5"]))
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    h5_out = create_step_file(output_h5)
    sample_idx = 0
    try:
        for _, batch in iter_samples(input_h5):
            # batch shape: (B, C, H, W, Z)
            if batch.ndim == 5:
                for i in range(batch.shape[0]):
                    write_sample(h5_out, sample_idx, batch[i])
                    sample_idx += 1
            else:
                # Already individual
                write_sample(h5_out, sample_idx, batch)
                sample_idx += 1
    finally:
        h5_out.attrs["num_samples"] = sample_idx
        h5_out.close()

    logger.info(f"Extracted {sample_idx} individual samples to {output_h5}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
