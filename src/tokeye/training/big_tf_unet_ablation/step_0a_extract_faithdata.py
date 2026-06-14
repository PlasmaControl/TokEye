from __future__ import annotations

import logging
import sys
from pathlib import Path

from tqdm.auto import tqdm

from .preprocess import index_dataset, pipeline
from .utils.configuration import load_settings, setup_directory

logger = logging.getLogger(__name__)

default_settings = {
    "shots_path": Path("data/settings/shots.txt"),
    "faith_cfg_path": Path("data/settings/faith_dataset.yaml"),
    "output_dir": Path("data/cache/step_0a_extract_faithdata"),
    "overwrite": False,
}


def get_files(
    shots_path: Path,
    fusion_data_dir: Path,
) -> list[int]:
    """Get valid shot numbers from shots.txt file."""
    shot_numbers = [int(f.strip()) for f in shots_path.read_text().splitlines()]
    valid_shots = [
        num for num in shot_numbers if (fusion_data_dir / f"{num}.h5").exists()
    ]
    valid_shots.sort()
    return valid_shots


def load_data(
    valid_shots: list[int],
    faith_cfg: dict,
    output_dir: Path,
    settings: dict,
) -> None:
    """Load data from fusion data directory."""
    for shot in tqdm(valid_shots, desc="Processing Stage 0"):
        pipeline(
            shot, cfg=faith_cfg, out_dir=output_dir, override=settings["overwrite"]
        )
    index_dataset(output_dir)


def main(
    config_path: Path | str | None = None,
    settings: dict | None = None,
) -> None:
    if settings is None:
        settings = load_settings(config_path, default_settings)
    output_dir = setup_directory(
        path=settings["output_dir"],
        overwrite=settings["overwrite"],
    )

    faith_cfg_path = settings.get("faith_cfg_path", default_settings["faith_cfg_path"])
    faith_cfg = load_settings(faith_cfg_path)

    valid_shots = get_files(
        settings.get("shots_path", default_settings["shots_path"]),
        Path(faith_cfg["raw_data_dir"]),
    )

    load_data(
        valid_shots=valid_shots,
        faith_cfg=faith_cfg,
        output_dir=output_dir,
        settings=settings,
    )


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
