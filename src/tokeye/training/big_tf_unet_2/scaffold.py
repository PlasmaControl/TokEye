"""Create a new run: workspace (run.yaml + notebooks), cache dirs, scale lock.

Usage (from the repo root):

    python -m tokeye.training.big_tf_unet_2.scaffold --nfft 512 --hop 128
    python -m tokeye.training.big_tf_unet_2.scaffold --nfft 512 --hop 128 --suffix v2

Refuses to overwrite an existing run.yaml — a run's knob file is its
experiment record.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from .paths import (
    NOTEBOOK_TEMPLATE_DIR,
    RUN_TEMPLATE_YAML,
    RunPaths,
    run_id_for,
)
from .run_config import ConfigError, RunSection


def scaffold_run(
    nfft: int, hop: int, suffix: str = "", allow_custom_scale: bool = False
) -> RunPaths:
    # Validate the scale with the same rules run.yaml will be held to later.
    try:
        RunSection(nfft=nfft, hop=hop, allow_custom_scale=allow_custom_scale)
    except Exception as err:  # pydantic ValidationError
        raise ConfigError(str(err)) from None

    run_id = run_id_for(nfft, hop, suffix)
    paths = RunPaths(run_id)

    if paths.run_yaml.exists():
        raise FileExistsError(
            f"{paths.run_yaml} already exists — refusing to overwrite. "
            f"Use --suffix for a fresh variant of this scale."
        )

    paths.workspace.mkdir(parents=True, exist_ok=True)
    paths.cache_root.mkdir(parents=True, exist_ok=True)
    paths.slurm_dir.mkdir(parents=True, exist_ok=True)

    yaml_text = (
        RUN_TEMPLATE_YAML.read_text()
        .replace("{{RUN_ID}}", run_id)
        .replace("{{NFFT}}", str(nfft))
        .replace("{{HOP}}", str(hop))
    )
    paths.run_yaml.write_text(yaml_text)

    notebooks = sorted(NOTEBOOK_TEMPLATE_DIR.glob("*.ipynb"))
    for nb in notebooks:
        target = paths.workspace / nb.name
        target.write_text(nb.read_text().replace("__RUN_ID__", run_id))

    paths.run_meta.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "nfft": nfft,
                "hop": hop,
                "created": datetime.now().isoformat(),
            },
            indent=2,
        )
    )

    print(f"Run {run_id} created.")
    print(f"  knobs:     {paths.run_yaml}")
    print(f"  notebooks: {paths.workspace}/")
    if not notebooks:
        print("  (no notebook templates found — package notebooks/ is empty)")
    print(f"  cache:     {paths.cache_root}")
    print(f"Start with {paths.workspace}/00_setup.ipynb")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfft", type=int, required=True)
    parser.add_argument("--hop", type=int, required=True)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--allow-custom-scale", action="store_true")
    args = parser.parse_args()
    scaffold_run(args.nfft, args.hop, args.suffix, args.allow_custom_scale)


if __name__ == "__main__":
    main()
