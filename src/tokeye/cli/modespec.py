"""``tokeye modespec`` — classic Mirnov mode-number analysis (vendored pymodespec)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "modespec",
        help=(
            "Classic DIII-D mode analysis: Mirnov spectrograms, toroidal "
            "mode-number fits, per-shot mode CSVs (needs MDSplus or a cache)."
        ),
    )
    parser.add_argument(
        "config",
        metavar="CONFIG",
        help="YAML config listing shots and analysis parameters (see modes.yaml).",
    )
    parser.add_argument(
        "--engine",
        choices=["classic"],
        default="classic",
        help="Analysis engine (only 'classic' today; 'deep' is planned).",
    )
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    from pathlib import Path

    import matplotlib as mpl

    mpl.use("Agg")  # vendored modules import pyplot at module load

    from tokeye.modespec.classic import run_config

    config_path = Path(args.config)
    if not config_path.exists():
        example = Path(__file__).parent.parent / "modespec" / "classic" / "modes.yaml"
        print(
            f"error: config not found: {config_path} "
            f"(example config: {example})",
            file=sys.stderr,
        )
        return 2

    try:
        return run_config(config_path)
    except (KeyError, ValueError) as exc:
        print(f"error: bad config {config_path}: {exc}", file=sys.stderr)
        return 2
