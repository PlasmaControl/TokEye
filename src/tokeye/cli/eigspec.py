"""``tokeye eigspec`` — modal identification / spectral analysis (vendored eigspec)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "eigspec",
        help=(
            "Interactive modal identification and spectral analysis "
            "(SSI, AR/PCA, random-projection; MATLAB eigspec port)."
        ),
    )
    parser.add_argument(
        "script",
        nargs="?",
        default=None,
        metavar="SCRIPT",
        help="Optional eigspec script file to execute instead of the prompt.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed error messages.",
    )
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    import os

    import matplotlib as mpl

    mpl.use("Agg")  # vendored vis modules import pyplot at module load

    if args.debug:
        os.environ["EIGSPEC_DEBUG"] = "1"

    from tokeye.eigspec.cli import EigspecCLI

    EigspecCLI().run(script_file=args.script)
    return 0
