"""``tokeye example`` — write a synthetic demo signal."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "example", help="Write a synthetic example signal to a .npy file."
    )
    parser.add_argument("--output", default="tokeye_example.npy")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--fs", type=float, default=200_000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    import numpy as np

    from tokeye.examples import make_example_signal

    output_path = Path(args.output)
    if output_path.suffix != ".npy":
        # np.save silently appends ".npy" to paths without that suffix;
        # normalize up-front so the printed path is the file that exists.
        output_path = output_path.with_suffix(".npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal = make_example_signal(duration_s=args.duration, fs=args.fs, seed=args.seed)
    np.save(output_path, signal)
    print(output_path)
    return 0
