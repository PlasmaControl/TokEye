"""``tokeye modesearch`` — mode database (design stage, prints the vision)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

_DESCRIPTION = """\
modesearch is not implemented yet. The plan:

  1. An offline crawler runs the TokEye suite (big_tf_unet, modespec,
     elmspec, alfvenspec) over shot archives and indexes every detected
     mode: shot, time interval, frequency band, mode numbers, amplitude,
     detector provenance.
  2. Researchers query the index -- e.g. "shots with an n=2 tearing mode
     between 2-4 kHz during an ELM-free period" -- instead of re-scanning
     raw data.
  3. The same index feeds the fusion-world-model shot designer with mode
     occurrence statistics.

Design notes: src/tokeye/modesearch/README.md and docs/ROADMAP.md.
"""


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "modesearch",
        help="Mode database + queries (design stage; prints the plan).",
    )
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    print(_DESCRIPTION)
    return 0
