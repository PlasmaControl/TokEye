"""``tokeye download`` — pre-fetch model checkpoints."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("download", help="Download one or more model checkpoints.")
    parser.add_argument(
        "models",
        nargs="*",
        default=None,
        metavar="MODEL",
        help="Model registry name(s) to download (default: big_tf_unet).",
    )
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    from huggingface_hub.errors import HfHubHTTPError

    from tokeye.cli._errors import print_hub_error
    from tokeye.hub import DEFAULT_MODEL, download_model

    names = args.models or [DEFAULT_MODEL]
    for name in names:
        try:
            path = download_model(name)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except (HfHubHTTPError, OSError) as exc:
            print_hub_error(name, exc)
            return 2
        print(path)
    return 0
