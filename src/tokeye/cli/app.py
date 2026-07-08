"""``tokeye app`` — launch the Gradio web app."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("app", help="Launch the TokEye Gradio web app.")
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to serve the app on."
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link."
    )
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Open the app in a browser on launch.",
    )
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    try:
        from tokeye.app.__main__ import main as app_main
    except ImportError as exc:
        print(
            "`tokeye app` needs the 'app' extra (gradio), which is not installed.\n"
            "Install it with:\n"
            "    pip install 'tokeye[app]'      # or:  uv sync --extra app\n"
            f"(underlying import error: {exc})",
            file=sys.stderr,
        )
        return 1

    app_main(port=args.port, share=args.share, open_browser=args.open_browser)
    return 0
