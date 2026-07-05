"""``tokeye`` console entry point.

Argparse only (no new dependencies). Heavy imports (torch, ``tokeye.batch``,
``tokeye.app``) are deferred into each subcommand handler so ``tokeye --help``
returns instantly and ``tokeye run`` never imports gradio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tokeye.transforms import (
    DEFAULT_CLIP_HIGH,
    DEFAULT_CLIP_LOW,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _add_app_subcommand(subparsers: argparse._SubParsersAction) -> None:
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
    parser.set_defaults(handler=_handle_app)


def _add_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run", help="Run batch inference on one or more inputs.")
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="Files, directories of .npy files, or glob patterns.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Registry name or path to a model checkpoint (default: big_tf_unet).",
    )
    parser.add_argument(
        "--output-dir",
        default="tokeye_output",
        help="Directory to write masks and previews to.",
    )
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop", type=int, default=DEFAULT_HOP)
    parser.add_argument(
        "--keep-dc",
        action="store_true",
        help="Do not clip the DC bin (clipped by default).",
    )
    parser.add_argument("--clip-low", type=float, default=DEFAULT_CLIP_LOW)
    parser.add_argument("--clip-high", type=float, default=DEFAULT_CLIP_HIGH)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--no-png",
        dest="save_png",
        action="store_false",
        help="Skip PNG overlay previews.",
    )
    parser.add_argument("--device", default="auto")
    parser.set_defaults(handler=_handle_run)


def _add_download_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("download", help="Download one or more model checkpoints.")
    parser.add_argument(
        "models",
        nargs="*",
        default=None,
        metavar="MODEL",
        help="Model registry name(s) to download (default: big_tf_unet).",
    )
    parser.set_defaults(handler=_handle_download)


def _add_example_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "example", help="Write a synthetic example signal to a .npy file."
    )
    parser.add_argument("--output", default="tokeye_example.npy")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--fs", type=float, default=200_000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(handler=_handle_example)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokeye",
        description=(
            "Automatic classification and localization of fluctuating signals "
            "in spectrograms."
        ),
    )
    parser.add_argument(
        "--version", action="store_true", help="Print the tokeye version and exit."
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_app_subcommand(subparsers)
    _add_run_subcommand(subparsers)
    _add_download_subcommand(subparsers)
    _add_example_subcommand(subparsers)
    return parser


def _handle_app(args: argparse.Namespace) -> int:
    from tokeye.app.__main__ import main as app_main

    app_main(port=args.port, share=args.share, open_browser=args.open_browser)
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    from huggingface_hub.errors import HfHubHTTPError

    from tokeye import batch
    from tokeye.hub import DEFAULT_MODEL, DEFAULT_REPO_ID

    stft_kwargs = {
        "n_fft": args.n_fft,
        "hop": args.hop,
        "clip_dc": not args.keep_dc,
        "clip_low": args.clip_low,
        "clip_high": args.clip_high,
    }
    model = args.model if args.model is not None else DEFAULT_MODEL

    try:
        return batch.run_batch(
            args.inputs,
            model=model,
            out_dir=Path(args.output_dir),
            stft_kwargs=stft_kwargs,
            save_png=args.save_png,
            threshold=args.threshold,
            device=args.device,
        )
    except (ValueError, FileNotFoundError) as exc:
        hint = (
            " (no data yet? create a demo signal with: tokeye example)"
            if "No input files found" in str(exc)
            else ""
        )
        print(f"error: {exc}{hint}", file=sys.stderr)
        return 2
    except (HfHubHTTPError, OSError) as exc:
        print(
            f"error: could not download model {model!r} from Hugging Face "
            f"repo {DEFAULT_REPO_ID!r}: {exc}. If the repo has moved, set "
            "TOKEYE_HF_REPO to override.",
            file=sys.stderr,
        )
        return 2


def _handle_download(args: argparse.Namespace) -> int:
    from huggingface_hub.errors import HfHubHTTPError

    from tokeye.hub import DEFAULT_MODEL, DEFAULT_REPO_ID, download_model

    names = args.models or [DEFAULT_MODEL]
    for name in names:
        try:
            path = download_model(name)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except (HfHubHTTPError, OSError) as exc:
            print(
                f"error: could not download model {name!r} from Hugging Face "
                f"repo {DEFAULT_REPO_ID!r}: {exc}. If the repo has moved, set "
                "TOKEYE_HF_REPO to override.",
                file=sys.stderr,
            )
            return 2
        print(path)
    return 0


def _handle_example(args: argparse.Namespace) -> int:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from importlib.metadata import version

        print(version("tokeye"))
        return 0

    if args.command is None:
        parser.print_help()
        return 2

    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
