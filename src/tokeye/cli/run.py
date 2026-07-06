"""``tokeye run`` — headless batch inference."""

from __future__ import annotations

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
    import argparse


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
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
    parser.add_argument(
        "--log",
        action="store_true",
        help=(
            "Apply log1p to 2D spectrogram inputs stored in linear scale "
            "(1D signals are always log-scaled during the STFT)."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--no-png",
        dest="save_png",
        action="store_false",
        help="Skip PNG overlay previews.",
    )
    parser.add_argument("--device", default="auto")
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    from huggingface_hub.errors import HfHubHTTPError

    from tokeye import batch
    from tokeye.cli._errors import print_hub_error
    from tokeye.hub import DEFAULT_MODEL

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
            log=args.log,
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
        print_hub_error(model, exc)
        return 2
