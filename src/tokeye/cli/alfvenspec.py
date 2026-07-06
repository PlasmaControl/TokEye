"""``tokeye alfvenspec`` — Alfvén-eigenmode detection with ae_tf_maskrcnn."""

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
    parser = subparsers.add_parser(
        "alfvenspec",
        help="Detect Alfvén-eigenmode activity (boxes + masks via ae_tf_maskrcnn).",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="Files, directories of .npy files, or glob patterns.",
    )
    parser.add_argument(
        "--model",
        default="ae_tf_maskrcnn",
        help="Registry name or path to a model checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="tokeye_ae",
        help="Directory to write detections CSV, masks, and previews to.",
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
    parser.add_argument(
        "--score-min",
        type=float,
        default=0.5,
        help="Keep detections with at least this score (default: %(default)s).",
    )
    parser.add_argument(
        "--window-cols",
        type=int,
        default=710,
        help=(
            "Process wide spectrograms in windows of this many columns "
            "(training width; 0 disables windowing; default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=None,
        help="Standardization mean (default: per-input statistics).",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=None,
        help="Standardization std (default: per-input statistics).",
    )
    parser.add_argument(
        "--no-masks",
        dest="save_masks",
        action="store_false",
        help="Skip writing per-input instance masks (.npy).",
    )
    parser.add_argument("--device", default="auto")
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    from huggingface_hub.errors import HfHubHTTPError

    from tokeye import batch
    from tokeye.alfvenspec import detect_windowed, write_detections_csv
    from tokeye.cli._errors import print_hub_error
    from tokeye.hub import load_model

    stft_kwargs = {
        "n_fft": args.n_fft,
        "hop": args.hop,
        "clip_dc": not args.keep_dc,
        "clip_low": args.clip_low,
        "clip_high": args.clip_high,
    }

    try:
        paths = batch.collect_inputs(args.inputs)
    except (ValueError, FileNotFoundError) as exc:
        hint = (
            " (no data yet? create a demo signal with: tokeye example)"
            if "No input files found" in str(exc)
            else ""
        )
        print(f"error: {exc}{hint}", file=sys.stderr)
        return 2

    try:
        model = load_model(args.model, args.device)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (HfHubHTTPError, OSError) as exc:
        print_hub_error(args.model, exc)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    all_detections = []
    failures = 0
    for path in paths:
        try:
            spectrogram = batch.load_input(path, stft_kwargs, log=args.log)
            detections = detect_windowed(
                spectrogram,
                model,
                window_cols=args.window_cols,
                score_min=args.score_min,
                mean=args.mean,
                std=args.std,
            )
        except Exception as exc:  # noqa: BLE001 - mirror `tokeye run`: keep batch going
            print(f"error: failed to process {path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        all_detections.append((str(path), detections))
        print(f"{path}: {len(detections['boxes'])} detection(s)")

        masks = detections["masks"]  # None when the input was windowed
        if args.save_masks and masks is not None and len(masks):
            np.save(out_dir / f"{path.stem}_ae_masks.npy", masks)

    detections_csv = out_dir / "ae_detections.csv"
    write_detections_csv(detections_csv, all_detections)
    print(detections_csv)
    return failures
