"""``tokeye elmspec`` — detect ELM events via the transient-activity channel."""

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
        "elmspec",
        help="Detect ELM events (transient-channel intervals, count, frequency).",
    )
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
        default="tokeye_elms",
        help="Directory to write event/summary CSVs (and previews) to.",
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
        "--fs",
        type=float,
        default=None,
        help=(
            "Sampling rate in Hz of the original signals; enables absolute "
            "event times and ELM frequency in the CSVs."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Mask binarization threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--activity-min",
        type=float,
        default=0.1,
        help=(
            "Minimum fraction of active frequency bins for a time column to "
            "belong to an ELM (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--min-gap-cols",
        type=int,
        default=3,
        help="Merge events separated by at most this many columns (default: %(default)s).",
    )
    parser.add_argument(
        "--min-duration-cols",
        type=int,
        default=1,
        help="Drop events shorter than this many columns (default: %(default)s).",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also write a mask-overlay preview PNG per input.",
    )
    parser.add_argument("--device", default="auto")
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    from huggingface_hub.errors import HfHubHTTPError

    from tokeye import batch
    from tokeye.cli._errors import print_hub_error
    from tokeye.elmspec import (
        extract_elm_events,
        summarize,
        write_events_csv,
        write_summary_csv,
    )
    from tokeye.hub import DEFAULT_MODEL, load_model
    from tokeye.inference import model_infer

    stft_kwargs = {
        "n_fft": args.n_fft,
        "hop": args.hop,
        "clip_dc": not args.keep_dc,
        "clip_low": args.clip_low,
        "clip_high": args.clip_high,
    }
    model_name = args.model if args.model is not None else DEFAULT_MODEL

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
        model = load_model(model_name, args.device)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (HfHubHTTPError, OSError) as exc:
        print_hub_error(model_name, exc)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_events = []
    all_summaries = []
    failures = 0
    for path in paths:
        try:
            spectrogram = batch.load_input(path, stft_kwargs, log=args.log)
            mask = model_infer(spectrogram, model)
            events = extract_elm_events(
                mask[1],
                threshold=args.threshold,
                activity_min=args.activity_min,
                min_gap_cols=args.min_gap_cols,
                min_duration_cols=args.min_duration_cols,
            )
        except Exception as exc:  # noqa: BLE001 - mirror `tokeye run`: keep batch going
            print(f"error: failed to process {path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        summary = summarize(events, n_cols=mask.shape[-1], hop=args.hop, fs=args.fs)
        all_events.append((str(path), events))
        all_summaries.append((str(path), summary))
        freq = summary["elm_freq_hz"]
        freq_text = f", {freq:.1f} Hz" if freq is not None else ""
        print(f"{path}: {summary['n_events']} ELM event(s){freq_text}")

        if args.png:
            preview_path = out_dir / f"{path.stem}_elm_preview.png"
            batch.save_overlay_png(spectrogram, mask, preview_path, threshold=args.threshold)

    events_csv = out_dir / "elm_events.csv"
    summary_csv = out_dir / "elm_summary.csv"
    write_events_csv(events_csv, all_events, hop=args.hop, fs=args.fs)
    write_summary_csv(summary_csv, all_summaries)
    print(events_csv)
    print(summary_csv)
    return failures
