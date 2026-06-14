"""Re-express the preserved ``raw_fast`` shots in ``foundation_model`` layout.

Counterpart to :mod:`preserve_raw_fast` (which built ``raw_fast`` from the ~18 GB
MDSplus dumps): this converts each ``data/autoprocess/raw_fast/{shot}.h5`` into the
same on-disk layout as ``/scratch/gpfs/EKOLEMEN/foundation_model/{shot}_processed.h5``
so the foundation loader (``step_0f_foundation``) can read a single, self-contained,
project-local copy of every training shot. Only 5 of the 23 training shots exist in
the external ``foundation_model`` dir, so this local copy is the only complete source.

Source layout (``raw_fast/{shot}.h5``)::

    /{modality}   group, attrs: t0_ms, dt_ms, n, rate_khz, units, [source_tree]
        {channel}  float32 (n,)        one dataset per channel

Target layout (``{out_dir}/{shot}_processed.h5``), matching foundation_model::

    /{modality}   group, NO attrs
        xdata  float32 (N',)           time base in SECONDS
        ydata  float32 (C, N')         channels x samples, sorted channel-name order

Each modality is resampled to ``--target-khz`` (default 500 -- the uniform rate the
foundation pipeline uses: ece already 500, mhr 2000->500, bes 1000->500, co2
~1667->500) with an anti-aliasing polyphase filter. Pass ``--no-resample`` to keep
native rates instead (the loader can then resample at read time).

Run::

    python -m tokeye.training.big_tf_unet_ablation.preprocess.raw_fast_to_foundation
    python -m ...raw_fast_to_foundation --overwrite --shots-file data/autoprocess/settings/shots.txt
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

DEFAULT_RAW_FAST = Path("data/autoprocess/raw_fast")
DEFAULT_OUT = Path("data/autoprocess/foundation")
DEFAULT_SHOTS = Path("data/autoprocess/settings/shots.txt")
DEFAULT_TARGET_KHZ = 500.0


def _resample_to(
    sig: np.ndarray, src_khz: float, target_khz: float
) -> tuple[np.ndarray, float]:
    """Resample ``(C, N)`` from ``src_khz`` to ``target_khz`` (anti-aliased).

    Returns ``(resampled, actual_rate_khz)``. A 1:1 ratio is a no-op. Mirrors
    ``step_0g_raw_fast._resample_to`` (same rational up/down, same filter).
    """
    frac = Fraction(int(round(target_khz)), int(round(src_khz))).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    if (up, down) == (1, 1):
        return sig, src_khz
    out = resample_poly(sig, up, down, axis=1)
    return out, src_khz * up / down


def convert_shot(
    shot: str,
    raw_dir: Path,
    out_dir: Path,
    target_khz: float | None,
    overwrite: bool,
) -> dict:
    src = raw_dir / f"{shot}.h5"
    out = out_dir / f"{shot}_processed.h5"
    if not src.exists():
        return {"shot": shot, "status": "src_missing"}
    if out.exists() and not overwrite:
        return {"shot": shot, "status": "skip_exists"}

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.tmp")
    shapes: dict[str, tuple[int, int]] = {}
    try:
        with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fout:
            for mod in sorted(fin.keys()):
                g = fin[mod]
                if not isinstance(g, h5py.Group):
                    continue
                chan_names = sorted(g.keys())
                if not chan_names:
                    continue
                src_khz = float(g.attrs["rate_khz"])
                t0_ms = float(g.attrs["t0_ms"])
                # Drop degenerate/truncated channels: some shots have stub channels
                # of length 1 alongside the real ones (e.g. 170796 ece TECEF41-48),
                # which would break np.stack. Keep channels matching the group's
                # declared sample count (fall back to the most common length).
                lengths = [g[c].shape[0] for c in chan_names]
                n_decl = int(g.attrs["n"]) if "n" in g.attrs else None
                if n_decl is not None and n_decl in lengths:
                    target_len = n_decl
                else:
                    target_len = Counter(lengths).most_common(1)[0][0]
                kept = [c for c, n_c in zip(chan_names, lengths) if n_c == target_len]
                dropped = [c for c, n_c in zip(chan_names, lengths) if n_c != target_len]
                if dropped:
                    logger.warning(
                        f"{shot}/{mod}: dropping {len(dropped)} off-length channels "
                        f"{dropped[:4]}{'...' if len(dropped) > 4 else ''} "
                        f"(kept {len(kept)} @ {target_len})"
                    )
                if not kept:
                    continue
                # stack channels as float32 (already float32 on disk) -> (C, N)
                ydata = np.stack([g[c][:] for c in kept]).astype(np.float32)
                if target_khz is not None:
                    ydata, rate_khz = _resample_to(ydata, src_khz, target_khz)
                else:
                    rate_khz = src_khz
                ydata = np.ascontiguousarray(ydata, dtype=np.float32)
                n = ydata.shape[1]
                dt_ms = 1.0 / rate_khz
                # time base in SECONDS (foundation convention)
                xdata = (
                    (t0_ms + np.arange(n, dtype=np.float64) * dt_ms) / 1000.0
                ).astype(np.float32)
                go = fout.create_group(mod)
                go.create_dataset("xdata", data=xdata)
                go.create_dataset("ydata", data=ydata)
                shapes[mod] = ydata.shape
                if not np.any(ydata):
                    logger.warning(f"{shot}/{mod}: all-zero ydata {ydata.shape}")
    except Exception as e:  # noqa: BLE001 - report per-shot, keep going
        tmp.unlink(missing_ok=True)
        return {"shot": shot, "status": f"error: {e}"}
    tmp.replace(out)  # atomic: only a complete file appears at the final path
    return {
        "shot": shot,
        "status": "ok",
        "size_gb": out.stat().st_size / 1e9,
        "shapes": shapes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots-file", type=Path, default=DEFAULT_SHOTS)
    ap.add_argument("--raw-fast-dir", type=Path, default=DEFAULT_RAW_FAST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--target-khz",
        type=float,
        default=DEFAULT_TARGET_KHZ,
        help="resample every modality to this rate (default 500, foundation's rate)",
    )
    ap.add_argument(
        "--no-resample",
        action="store_true",
        help="keep each modality's native rate instead of resampling to --target-khz",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    target_khz = None if args.no_resample else args.target_khz

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shots = args.shots_file.read_text().split()
    rate_desc = "native" if target_khz is None else f"{target_khz}kHz"
    logger.info(
        f"Converting {len(shots)} shots ({rate_desc}): "
        f"{args.raw_fast_dir} -> {args.out_dir}"
    )

    results = []
    for i, shot in enumerate(shots, 1):
        r = convert_shot(
            shot, args.raw_fast_dir, args.out_dir, target_khz, args.overwrite
        )
        results.append(r)
        shapes = r.get("shapes", {})
        shape_str = " ".join(f"{m}{tuple(s)}" for m, s in shapes.items())
        logger.info(
            f"[{i}/{len(shots)}] {shot}: {r.get('status')} "
            f"{r.get('size_gb', 0):.2f}GB {shape_str}"
        )

    ok = [r for r in results if r["status"] == "ok"]
    total = sum(r.get("size_gb", 0) for r in ok)
    logger.info(f"DONE: {len(ok)}/{len(shots)} converted, {total:.1f}GB total")
    bad = [r for r in results if r["status"] not in ("ok", "skip_exists")]
    if bad:
        logger.warning(f"problems: {bad}")


if __name__ == "__main__":
    main()
