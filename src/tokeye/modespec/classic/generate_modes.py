#!/usr/bin/env python
"""
generate_modes.py — turn a config into a list of detected MHD modes.

Reads a YAML config that lists shots + analysis parameters, runs the modespec
pipeline on each shot, detects coherent mode "events", and writes one CSV per
shot to `<output_dir>/<shot>_modes.csv`. Optionally saves a four-panel
spectrogram figure per shot via plot_modespec().

Usage::

    pixi run modes modes.yaml
    # or
    python generate_modes.py modes.yaml

A "mode event" is a contiguous stretch of time where a single toroidal mode
number n is the dominant, statistically-significant mode. Significance per
(time, frequency) bin is:  n_dominant == n  AND  coherence >= threshold
AND  mode_amp[n] >= amp_min_G.  The threshold defaults to the 95% coherence
confidence level (result['c95']) computed by mode_spectrogram().
"""

import sys
import csv
import argparse
from pathlib import Path

import yaml
import numpy as np

# Headless backend BEFORE modespec imports pyplot (modespec.py imports it at module load).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .modespec import fetch_mirnov, mode_spectrogram, plot_modespec


# ── Config handling ──────────────────────────────────────────────────────────────

# Per-shot parameters and their defaults. A shot entry in the config may override
# any of these; anything omitted falls back to the config's `defaults` block, then
# to these built-in values.
PARAM_DEFAULTS = {
    "array":           "toroidal",   # 'toroidal' (14-probe) or 'poloidal' (31-probe)
    "integrated":      False,        # False = B-dot (200 kHz), True = integrated B (~20 kHz)
    "t_min_ms":        None,         # optional signal time crop
    "t_max_ms":        None,
    "dt_window_ms":    4.0,          # FFT window length
    "overlap_frac":    0.75,         # window overlap fraction
    "f_min_khz":       5.0,          # analysis band
    "f_max_khz":       50.0,
    "f_smooth_khz":    1.0,          # frequency smoothing bandwidth
    "n_range":         [1, 5],       # [n_min, n_max] mode numbers to test
    # detection
    "coherence_min":   None,         # None -> use result['c95']
    "amp_min_G":       0.5,          # mode_amp floor for a bin to count (units: G if
                                     #   integrated=True, else G/s — raise it for B-dot)
    "merge_gap_ms":    2.0,          # bridge sub-threshold gaps shorter than this
    "min_duration_ms": 5.0,          # discard events shorter than this
    "make_figure":     True,         # save <shot>_modespec.png
}

CSV_COLUMNS = [
    "array", "mode_label", "mode_number",
    "t_start_ms", "t_end_ms", "duration_ms",
    "peak_freq_khz", "peak_amp_G", "mean_coherence",
    "f_min_khz", "f_max_khz", "coherence_thresh",
]


def load_config(path):
    """Load YAML config -> (global_cfg, list_of_resolved_shot_cfgs)."""
    with Path(path).open() as fh:
        cfg = yaml.safe_load(fh) or {}

    defaults = {**PARAM_DEFAULTS, **(cfg.get("defaults") or {})}
    shots = cfg.get("shots") or []
    if not shots:
        raise SystemExit(f"No 'shots' listed in {path}")

    resolved = []
    for entry in shots:
        if isinstance(entry, int):          # allow a bare shot number
            entry = {"shot": entry}
        if "shot" not in entry:
            raise SystemExit(f"Shot entry missing 'shot': {entry}")
        merged = {**defaults, **entry}
        resolved.append(merged)

    global_cfg = {
        "output_dir": cfg.get("output_dir", "mode_analysis"),
        "atlas":      cfg.get("atlas", "atlas.gat.com"),
    }
    return global_cfg, resolved


# ── Detection ────────────────────────────────────────────────────────────────────

def _contiguous_runs(mask):
    """Yield (start, end) inclusive index pairs for each run of True in a 1-D bool array."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    for s, e in zip(starts, ends):
        yield int(s), int(e)


def _fill_gaps(mask, max_gap):
    """Bridge runs of False shorter than `max_gap` windows (morphological closing)."""
    if max_gap < 1:
        return mask
    out = mask.copy()
    inside = np.flatnonzero(mask)
    if inside.size == 0:
        return out
    for s, e in _contiguous_runs(~mask):
        if s > inside[0] and e < inside[-1] and (e - s + 1) <= max_gap:
            out[s:e + 1] = True       # gap is enclosed by True on both sides
    return out


def detect_modes(result, cfg):
    """Return a list of mode-event dicts from a mode_spectrogram() result."""
    t = result["t_win_ms"]
    f = result["freq_khz"]
    nd = result["n_dominant"]
    coh = result["coherence"]
    mode_amp = result["mode_amp"]
    n_lo, n_hi = result["n_range"]

    thresh = cfg["coherence_min"]
    if thresh is None:
        thresh = result["c95"]
    amp_min = cfg["amp_min_G"]
    min_dur = cfg["min_duration_ms"]

    step_ms = float(np.mean(np.diff(t))) if t.size > 1 else 1.0
    max_gap = int(round(cfg["merge_gap_ms"] / step_ms))

    events = []
    for n in range(int(n_lo), int(n_hi) + 1):
        if n == 0:
            continue  # n=0 is axisymmetric, not a rotating mode of interest
        amp_n = mode_amp[n]
        sig = (nd == n) & (coh >= thresh) & (amp_n >= amp_min)   # (n_win, n_freq)
        present = _fill_gaps(sig.any(axis=1), max_gap)            # (n_win,)
        for i0, i1 in _contiguous_runs(present):
            t0, t1 = float(t[i0]), float(t[i1])
            if (t1 - t0) < min_dur:
                continue
            # Peak (amplitude) bin among significant bins inside the event window.
            sub = np.where(sig[i0:i1 + 1], amp_n[i0:i1 + 1], -np.inf)
            iw, jf = np.unravel_index(np.argmax(sub), sub.shape)
            events.append({
                "mode_number":    n,
                "t_start_ms":     round(t0, 3),
                "t_end_ms":       round(t1, 3),
                "duration_ms":    round(t1 - t0, 3),
                "peak_freq_khz":  round(float(f[jf]), 4),
                "peak_amp_G":     round(float(amp_n[i0 + iw, jf]), 5),
                "mean_coherence": round(float(coh[i0:i1 + 1][sig[i0:i1 + 1]].mean()), 4),
                "coherence_thresh": round(float(thresh), 4),
            })
    return events


# ── Per-shot driver ──────────────────────────────────────────────────────────────

def process_shot(cfg, atlas, output_dir):
    """Run the pipeline for one resolved shot config; return list of CSV rows."""
    shot = cfg["shot"]
    array = cfg["array"]
    mode_label = "n" if array == "toroidal" else "m"
    print(f"[{shot}] fetching {array} array ...")

    signals, t_ms, angles, names = fetch_mirnov(
        shot, array=array, integrated=cfg["integrated"], atlas=atlas,
        t_min_ms=cfg["t_min_ms"], t_max_ms=cfg["t_max_ms"],
    )

    result = mode_spectrogram(
        signals, t_ms, angles,
        dt_window_ms=cfg["dt_window_ms"], overlap_frac=cfg["overlap_frac"],
        f_min_khz=cfg["f_min_khz"], f_max_khz=cfg["f_max_khz"],
        f_smooth_khz=cfg["f_smooth_khz"], n_range=tuple(cfg["n_range"]),
    )

    events = detect_modes(result, cfg)
    print(f"[{shot}] detected {len(events)} mode event(s) "
          f"(coh>={result['c95']:.3f}, {cfg['f_min_khz']}-{cfg['f_max_khz']} kHz)")

    if cfg["make_figure"]:
        try:
            fig = plot_modespec(result, shot=shot, mode_label=mode_label)
            fig_path = output_dir / f"{shot}_modespec.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[{shot}] figure -> {fig_path}")
        except Exception as exc:                       # never lose CSV rows over a plot
            print(f"[{shot}] WARNING: figure failed: {exc}", file=sys.stderr)

    rows = []
    for ev in events:
        rows.append({
            "array":       array,
            "mode_label":  mode_label,
            "f_min_khz":   cfg["f_min_khz"],
            "f_max_khz":   cfg["f_max_khz"],
            **ev,
        })

    csv_path = output_dir / f"{shot}_modes.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{shot}] wrote {len(rows)} mode(s) -> {csv_path}")
    return len(rows)


def run_config(config_path):
    """Process every shot in a YAML config. Returns the number of failed shots."""
    global_cfg, shot_cfgs = load_config(config_path)
    output_dir = Path(global_cfg["output_dir"])
    atlas = global_cfg["atlas"]
    output_dir.mkdir(parents=True, exist_ok=True)

    total = ok = 0
    for cfg in shot_cfgs:
        try:
            total += process_shot(cfg, atlas, output_dir)   # writes <shot>_modes.csv
            ok += 1
        except Exception as exc:                       # one bad shot shouldn't kill the run
            print(f"[{cfg['shot']}] ERROR: {exc}", file=sys.stderr)

    print(f"\nWrote {total} mode(s) across {ok}/{len(shot_cfgs)} shot(s) "
          f"-> {output_dir}/<shot>_modes.csv")
    return len(shot_cfgs) - ok


def main():
    ap = argparse.ArgumentParser(description="Generate a list of MHD modes from a config.")
    ap.add_argument("config", nargs="?", default="modes.yaml",
                    help="YAML config (default: modes.yaml)")
    args = ap.parse_args()
    return run_config(args.config)


if __name__ == "__main__":
    main()
