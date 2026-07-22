"""Shared "save results" backbone: npz bundle schemas + CSV export.

Every TokEye surface that offers a "Save results" feature (Gradio tabs, the
native Qt GUI, CLI batch) builds its output through this module so every
saved ``.npz`` follows one consistent schema. **numpy-only** at import time:
the single vendored import (``tokeye.modespec.classic.generate_modes``)
happens lazily inside :func:`modes_csv_text`, so importing this module never
pulls in torch, gradio, Qt, matplotlib, or plotly.

Two schemas:

- ``tokeye-analysis/v1`` — spectrogram surfaces (:func:`analysis_bundle`).
- ``tokeye-modespec/v1`` — mode-spectrogram surfaces (:func:`modespec_bundle`).

Both share four keys: ``schema``, ``created_utc``, ``source``,
``params_json``. Bundles are flat ``str -> np.ndarray``-ish dicts saved with
:func:`save_npz`; optional keys are simply absent when their inputs are
absent — bundles never store ``None``.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

SCHEMA_ANALYSIS = "tokeye-analysis/v1"
SCHEMA_MODESPEC = "tokeye-modespec/v1"

# A bundle value is either an array, a plain str (schema/source/params_json/
# created_utc), or a numpy scalar (e.g. the c95/coh_thresh floats).
NpzBundle = dict[str, np.ndarray | str | np.generic]

_DEFAULT_N_FFT = 1024
_DEFAULT_HOP = 256
_DEFAULT_T0_MS = 0.0
_DEFAULT_CLIP_DC = True


def _now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_default(obj: object) -> object:
    """``json.dumps`` fallback: unwrap numpy scalars/arrays to plain Python."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _params_json(params: dict | None) -> str:
    """Serialize a params dict to JSON, tolerating numpy-typed values."""
    return json.dumps(params or {}, default=_json_default)


def stft_axes(
    n_rows: int, n_cols: int, stft_meta: dict | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Pixel-centre axes (time_ms, freq_khz) for an STFT image.

    ``stft_meta`` keys: ``fs`` [Hz], ``t0_ms``, ``n_fft``, ``hop``,
    ``clip_dc`` (bool). Math (must match the GUI/plotly renderers exactly)::

        offset = 1 if clip_dc else 0
        df = fs / n_fft / 1e3          # kHz per row
        dt = hop / fs * 1e3            # ms per column
        time_ms[i] = t0_ms + i * dt        (column centres, i in [0, n_cols))
        freq_khz[r] = (r + offset) * df    (row centres, r in [0, n_rows))

    Returns ``(None, None)`` when ``stft_meta`` is falsy or ``fs <= 0``.
    Defaults when keys are absent: ``n_fft=1024``, ``hop=256``,
    ``t0_ms=0.0``, ``clip_dc=True``.
    """
    if not stft_meta:
        return None, None

    fs = stft_meta.get("fs", 0.0)
    if not fs or fs <= 0:
        return None, None

    n_fft = stft_meta.get("n_fft", _DEFAULT_N_FFT)
    hop = stft_meta.get("hop", _DEFAULT_HOP)
    t0_ms = stft_meta.get("t0_ms", _DEFAULT_T0_MS)
    clip_dc = stft_meta.get("clip_dc", _DEFAULT_CLIP_DC)

    offset = 1 if clip_dc else 0
    df = fs / n_fft / 1e3
    dt = hop / fs * 1e3

    time_ms = t0_ms + np.arange(n_cols, dtype=np.float64) * dt
    freq_khz = (np.arange(n_rows, dtype=np.float64) + offset) * df
    return time_ms, freq_khz


def analysis_bundle(
    *,
    spectrogram: np.ndarray,
    mask: np.ndarray | None = None,
    stft_meta: dict | None = None,
    raw: tuple[np.ndarray, np.ndarray] | None = None,
    params: dict | None = None,
    source: str = "",
) -> NpzBundle:
    """Build a ``tokeye-analysis/v1`` npz bundle (key -> array/str).

    ``raw``, when given, is a ``(t_ms, x)`` tuple of the raw signal trace,
    stored as ``raw_t_ms``/``raw_x``. Axes come from :func:`stft_axes`
    applied to ``spectrogram``'s shape; optional keys are simply omitted
    when their inputs are unavailable (never stored as ``None``).
    """
    spectrogram = np.asarray(spectrogram, dtype=np.float32)

    bundle: NpzBundle = {
        "schema": SCHEMA_ANALYSIS,
        "created_utc": _now_utc_iso(),
        "source": source,
        "params_json": _params_json(params),
        "spectrogram": spectrogram,
    }

    if mask is not None:
        bundle["mask"] = np.asarray(mask, dtype=np.float32)

    n_rows, n_cols = spectrogram.shape
    time_ms, freq_khz = stft_axes(n_rows, n_cols, stft_meta)
    if time_ms is not None:
        bundle["time_ms"] = time_ms
    if freq_khz is not None:
        bundle["freq_khz"] = freq_khz

    if raw is not None:
        raw_t_ms, raw_x = raw
        bundle["raw_t_ms"] = np.asarray(raw_t_ms, dtype=np.float32)
        bundle["raw_x"] = np.asarray(raw_x, dtype=np.float32)

    return bundle


def modespec_bundle(
    *,
    result: dict,
    nd: np.ndarray | None = None,
    tok_mask: np.ndarray | None = None,
    coh_thresh: float | None = None,
    params: dict | None = None,
    source: str = "",
) -> NpzBundle:
    """Build a ``tokeye-modespec/v1`` npz bundle from a mode-spectrogram
    analysis ``result`` (as returned by ``mode_spectrogram()``): keys
    ``t_win_ms``, ``freq_khz``, ``n_dominant``, ``coherence``, ``n_range``
    (2-seq), ``c95``. ``nd``, when given, is the TokEye-gated dominant-mode
    array and is stored as ``n_gated``.
    """
    bundle: NpzBundle = {
        "schema": SCHEMA_MODESPEC,
        "created_utc": _now_utc_iso(),
        "source": source,
        "params_json": _params_json(params),
        "n_dominant": np.asarray(result["n_dominant"], dtype=np.float32),
        "coherence": np.asarray(result["coherence"], dtype=np.float32),
        "t_win_ms": np.asarray(result["t_win_ms"], dtype=np.float64),
        "freq_khz": np.asarray(result["freq_khz"], dtype=np.float64),
        "n_range": np.asarray(result["n_range"], dtype=np.int64),
        "c95": np.float64(result["c95"]),
    }

    if nd is not None:
        bundle["n_gated"] = np.asarray(nd, dtype=np.float32)
    if tok_mask is not None:
        bundle["tok_mask"] = np.asarray(tok_mask, dtype=np.float32)
    if coh_thresh is not None:
        bundle["coh_thresh"] = np.float64(coh_thresh)

    return bundle


def save_npz(path: str | Path, bundle: NpzBundle) -> Path:
    """Save a bundle with ``np.savez_compressed``; mkdir parents; return the
    actual path written (numpy appends ``.npz`` if not already present)."""
    path = Path(path)
    if not path.name.endswith(".npz"):
        path = path.with_name(path.name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle)
    return path


def modes_csv_text(
    result: dict, *, array: str = "toroidal", f_min: float, f_max: float
) -> str:
    """Render detected mode events as CSV text: byte-compatible with the
    diiid-batch ``<shot>_modes.csv`` for toroidal arrays (identical output),
    and matching the vendored ``generate_modes`` driver's n/m ``mode_label``
    convention for other arrays. ``result`` is the live mode-spectrogram
    analysis result (must include ``mode_amp``, as ``detect_modes``
    requires)."""
    from tokeye.modespec.classic.generate_modes import (
        CSV_COLUMNS,
        PARAM_DEFAULTS,
        detect_modes,
    )

    cfg = {**PARAM_DEFAULTS, "n_range": list(result["n_range"])}
    rows = detect_modes(result, cfg)
    mode_label = "n" if array == "toroidal" else "m"

    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for ev in rows:
        writer.writerow(
            {
                "array": array,
                "mode_label": mode_label,
                "f_min_khz": f_min,
                "f_max_khz": f_max,
                **ev,
            }
        )
    return buf.getvalue()


def default_stem(kind: str, *parts: object) -> str:
    """Build a timestamped default filename stem:
    ``tokeye_<kind>[_<part>...]_<YYYYmmdd-HHMMSS>``. ``parts`` are
    stringified; falsy ones (``None``, ``""``, ``0``, ...) are skipped."""
    bits = [str(part) for part in parts if part]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return "_".join(["tokeye", kind, *bits, stamp])
