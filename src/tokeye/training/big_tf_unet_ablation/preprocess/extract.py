"""Minimal local replacement for fusionaihub.preprocess.

Extracts raw fusion data from HDF5 files, aligns signals, windows them,
and saves as joblib dicts. Based on FusionAIHub/src/faith/preprocess/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from warnings import simplefilter

import joblib
import numpy as np
import pandas as pd
from scipy.signal import resample

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


def extract_running_time(
    shot_number: int,
    directory: Path,
    ip_threshold: float = 0.1,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[float, float]:
    """Determine plasma running time from plasma current threshold.

    Reads the ``ip/ipsip`` key from the shot HDF5 file and finds the time
    range where plasma current exceeds *ip_threshold*.

    Returns:
        ``(start_ms, end_ms)`` in milliseconds.
    """
    path = (directory / str(shot_number)).with_suffix(".h5")
    with pd.HDFStore(path, "r") as store:
        df = store["ip"]["ipsip"]

    df = df.loc[df > ip_threshold]
    if start_time is not None:
        df = df.loc[df.index >= start_time]
    if end_time is not None:
        df = df.loc[df.index <= end_time]

    return float(df.index[0]), float(df.index[-1])


def extract_signal(
    shot_number: int,
    directory: Path,
    signal: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> pd.DataFrame:
    """Read a single signal from the shot HDF5 file."""
    path = (directory / str(shot_number)).with_suffix(".h5")
    return pd.DataFrame(pd.read_hdf(path, key=signal))


def align_signal(
    df: pd.DataFrame,
    start_time: float,
    end_time: float,
    fs: float,
) -> pd.DataFrame:
    """Align signal to a common timebase and sampling frequency.

    Crops to ``[start_time, end_time]``, resamples to *fs* kHz, and adds
    zero-padding + ``_state`` columns.
    """
    fs_raw = len(df) / (df.index[-1] - df.index[0])
    df = df.loc[(df.index >= start_time) & (df.index <= end_time)]

    num = int(len(df) * fs / fs_raw)
    df = pd.DataFrame(
        {col: resample(df[col].values, num) for col in df.columns},
        index=np.linspace(df.index[0], df.index[-1], num),
    )

    # Padding for start/end alignment
    start_nan = (df.index[0] - start_time) * fs
    end_nan = (end_time - df.index[-1]) * fs
    start_pad = pd.DataFrame(
        0, index=pd.RangeIndex(start=int(start_nan)), columns=df.columns
    )
    end_pad = pd.DataFrame(
        0,
        index=pd.RangeIndex(
            start=int(len(df) + start_nan),
            stop=int(len(df) + start_nan + end_nan),
        ),
        columns=df.columns,
    )

    df_state = pd.DataFrame(True, index=df.index, columns=df.columns)
    start_pad_state = pd.DataFrame(False, index=start_pad.index, columns=df.columns)
    end_pad_state = pd.DataFrame(False, index=end_pad.index, columns=df.columns)

    df = pd.concat([start_pad, df, end_pad], ignore_index=True).astype(np.float32)
    df_state = pd.concat(
        [start_pad_state, df_state, end_pad_state], ignore_index=True
    )
    df_state.columns = [f"{col}_state" for col in df.columns]

    return pd.concat([df, df_state], axis=1).rename_axis("time")


def _identity_transform(x: np.ndarray) -> np.ndarray:
    return np.expand_dims(x.astype(np.float32), axis=1)


def _split_samples(
    df: pd.DataFrame,
    shot_number: int,
    window_ms: int | None = None,
    hop_ms: int | None = None,
    fs_khz: float | None = None,
) -> list[dict[str, pd.DataFrame]]:
    """Split signal data into overlapping time windows."""
    if window_ms is None or hop_ms is None or fs_khz is None:
        return [{f"{shot_number}_0": df}]

    num_samples = int(window_ms * fs_khz)
    hop_samples = int(hop_ms * fs_khz)
    samples = []
    idx = 0
    for start in range(0, len(df) - num_samples + 1, hop_samples):
        sample = df.iloc[start : start + num_samples]
        if len(sample) == num_samples:
            samples.append({f"{shot_number}_{idx}": sample})
            idx += 1
    return samples


def _remove_empty_samples(
    samples: list[dict[str, pd.DataFrame]],
) -> list[dict[str, pd.DataFrame]]:
    """Remove samples where all state columns are False."""
    out = []
    for sample in samples:
        filtered = {}
        for key, value in sample.items():
            state_cols = [c for c in value.columns if c.endswith("_state")]
            if np.any(value[state_cols].to_numpy()):
                filtered[key] = value.drop(columns=state_cols)
        if filtered:
            out.append(filtered)
    return out


def pipeline(
    shot_number: int,
    cfg: dict,
    out_dir: Path,
    override: bool = False,
) -> None:
    """Process a single shot through the data preparation pipeline.

    1. Determine plasma running time
    2. Extract and align all configured signals
    3. Split into time windows
    4. Save as joblib dicts (one per window)
    """
    temp_path = out_dir / f"{shot_number}_0.joblib"
    if temp_path.exists() and not override:
        logger.warning(f"Shot {shot_number} already processed. Skipping.")
        return

    # 1. Running time
    try:
        start_time, end_time = extract_running_time(
            shot_number=shot_number,
            directory=Path(cfg["raw_data_dir"]),
            ip_threshold=cfg.get("ip_threshold", 0.1),
            start_time=cfg.get("start_time"),
            end_time=cfg.get("end_time"),
        )
    except Exception as e:
        logger.error(f"Could not determine running time for shot {shot_number}: {e}")
        return

    # 2. Extract and align signals
    dfs = []
    missing_signals = []
    for signal_name, signal_cfg in cfg["signal"].items():
        abbr = signal_cfg["abbr"]
        try:
            df = extract_signal(
                shot_number=shot_number,
                directory=Path(cfg["raw_data_dir"]),
                signal=signal_name,
                start_time=start_time,
                end_time=end_time,
            )
            df.columns = [f"{abbr}_{i}" for i in range(len(df.columns))]
            df = align_signal(df, start_time, end_time, cfg["fs_khz"])
            dfs.append(df)
        except Exception:
            for ch in range(int(signal_cfg.get("expected_channels", 0))):
                missing_signals.append((abbr, ch))

    try:
        df = pd.concat(dfs, axis=1, join="inner")
    except Exception as e:
        logger.error(f"Could not concatenate for shot {shot_number}: {e}")
        return

    # Add missing signals as NaN
    for abbr, ch in missing_signals:
        df[f"{abbr}_{ch}"] = np.nan
        df[f"{abbr}_{ch}_state"] = False

    df["time_ms"] = np.linspace(start_time, end_time, len(df))

    # 3. Split into samples
    samples = _split_samples(
        df, shot_number, cfg.get("window_ms"), cfg.get("hop_ms"), cfg.get("fs_khz")
    )
    samples = _remove_empty_samples(samples)

    # 4. Save — no STFT mode (raw time-domain signals as dicts)
    if not cfg.get("do_stft", False):
        for sample in samples:
            for key, value in sample.items():
                out = {}
                for signal_cfg in cfg["signal"].values():
                    abbr = signal_cfg["abbr"]
                    cols = [c for c in value.columns if c.startswith(f"{abbr}_")]
                    out[abbr] = _identity_transform(value[cols].to_numpy().T)
                out["time_ms"] = _identity_transform(
                    np.array([value["time_ms"].to_numpy()])
                )
                joblib.dump(out, out_dir / f"{key}.joblib")
        return

    # STFT mode
    import torch

    n_fft = cfg["stft"]["n_fft"]
    hop_length = cfg["stft"]["hop_length"]

    def _stft(x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x.astype(np.float32))
        y = torch.stft(
            t, n_fft=n_fft, hop_length=hop_length,
            window=torch.hann_window(n_fft), return_complex=True,
        )
        return torch.abs(y).numpy()

    # Determine reference shape from first sample
    first_value = list(samples[0].values())[0]
    first_arr = np.array([first_value.iloc[:, 0].values], dtype=np.float32)
    ref_shape = _stft(first_arr).shape

    for sample in samples:
        for key, value in sample.items():
            out = {}
            for signal_cfg in cfg["signal"].values():
                abbr = signal_cfg["abbr"]
                cols = [c for c in value.columns if c.startswith(f"{abbr}_")]
                arr = value[cols].to_numpy().T.astype(np.float32)
                if signal_cfg.get("make_stft", False):
                    out[abbr] = _stft(arr)
                else:
                    # Resample to match STFT time dimension
                    target_len = ref_shape[-1]
                    resampled = [resample(row, target_len) for row in arr]
                    out[abbr] = np.expand_dims(np.array(resampled), axis=1)

            # Time axis
            time_arr = value["time_ms"].to_numpy().astype(np.float32)
            target_len = ref_shape[-1]
            out["time_ms"] = np.expand_dims(
                [np.interp(
                    np.linspace(0, 1, target_len),
                    np.linspace(0, 1, len(time_arr)),
                    time_arr,
                )],
                axis=1,
            )
            joblib.dump(out, out_dir / f"{key}.joblib")


def index_dataset(out_dir: Path | str) -> None:
    """Create ``index.csv`` listing all joblib files in *out_dir*."""
    out_dir = Path(out_dir)
    files = list(out_dir.glob("*.joblib"))
    pd.DataFrame({"files": [str(f) for f in files]}).to_csv(
        out_dir / "index.csv", index=False
    )
    logger.info(f"Indexed {len(files)} files.")
