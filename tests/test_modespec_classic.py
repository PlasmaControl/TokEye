"""Tests for the vendored pymodespec (tokeye.modespec.classic).

Numeric behavior only — MDSplus fetch paths are untestable off the GA
cluster and are deliberately not covered.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

mpl.use("Agg")

from tokeye.modespec.classic import detect_modes, load_config, mode_spectrogram
from tokeye.modespec.classic.generate_modes import _contiguous_runs, _fill_gaps

CLASSIC_DIR = "src/tokeye/modespec/classic"


def test_mode_spectrogram_recovers_synthetic_n2_mode():
    # 6 toroidal probes (non-uniform angles, like a real Mirnov array —
    # uniform spacing would alias n and n±6), one rotating n=2 mode at 50 kHz
    n_true, f_khz = 2, 50.0
    fs_khz = 500.0
    t_ms = np.arange(0, 20.0, 1.0 / fs_khz)
    phi_deg = np.array([0.0, 40.0, 90.0, 140.0, 200.0, 250.0])
    phi_rad = np.deg2rad(phi_deg)

    rng = np.random.default_rng(0)
    signals = np.cos(
        2 * np.pi * f_khz * t_ms[None, :] + n_true * phi_rad[:, None]
    ) + 0.01 * rng.normal(size=(6, t_ms.size))

    # f_smooth=0: the complex frequency smoothing assumes finite-linewidth
    # modes; a zero-linewidth synthetic tone has alternating main-lobe bin
    # phases (Hann) that uniform smoothing cancels out.
    result = mode_spectrogram(signals, t_ms, phi_deg, f_smooth_khz=0.0)

    # at the injected frequency, mid-signal, the fit must pick n=+2 with
    # the largest matched-filter amplitude
    jf = int(np.argmin(np.abs(result["freq_khz"] - f_khz)))
    iw = result["t_win_ms"].size // 2
    assert int(result["n_dominant"][iw, jf]) == n_true
    # coherence = A_best / sum(A_n) over 11 tested n; steering-vector
    # crosstalk with 6 probes caps a perfect signal well below 1.0
    assert result["coherence"][iw, jf] > 0.2
    amp_true = result["mode_amp"][n_true][iw, jf]
    others = [
        result["mode_amp"][n][iw, jf]
        for n in range(*result["n_range"])
        if n != n_true
    ]
    assert amp_true > max(others)


def test_contiguous_runs_and_fill_gaps():
    mask = np.array([False, True, True, False, False, True, False])
    assert list(_contiguous_runs(mask)) == [(1, 2), (5, 5)]

    filled = _fill_gaps(mask, max_gap=2)
    assert list(_contiguous_runs(filled)) == [(1, 5)]

    # gaps at the edges are never bridged
    assert not filled[0]
    assert not filled[6]


def test_detect_modes_on_canned_result():
    n_win, n_freq = 20, 5
    t = np.arange(n_win, dtype=float)  # 1 ms steps
    zeros = np.zeros((n_win, n_freq))

    n_dominant = np.zeros((n_win, n_freq), dtype=int)
    coherence = zeros.copy()
    amp2 = zeros.copy()
    # an n=2 mode living in windows 5..12, frequency bin 3
    n_dominant[5:13, 3] = 2
    coherence[5:13, 3] = 0.9
    amp2[5:13, 3] = 1.5

    result = {
        "t_win_ms": t,
        "freq_khz": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        "n_dominant": n_dominant,
        "coherence": coherence,
        "mode_amp": {n: (amp2 if n == 2 else zeros) for n in range(-5, 6)},
        "n_range": (-5, 5),
        "c95": 0.1,
    }
    cfg = {
        "coherence_min": 0.5,
        "amp_min_G": 0.5,
        "min_duration_ms": 2.0,
        "merge_gap_ms": 2.0,
    }

    events = detect_modes(result, cfg)

    assert len(events) == 1
    event = events[0]
    assert event["mode_number"] == 2
    assert event["t_start_ms"] == 5.0
    assert event["t_end_ms"] == 12.0
    assert event["peak_freq_khz"] == 40.0
    assert event["peak_amp_G"] == 1.5


def test_load_config_roundtrip_on_vendored_example():
    global_cfg, shot_cfgs = load_config(f"{CLASSIC_DIR}/modes.yaml")

    assert "output_dir" in global_cfg
    assert "atlas" in global_cfg
    assert shot_cfgs, "vendored modes.yaml must list at least one shot"
    for cfg in shot_cfgs:
        assert "shot" in cfg
        assert "coherence_min" in cfg  # defaults merged in


def test_import_does_not_require_mdsplus():
    # the imports at the top of this file must succeed without MDSplus
    import sys

    assert "MDSplus" not in sys.modules
