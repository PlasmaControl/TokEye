"""Tests for the vendored eigspec (tokeye.eigspec).

The import-smoke test is the regression test for the vendoring fixes:
upstream had `lambda`-attribute SyntaxErrors and missing typing imports,
so `import eigspec` failed everywhere.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from tokeye.eigspec.utils.subspace_identification import covariance_driven_ssi


def test_package_imports_without_sklearn_in_subprocess():
    code = (
        "import tokeye.eigspec, tokeye.eigspec.utils.data_extraction, "
        "tokeye.eigspec.utils.subspace_identification, sys; "
        "assert 'sklearn' not in sys.modules; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_vis_modules_import():
    # spectral_plots carried five of the seven `.lambda` syntax errors
    import matplotlib as mpl

    mpl.use("Agg")
    from tokeye.eigspec.vis import spectral_plots  # noqa: F401


def test_covariance_driven_ssi_recovers_pole_frequency():
    # 2-channel lightly damped 5 Hz oscillation sampled at 100 Hz
    fs, f0, zeta = 100.0, 5.0, 0.02
    t = np.arange(0, 20.0, 1.0 / fs)
    rng = np.random.default_rng(0)
    envelope = np.exp(-zeta * 2 * np.pi * f0 * t)
    data = np.column_stack(
        [
            envelope * np.sin(2 * np.pi * f0 * t),
            envelope * np.cos(2 * np.pi * f0 * t),
        ]
    ) + 0.01 * rng.normal(size=(t.size, 2))

    result = covariance_driven_ssi(data, [10, 10, 2])

    poles = np.linalg.eigvals(result.state_matrix)
    freqs_hz = np.abs(np.angle(poles)) * fs / (2 * np.pi)
    dampings = -np.log(np.abs(poles)) / np.abs(np.angle(poles))
    assert np.any(np.abs(freqs_hz - f0) < 0.2), freqs_hz
    assert np.any(np.abs(dampings - zeta) < 0.01), dampings
