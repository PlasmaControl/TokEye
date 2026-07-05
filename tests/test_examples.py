from __future__ import annotations

import numpy as np

from tokeye.examples import make_example_signal, write_example_signal


def test_length_matches_duration_and_fs():
    duration_s, fs = 1.5, 1000.0
    sig = make_example_signal(duration_s=duration_s, fs=fs)
    assert sig.shape[0] == int(duration_s * fs)


def test_dtype_is_float():
    sig = make_example_signal(duration_s=0.5, fs=1000.0)
    assert np.issubdtype(sig.dtype, np.floating)


def test_all_finite():
    sig = make_example_signal(duration_s=0.5, fs=1000.0)
    assert np.all(np.isfinite(sig))


def test_same_seed_identical():
    sig_a = make_example_signal(duration_s=0.5, fs=1000.0, seed=42)
    sig_b = make_example_signal(duration_s=0.5, fs=1000.0, seed=42)
    np.testing.assert_array_equal(sig_a, sig_b)


def test_different_seed_differs():
    sig_a = make_example_signal(duration_s=0.5, fs=1000.0, seed=0)
    sig_b = make_example_signal(duration_s=0.5, fs=1000.0, seed=1)
    assert not np.array_equal(sig_a, sig_b)


def test_write_example_signal_creates_file(tmp_path):
    target_dir = tmp_path / "nested" / "output"
    out_path = write_example_signal(target_dir)
    assert out_path.exists()
    assert out_path.parent == target_dir
    loaded = np.load(out_path)
    np.testing.assert_array_equal(loaded, make_example_signal())


def test_write_example_signal_custom_filename(tmp_path):
    out_path = write_example_signal(tmp_path, filename="custom.npy")
    assert out_path.name == "custom.npy"
    assert out_path.exists()
