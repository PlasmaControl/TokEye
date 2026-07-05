from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn

from tokeye import batch


@pytest.fixture
def stub_model(monkeypatch):
    """A cheap Conv2d(1, 2, 1) stands in for the real (heavy) TokEye model."""
    model = nn.Conv2d(1, 2, kernel_size=1)
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)
    return model


@pytest.fixture
def signal_npy(tmp_path):
    path = tmp_path / "signal.npy"
    sig = np.random.default_rng(0).normal(size=8192).astype(np.float32)
    np.save(path, sig)
    return path


@pytest.fixture
def spectrogram_npy(tmp_path):
    path = tmp_path / "spectrogram.npy"
    spec = np.random.default_rng(1).normal(size=(64, 32)).astype(np.float32)
    np.save(path, spec)
    return path


class TestCollectInputs:
    def test_single_file(self, signal_npy):
        assert batch.collect_inputs([str(signal_npy)]) == [signal_npy]

    def test_directory_finds_only_npy_sorted(self, tmp_path):
        (tmp_path / "b.npy").touch()
        (tmp_path / "a.npy").touch()
        (tmp_path / "ignore.txt").touch()

        result = batch.collect_inputs([str(tmp_path)])

        assert result == [tmp_path / "a.npy", tmp_path / "b.npy"]

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "x1.npy").touch()
        (tmp_path / "x2.npy").touch()

        result = batch.collect_inputs([str(tmp_path / "x*.npy")])

        assert result == [tmp_path / "x1.npy", tmp_path / "x2.npy"]

    def test_mixed_inputs_dedup_preserving_order(self, tmp_path):
        (tmp_path / "a.npy").touch()
        (tmp_path / "b.npy").touch()

        # The directory glob would also match a.npy; it should appear once,
        # in its first-seen position.
        result = batch.collect_inputs(
            [str(tmp_path / "a.npy"), str(tmp_path)]
        )

        assert result == [tmp_path / "a.npy", tmp_path / "b.npy"]

    def test_empty_result_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="No input files found"):
            batch.collect_inputs([str(tmp_path / "does_not_exist_*.npy")])


class TestLoadInput:
    def test_1d_signal_becomes_spectrogram(self, signal_npy):
        spec = batch.load_input(signal_npy, {"n_fft": 256, "hop": 64})
        assert spec.ndim == 2

    def test_2d_spectrogram_passes_through(self, spectrogram_npy):
        spec = batch.load_input(spectrogram_npy, {})
        assert spec.shape == (64, 32)
        assert np.issubdtype(spec.dtype, np.floating)

    def test_3d_array_raises_value_error(self, tmp_path):
        path = tmp_path / "bad.npy"
        np.save(path, np.zeros((2, 3, 4)))

        with pytest.raises(ValueError, match="ndim=3"):
            batch.load_input(path, {})

    def test_log_applies_log1p_to_2d_input(self, tmp_path):
        arr = np.random.default_rng(0).random((16, 16))
        path = tmp_path / "linear.npy"
        np.save(path, arr)

        spec = batch.load_input(path, {}, log=True)

        np.testing.assert_allclose(spec, np.log1p(arr))

    def test_log_rejects_negative_2d_input(self, tmp_path):
        path = tmp_path / "db_scaled.npy"
        np.save(path, np.full((8, 8), -30.0))

        with pytest.raises(ValueError, match="negative"):
            batch.load_input(path, {}, log=True)


class TestRunBatch:
    def test_on_1d_signal_writes_mask_and_preview(
        self, stub_model, signal_npy, tmp_path
    ):
        out_dir = tmp_path / "out"
        failures = batch.run_batch(
            [str(signal_npy)],
            out_dir=out_dir,
            stft_kwargs={"n_fft": 256, "hop": 64},
        )

        assert failures == 0
        mask_path = out_dir / "signal_mask.npy"
        preview_path = out_dir / "signal_preview.png"
        assert mask_path.exists()
        assert preview_path.exists()
        assert preview_path.stat().st_size > 0

        mask = np.load(mask_path)
        assert mask.ndim == 3
        assert mask.shape[0] == 2
        assert mask.dtype == np.float32
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)

    def test_on_2d_spectrogram_writes_mask_and_preview(
        self, stub_model, spectrogram_npy, tmp_path
    ):
        out_dir = tmp_path / "out"
        failures = batch.run_batch([str(spectrogram_npy)], out_dir=out_dir)

        assert failures == 0
        mask = np.load(out_dir / "spectrogram_mask.npy")
        assert mask.shape == (2, 64, 32)
        assert mask.dtype == np.float32
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
        assert (out_dir / "spectrogram_preview.png").exists()

    def test_save_png_false_skips_preview(self, stub_model, spectrogram_npy, tmp_path):
        out_dir = tmp_path / "out"
        failures = batch.run_batch(
            [str(spectrogram_npy)], out_dir=out_dir, save_png=False
        )

        assert failures == 0
        assert (out_dir / "spectrogram_mask.npy").exists()
        assert not (out_dir / "spectrogram_preview.png").exists()

    def test_one_bad_file_among_good_ones_counts_as_failure(
        self, stub_model, spectrogram_npy, tmp_path
    ):
        bad_path = tmp_path / "bad.npy"
        np.save(bad_path, np.zeros((2, 3, 4)))

        out_dir = tmp_path / "out"
        failures = batch.run_batch(
            [str(spectrogram_npy), str(bad_path)], out_dir=out_dir
        )

        assert failures == 1
        assert (out_dir / "spectrogram_mask.npy").exists()
        assert not (out_dir / "bad_mask.npy").exists()

    def test_loads_model_once_via_hub(self, tmp_path, monkeypatch):
        """Multiple input files -> exactly one load_model call (per run, not
        per file)."""
        model = nn.Conv2d(1, 2, kernel_size=1)
        calls = []

        def fake_load_model(source, device):
            calls.append((source, device))
            return model

        monkeypatch.setattr("tokeye.hub.load_model", fake_load_model)

        rng = np.random.default_rng(2)
        for name in ("first.npy", "second.npy"):
            np.save(tmp_path / name, rng.normal(size=(64, 32)).astype(np.float32))

        out_dir = tmp_path / "out"
        failures = batch.run_batch(
            [str(tmp_path / "first.npy"), str(tmp_path / "second.npy")],
            out_dir=out_dir,
            device="cpu",
        )

        assert failures == 0
        assert (out_dir / "first_mask.npy").exists()
        assert (out_dir / "second_mask.npy").exists()
        assert calls == [(batch.hub.DEFAULT_MODEL, "cpu")]
