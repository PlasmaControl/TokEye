from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys

import numpy as np
import pytest

from tokeye.export import (
    SCHEMA_ANALYSIS,
    SCHEMA_MODESPEC,
    analysis_bundle,
    default_stem,
    modes_csv_text,
    modespec_bundle,
    save_npz,
    stft_axes,
)


def _fake_modespec_result():
    rng = np.random.default_rng(0)
    return {
        "t_win_ms": np.linspace(1000, 1020, 30),
        "freq_khz": np.linspace(5, 150, 25),
        "n_dominant": rng.integers(-3, 4, size=(30, 25)),
        "coherence": rng.random((30, 25)),
        "n_range": (-3, 3),
        "c95": 0.3,
    }


class TestAnalysisBundleRoundTrip:
    def test_full_bundle_round_trip(self, tmp_path):
        rng = np.random.default_rng(0)
        spectrogram = rng.random((10, 20)).astype(np.float32)
        mask = rng.random((2, 10, 20)).astype(np.float32)
        stft_meta = {
            "fs": 2.0e6,
            "t0_ms": 1000.0,
            "n_fft": 512,
            "hop": 256,
            "clip_dc": True,
        }
        raw_t_ms = np.arange(100, dtype=np.float64)
        raw_x = rng.random(100)
        params = {"n_fft": 512, "hop": 256, "model": "big_tf_unet"}

        bundle = analysis_bundle(
            spectrogram=spectrogram,
            mask=mask,
            stft_meta=stft_meta,
            raw=(raw_t_ms, raw_x),
            params=params,
            source="analyze",
        )
        path = save_npz(tmp_path / "result.npz", bundle)

        assert path == tmp_path / "result.npz"
        assert path.exists()

        loaded = np.load(path, allow_pickle=False)
        expected_keys = {
            "schema",
            "created_utc",
            "source",
            "params_json",
            "spectrogram",
            "mask",
            "time_ms",
            "freq_khz",
            "raw_t_ms",
            "raw_x",
        }
        assert set(loaded.files) == expected_keys

        assert str(loaded["schema"]) == SCHEMA_ANALYSIS
        assert str(loaded["source"]) == "analyze"
        assert json.loads(str(loaded["params_json"])) == params

        assert loaded["spectrogram"].dtype == np.float32
        assert loaded["spectrogram"].shape == (10, 20)
        assert loaded["mask"].dtype == np.float32
        assert loaded["mask"].shape == (2, 10, 20)
        assert loaded["time_ms"].dtype == np.float64
        assert loaded["time_ms"].shape == (20,)
        assert loaded["freq_khz"].dtype == np.float64
        assert loaded["freq_khz"].shape == (10,)
        assert loaded["raw_t_ms"].dtype == np.float32
        assert loaded["raw_x"].dtype == np.float32

        # ISO-8601 UTC timestamp round-trips.
        from datetime import datetime

        datetime.fromisoformat(str(loaded["created_utc"]))

    def test_minimal_bundle_omits_optional_keys(self, tmp_path):
        spectrogram = np.zeros((4, 6), dtype=np.float32)

        bundle = analysis_bundle(spectrogram=spectrogram, source="cli-batch")
        path = save_npz(tmp_path / "min.npz", bundle)
        loaded = np.load(path, allow_pickle=False)

        assert set(loaded.files) == {
            "schema",
            "created_utc",
            "source",
            "params_json",
            "spectrogram",
        }
        assert json.loads(str(loaded["params_json"])) == {}

    def test_save_npz_creates_parent_dirs(self, tmp_path):
        bundle = analysis_bundle(spectrogram=np.zeros((2, 2), dtype=np.float32))
        nested = tmp_path / "a" / "b" / "c.npz"

        path = save_npz(nested, bundle)

        assert path == nested
        assert path.exists()

    def test_save_npz_appends_npz_suffix(self, tmp_path):
        bundle = analysis_bundle(spectrogram=np.zeros((2, 2), dtype=np.float32))

        path = save_npz(tmp_path / "no_suffix", bundle)

        assert path.name == "no_suffix.npz"
        assert path.exists()


class TestParamsJsonNumpyTolerance:
    _PARAMS = {
        "n_fft": np.int64(512),
        "thresh": np.float32(0.25),
        "clip_dc": np.bool_(True),
        "n_range": np.array([-3, 3]),
        "plain": "str",
    }
    _EXPECTED = {
        "n_fft": 512,
        "thresh": 0.25,
        "clip_dc": True,
        "n_range": [-3, 3],
        "plain": "str",
    }

    def test_analysis_bundle_accepts_numpy_typed_params(self):
        bundle = analysis_bundle(
            spectrogram=np.zeros((2, 2), dtype=np.float32),
            params=self._PARAMS,
        )
        assert json.loads(str(bundle["params_json"])) == self._EXPECTED

    def test_modespec_bundle_accepts_numpy_typed_params(self):
        bundle = modespec_bundle(result=_fake_modespec_result(), params=self._PARAMS)
        assert json.loads(str(bundle["params_json"])) == self._EXPECTED

    def test_unserializable_param_still_raises(self):
        with pytest.raises(TypeError):
            analysis_bundle(
                spectrogram=np.zeros((2, 2), dtype=np.float32),
                params={"bad": object()},
            )


class TestStftAxes:
    def test_hand_computed_axes(self):
        meta = {
            "fs": 2.0e6,
            "t0_ms": 1000.0,
            "n_fft": 512,
            "hop": 256,
            "clip_dc": True,
        }
        time_ms, freq_khz = stft_axes(n_rows=50, n_cols=100, stft_meta=meta)

        dt = 0.128
        df = 2.0e6 / 512 / 1e3  # 3.90625 kHz

        assert time_ms.dtype == np.float64
        assert freq_khz.dtype == np.float64
        assert time_ms.shape == (100,)
        assert freq_khz.shape == (50,)
        assert time_ms[0] == pytest.approx(1000.0)
        np.testing.assert_allclose(np.diff(time_ms), dt)
        assert freq_khz[0] == pytest.approx(df)
        np.testing.assert_allclose(np.diff(freq_khz), df)

    def test_clip_dc_false_starts_freq_axis_at_zero(self):
        meta = {"fs": 2.0e6, "n_fft": 512, "hop": 256, "clip_dc": False}
        _, freq_khz = stft_axes(n_rows=50, n_cols=100, stft_meta=meta)
        assert freq_khz[0] == 0.0

    def test_none_meta_returns_none_none(self):
        assert stft_axes(10, 10, None) == (None, None)

    def test_empty_meta_returns_none_none(self):
        assert stft_axes(10, 10, {}) == (None, None)

    def test_zero_fs_returns_none_none(self):
        assert stft_axes(10, 10, {"fs": 0}) == (None, None)

    def test_missing_fs_returns_none_none(self):
        assert stft_axes(10, 10, {"n_fft": 512, "hop": 256}) == (None, None)


class TestModespecBundle:
    def test_required_keys_and_dtypes(self, tmp_path):
        result = _fake_modespec_result()

        bundle = modespec_bundle(result=result, source="diiid-modespec")
        path = save_npz(tmp_path / "modespec.npz", bundle)
        loaded = np.load(path, allow_pickle=False)

        expected_keys = {
            "schema",
            "created_utc",
            "source",
            "params_json",
            "n_dominant",
            "coherence",
            "t_win_ms",
            "freq_khz",
            "n_range",
            "c95",
        }
        assert set(loaded.files) == expected_keys
        assert str(loaded["schema"]) == SCHEMA_MODESPEC

        assert loaded["n_dominant"].dtype == np.float32
        assert loaded["n_dominant"].shape == (30, 25)
        assert loaded["coherence"].dtype == np.float32
        assert loaded["coherence"].shape == (30, 25)
        assert loaded["t_win_ms"].dtype == np.float64
        assert loaded["t_win_ms"].shape == (30,)
        assert loaded["freq_khz"].dtype == np.float64
        assert loaded["freq_khz"].shape == (25,)
        assert loaded["n_range"].dtype == np.int64
        assert loaded["n_range"].shape == (2,)
        np.testing.assert_array_equal(loaded["n_range"], [-3, 3])
        assert loaded["c95"].dtype == np.float64
        assert float(loaded["c95"]) == pytest.approx(0.3)

    def test_nd_produces_n_gated(self, tmp_path):
        result = _fake_modespec_result()
        nd = result["n_dominant"].astype(np.float32).copy()
        nd[0, 0] = np.nan

        bundle = modespec_bundle(result=result, nd=nd, source="gui-modespec")
        path = save_npz(tmp_path / "gated.npz", bundle)
        loaded = np.load(path, allow_pickle=False)

        assert "n_gated" in loaded.files
        assert loaded["n_gated"].dtype == np.float32
        assert np.isnan(loaded["n_gated"][0, 0])

    def test_tok_mask_and_coh_thresh_optional(self, tmp_path):
        result = _fake_modespec_result()
        tok_mask = np.ones((30, 25), dtype=np.float32)

        bundle = modespec_bundle(
            result=result, tok_mask=tok_mask, coh_thresh=0.42, source="diiid-modespec"
        )
        path = save_npz(tmp_path / "tokmask.npz", bundle)
        loaded = np.load(path, allow_pickle=False)

        assert "tok_mask" in loaded.files
        assert loaded["tok_mask"].dtype == np.float32
        assert "coh_thresh" in loaded.files
        assert float(loaded["coh_thresh"]) == pytest.approx(0.42)

    def test_optional_keys_absent_by_default(self, tmp_path):
        result = _fake_modespec_result()
        bundle = modespec_bundle(result=result, source="diiid-modespec")
        path = save_npz(tmp_path / "plain.npz", bundle)
        loaded = np.load(path, allow_pickle=False)

        assert "n_gated" not in loaded.files
        assert "tok_mask" not in loaded.files
        assert "coh_thresh" not in loaded.files


class TestModesCsvText:
    def _fake_result_with_mode_amp(self):
        # detect_modes() (vendored, called inside modes_csv_text) requires
        # result["mode_amp"]: dict {n: (n_win, n_freq)} — not part of the
        # npz schema, but required for the real function to run at all. See
        # tests/test_export.py module docstring / T3 report for the
        # rationale (the brief's fake result omits it).
        result = _fake_modespec_result()
        n_lo, n_hi = result["n_range"]
        n_win, n_freq = result["n_dominant"].shape
        result["mode_amp"] = {
            n: np.ones((n_win, n_freq)) for n in range(int(n_lo), int(n_hi) + 1)
        }
        return result

    def test_header_matches_csv_columns(self):
        from tokeye.modespec.classic.generate_modes import CSV_COLUMNS

        result = self._fake_result_with_mode_amp()
        text = modes_csv_text(result, array="toroidal", f_min=5.0, f_max=50.0)

        header = text.split("\r\n", 1)[0]
        assert header == ",".join(CSV_COLUMNS)

    def test_rows_populated(self):
        from tokeye.modespec.classic.generate_modes import CSV_COLUMNS

        result = self._fake_result_with_mode_amp()
        text = modes_csv_text(result, array="toroidal", f_min=5.0, f_max=50.0)

        reader = csv.DictReader(io.StringIO(text))
        assert reader.fieldnames == CSV_COLUMNS
        rows = list(reader)
        assert len(rows) > 0
        for row in rows:
            assert row["array"] == "toroidal"
            assert row["mode_label"] == "n"
            assert row["f_min_khz"] == "5.0"
            assert row["f_max_khz"] == "50.0"

    def test_poloidal_array_uses_m_label(self):
        # Matches the vendored generate_modes driver:
        # mode_label = "n" if array == "toroidal" else "m"
        result = self._fake_result_with_mode_amp()
        text = modes_csv_text(result, array="poloidal", f_min=5.0, f_max=50.0)

        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) > 0
        for row in rows:
            assert row["array"] == "poloidal"
            assert row["mode_label"] == "m"

    def test_runs_without_error_on_fake_result(self):
        result = self._fake_result_with_mode_amp()
        text = modes_csv_text(result, array="poloidal", f_min=1.0, f_max=2.0)
        assert isinstance(text, str)
        assert text  # at least the header


class TestDefaultStem:
    _PATTERN = re.compile(r"^tokeye_[\w-]+(?:_[\w.-]+)*_\d{8}-\d{6}$")

    def test_kind_only(self):
        stem = default_stem("analysis")
        assert self._PATTERN.match(stem)
        assert stem.startswith("tokeye_analysis_")

    def test_kind_with_parts(self):
        stem = default_stem("modespec", "shot12345", "toroidal")
        assert self._PATTERN.match(stem)
        assert stem.startswith("tokeye_modespec_shot12345_toroidal_")

    def test_falsy_parts_skipped(self):
        stem = default_stem("kind", 0, "", None, "keep")
        assert "_keep_" in stem
        assert "__" not in stem


class TestImportHygiene:
    def test_export_import_stays_numpy_only(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import tokeye.export, sys; "
                "assert 'torch' not in sys.modules; "
                "assert 'gradio' not in sys.modules; "
                "assert 'matplotlib' not in sys.modules; "
                "assert not any('PySide' in m or 'PyQt' in m for m in sys.modules); "
                "assert 'plotly' not in sys.modules; "
                "print('ok')",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout
