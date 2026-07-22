"""Tests for the Utilities tab (src/tokeye/app/tabs/utilities.py).

Covers three defects fixed together:
1. ``convert_audio_to_npy`` dropped the sample rate on the floor (the status
   text claimed it was "saved separately in metadata" - it wasn't saved
   anywhere) instead of encoding it in the filename like the batch path does.
2. ``process_recording_btn`` was wired to TWO separate ``.click()`` handlers
   (process then update-playback) with no ordering guarantee between them -
   merged into one handler that returns all four outputs together.
3. ``batch_convert_audio_files`` had no progress feedback for long batches.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np

from tokeye.app.tabs import utilities

# ============================================================================
# convert_audio_to_npy - sample rate encoded in filename, honest status text
# ============================================================================


def test_convert_audio_to_npy_filename_encodes_sample_rate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    waveform = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    sample_rate = 44100

    filepath, status = utilities.convert_audio_to_npy(
        waveform, sample_rate, normalize=True
    )

    assert filepath is not None
    assert f"_sr{sample_rate}" in Path(filepath).name
    assert Path(filepath).exists()


def test_convert_audio_to_npy_status_text_does_not_claim_metadata(
    tmp_path, monkeypatch
):
    """Pre-fix, the status text claimed the rate was 'saved separately in
    metadata' when nothing was ever saved - the rate only ever lived in the
    filename convention. The honest fix says so and drops the bogus
    'Consider saving as' suggestion (which just repeated the file that was
    already written).
    """
    monkeypatch.chdir(tmp_path)
    waveform = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    sample_rate = 16000

    filepath, status = utilities.convert_audio_to_npy(
        waveform, sample_rate, normalize=True
    )

    assert "metadata" not in status.lower()
    assert "consider saving as" not in status.lower()
    assert str(sample_rate) in status
    assert Path(filepath).name in status or Path(filepath).stem in status


def test_convert_audio_to_npy_no_waveform_returns_none():
    filepath, status = utilities.convert_audio_to_npy(None, 44100)

    assert filepath is None
    assert "no audio" in status.lower()


# ============================================================================
# process_and_preview_recording - merged single handler (no click race)
# ============================================================================


def test_process_and_preview_recording_returns_full_output_tuple():
    sample_rate = 8000
    raw = (np.sin(np.linspace(0, 2 * np.pi, sample_rate)) * 20000).astype(np.int16)
    audio_data = (sample_rate, raw)

    waveform, sr_out, info, playback = utilities.process_and_preview_recording(
        audio_data
    )

    assert sr_out == sample_rate
    assert isinstance(waveform, np.ndarray)
    assert "Recording Information" in info
    assert playback is not None
    playback_sr, playback_waveform = playback
    assert playback_sr == sample_rate
    assert np.array_equal(playback_waveform, waveform)


def test_process_and_preview_recording_playback_reflects_this_recording():
    """The merged handler must not leak state across calls - the playback
    tuple returned for a given call must be built from THAT call's waveform,
    not a previous one (which is exactly what the two-.click race risked).
    """
    sr_a = 8000
    raw_a = np.full(4000, 100, dtype=np.int16)
    sr_b = 16000
    raw_b = np.full(2000, -200, dtype=np.int16)

    _first = utilities.process_and_preview_recording((sr_a, raw_a))
    waveform_b, sr_out_b, _info_b, playback_b = utilities.process_and_preview_recording(
        (sr_b, raw_b)
    )

    assert sr_out_b == sr_b
    playback_sr_b, playback_waveform_b = playback_b
    assert playback_sr_b == sr_b
    assert np.array_equal(playback_waveform_b, waveform_b)
    assert len(playback_waveform_b) == len(raw_b)


def test_process_and_preview_recording_no_audio_returns_none_playback():
    waveform, sr_out, info, playback = utilities.process_and_preview_recording(None)

    assert waveform is None
    assert sr_out is None
    assert "no recording" in info.lower()
    assert playback is None


# ============================================================================
# batch_convert_audio_files - progress reporting
# ============================================================================


class _StubProgress:
    """Minimal stand-in for gr.Progress in offline tests."""

    def tqdm(self, iterable, **kwargs):
        return iterable


def test_batch_convert_audio_files_accepts_progress_kwarg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class _FakeAudioFile:
        def __init__(self, name):
            self.name = name

    def fake_librosa_load(path, sr=None, mono=True):
        return np.linspace(-1.0, 1.0, 50, dtype=np.float32), 22050

    import types

    fake_librosa_module = types.SimpleNamespace(load=fake_librosa_load)
    fake_soundfile_module = types.SimpleNamespace(read=lambda *a, **k: (None, None))

    import sys

    monkeypatch.setitem(sys.modules, "librosa", fake_librosa_module)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile_module)

    files = [_FakeAudioFile("clip_one.wav"), _FakeAudioFile("clip_two.wav")]

    status, output_files = utilities.batch_convert_audio_files(
        files, progress=_StubProgress()
    )

    assert "Successful:** 2" in status
    assert len(output_files) == 2
    for f in output_files:
        assert "_sr22050" in Path(f).name


def test_batch_convert_audio_files_progress_default_is_gr_progress():
    import inspect

    params = inspect.signature(utilities.batch_convert_audio_files).parameters
    assert "progress" in params
    param = params["progress"]
    assert param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert isinstance(param.default, gr.Progress)
    assert next(reversed(params)) == "progress"
