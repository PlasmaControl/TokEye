from __future__ import annotations

import csv

import numpy as np

from tokeye.elmspec import (
    ElmEvent,
    column_activity,
    extract_elm_events,
    summarize,
    write_events_csv,
)


def _mask_with_bursts(bursts: list[tuple[int, int]], width: int = 50, height: int = 10):
    """Transient-channel mask with full-column bursts over [start, end] cols."""
    mask = np.zeros((height, width), dtype=np.float32)
    for start, end in bursts:
        mask[:, start : end + 1] = 1.0
    return mask


def test_column_activity_is_fraction_of_active_bins():
    mask = np.zeros((10, 4), dtype=np.float32)
    mask[:5, 1] = 1.0  # half the bins in column 1
    mask[:, 2] = 1.0  # all bins in column 2
    activity = column_activity(mask, threshold=0.5)
    assert activity.shape == (4,)
    assert activity[0] == 0.0
    assert activity[1] == 0.5
    assert activity[2] == 1.0


def test_extract_merges_events_across_small_gaps():
    mask = _mask_with_bursts([(10, 12), (15, 17)])  # gap of 2 columns
    events = extract_elm_events(mask, min_gap_cols=3)
    assert len(events) == 1
    assert events[0].start_col == 10
    assert events[0].end_col == 17


def test_extract_keeps_events_across_large_gaps():
    mask = _mask_with_bursts([(10, 12), (30, 32)])
    events = extract_elm_events(mask, min_gap_cols=3)
    assert [(e.start_col, e.end_col) for e in events] == [(10, 12), (30, 32)]


def test_extract_drops_short_events():
    mask = _mask_with_bursts([(5, 5), (20, 24)])
    events = extract_elm_events(mask, min_gap_cols=1, min_duration_cols=2)
    assert [(e.start_col, e.end_col) for e in events] == [(20, 24)]


def test_extract_empty_mask_gives_no_events():
    mask = np.zeros((10, 50), dtype=np.float32)
    assert extract_elm_events(mask) == []


def test_extract_records_peak_activity():
    mask = np.zeros((10, 50), dtype=np.float32)
    mask[:5, 10:13] = 1.0  # activity 0.5
    events = extract_elm_events(mask, activity_min=0.3)
    assert len(events) == 1
    assert events[0].peak_activity == 0.5


def test_summarize_counts_and_frequency():
    events = [ElmEvent(10, 12, 1.0), ElmEvent(30, 32, 1.0)]
    # 100 columns at hop=256, fs=200_000 -> 0.128 s total
    summary = summarize(events, n_cols=100, hop=256, fs=200_000.0)
    assert summary["n_events"] == 2
    assert np.isclose(summary["elm_freq_hz"], 2 / (100 * 256 / 200_000.0))
    assert np.isclose(summary["duty_cycle"], 6 / 100)


def test_summarize_without_fs_leaves_frequency_unset():
    summary = summarize([ElmEvent(0, 1, 1.0)], n_cols=10, hop=256, fs=None)
    assert summary["n_events"] == 1
    assert summary["elm_freq_hz"] is None


def test_write_events_csv(tmp_path):
    out = tmp_path / "elm_events.csv"
    events = [ElmEvent(10, 12, 0.75)]
    write_events_csv(out, [("shot1.npy", events)], hop=256, fs=200_000.0)

    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["input"] == "shot1.npy"
    assert rows[0]["event"] == "0"
    assert rows[0]["start_col"] == "10"
    assert rows[0]["end_col"] == "12"
    assert float(rows[0]["t_start_s"]) == 10 * 256 / 200_000.0
    assert float(rows[0]["peak_activity"]) == 0.75


def test_write_events_csv_without_fs_leaves_times_blank(tmp_path):
    out = tmp_path / "elm_events.csv"
    write_events_csv(out, [("shot1.npy", [ElmEvent(0, 3, 1.0)])], hop=256, fs=None)
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["t_start_s"] == ""
