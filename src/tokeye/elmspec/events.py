"""Pure-numpy ELM event extraction from a transient-activity mask.

Input is the transient channel of a TokEye mask (``mask[1]``, shape ``(H, W)``,
values in [0, 1]). An ELM shows up as a broadband vertical stripe: many
frequency bins active in the same time column. Detection is therefore
column-wise: threshold the mask, measure the active fraction per column,
mark columns above ``activity_min``, close small gaps, and report the
remaining contiguous runs as events.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ElmEvent:
    start_col: int
    end_col: int  # inclusive
    peak_activity: float

    @property
    def duration_cols(self) -> int:
        return self.end_col - self.start_col + 1


def column_activity(transient_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Fraction of frequency bins at or above ``threshold``, per time column."""
    return (transient_mask >= threshold).mean(axis=0)


def _contiguous_runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of each True run."""
    padded = np.concatenate(([False], active, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    starts, ends = edges[::2], edges[1::2] - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _fill_gaps(active: np.ndarray, max_gap: int) -> np.ndarray:
    """Close False gaps of at most ``max_gap`` columns between True runs."""
    if max_gap <= 0:
        return active
    filled = active.copy()
    runs = _contiguous_runs(active)
    for (_, prev_end), (next_start, _) in zip(runs, runs[1:], strict=False):
        if next_start - prev_end - 1 <= max_gap:
            filled[prev_end : next_start + 1] = True
    return filled


def extract_elm_events(
    transient_mask: np.ndarray,
    *,
    threshold: float = 0.5,
    activity_min: float = 0.1,
    min_gap_cols: int = 3,
    min_duration_cols: int = 1,
) -> list[ElmEvent]:
    """Detect ELM events in a ``(H, W)`` transient-activity mask.

    ``threshold`` binarizes mask values; ``activity_min`` is the minimum
    active-bin fraction for a column to count as part of an event;
    runs separated by gaps of at most ``min_gap_cols`` columns are merged;
    events shorter than ``min_duration_cols`` are dropped.
    """
    activity = column_activity(transient_mask, threshold=threshold)
    active = _fill_gaps(activity >= activity_min, min_gap_cols)
    return [
        ElmEvent(start, end, float(activity[start : end + 1].max()))
        for start, end in _contiguous_runs(active)
        if end - start + 1 >= min_duration_cols
    ]


def summarize(
    events: list[ElmEvent], n_cols: int, hop: int, fs: float | None
) -> dict[str, float | int | None]:
    """Per-input summary: event count, ELM frequency (needs ``fs``), duty cycle.

    ``elm_freq_hz`` is events per second of analyzed signal; ``None`` when the
    sampling rate is unknown (spectrogram columns have no absolute timebase).
    """
    active_cols = sum(event.duration_cols for event in events)
    total_s = n_cols * hop / fs if fs else None
    return {
        "n_events": len(events),
        "elm_freq_hz": len(events) / total_s if total_s else None,
        "duty_cycle": active_cols / n_cols if n_cols else 0.0,
    }


def _col_to_s(col: int, hop: int, fs: float | None) -> float | str:
    return col * hop / fs if fs else ""


EVENT_FIELDS = (
    "input",
    "event",
    "start_col",
    "end_col",
    "duration_cols",
    "t_start_s",
    "t_end_s",
    "duration_s",
    "peak_activity",
)

SUMMARY_FIELDS = ("input", "n_events", "elm_freq_hz", "duty_cycle")


def write_events_csv(
    path: Path,
    per_input: list[tuple[str, list[ElmEvent]]],
    hop: int,
    fs: float | None,
) -> None:
    """One row per detected event; time columns blank when ``fs`` is unknown."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for name, events in per_input:
            for index, event in enumerate(events):
                writer.writerow(
                    {
                        "input": name,
                        "event": index,
                        "start_col": event.start_col,
                        "end_col": event.end_col,
                        "duration_cols": event.duration_cols,
                        "t_start_s": _col_to_s(event.start_col, hop, fs),
                        "t_end_s": _col_to_s(event.end_col + 1, hop, fs),
                        "duration_s": _col_to_s(event.duration_cols, hop, fs),
                        "peak_activity": event.peak_activity,
                    }
                )


def write_summary_csv(
    path: Path, per_input: list[tuple[str, dict[str, float | int | None]]]
) -> None:
    """One row per input file with its :func:`summarize` result."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for name, summary in per_input:
            row = {"input": name, **summary}
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
