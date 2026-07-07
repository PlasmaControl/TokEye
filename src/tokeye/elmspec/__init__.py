"""ELM detection from TokEye's transient-activity channel.

``tokeye elmspec`` runs the segmentation model on spectrograms and turns the
transient channel (``mask[1]``) into discrete ELM events: time intervals,
counts, and ELM frequency. Pure-numpy event extraction lives in
:mod:`tokeye.elmspec.events`; the model plumbing is in the CLI handler.
"""

from __future__ import annotations

from tokeye.elmspec.events import (
    ElmEvent,
    column_activity,
    extract_elm_events,
    summarize,
    write_events_csv,
    write_summary_csv,
)

__all__ = [
    "ElmEvent",
    "column_activity",
    "extract_elm_events",
    "summarize",
    "write_events_csv",
    "write_summary_csv",
]
