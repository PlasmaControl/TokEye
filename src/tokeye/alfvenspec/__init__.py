"""Alfvén-eigenmode detection tools (energetic-particle group).

Deliberately thin for now: ``tokeye alfvenspec`` runs the ``ae_tf_maskrcnn``
instance-detection model over spectrograms and writes per-detection boxes,
scores, and masks. Deeper EP-group workflows (AE taxonomy, cross-diagnostic
checks) land here once their requirements are gathered — see docs/ROADMAP.md.
"""

from __future__ import annotations

from tokeye.alfvenspec.inference import (
    DEFAULT_WINDOW_COLS,
    detect,
    detect_windowed,
    write_detections_csv,
)

__all__ = ["DEFAULT_WINDOW_COLS", "detect", "detect_windowed", "write_detections_csv"]
