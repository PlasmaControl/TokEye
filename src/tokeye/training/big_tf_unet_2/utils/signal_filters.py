"""Signal-processing filters shared by the intake steps.

``Preemphasis`` used to be re-exported from ``torchaudio.transforms`` by each
step module that needed it (``step_0b_filter_faithdata.py`` here and in the
ablation pipeline, ``step_0f_foundation.py``). It now lives in one place so
steps depend on this module instead of on each other.
"""

from __future__ import annotations

from torchaudio.transforms import Preemphasis

__all__ = ["Preemphasis"]
