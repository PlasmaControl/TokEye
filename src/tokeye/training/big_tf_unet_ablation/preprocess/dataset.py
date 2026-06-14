"""Minimal JoblibDataset replacing faith.train.data.datasets.file_based.

Loads joblib dict files and provides windowed subsequence access via a
PyTorch-compatible interface (iterable with ``__getitem__``/``__len__``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

logger = logging.getLogger(__name__)


class JoblibDataset:
    """Dataset that loads joblib dict files and extracts fixed-length windows.

    Each joblib file is a dict mapping signal names to numpy arrays.  The
    dataset builds a flat index of non-overlapping windows of length
    *subseq_len* across all files and exposes them via ``__getitem__``.

    Parameters
    ----------
    file_paths : list of paths
        Joblib files to load.
    input_key : str or list[str]
        Key(s) to read from each joblib dict.
    subseq_len : int
        Number of time-domain samples per window.
    validate_on_init : bool
        If True, inspect every file on init to build the index.
    """

    def __init__(
        self,
        file_paths: str | list[str] | Path,
        input_key: str | list[str] | None = None,
        subseq_len: int = 1024,
        validate_on_init: bool = True,
    ) -> None:
        if isinstance(file_paths, (str, Path)):
            p = Path(file_paths)
            if p.is_dir():
                file_paths = sorted(p.glob("*.joblib"), key=lambda x: int(x.stem))
            else:
                file_paths = sorted(p.parent.glob(p.name))
        self.file_paths = [str(p) for p in file_paths]

        if isinstance(input_key, str):
            input_key = [input_key]
        self.input_key = input_key
        self.subseq_len = subseq_len

        # Per-file handles (set in worker_init)
        self._file_handles: list[dict[str, Any] | None] = [None] * len(self.file_paths)

        # Build index: list of (file_idx, start, end)
        self.subseq_index: list[tuple[int, int, int]] = []
        if validate_on_init:
            self._build_index()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """Scan files and build non-overlapping window index."""
        self.subseq_index = []
        for file_idx, fp in enumerate(self.file_paths):
            try:
                data = joblib.load(fp, mmap_mode="r")
            except Exception as e:
                logger.warning(f"Could not inspect {fp}: {e}")
                continue

            seq_len = self._infer_seq_len(data)
            n_windows = seq_len // self.subseq_len
            for w in range(n_windows):
                start = w * self.subseq_len
                self.subseq_index.append((file_idx, start, start + self.subseq_len))

        logger.info(
            f"JoblibDataset: {len(self.file_paths)} files, "
            f"{len(self.subseq_index)} windows of length {self.subseq_len}"
        )

    def _infer_seq_len(self, data: dict[str, Any]) -> int:
        """Infer sequence length from the first matching key."""
        keys = self.input_key or [k for k in data if k != "time_ms"]
        for k in keys:
            arr = data[k] if isinstance(data, dict) else data
            if isinstance(arr, np.ndarray):
                # Time is typically the longest dimension
                return max(arr.shape)
        return 0

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------
    def worker_init(self) -> None:
        """Open all files with memory mapping. Call before iteration."""
        for i, fp in enumerate(self.file_paths):
            try:
                self._file_handles[i] = joblib.load(fp, mmap_mode="r")
            except Exception as e:
                logger.warning(f"Could not open {fp}: {e}")
                self._file_handles[i] = None

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.subseq_index)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor]]:
        file_idx, start, end = self.subseq_index[idx]

        data = self._file_handles[file_idx]
        if data is None:
            data = joblib.load(self.file_paths[file_idx], mmap_mode="r")

        keys = self.input_key or [k for k in data if k != "time_ms"]
        out: dict[str, torch.Tensor] = {}
        for k in keys:
            arr = np.asarray(data[k])
            t = self._extract(arr, start, end)
            out[k] = t

        return (out,)

    @staticmethod
    def _extract(arr: np.ndarray, start: int, end: int) -> torch.Tensor:
        """Slice the time dimension (assumed to be the longest axis)."""
        if arr.ndim == 1:
            return torch.from_numpy(arr[start:end].copy()).float()
        if arr.ndim == 2:
            # (channels, time) or (time, channels)
            if arr.shape[-1] >= arr.shape[0]:
                return torch.from_numpy(arr[:, start:end].copy()).float()
            return torch.from_numpy(arr[start:end, :].copy()).float()
        if arr.ndim >= 3:
            # Assume time is last dimension
            return torch.from_numpy(arr[..., start:end].copy()).float()
        return torch.from_numpy(arr.copy()).float()

    # ------------------------------------------------------------------
    # Iteration (for step_0c compatibility)
    # ------------------------------------------------------------------
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
