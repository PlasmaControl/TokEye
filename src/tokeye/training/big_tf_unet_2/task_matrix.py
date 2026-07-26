"""Persistent per-run progress tracker with staleness.

JSON-backed (atomic write + flock, safe for concurrent SLURM jobs), one file
per run at ``<cache_root>/task_matrix.json``. Entries are keyed ``step_N`` for
combined steps and ``step_N:<modality>`` for per-modality steps.

This is written by the job itself, on the compute node, so progress survives
losing whatever shell or notebook submitted the work.

Beyond the ablation tracker this adds:

- a status enum (``pending``/``running``/``complete``/``failed``/``stale``),
- ``params_hash`` — hash of the resolved settings that produced the artifact;
  rerunning a step marks everything downstream ``stale`` so nothing can train
  on outputs that no longer match their inputs,
- ``job_id`` — the SLURM job that is producing (or produced) the artifact,
  taken from ``$SLURM_JOB_ID`` inside the job, and the key to finding its log.

There is no sign-off flag: pacing is the operator's, enforced by submitting one
step at a time rather than by a gate in code.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import get_step, steps_after

STATUSES = ("pending", "submitted", "running", "complete", "failed", "stale")


def params_hash(settings: dict[str, Any]) -> str:
    """Stable hash of a step's resolved settings dict."""
    blob = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


class RunTaskMatrix:
    """JSON-backed progress tracker for one pipeline run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"entries": {}}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        content = json.dumps(self._data, indent=2)
        with tmp.open("w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(content)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
        tmp.rename(self.path)

    def reload(self) -> None:
        self._data = self._load()

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    @staticmethod
    def key(step: str, modality: str | None = None) -> str:
        spec = get_step(step)
        if spec.per_modality:
            if modality is None:
                raise ValueError(f"{step} is per-modality; a modality is required")
            return f"{step}:{modality}"
        return step

    def keys_for(self, step: str, modalities: list[str]) -> list[str]:
        spec = get_step(step)
        if spec.per_modality:
            return [f"{step}:{m}" for m in modalities]
        return [step]

    def _entry(self, key: str) -> dict:
        return self._data["entries"].get(key, {})

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _set(self, key: str, **fields: Any) -> None:
        entry = self._data["entries"].setdefault(key, {})
        entry.update(fields, timestamp=datetime.now().isoformat())

    def mark_running(
        self,
        step: str,
        modality: str | None = None,
        job_id: str | None = None,
        step_params_hash: str | None = None,
    ) -> None:
        """Record a step as started.

        The params hash is stored here, not only on completion, so a step that
        died partway can be compared against the knobs of the next attempt —
        that comparison is what makes resuming a partial result safe.
        """
        self.reload()
        entry = self._entry(self.key(step, modality))
        job_id = job_id or entry.get("job_id")
        self._set(
            self.key(step, modality),
            status="running",
            job_id=job_id,
            params_hash=step_params_hash or entry.get("params_hash"),
        )
        self._save()

    def params_match(
        self, step: str, modality: str | None, step_params_hash: str
    ) -> bool:
        """True if the last attempt at this step ran with these same knobs."""
        return self._entry(self.key(step, modality)).get(
            "params_hash"
        ) == step_params_hash

    def mark_complete(
        self,
        step: str,
        modality: str | None,
        step_params_hash: str,
        modalities: list[str],
    ) -> None:
        """Record completion and mark all downstream complete work stale."""
        self.reload()
        self._set(
            self.key(step, modality),
            status="complete",
            params_hash=step_params_hash,
        )
        self._invalidate_downstream(step, modality, modalities)
        self._save()

    def mark_failed(self, step: str, modality: str | None = None) -> None:
        self.reload()
        self._set(self.key(step, modality), status="failed")
        self._save()

    def mark_pending(self, step: str, modalities: list[str]) -> None:
        """Forget a step entirely (used by clearing) and stale its downstream."""
        self.reload()
        for key in self.keys_for(step, modalities):
            self._data["entries"].pop(key, None)
        self._invalidate_downstream(step, None, modalities)
        self._save()

    def _invalidate_downstream(
        self, step: str, modality: str | None, modalities: list[str]
    ) -> None:
        """Stale every downstream entry that currently claims completion.

        A per-modality change only stales that modality's own downstream chain
        plus every combined step; other modalities' per-modality work survives.
        """
        for spec in steps_after(step):
            if spec.per_modality:
                mods = [modality] if modality is not None else modalities
                keys = [f"{spec.name}:{m}" for m in mods]
            else:
                keys = [spec.name]
            for key in keys:
                if self._entry(key).get("status") == "complete":
                    self._set(key, status="stale")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def status(self, step: str, modality: str | None = None) -> str:
        return self._entry(self.key(step, modality)).get("status", "pending")

    def is_complete(self, step: str, modality: str | None = None) -> bool:
        return self.status(step, modality) == "complete"

    def all_complete(self, step: str, modalities: list[str]) -> bool:
        spec = get_step(step)
        if spec.per_modality:
            return all(self.is_complete(step, m) for m in modalities)
        return self.is_complete(step)

    def job_id(self, step: str, modality: str | None = None) -> str | None:
        return self._entry(self.key(step, modality)).get("job_id")

    def to_rows(self, modalities: list[str]) -> list[dict[str, Any]]:
        """One row per (step, modality) for status display."""
        from .paths import STEPS

        rows = []
        for spec in STEPS:
            mods = modalities if spec.per_modality else [None]
            for mod in mods:
                entry = self._entry(self.key(spec.name, mod))
                rows.append(
                    {
                        "step": spec.name,
                        "title": spec.title,
                        "modality": mod or "-",
                        "status": entry.get("status", "pending"),
                        "job_id": entry.get("job_id"),
                        "updated": entry.get("timestamp"),
                    }
                )
        return rows
