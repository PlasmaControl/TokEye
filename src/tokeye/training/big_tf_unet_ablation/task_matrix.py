"""Persistent task matrix tracking pipeline progress across all combos.

Stored as a JSON file with atomic writes and file-locking for safe
concurrent access from SLURM array jobs.
"""

from __future__ import annotations

import fcntl
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Steps that define a fully-complete combo pipeline run
_COMBO_STEPS = [
    "step_0c", "step_1a",
    "step_2a", "step_2b",
    "step_3a", "step_3b",
    "step_4a",
    "step_5a",
    "step_6a", "step_6b", "step_6c", "step_6d",
]

_SHARED_STEPS = ["step_0a", "step_0b"]


class TaskMatrix:
    """JSON-backed progress tracker for the multiscale pipeline."""

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
        return {
            "shared_steps": {},
            "combos": {},
            "final_trained": False,
        }

    def _save(self) -> None:
        """Atomic write with file-locking for concurrent SLURM tasks."""
        tmp = self.path.with_suffix(".tmp")
        content = json.dumps(self._data, indent=2)
        with tmp.open("w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(content)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
        tmp.rename(self.path)

    def reload(self) -> None:
        """Re-read from disk (useful in long-running jobs)."""
        self._data = self._load()

    # ------------------------------------------------------------------
    # Shared steps
    # ------------------------------------------------------------------

    def is_shared_step_complete(self, step_name: str) -> bool:
        return self._data["shared_steps"].get(step_name, {}).get("status") == "complete"

    def mark_shared_step_complete(self, step_name: str) -> None:
        self._data["shared_steps"][step_name] = {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    # ------------------------------------------------------------------
    # Per-combo steps
    # ------------------------------------------------------------------

    def is_step_complete(self, combo_id: str, step_name: str) -> bool:
        combo = self._data["combos"].get(combo_id, {})
        return combo.get(step_name, {}).get("status") == "complete"

    def mark_step_complete(self, combo_id: str, step_name: str) -> None:
        if combo_id not in self._data["combos"]:
            self._data["combos"][combo_id] = {}
        self._data["combos"][combo_id][step_name] = {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    def is_combo_complete(self, combo_id: str) -> bool:
        # In the ablation pipeline per-modality steps are keyed "step:modality"
        # and the final combined step is "step_6d"; a variant is complete iff its
        # final surrogate (step_6d) finished.
        return self.is_step_complete(combo_id, "step_6d")

    # ------------------------------------------------------------------
    # Final model
    # ------------------------------------------------------------------

    def is_final_trained(self) -> bool:
        return bool(self._data.get("final_trained", False))

    def mark_final_trained(self) -> None:
        self._data["final_trained"] = True
        self._save()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_incomplete_combos(self, all_combo_ids: list[str]) -> list[str]:
        """Return combo IDs that are not fully complete."""
        return [cid for cid in all_combo_ids if not self.is_combo_complete(cid)]

    def get_summary(self, all_combo_ids: list[str] | None = None) -> dict:
        """Return a summary dict for display."""
        combos = self._data.get("combos", {})
        total = len(all_combo_ids) if all_combo_ids else len(combos)
        complete = sum(
            1 for cid in (all_combo_ids or combos)
            if self.is_combo_complete(cid)
        )
        return {
            "total_combos": total,
            "complete_combos": complete,
            "shared_steps": {
                k: v.get("status", "unknown")
                for k, v in self._data.get("shared_steps", {}).items()
            },
            "final_trained": self.is_final_trained(),
        }

    def print_status(self, all_combo_ids: list[str] | None = None) -> None:
        """Print a human-readable status table."""
        summary = self.get_summary(all_combo_ids)
        print("Pipeline Status:")
        print(f"  Shared steps: {summary['shared_steps']}")
        print(f"  Combos: {summary['complete_combos']}/{summary['total_combos']} complete")
        print(f"  Final trained: {summary['final_trained']}")

        if all_combo_ids:
            print(f"\n  {'Combo':<20} ", end="")
            for s in _COMBO_STEPS:
                print(f"{s[-2:]} ", end="")
            print()
            for cid in all_combo_ids:
                print(f"  {cid:<20} ", end="")
                for s in _COMBO_STEPS:
                    done = self.is_step_complete(cid, s)
                    print(f"{'OK' if done else '--':>2} ", end="")
                print()
