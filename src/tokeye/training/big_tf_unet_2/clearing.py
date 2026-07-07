"""Safe, file-granular clearing of step artifacts.

Replaces the legacy ``setup_directory(overwrite=True)`` pattern (which
``shutil.rmtree``'d whole directories and could wipe sibling modalities).
Every deletion here is resolved against an explicit artifact list from
``RunPaths.artifacts`` and fenced to the run's own cache/model roots —
clearing can never touch another run or anything outside the pipeline dirs.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from .paths import RunPaths, get_step
from .task_matrix import RunTaskMatrix

if TYPE_CHECKING:
    from pathlib import Path

    from .run_config import RunConfig


def _assert_fenced(target: Path, roots: list[Path]) -> None:
    resolved = target.resolve()
    for root in roots:
        if resolved.is_relative_to(root.resolve()):
            return
    raise RuntimeError(
        f"Refusing to delete {target}: outside the run's cache/model roots"
    )


def clear_step(
    paths: RunPaths,
    cfg: RunConfig,
    step: str,
    modality: str | None = None,
) -> list[Path]:
    """Delete one step's artifacts (optionally one modality) and mark it
    pending; everything downstream that was complete becomes stale.

    Returns the paths that were removed.
    """
    spec = get_step(step)
    if modality is not None and not spec.per_modality:
        raise ValueError(f"{step} is not per-modality; drop the modality argument")
    modalities = [modality] if modality is not None else cfg.modality_names

    roots = [paths.cache_root, paths.model_dir]
    removed: list[Path] = []
    for target in paths.artifacts(step, modalities):
        _assert_fenced(target, roots)
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(target)
        elif target.exists():
            target.unlink()
            removed.append(target)

    matrix = RunTaskMatrix(paths.task_matrix_path)
    matrix.mark_pending(step, modalities)
    return removed


def clear_run(paths: RunPaths, confirm: str) -> None:
    """Delete ALL cache and model artifacts of a run. Keeps the workspace
    (run.yaml + notebooks). ``confirm`` must equal the run_id, typed out.
    """
    if confirm != paths.run_id:
        raise ValueError(
            f"clear_run needs confirm={paths.run_id!r} (got {confirm!r}). "
            f"Type the run id exactly to confirm you mean it."
        )
    for root in (paths.cache_root, paths.model_dir):
        if root.exists():
            _assert_fenced(root, [root])
            shutil.rmtree(root)
    paths.cache_root.mkdir(parents=True, exist_ok=True)
