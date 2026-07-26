"""Read-only inspection of a run — everything the diagnosis notebooks call.

The pipeline is driven from the shell (one ``sbatch step_N_*.sh`` at a time), so
nothing here submits, runs, clears, or signs off. A notebook opens the run it
sits in and looks at what already happened:

    from tokeye.training.big_tf_unet_2 import view
    v = view.here()          # run folder is the cwd
    v.status()
    v.gallery("step_6")
    v.log("step_6")

``here()`` deriving the run from the working directory is what lets the template
be copied without substituting anything: the notebook, like the ``.sh``, learns
which run it belongs to from where it sits.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .paths import RunPaths, get_step
from .run_config import check_scale_lock, load_run_config
from .task_matrix import RunTaskMatrix

if TYPE_CHECKING:
    import pandas as pd

# pandas, h5py and matplotlib are train-group deps, and `gallery` and
# `auto_resolve` pull them in. Importing them here would make this module
# unimportable under a plain `uv sync --dev` (what CI installs), so they are
# deferred into the two methods that actually need them.

_ACTIVE = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING")


def job_state(job_id: str) -> str:
    """Live SLURM state via squeue, falling back to sacct for finished jobs."""
    result = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
        capture_output=True,
        text=True,
    )
    state = result.stdout.strip()
    if state:
        return state.splitlines()[0].strip()
    result = subprocess.run(
        ["sacct", "-j", str(job_id), "--format=State", "-n", "-P", "-X"],
        capture_output=True,
        text=True,
    )
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    return lines[0] if lines else "UNKNOWN"


class RunView:
    """Read-only handle on one pipeline run, addressed by its run_id."""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.run_id = run_id
        self.paths = RunPaths(run_id) if root is None else RunPaths(run_id, root=root)
        if not self.paths.run_yml.exists():
            raise FileNotFoundError(
                f"No run.yml at {self.paths.run_yml} — create the run by copying "
                f"the template: cp -r scripts/big_tf_unet_2 "
                f"dev/training/{run_id}"
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def cfg(self):
        cfg = load_run_config(self.paths.run_yml)
        check_scale_lock(cfg, self.paths.run_meta)
        return cfg

    @property
    def _matrix(self) -> RunTaskMatrix:
        return RunTaskMatrix(self.paths.task_matrix_path)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> pd.DataFrame:
        """Progress table; SLURM state is refreshed for jobs that recorded one."""
        import pandas as pd

        rows = self._matrix.to_rows(self.cfg.modality_names)
        for row in rows:
            job_id = row.get("job_id")
            running = row["status"] == "running"
            row["slurm"] = job_state(job_id) if (job_id and running) else ""
        return pd.DataFrame(rows)

    def suggest(self, step: str) -> None:
        """Show auto-resolved values + which file holds this step's knobs."""
        from . import auto_resolve

        cfg = self.cfg
        spec = get_step(step)
        mods = cfg.modality_names if spec.per_modality else [None]
        for mod in mods:
            values = auto_resolve.suggest(cfg, step, mod, self.paths)
            label = f"{step}" + (f" [{mod}]" if mod else "")
            if values:
                print(f"{label} auto suggestions:")
                for k, v in values.items():
                    print(f"  {k} = {v}")
            else:
                print(f"{label}: no auto values (or upstream not run yet)")
        print(
            f"knobs: section '{spec.knob_section}:' in "
            f"{self.paths.step_yml(step).name}"
        )

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def logs_for(self, step: str) -> list[Path]:
        """This step's job output files, oldest first.

        ``logs/`` is flat and job-id-first (``<jobid>[_<arraytask>]_<jobname>``),
        and the job name is the step's module, so a suffix match finds a step's
        logs across every run. Sorted by mtime because job ids sort as strings.
        """
        module = get_step(step).module
        return sorted(
            self.paths.log_dir.glob(f"*_{module}.out"), key=lambda p: p.stat().st_mtime
        )

    def log(self, step: str, n: int = 40) -> None:
        """Tail the newest log for a step (all array tasks, if it was an array)."""
        found = self.logs_for(step)
        if not found:
            print(f"(no log yet for {step} in {self.paths.log_dir} — still queued?)")
            return
        newest = found[-1]
        # An array job writes one file per task at nearly the same time; show
        # every task that shares the newest job id rather than an arbitrary one.
        job_id = newest.name.split("_")[0]
        for path in [p for p in found if p.name.startswith(f"{job_id}_")]:
            print(f"===== {path.name} =====")
            print("\n".join(path.read_text().splitlines()[-n:]))

    def jobstats(self, job_id: str) -> None:
        """Cluster efficiency report for a finished/running job."""
        result = subprocess.run(
            ["jobstats", str(job_id)], capture_output=True, text=True
        )
        print(result.stdout or result.stderr)

    # ------------------------------------------------------------------
    # Pictures
    # ------------------------------------------------------------------

    def gallery(self, step: str, modality: str | None = None, n: int = 6) -> None:
        from . import gallery

        gallery.show(
            self.paths, step, modality, modalities=self.cfg.modality_names, n=n
        )


def open(run_id: str) -> RunView:  # noqa: A001 — mirrors the old Run.open
    """Open a run by id (the name of its folder under ``dev/training/``)."""
    return RunView(run_id)


def here() -> RunView:
    """Open the run whose folder is the current working directory.

    A notebook's cwd is its own directory, which for these templates is the run
    folder — so the folder name is the run id and nothing needs typing.
    """
    cwd = Path.cwd()
    if not (cwd / "run.yml").exists():
        raise FileNotFoundError(
            f"No run.yml in {cwd} — view.here() expects the notebook to sit in a "
            f"run folder copied from scripts/big_tf_unet_2. Use "
            f"view.open('<run_id>') instead."
        )
    return RunView(cwd.name)
