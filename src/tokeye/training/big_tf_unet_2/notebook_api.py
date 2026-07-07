"""The ``Run`` facade — everything the notebooks call, nothing else.

Every notebook cell is 1-3 calls into this class. The loop per step:

    run.suggest("step_2")   # what the autos picked + which knobs matter
    run.run("step_2")       # or run.submit(...) for GPU/heavy steps
    run.status()            # table; rerun the cell to refresh
    run.gallery("step_2")   # look at the pictures
    run.accept("step_2")    # sign off — or edit run.yaml, run.clear, redo

Guardrails: inline execution refuses GPU steps and steps whose upstream
isn't complete+accepted; every error surfaces as a one-line message naming
what to do next.
"""

from __future__ import annotations

import subprocess
import time

import pandas as pd

from . import auto_resolve, gallery, slurm
from .clearing import clear_run, clear_step
from .paths import STEP_ORDER, RunPaths, get_step
from .run_config import check_scale_lock, load_run_config
from .runner import run_step
from .scaffold import scaffold_run
from .task_matrix import RunTaskMatrix


class Run:
    """Handle on one pipeline run, addressed by its run_id."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.paths = RunPaths(run_id)
        if not self.paths.run_yaml.exists():
            raise FileNotFoundError(
                f"No run.yaml at {self.paths.run_yaml} — create the run first: "
                f"python -m tokeye.training.big_tf_unet_2.scaffold "
                f"--nfft <nfft> --hop <hop>"
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, run_id: str) -> Run:
        return cls(run_id)

    @classmethod
    def create(cls, nfft: int, hop: int, suffix: str = "") -> Run:
        paths = scaffold_run(nfft, hop, suffix)
        return cls(paths.run_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def cfg(self):
        cfg = load_run_config(self.paths.run_yaml)
        check_scale_lock(cfg, self.paths.run_meta)
        return cfg

    @property
    def _matrix(self) -> RunTaskMatrix:
        return RunTaskMatrix(self.paths.task_matrix_path)

    def _upstream_unaccepted(self, step: str) -> list[str]:
        mods = self.cfg.modality_names
        matrix = self._matrix
        return [
            name
            for name in STEP_ORDER[: STEP_ORDER.index(step)]
            if not matrix.is_accepted(name, mods)
        ]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> pd.DataFrame:
        """Progress table; SLURM state is refreshed for submitted jobs."""
        rows = self._matrix.to_rows(self.cfg.modality_names)
        for row in rows:
            if row["status"] in ("submitted", "running") and row["job_id"]:
                row["slurm"] = slurm.job_state(str(row["job_id"]))
            else:
                row["slurm"] = ""
        return pd.DataFrame(rows)

    def suggest(self, step: str) -> None:
        """Show auto-resolved values + where the step's knobs live."""
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
        self.edit_hint(step)

    def edit_hint(self, step: str) -> None:
        section = get_step(step).knob_section
        print(f"knobs: section '{section}:' in {self.paths.run_yaml}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, step: str, modality: str | None = None, force: bool = False):
        """Run a step inline. Refuses GPU steps and unaccepted upstream."""
        spec = get_step(step)
        if spec.exec_mode == "sbatch_gpu" and not force:
            print(f"{step} is a GPU step — use run.submit({step!r}) instead")
            return
        unaccepted = [] if force else self._upstream_unaccepted(step)
        if unaccepted:
            print(
                f"{step} needs your sign-off on {', '.join(unaccepted)} first "
                f"(run.accept(...) after checking the gallery), or force=True"
            )
            return
        run_step(self.paths.run_yaml, step, modality, force)

    def submit(self, steps: str | list[str], modality: str | None = None) -> str:
        """Submit step(s) as one SLURM job; returns the job id."""
        first = steps.split(",")[0] if isinstance(steps, str) else steps[0]
        unaccepted = self._upstream_unaccepted(first.strip())
        if unaccepted:
            print(
                f"heads-up: upstream not signed off yet: {', '.join(unaccepted)}"
            )
        job_id = slurm.submit_step(self.paths.run_yaml, steps, modality)
        print(f"submitted job {job_id} — run.status() / run.log({job_id!r})")
        return job_id

    def wait(self, job_id: str, poll: int = 30) -> str:
        """Block until a SLURM job leaves the queue; returns the final state."""
        while True:
            state = slurm.job_state(job_id)
            if state in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING"):
                time.sleep(poll)
                continue
            print(f"job {job_id}: {state}")
            return state

    def log(self, job_id: str, n: int = 40) -> None:
        print(slurm.tail_log(self.paths, str(job_id), n))

    # ------------------------------------------------------------------
    # Inspection + sign-off
    # ------------------------------------------------------------------

    def gallery(self, step: str, modality: str | None = None, n: int = 6) -> None:
        gallery.show(
            self.paths, step, modality, modalities=self.cfg.modality_names, n=n
        )

    def accept(self, step: str) -> None:
        try:
            self._matrix.accept(step, self.cfg.modality_names)
            print(f"{step} accepted — next step unlocked")
        except ValueError as err:
            print(err)

    def clear(self, step: str, modality: str | None = None) -> None:
        removed = clear_step(self.paths, self.cfg, step, modality)
        print(
            f"cleared {len(removed)} artifact(s) for {step}"
            + (f" [{modality}]" if modality else "")
            + "; downstream marked stale"
        )

    def clear_all(self, confirm: str) -> None:
        """Delete every cache/model artifact. confirm must be the run_id."""
        clear_run(self.paths, confirm)
        print(f"run {self.run_id} cache cleared (run.yaml + notebooks kept)")

    def jobstats(self, job_id: str) -> None:
        """Cluster efficiency report for a finished/running job."""
        result = subprocess.run(
            ["jobstats", str(job_id)], capture_output=True, text=True
        )
        print(result.stdout or result.stderr)
