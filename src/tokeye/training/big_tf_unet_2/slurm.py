"""SLURM integration: render, submit, and monitor per-step batch jobs.

GPU steps go to the GPU partition with a gres; CPU-heavy steps go to a CPU
node with NO gres (CPU work must never hold a GPU). The sbatch payload is the
same runner CLI the notebooks use inline, so behavior is identical either
way. Scripts + logs live under the run's ``slurm/`` cache dir.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from .paths import RunPaths, get_step
from .run_config import load_run_config
from .task_matrix import RunTaskMatrix

if TYPE_CHECKING:
    from pathlib import Path

    from .run_config import RunConfig


def _needs_gpu(steps: list[str]) -> bool:
    return any(get_step(s).exec_mode == "sbatch_gpu" for s in steps)


def _render_script(
    cfg: RunConfig,
    paths: RunPaths,
    run_yaml: Path,
    steps: list[str],
    modality: str | None,
    tag: str,
) -> str:
    gpu = _needs_gpu(steps)
    slurm = cfg.slurm
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=bt2_{tag}",
        f"#SBATCH --output={paths.slurm_dir.resolve()}/%j_{tag}.out",
        f"#SBATCH --time={slurm.gpu_time if gpu else slurm.cpu_time}",
        f"#SBATCH --mem={slurm.gpu_mem if gpu else slurm.cpu_mem}",
        f"#SBATCH --cpus-per-task={slurm.gpu_cpus if gpu else slurm.cpu_cpus}",
    ]
    if gpu:
        lines.append("#SBATCH --gres=gpu:1")
        if slurm.gpu_partition:
            lines.append(f"#SBATCH --partition={slurm.gpu_partition}")
    elif slurm.cpu_partition:
        lines.append(f"#SBATCH --partition={slurm.cpu_partition}")
    if slurm.account:
        lines.append(f"#SBATCH --account={slurm.account}")

    cmd = (
        "srun python -m tokeye.training.big_tf_unet_2.runner "
        f"--run-config {run_yaml.resolve()} --steps {','.join(steps)}"
    )
    if modality:
        cmd += f" --modalities {modality}"
    lines += [
        "",
        f'cd "{paths.root.resolve()}" && source .venv/bin/activate',
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        cmd,
        "",
    ]
    return "\n".join(lines)


def submit_step(
    run_yaml: str | Path,
    steps: list[str] | str,
    modality: str | None = None,
) -> str:
    """Render + sbatch one job running the given steps; returns the job id."""
    from pathlib import Path as _Path

    run_yaml = _Path(run_yaml)
    if isinstance(steps, str):
        steps = [s.strip() for s in steps.split(",")]
    cfg = load_run_config(run_yaml)
    paths = RunPaths(run_yaml.parent.name)
    paths.slurm_dir.mkdir(parents=True, exist_ok=True)

    tag = "_".join(s.removeprefix("step_") for s in steps) + (
        f"_{modality}" if modality else ""
    )
    script = paths.slurm_dir / f"{tag}.sh"
    script.write_text(_render_script(cfg, paths, run_yaml, steps, modality, tag))

    result = subprocess.run(
        ["sbatch", "--parsable", str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    job_id = result.stdout.strip().split(";")[0]

    matrix = RunTaskMatrix(paths.task_matrix_path)
    for step in steps:
        mods = (
            [modality] if modality else cfg.modality_names
        ) if get_step(step).per_modality else [None]
        for mod in mods:
            matrix.record_job(step, mod, job_id)
    return job_id


def job_state(job_id: str) -> str:
    """Live state via squeue, falling back to sacct for finished jobs."""
    result = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%T"], capture_output=True, text=True
    )
    state = result.stdout.strip()
    if state:
        return state
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "-n", "-P", "-X"],
        capture_output=True,
        text=True,
    )
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    return lines[0] if lines else "UNKNOWN"


def tail_log(paths: RunPaths, job_id: str, n: int = 40) -> str:
    """Last n lines of a submitted job's log file."""
    logs = sorted(paths.slurm_dir.glob(f"{job_id}_*.out"))
    if not logs:
        return f"(no log yet for job {job_id} — still queued?)"
    return "\n".join(logs[-1].read_text().splitlines()[-n:])
