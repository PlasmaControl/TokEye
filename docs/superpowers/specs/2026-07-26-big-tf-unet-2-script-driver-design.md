# big_tf_unet_2 — script driver replacing the notebook driver

Date: 2026-07-26
Status: approved, implementing

## Problem

`big_tf_unet_2` is driven from six Jupyter notebooks. On the cluster that means
`ssh` + `jupyter lab` on a login node, which is fragile: the kernel dies with the
SSH session, a blocking `run.wait()` poll loop dies with it, and the operator
concludes the SLURM job was cancelled. (It wasn't — `sbatch --parsable` is
detached and nothing in the package calls `scancel`. The step_6 deaths trace to
`gpu_time` being cut from 18h to 5h, against a partition limit of 15 days.)

The notebook layer is also thin. `runner.py::run_step` already does the real
work, including per-modality fan-out and upstream gating; `notebook_api.py` is a
7.5 KB facade over it and `slurm.py` renders sbatch scripts from YAML.

## Goal

A flat, self-contained run folder of plain files an operator runs from the shell,
one step at a time, checking each before submitting the next. Notebooks survive
for **diagnosis only**. SLURM scripts must be short enough to read end to end.

## Design

### Template, not scaffold

The run folder is a template at `scripts/big_tf_unet_2/`, copied into place:

```bash
cp -r scripts/big_tf_unet_2 dev/training/nfft512_hop128
cd dev/training/nfft512_hop128
```

`cp -r` replaces `scaffold.py`, which is deleted. The **directory name becomes
the run id**, because `RunPaths` already derives it from the config file's parent
directory — so `data/cache/big_tf_unet_2/<dirname>/` needs no extra plumbing.

Consequences:

- `run_meta.json` (the nfft/hop scale lock) was written by scaffold. `runner.py`
  now writes it on first run. `check_scale_lock` already early-returns when the
  file is absent, so it still catches the case it exists for: changing `nfft`
  midway through a run and silently mixing artifacts.
- `run_id_for()` and the `SCALE_CONFIGS` grid stop being enforced at creation,
  since nothing generates the name. A folder named `nfft512_hop128` may contain
  `nfft: 1024`. Accepted trade for `cp -r`; the scale lock still catches drift
  *within* a run.

### Flat triplets

Nine steps, each with three files sharing a stem, no subdirectories:

```
run.yml                       # shared: run, modalities, paths, smoke, overwrite
step_0_intake.{yml,sh,ipynb}
step_1_spectrogram.{yml,sh,ipynb}
step_2_baseline.{yml,sh,ipynb}
step_3_denoise.{yml,sh,ipynb}
step_4_labels.{yml,sh,ipynb}
step_5_dataset.{yml,sh,ipynb}
step_6_refine.{yml,sh,ipynb}
step_7_final.{yml,sh,ipynb}
step_8_eval.{yml,sh,ipynb}
```

`.yml` = what it runs with, `.sh` = how it's submitted, `.ipynb` = what you look
at afterward. Filenames are `StepSpec.module`; each `.yml` owns exactly the
section named by that step's `StepSpec.knob_section`. Both fields already exist
in the registry, so the file set is derived, not hand-maintained.

| File | Section |
|---|---|
| `step_0_intake.yml` | `extraction` |
| `step_1_spectrogram.yml` | `window_filter` |
| `step_2_baseline.yml` | `baseline` |
| `step_3_denoise.yml` | `denoise` |
| `step_4_labels.yml` | `labels` |
| `step_5_dataset.yml` | `dataset` |
| `step_6_refine.yml` | `refine` |
| `step_7_final.yml` | `final` |
| `step_8_eval.yml` | `eval` |

### Config merge

```
config/defaults.yaml  ←  run.yml  ←  step_N_*.yml
```

One `OmegaConf.merge` chain, deep-merged, validated into a single whole
`RunConfig`. Splitting the *files* does not split the *config object*: `step_3`
still reads `run.nfft` and `modalities`. The runner finds `run.yml` as a sibling
of the `--config` file it was handed.

`run.yml` holds what is not per-step: `run` (nfft/hop), `modalities`, `paths`,
`smoke`, `overwrite`. The `slurm:` section is deleted — those numbers now live in
`#SBATCH` headers.

### The sbatch files

```bash
#!/bin/bash
#SBATCH --job-name=step_3_denoise
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --chdir=/scratch/gpfs/nc1514/tokeye
#SBATCH --time=18:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

source .venv/bin/activate
MODS=(ece mhr bes co2)
python -m tokeye.training.big_tf_unet_2.runner \
    --config "$SLURM_SUBMIT_DIR/step_3_denoise.yml" \
    --modalities "${MODS[$SLURM_ARRAY_TASK_ID]}"
```

- `--chdir` is the repo root. It is what makes a static template work: SLURM
  resolves `--output` relative to `--chdir`, so `logs/` is the repo's `logs/`,
  and `source .venv/bin/activate` resolves. Setting cwd in the body would be too
  late — `--output` is resolved by slurmd before the first command runs.
- `$SLURM_SUBMIT_DIR` supplies the run folder. The act of submitting from that
  folder tells the script which run it belongs to, so one template serves every
  scale with no substitution.
- No `srun`: single-task job, and it removes the inherited-step-environment
  failure mode.
- Per-modality steps (0–4) use `--array`, so modalities run in parallel.

Log files are flat in the repo's `logs/`, job id first:

```
logs/2884567_0_step_3_denoise.out     # %A_%a_%x — array
logs/2884571_step_5_dataset.out       # %j_%x
```

Python logging goes to stdout only, which SLURM captures into that same file. The
shared `runner.log` is dropped — a single append-mode file under a flat `logs/`
would interleave concurrent array tasks. The runner logs the run id as its first
line, since the filename carries no run identity.

Resource headers:

| Step | time | mem | cpus | gres | array |
|---|---|---|---|---|---|
| 0 intake | 04:00:00 | 43G | 42 | — | 0-3 |
| 1 spectrogram | 04:00:00 | 43G | 42 | — | 0-3 |
| 2 baseline | 08:00:00 | 43G | 42 | — | 0-3 |
| 3 denoise | 18:00:00 | 48G | 16 | gpu:1 | 0-3 |
| 4 labels | 01:00:00 | 43G | 8 | — | 0-3 |
| 5 dataset | 02:00:00 | 43G | 42 | — | — |
| 6 refine | 23:59:00 | 48G | 16 | gpu:1 | — |
| 7 final | 18:00:00 | 48G | 16 | gpu:1 | — |
| 8 eval | 02:00:00 | 43G | 8 | — | — |

step_6 gets 23:59:00, not the 5h that has been killing it.

### No chaining, no sign-off gate

There is no `submit_all.sh`. The operator submits one step, inspects it, submits
the next. `task_matrix.accept` / `is_accepted` are deleted along with the
`accepted` field; `_check_upstream` keeps only its "upstream complete" check.
Submitting out of order starts a job that dies in seconds naming the missing
upstream, so no GPU hours burn.

### Overwrite instead of clear

`runner.py::run_step` already calls `clear_step` unconditionally before every
step, so overwrite-by-default is existing behavior. This design only exposes the
knob to turn it *off*:

```yaml
overwrite: true     # run.yml; a resubmit replaces that step's artifacts
```

With `overwrite: false` a resubmit refuses when artifacts exist. Because the
merge chain is `defaults ← run.yml ← step_N.yml`, a per-step override is free:
`overwrite: false` in `step_6_refine.yml` protects twenty hours of folds without
new machinery.

The legacy `big_tf_unet` `overwrite=True` semantics — `setup_directory` calling
`shutil.rmtree` on the output directory — are **not** reinstated. `clearing.py`
exists specifically to replace that pattern, which "could wipe sibling
modalities". Clearing stays file-granular and `_assert_fenced` to the run's own
cache/model roots, so rerunning `step_3` for `co2` touches only `co2`.

### Notebooks

`notebook_api.py` becomes `view.py`, read-only. `submit`, `wait`, `run`,
`accept`, `clear`, `clear_all`, `create` are removed; `status`, `gallery`, `log`,
`suggest`, `jobstats` remain. `gallery.py` is untouched — `plt.show()` still
works because a notebook is displaying it.

Each of the nine notebooks is four cells:

```python
from tokeye.training.big_tf_unet_2 import view
v = view.here()          # run folder is the cwd — no run id to type
v.status()
```
```python
v.gallery("step_6")
```
```python
v.log("step_6")          # newest logs/<jobid>_*_step_6_refine.out
```

`view.here()` deriving the run from cwd is what keeps the template
substitution-free.

### co2

`modalities.co2` is re-enabled (uncommented in `defaults.yaml`) and arrays become
`0-3`. This is a deliberate override of the recorded reason for disabling it:

> co2 disabled: the preserved co2 top-up data is all-zero for every shot/chord.

`step_0_intake.py` already detects all-zero shots and `continue`s silently, which
would yield an empty h5 and a confusing step_1 failure. It now **raises after the
shot loop when a modality produced zero windows**, reporting how many shots were
skipped as all-zero. If the co2 re-fetch has not happened, that surfaces in
step_0 rather than after step_7 trains on zeros.

### Deleted

| What | Why |
|---|---|
| `slurm.py` | scripts are static files, not rendered |
| `scaffold.py` | `cp -r` replaces it |
| `notebooks/00_setup … 05_eval.ipynb` | replaced by per-step `.sh` + `.ipynb` |
| `config/run_template.yaml` | template folder replaces it |
| `notebook_api.py` | becomes read-only `view.py` |
| `RunConfig.slurm` / `SlurmSection` | resources live in `#SBATCH` headers |
| `task_matrix.accept` / `is_accepted` / `accepted` field | gate dropped |
| `RunPaths.slurm_dir` | nothing generates scripts |

## Error handling

| Situation | Behavior |
|---|---|
| step submitted before upstream complete | dies in seconds, names the missing step + modality |
| `--modalities` names something absent from `run.yml` | hard error (catches `--array` range drift) |
| `overwrite: false` with artifacts present | refuses, names the artifacts |
| modality yields zero windows | raises, reports all-zero shot count |
| venv not activated before `sbatch` | `No module named tokeye` in the log |
| `nfft` changed mid-run | `check_scale_lock` raises |

## Tests

Parametrized over the `STEPS` registry so they cannot drift:

- every step has a `.yml`/`.sh`/`.ipynb` triplet in the template
- each `.sh` references its own `.yml`; `--array` range matches `run.yml`'s
  modality count; every `.sh` has `--chdir` and a `logs/` `--output`
- each `.yml` contains exactly its step's `knob_section`
- the merge chain yields the same `RunConfig` as a single combined file
- the step is inferred correctly from each `.yml` stem
- `overwrite: false` refuses; `overwrite: true` clears only its own
  step + modality
- zero-window modality raises in step_0

### Fold-level resume for step_6

Included after all. step_6 is the only step marked `StepSpec.resumable`: K folds
trained in sequence, each worth hours, and each fold's slice of the output is
complete and independently valid the moment it is written.

- `step_6_refine.py::_open_output` reuses an existing output file when its shape
  and fold count match, returning the `folds_done` list and accumulated metrics.
  The fold loop skips those folds. Both attributes are written and `flush()`ed
  after each fold, so what a resumed run reads back is always committed.
- Correctness rests on the split being seeded (`random_state=42`): fold *k* holds
  out the same samples on every attempt.
- `runner._replace_artifacts` does not clear a resumable step **when the previous
  attempt ran with the same knobs**. `task_matrix.mark_running` now stores the
  params hash (not just `mark_complete`), and `params_match` compares it. Any knob
  change invalidates the partial and it is cleared — half a result computed under
  different settings is not a result.
- A fold interrupted mid-training restarts from scratch. Only whole folds are
  checkpointed; per-epoch resume is not attempted.
- `main` raises if any fold never finished, so a timed-out job fails rather than
  reporting success on a partial file.

Operationally: hit the wall clock, resubmit the same `.sh`, and it continues.
