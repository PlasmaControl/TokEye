# big_tf_unet_2 — run folder

This folder is one training run: one STFT resolution, one set of results. It is a
template until you copy it.

```bash
cd /scratch/gpfs/nc1514/tokeye
cp -r scripts/big_tf_unet_2 dev/training/nfft512_hop128
cd dev/training/nfft512_hop128
```

**The folder name is the run id.** Artifacts land in
`data/cache/big_tf_unet_2/<folder name>/` and `model/big_tf_unet_2/<folder
name>/`. Copy the template again under a different name for a different
resolution or a throwaway experiment; runs cannot touch each other.

## Running it

Activate the environment once per login — the `.sh` files rely on inheriting it:

```bash
source /scratch/gpfs/nc1514/tokeye/.venv/bin/activate
```

Then, from inside the run folder, one step at a time:

```bash
sbatch step_0_intake.sh
```

Check it before submitting the next one — `squeue -u $USER`, the log in
`logs/`, or the step's notebook. Then continue: `step_1_spectrogram.sh`,
`step_2_baseline.sh`, … through `step_8_eval.sh`.

Nothing chains the steps and nothing gates them. Submitting out of order is safe:
the job dies within seconds naming the upstream step it needs.

## The three files per step

| Extension | What it is |
|---|---|
| `.yml` | the knobs this step runs with — the only thing you edit |
| `.sh` | how it is submitted; resource requests are the `#SBATCH` lines |
| `.ipynb` | read-only: look at what the step produced |

`run.yml` holds what is not per-step: the STFT scale, which diagnostics to use,
where the raw data lives.

To change a knob, edit that step's `.yml` and resubmit. A resubmit replaces that
step's artifacts and marks everything downstream stale, so nothing can train on
outdated intermediates.

```bash
vim step_4_labels.yml
sbatch step_4_labels.sh
```

Set `overwrite: false` in a step's `.yml` once its result is expensive and good —
a resubmit will then refuse rather than clobber it.

`step_6_refine` is the exception, because it is the long one: it trains `n_folds`
models in sequence. If its job hits the wall clock, **just resubmit the same
script** — the folds it already finished are kept and it continues from the one it
died on. Changing a knob in `step_6_refine.yml` invalidates that partial result
and restarts from fold 0, which is the point: half a result computed under
different settings is not a result.

## Logs

Job output goes to the repo's flat `logs/`, job id first:

```
logs/2884567_0_step_3_denoise.out     # array task 0 of job 2884567
logs/2884571_step_6_refine.out
```

`logs/` must exist before the first `sbatch` — SLURM creates the file, not the
directory:

```bash
mkdir -p /scratch/gpfs/nc1514/tokeye/logs
```

## Looking at results

```bash
jupyter lab step_3_denoise.ipynb
```

Each notebook opens the run it sits in (`view.here()` — no run id to type) and
shows status, the gallery, the log tail, and the auto-resolved knob values. It
cannot submit or delete anything.

Because the pipeline is driven by `sbatch`, losing your SSH session costs you
nothing: the job is owned by the scheduler, and progress is recorded by the job
itself into `data/cache/big_tf_unet_2/<run>/task_matrix.json`. Reconnect and
carry on.

## The steps

| Step | What it does | Where |
|---|---|---|
| `step_0_intake` | load raw signals, resample, cut into windows | CPU, per modality |
| `step_1_spectrogram` | spectrograms + keep the most active windows | CPU, per modality |
| `step_2_baseline` | remove the smooth background | CPU, per modality |
| `step_3_denoise` | self-supervised cross-channel denoiser | GPU, per modality |
| `step_4_labels` | threshold into coherent/transient masks | CPU, per modality |
| `step_5_dataset` | pack all diagnostics into one training set | CPU |
| `step_6_refine` | 5-fold out-of-fold second opinion on the masks | GPU |
| `step_7_final` | train + export the final model | GPU |
| `step_8_eval` | TJ-II benchmark number | CPU |

Per-modality steps run as a job array, one task per diagnostic, in parallel. If
you remove a diagnostic from `run.yml`, narrow `--array` in those five `.sh`
files to match — a task index past the end of the modality list fails loudly
rather than silently doing the wrong thing.

## Using a different checkout

Every `.sh` hardcodes the repo root in `#SBATCH --chdir`, which is what lets the
scripts find `logs/` and `.venv/` with no path arithmetic. If your checkout is
elsewhere, rewrite them once:

```bash
sed -i 's|--chdir=/scratch/gpfs/nc1514/tokeye|--chdir=/your/path/tokeye|' *.sh
```
