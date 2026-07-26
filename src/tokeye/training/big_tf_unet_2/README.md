# big_tf_unet_2 — single-scale teacher training

This pipeline trains **one model per STFT resolution** (an `(nfft, hop)` pair).
You run it once per scale; the resulting per-scale "teacher" models are later
distilled into one multiscale student (not part of this pipeline).

You drive it from the shell, one `sbatch` per step. You never edit code — every
adjustable value (a "knob") lives in a YAML file next to the script that uses it,
and every step has a notebook that shows you pictures of what it produced.

## Starting a run

```bash
cd /scratch/gpfs/nc1514/tokeye
uv sync --group train                        # once
mkdir -p logs                                # SLURM needs it to exist

cp -r scripts/big_tf_unet_2 dev/training/nfft512_hop128
cd dev/training/nfft512_hop128
```

The copy **is** the setup — there is no scaffold command, and the folder name
becomes the run id. See `scripts/big_tf_unet_2/README.md` (which travels with the
copy) for the full operating instructions.

## The run folder

Flat, three files per step plus one shared:

```
run.yml                        # scale, modalities, data paths, overwrite
step_0_intake.{yml,sh,ipynb}
step_1_spectrogram.{yml,sh,ipynb}
...
step_8_eval.{yml,sh,ipynb}
```

`.yml` = what the step runs with, `.sh` = how it is submitted, `.ipynb` = what you
look at afterward. Each `.yml` owns exactly the config section named by that
step's `StepSpec.knob_section`; filenames are `StepSpec.module`. Both come from
the registry in `paths.py`, so the file set is derived rather than maintained.

## The loop

1. **Edit** — the knob you want, in that step's `.yml`.
2. **Submit** — `sbatch step_2_baseline.sh`, from inside the run folder.
3. **Look** — open `step_2_baseline.ipynb`; `v.gallery("step_2")` shows the
   results as images, `v.log("step_2")` tails the job output, and
   `v.suggest("step_2")` reports what the `auto` knobs resolved to.
4. **Decide** — good, move on; not good, edit the knob and resubmit. A resubmit
   replaces that step's artifacts and marks everything downstream stale.

Nothing chains the steps and nothing gates them — the pacing is yours. Submitting
out of order fails in seconds naming the missing upstream, so no GPU time burns.

## Config

```
config/defaults.yaml   <-   run.yml   <-   step_N_*.yml
```

A deep `OmegaConf.merge`, validated as one whole `RunConfig`, so a step still
reads `run.nfft` and `modalities` even though its own file holds one section.
Splitting the files does not split the config object. Bad values die before any
compute, naming the field and its allowed range.

Because the chain ends at the step's own file, any global setting can be
overridden for one step: `overwrite: false` in `step_6_refine.yml` protects that
step alone.

Resource requests are **not** in YAML. They are the `#SBATCH` lines of each `.sh`,
so `head -8 step_6_refine.sh` answers "what will this ask for?".

## Knob glossary (the ones you'll actually touch)

| Knob | File | Meaning |
|---|---|---|
| `baseline.lam` | `step_2_baseline.yml` | Background smoothness. Bigger = smoother estimate. `auto` scales it to your resolution. |
| `baseline.edge_k` | `step_2_baseline.yml` | How aggressively noisy edge frequency bins are cut. Bigger = fewer bins cut. |
| `labels.knee_sensitivity` | `step_4_labels.yml` | Mask threshold strictness. Bigger = higher threshold = fewer labeled pixels. |
| `labels.delta` | `step_4_labels.yml` | Direct threshold offset in noise-sigma units. |
| `labels.min_size_fraction` | `step_4_labels.yml` | Smallest object kept, as a fraction of the image. |
| `refine.model_trust` | `step_6_refine.yml` | 0 = trust your step_4 masks, 1 = trust the cross-validation models. Changing it only re-runs step_7. |
| `denoise.max_epochs` | `step_3_denoise.yml` | Denoiser training length. More = cleaner, slower. |
| `overwrite` | `run.yml` or any step | `false` makes a resubmit refuse instead of replacing artifacts. |

## If something fails

- Read the log: `logs/<jobid>_<step>.out`, or `v.log("step_3")` in the notebook.
  Most failures are a knob change plus a resubmit.
- `v.status()` shows every step's state, and the SLURM state of anything running.
- A job seems slow or stuck — `v.jobstats(<jobid>)` shows whether it is actually
  using its CPUs and GPU.
- Losing your SSH session costs nothing. The job belongs to the scheduler, and it
  records its own progress into `task_matrix.json` on shared storage from the
  compute node. Reconnect and carry on.

## For developers

Steps expose `main(settings: dict)`; settings are built centrally in `runner.py`
from the merged config, validated by `run_config.py`. `"auto"` knobs resolve in
`auto_resolve.py` and every resolved value is recorded with its source in the
run's `resolved_params.yaml`. Progress and staleness live in `task_matrix.json`
(`task_matrix.py`), written by the job itself so it survives losing the shell that
submitted it. `paths.py` is the single source of truth for the step registry and
artifact locations, including which workspace filenames a step owns. `view.py` is
the read-only notebook surface — it cannot submit or delete.

Every step is a batch job, including the light ones: uniform submission is worth
more than saving a queue wait, and step_0/step_1/step_5 do enough I/O that they do
not belong on a login node.
