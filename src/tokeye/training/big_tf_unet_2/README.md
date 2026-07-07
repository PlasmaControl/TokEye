# big_tf_unet_2 — single-scale teacher training, human-in-the-loop

This pipeline trains **one model per STFT resolution** (an `(nfft, hop)` pair).
You run it once per scale; the resulting per-scale "teacher" models are later
distilled into one multiscale student (not part of this pipeline).

You drive the pipeline from Jupyter notebooks. You never edit code — every
adjustable value (a "knob") lives in one YAML file, and every step shows you
pictures so you can decide whether to keep its output or tweak a knob and
rerun it.

## One-time setup

```bash
cd /scratch/gpfs/nc1514/tokeye
uv sync --group train          # installs everything the pipeline needs
```

Open JupyterLab through the cluster's OnDemand portal (or however you usually
start notebooks), with this repository as the working directory. The kernel
must use the project environment (`.venv`).

## Starting a run

From the repository root:

```bash
uv run python -m tokeye.training.big_tf_unet_2.scaffold --nfft 512 --hop 128
```

This creates `dev/training/nfft512_hop128/` containing:

- **`run.yaml`** — the only file you ever edit. It starts with sensible
  defaults; most knobs say `auto`, meaning the pipeline computes a suggested
  value from your data and records what it picked.
- **`00_setup.ipynb` … `05_eval.ipynb`** — the six notebooks you work
  through, in order.

To try a different resolution, scaffold again with different numbers — each
scale is a completely separate run with its own folder, cache, and model.
Nothing you do in one run can affect another.

## The loop (same for every step)

1. **Suggest** — `run.suggest("step_2")` shows the auto-chosen values and
   tells you which section of `run.yaml` holds this step's knobs.
2. **Run** — light steps run right in the notebook
   (`run.run("step_4")`); heavy steps are submitted to the cluster
   (`job = run.submit("step_3")`). Re-run `run.status()` until the step says
   `complete`; `run.log(job)` shows the job's output.
3. **Look** — `run.gallery("step_2")` shows the step's results as images.
4. **Decide** — if it looks right, `run.accept("step_2")` unlocks the next
   step. If not: edit the knob in `run.yaml`, `run.clear("step_2")`, and run
   it again. Clearing a step automatically marks everything after it
   `stale`, so you can't accidentally train on outdated intermediate data.

## What the steps do

| Step | What it does | Where it runs |
|---|---|---|
| step_0 | load raw signals, cut into windows | notebook |
| step_1 | spectrograms + keep the most active windows | notebook |
| step_2 | remove the smooth background (baseline) | cluster (CPU) |
| step_3 | self-supervised denoiser | cluster (GPU) |
| step_4 | threshold into coherent/transient masks | notebook (seconds) |
| step_5 | pack all diagnostics into a training set | notebook |
| step_6 | 5-fold cross-validation "second opinion" on the masks | cluster (GPU) |
| step_7 | train + export the final model | cluster (GPU) |
| step_8 | quick TJ-II benchmark number | notebook |

## Knob glossary (the ones you'll actually touch)

| Knob (run.yaml) | Meaning |
|---|---|
| `baseline.lam` | Background smoothness. Bigger = smoother background estimate. `auto` scales it to your resolution. |
| `baseline.edge_k` | How aggressively noisy frequency bins at the spectrogram's top/bottom edges are cut. Bigger = fewer bins cut. |
| `labels.knee_sensitivity` | Mask threshold strictness. Bigger = higher threshold = fewer labeled pixels. |
| `labels.delta` | Direct threshold offset in noise-sigma units (0.5 = half a sigma stricter). |
| `labels.min_size_fraction` | Smallest object kept in a mask, as a fraction of the image. Bigger = more small specks removed. |
| `refine.model_trust` | 0 = trust your step_4 masks completely; 1 = trust the cross-validation models completely. Changing it only re-runs step_7 (cheap). |
| `denoise.max_epochs` | Denoiser training length. More = cleaner, slower. |

Bad values are caught the moment a step starts, with a message naming the
field and its allowed range — you cannot break anything by mistyping a knob.

## If something fails

- `run.status()` shows `failed` — `run.log(job)` (or the printed error for
  notebook steps) says why. Most failures are fixed by a knob change +
  `run.clear(step)` + rerun.
- A cluster job seems slow or stuck — `run.jobstats(job)` shows whether it's
  actually using its CPUs/GPU.
- You want to start a scale completely over —
  `run.clear_all(confirm="<run_id>")` (it makes you type the run id so it
  can't happen by accident). Your `run.yaml` and notebooks are kept.

## For developers

Steps expose `main(settings: dict)`; the settings are built centrally in
`runner.py` from the merged config (`config/defaults.yaml` ← `run.yaml`,
validated by `run_config.py`). `"auto"` knobs resolve in `auto_resolve.py`
and every resolved value is recorded with its source in the run's
`resolved_params.yaml`. Progress, human sign-off, and staleness live in
`task_matrix.json` (`task_matrix.py`). `paths.py` is the single source of
truth for step registry and artifact locations. Deployment inputs
(normalization stats, edge bins, scale) are exported per run in
`model/big_tf_unet_2/<run_id>/deploy_manifest.yaml`.
