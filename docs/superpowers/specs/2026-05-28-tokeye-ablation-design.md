# TokEye ablation pipeline + TJ-II uncertainty quantification — Design

**Date:** 2026-05-28
**Author:** Nathaniel Chen (with Claude)
**Status:** Approved-pending-spec-review

## 1. Context & motivation

`tokeyeball-2.pdf` makes a mechanistic central claim (Methods §4.2.3, Discussion):
the self-supervised denoising pipeline must operate on the **complex (real+imag)
STFT representation**, because that is "the only representation under which the
noise assumptions required by self-supervised denoising hold pointwise" (real and
imaginary parts of complex Gaussian noise are each zero-mean Gaussian; magnitude
is biased-Rayleigh, phase wraps mod 2π).

Two reviewer asks (the "single biggest acceptance lever" and a Nature requirement):

1. **Ablation of the central claim** on TJ-II (which has pixel ground truth):
   complex vs magnitude-only representation; ± baseline removal; ± multichannel
   denoising. (The reviewer's original list also included magnitude+phase and
   ±MPMS; per author decision these two axes are **dropped**.)
2. **Uncertainty quantification**: report 95% CI / std across the 5 CV folds **and**
   across images on recall / F1 / IoU, with explicit *n* — not point estimates.

The original training pipeline (`src/tokeye/training/big_tf_unet/`) was ported from
a custom repo and is **not automatable as-is**: it has eyeballed constants
(`step_3a` clamp `(0.01,1.9)` and `data[:, :4]=0.75`; `step_4a` `adjust=0.05`;
`step_6c` thresholds `0.25/0.45`), a **blocking manual Tkinter GUI** (`step_5b`),
and **file proliferation** (per-window joblibs that doubled in `step_2b`) that
previously exhausted the 3 TB / max-file-count quota.

**Key leverage:** the sibling `src/tokeye/training/big_tf_unet_multiscale/` already
solved the automation and storage problems — `utils/auto_params.py` replaces every
eyeballed constant with data-driven detection; everything is HDF5 (not millions of
joblibs); the manual GUI step is gone; and a config-driven orchestrator +
file-locked `task_matrix.json` already ran end-to-end through `step_2b` on a 35 GB
cache. We **reuse that infrastructure** and strip the multiscale-specific parts.

## 2. Goals / non-goals

**Goals**
- A fully-automated, no-manual-intervention pipeline `big_tf_unet_ablation` that runs
  the recipe end-to-end (label-gen → 5-fold surrogate train → TJ-II eval) under each
  ablation toggle.
- A TJ-II ablation eval that reports across-fold and across-image uncertainty.
- Publication-ready CSV + LaTeX table + figure with error bars.
- Storage stays within quota via HDF5 + window reduction + cleanup.

**Non-goals**
- The standalone multiscale (44-combo) training pipeline — **not** going in the paper.
  MPMS as a *standalone* training regime is out of scope. (Note: ±MPMS as an
  ablation axis is also dropped per author decision.)
- Re-deriving the paper's headline TJ-II numbers at full ~5000-shot scale per
  variant. Headline numbers remain attributed to the existing full model; the
  ablation measures **relative** component contributions at a fixed reduced scale.

## 3. Ablation variants (leave-one-out from the full recipe)

All variants share the same source data, same window selection, same STFT scale
(`nfft=1024, hop=128`), and the same training budget — only the named component
changes, so differences are attributable to that component.

| ID           | Denoiser representation | Baseline removal (§4.2.2) | Multichannel denoise (§4.2.3) | What it isolates |
|--------------|-------------------------|---------------------------|-------------------------------|------------------|
| `full`       | complex `(Re,Im)`       | ✓                         | ✓                             | reference recipe |
| `mag`        | **magnitude `|Z|`**     | ✓                         | ✓                             | **central thesis**: complex vs magnitude |
| `nobaseline` | complex `(Re,Im)`       | ✗                         | ✓                             | broadband-coherent separation stage |
| `nodenoise`  | — (n/a)                 | ✓                         | ✗                             | whether self-supervised denoising helps at all |

`mag` definition (confirmed): the denoiser U-Net consumes the **magnitude**
`|Z| = sqrt(Re²+Im²)` as a single component per channel (k-channel in → 1-channel
out, MAE loss on magnitude), versus the complex path's two components `(Re,Im)`
(2k-channel in → 2-channel out). Downstream stages (threshold/combine/refine/final)
are identical — they already operate on magnitude.

**4 variants × 5 CV folds = 20 surrogate trainings.**

## 4. Architecture

New package `src/tokeye/training/big_tf_unet_ablation/`. Reuses multiscale `utils/`
wholesale; replaces the STFT-grid combo matrix with an ablation matrix; adds a
shared model-based window filter; makes the final surrogate 5-fold for UQ.

### 4.1 Shared-prefix / per-variant-branch execution graph

```
SHARED (computed once, identical for all variants):
  step_0a  extract timeseries (DONE in existing cache)
  step_0b  preemphasis filter   (DONE)
  step_0c  window into subsequences   (+ generous pre-cap)
  step_2a  complex STFT (C,F,T,2), single scale 1024/128
  step_2f  *** NEW: model-based window activity filter → keep ≤25 windows/shot ***
            (writes a reduced step_2a containing only kept windows)

PER VARIANT v in {full, mag, nobaseline, nodenoise} (branch from shared step_2a):
  step_2b  baseline removal           → step_2b.h5 (coherent input) + step_2b_baseline.h5 (transient input)
                                       [baseline toggle: if nobaseline, DON'T subtract baseline
                                        from the coherent path; still compute baseline.h5 for transient]
  step_3a  multichannel denoiser      [toggle: skip if v == nodenoise;
                                       repr=complex|magnitude per v]  reads step_2b.h5
  step_3b  unbatch denoiser output    [skip if nodenoise]
  step_4a-coherent  knee threshold    → step_4a_threshold.h5   (input = step_3b, OR step_2b if nodenoise)
  step_4a-transient knee threshold    → step_4a_threshold_baseline.h5  (input = step_2b_baseline)
  step_5a  combine per-shot/channel   img=shared step_2a_filtered; coherent + transient masks
  step_6a  → TIF + per-dir norm stats; DUAL mask [ch0=coherent, ch1=transient]
  step_6b  refiner (n_folds CV, MC-dropout) → refined labels
  step_6c  convert predictions → refined label set
  step_6d  *** final surrogate trained as 5 CV folds → 5 checkpoints ***

CANONICAL DATAFLOW NOTE (verified against original big_tf_unet step_6a.validate_directory):
  the 2-channel target is [coherent = threshold(denoised step_3b),
  transient = threshold(step_2b_baseline)]. step_4a therefore runs TWICE per variant
  (coherent + transient branches). The rough multiscale orchestrator wired ONLY the
  transient branch (step_4a input = step_2b_baseline) and never consumed the denoiser
  output — that is a bug we must NOT inherit. TJ-II GT is coherent modes and eval uses
  sigmoid(out[:,0]); the coherent branch is what the central claim hinges on.

EVAL:
  TJII2021_ablation.py → per-variant × per-fold TJ-II metrics + bootstrap CIs
```

Note: `step_1a` (the original copy/branch step) is folded away — `step_2a` reads
directly from `step_0c` output as in the multiscale flow.

### 4.2 Directory layout

```
src/tokeye/training/big_tf_unet_ablation/
  __init__.py
  README.md
  config/
    ablation.yaml          # single source of truth (see §10)
    variants.py            # AblationVariant dataclass + build_variants()
  ablation_matrix.py       # enumerate the 4 variants, map index→variant
  orchestrator.py          # CLI: --shared-steps / --variant-index / --eval / --status
  task_matrix.py           # reused/adapted from multiscale (file-locked progress)
  window_filter.py         # NEW: load existing TokEye model, score+select windows
  step_0a_*.py … step_6d_*.py   # adapted from multiscale; toggles added to 2b/3a/4a/6d
  utils/                   # symlink/import of multiscale utils (auto_params, hdf5_io, …)
```

Cache: `data/cache/ablation/{shared, full, mag, nobaseline, nodenoise}/…`
Models: `model/ablation/{variant}/fold_{k}/best_model.ckpt`
Results: `data/eval/results/TJII2021_ablation*.csv`

## 5. Component toggles (precise wiring)

- **± baseline (`step_2b`)** — config `baseline.enabled`. When false, output
  `log1p(|component|)` with the same auto edge-masking but **skip** `_fit_baseline`
  and the `(sxx - baseline)/(baseline+ε)` normalization. Output keeps `(C,F,T,2)`
  shape so downstream is unchanged. (Per-variant `input_h5`/`output_h5` set by
  orchestrator.)
- **± denoise (`step_3a`/`3b`/`4a`)** — config `denoise.enabled`. When false, the
  orchestrator skips `step_3a`/`step_3b` and points `step_4a.input_h5` at the
  `step_2b` output (or shared `step_2a` if also `nobaseline`).
- **representation (`step_3a`)** — config `denoise.representation: complex|magnitude`.
  `complex` = existing `BTN` (real/imag through shared U-Net, `flip(-1)` imag).
  `magnitude` = new `BTNMag`: compute `|Z|` from `(Re,Im)`, single-component U-Net
  that **mirrors the complex path's channel arithmetic** (adjacent-channel input →
  target-channel output, same width/depth/budget so the contrast is representation,
  not capacity), MAE loss without the imag `flip(-1)`; `SpecDatasetMag` returns
  `(C,F,T)`.

Toggles are pure config; no code path is chosen by environment/heuristic.

## 6. NEW: model-based window activity filter (`step_2f` / `window_filter.py`)

**Why:** many windows are near-empty; statistical thresholding can't distinguish
them from noise, and the paper's knee-point method *requires* ≥1 signal present —
empty windows produce garbage labels. Filtering both cuts volume and improves label
quality.

**Fairness:** the filter uses the **existing full-recipe TokEye model** and is
applied **identically across all 4 variants** (it selects *which* windows everyone
trains on; it does not generate labels). It therefore cannot bias the *relative*
comparison — it shifts all variants' absolute numbers equally. This is documented
as a fixed data-selection step, analogous to "train on the informative subset."

**Mechanism (per modality, per shot):**
1. For each window's complex spectrogram `(C,F,T,2)`, compute magnitude per channel
   `|Z|`, log1p, normalize (per-sensor or eval MEAN/STD).
2. Run the existing model (`build_model`, weights resolved with the eval's candidate
   list + local fallback `model/big_tf_unet_251210_weights.pt`), 1 channel at a time;
   take `sigmoid(out[:,0]) + sigmoid(out[:,1])` (coherent+transient).
3. Activity score per window = aggregate over channels of `mean(active_pixels)`
   above a fixed score threshold (config `window_filter.activity_threshold`).
4. Drop windows below `window_filter.min_activity` (near-empty floor), then keep the
   top `window_filter.max_windows_per_shot` (= **25**) by score.
5. Write the kept windows to a reduced shared HDF5 `step_2a_filtered.h5` (only kept
   windows, same `(C,F,T,2)` format); every variant's `step_2b` reads this file.

Config-gated (`window_filter.enabled`, default true). Runs on the V100 head node
(model inference ≈ 0.5 s/shot per paper).

## 7. Data-volume reduction

The blowup is over-cutting + 500 kHz oversampling, not shot count (already 22 shots).
Levers, applied identically across variants:
- **`step_2f` window filter → ≤25 windows/shot** (primary lever; also quality).
- **Single STFT scale** (1024/128) vs the 11-scale multiscale grid → ~11× less.
- **Cleanup policy** (from multiscale config): keep only `step_6a`–`6d`, delete
  `step_0c`–`step_5a` after each variant completes. HDF5 throughout.
- Optional generous pre-cap in `step_0c` to bound STFT work before filtering.

## 8. Uncertainty quantification

- **Across folds (model variability):** `step_6d` trains the final surrogate as
  **5 CV-fold models** per variant (each on 4/5 of that variant's self-supervised
  label set). Evaluate all 5 on TJ-II → 5 metric sets → report **mean ± std and
  95% CI across folds** for recall, F1, per-image IoU.
- **Across images (data variability):** existing image-level bootstrap in
  `src/tokeye/extra/eval/sweep.py` (`PRSweep.bootstrap_ci`) per fold; pooled.
- **Explicit n:** n_folds = 5, n_images = 493 (TJ-II), reported in table/caption.
- Operating points: recall at default thr 0.5; F1 & IoU at the F1-optimal thr from
  the PR sweep (consistent with the paper's existing protocol).

## 9. Evaluation & outputs

- `scripts/eval/TJII2021_ablation.py` (mirrors `RadDet_ablation.py`): loops variants
  × folds, runs TJ-II, writes:
  - `TJII2021_ablation.csv` — per variant: recall/F1/IoU mean ± std/CI across folds.
  - `TJII2021_ablation_folds.csv` — raw per-fold rows (for transparency).
  - `TJII2021_ablation_ci.csv` — image-bootstrap CIs.
- `scripts/eval/ablation_figure.py` (extend existing): forest/bar plot with error
  bars (recall, F1, IoU per variant) + LaTeX table for the paper.
- `scripts/commands/ablation_train.sh`, `ablation_eval.sh` — SLURM wrappers.

## 10. Config schema (`config/ablation.yaml`, sketch)

```yaml
stft: { nfft: 1024, hop_length: 128 }            # single scale
modalities: { co2: [...], mhr: [...], ece: [...], bes: [...] }   # reuse multiscale
extraction: { subseq_len: 66000, preemphasis_coeff: 0.99, fs_khz: 500, ... }
window_filter:
  enabled: true
  weights: "/scratch/gpfs/nc1514/aemodes/model/big_mode_v1-5_weights.pt"
  weights_fallback: "model/big_tf_unet_251210_weights.pt"
  max_windows_per_shot: 25
  activity_threshold: 0.5
  min_activity: 0.0005          # fraction of pixels; drop near-empty windows
baseline: { method: fabc, method_kwargs: { lam: 1.0e5 }, bin_cutting: auto }
correlation: { first_layer_size: 32, num_layers: 5, clamp_range: auto, ... }
threshold: { min_size: auto, remove_bottom_rows: auto, ... }
refiner:  { n_folds: 5, loss_type: symmetric_bce_dice, max_epochs: 200, ... }
final:    { n_folds: 5, loss_type: focal, max_epochs: 100, ... }   # 5-fold for UQ
variants:
  - { id: full,       baseline: true,  denoise: true,  representation: complex }
  - { id: mag,        baseline: true,  denoise: true,  representation: magnitude }
  - { id: nobaseline, baseline: false, denoise: true,  representation: complex }
  - { id: nodenoise,  baseline: true,  denoise: false, representation: complex }
paths: { cache_dir: data/cache/ablation, model_dir: model/ablation, ... }
cleanup: { enabled: true, keep_steps: [step_6a, step_6b, step_6c, step_6d] }
smoke:   { enabled: false, n_shots: 2, max_windows_per_shot: 2, n_folds: 2, max_epochs: 1 }
```

## 11. Orchestration / SLURM / storage

- `orchestrator.py` modes: `--shared-steps` (0a–2f, V100/CPU), `--variant-index N`
  (SLURM array 0–3: runs 2b–6d for one variant), `--eval`, `--status`.
- `task_matrix.json` (file-locked) tracks shared steps + per-variant/per-fold
  completion → fully resumable; re-running skips done work.
- Heavy training (`step_3a`, `6b`, `6d`) on **A100** SLURM; STFT/filter/threshold/
  eval on **V100** head node. Smoke config runs entirely on V100s.

## 12. Execution plan

1. Build the package, toggles, window filter, 5-fold final, eval, SLURM scripts.
2. **Smoke test** on V100s (`smoke.enabled=true`): 2 shots, 2 windows, 2 folds,
   1 epoch, all 4 variants → verify end-to-end and that eval emits a table.
3. On green smoke: **launch the full ablation** on A100 SLURM; monitor `task_matrix`;
   iterate on failures.
4. Produce `TJII2021_ablation.csv`, the figure, and the LaTeX table.

## 13. Risks & mitigations

- **`magnitude` denoiser underperforms trivially** (different #channels/capacity):
  match U-Net width/depth and training budget to the complex path so the contrast is
  representation, not capacity. Document.
- **Filter leaks full-recipe bias:** applied identically to all variants → relative
  comparison unaffected; documented as fixed data selection.
- **Storage:** HDF5 + 25-window cap + per-variant cleanup; monitor `du` between
  variants; `task_matrix` lets us run variants serially if needed.
- **Multiscale `step_6a` had path bugs / explorer flagged stubs:** `6b`/`6c` are
  actually complete (727/229 lines w/ `main()`); `6a` path construction will be
  rewritten for the ablation cache layout regardless.
- **Refiner cost (5-fold × 200 ep × 4):** early-stopping + reduced data keep epochs
  fast; `n_folds`/`max_epochs` are config knobs; smoke uses 2/1.

## 14. Reuse map

| Reused as-is (from multiscale `utils/`) | Adapted | New |
|---|---|---|
| `auto_params.py`, `hdf5_io.py`, `configuration.py`, `losses.py`, `augmentations.py`, `parmap.py` | `orchestrator.py`, `task_matrix.py`, `step_0c/2a/2b/3a/3b/4a/5a/6a/6b/6c/6d` | `config/ablation.yaml`, `config/variants.py`, `ablation_matrix.py`, `window_filter.py`, `BTNMag` (mag denoiser), 5-fold `step_6d`, `scripts/eval/TJII2021_ablation.py`, `scripts/eval/ablation_figure.py` (extend), SLURM wrappers |
```
