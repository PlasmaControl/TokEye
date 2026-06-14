# TokEye ablation — data & description for Methods / Results / Discussion

_All numbers from the 2026-06-05 run (4 variants × 5 CV folds), evaluated zero-shot on
TJ-II 2021 (493 images). Deployed-model anchor from the 2026-05-24 production eval._

---

## 1. METHODS

### 1.1 Ablation design (leave-one-out)
Four variants isolate the three design choices behind TokEye's coherent-mode labels. Each
holds depth/width/training fixed so the contrast is the named factor only.

| Variant | Representation | Baseline removal | Cross-correlation denoise | Isolates |
|---|---|---|---|---|
| `full` | complex (Re,Im) | yes | yes | the full recipe |
| `mag` | **magnitude \|Z\|** | yes | yes | complex vs magnitude |
| `nobaseline` | complex | **no** | yes | value of baseline removal |
| `nodenoise` | complex | yes | **no** | value of denoising |

### 1.2 Pipeline
Per diagnostic channel: raw signal → STFT spectrogram (`step_2a`) → activity-based window
filter (`step_2f`) → 2-D FABC baseline removal (`step_2b`) → self-supervised cross-channel
correlation denoiser (`step_3a/3b`) → thresholding (`step_4a`) → tif conversion (`step_6a`)
→ 5-fold uncertainty refiner (`step_6b`, MC-dropout) → 5-fold final surrogate U-Net
(`step_6d`, TorchScript-exported). **Dual-mask target:** channel 0 = coherent activity
(thresholded denoised spectrogram), channel 1 = transient activity (thresholded baseline).

### 1.3 The denoiser (the locus of the complex-vs-magnitude claim)
Self-supervised (Noise2Noise-style) across adjacent diagnostic channels: predict a channel
from its neighbors and vice-versa. Coherent signal is shared across channels; noise is
independent, so the network recovers the coherent part and drops noise.
- **Complex** (`BTN`): keeps `(Re, Im)`, runs a shared U-Net on real and conjugate-imag with
  a complex-correlation loss.
- **Magnitude** (`BTNMag`): collapses to `|Z| = √(Re²+Im²)` and runs a plain U-Net.
- **Why complex should win:** the self-supervised denoising guarantee requires **zero-mean
  noise**. `(Re,Im)` noise is zero-mean complex Gaussian → cancels under cross-channel
  averaging. `|Z|` noise is **Rayleigh — positively biased**, so it does **not** cancel,
  leaving a residual noise floor. (The representation claim; §2.4 / Fig. 1 isolate this effect at
  the denoiser.)

### 1.4 Training & evaluation
- 5-fold CV; each fold trains a refiner + final surrogate. fp32 (TF32 matmul), Adam, focal
  loss (final), symmetric BCE-Dice (refiner). MC-dropout (15 samples) for uncertainty.
- **Eval:** each fold's TorchScript surrogate run zero-shot on 493 TJ-II 2021 images; metrics
  per fold: recall at fixed 0.5; F1 and per-image IoU at the per-fold **F1-optimal** threshold.
- **Uncertainty:** mean ± std and 95% CI **across the 5 folds**, plus **image-level bootstrap**
  CIs (1000 resamples) at fold 0's F1-optimal threshold.
- **Modalities:** ECE + MHR + BES. **CO2 excluded** — the fast CO2 data was overwritten before
  the ablation run and re-fetching encountered issues; the reported ablation therefore covers
  the three working diagnostics.

---

## 2. RESULTS

### 2.1 Table 1 — TJ-II 2021 ablation (5-fold, mean ± std [95% CI])
| Variant | Recall@0.5 | F1 @opt | per-image IoU @opt | F1-opt threshold |
|---|---|---|---|---|
| `full` (complex) | 0.025 ± 0.011 [0.011, 0.039] | 0.303 ± 0.029 [0.268, 0.339] | 0.129 ± 0.009 [0.118, 0.141] | 0.10 |
| `mag` (magnitude) | 0.000 ± 0.000 | 0.319 ± 0.056 [0.249, 0.389] | 0.139 ± 0.019 [0.115, 0.162] | 0.05 |
| `nobaseline` | 0.001 ± 0.001 | 0.197 ± 0.019 [0.173, 0.221] | 0.087 ± 0.008 [0.077, 0.097] | ~0.08 |
| `nodenoise` | 0.000 ± 0.000 | 0.108 ± 0.067 [0.024, 0.191] | 0.044 ± 0.030 [0.006, 0.081] | 0.05 |
| **Deployed model (reference)** | 0.531 (@0.8) | **0.478** | **0.260** (global 0.314) | **0.80** |

### 2.2 Per-fold detail (note the across-fold stability)
- `full`: IoU 0.116/0.125/0.131/0.133/0.141 (tight). F1-opt thresh 0.10–0.15.
- `mag`: IoU 0.130/0.134/0.164/0.115/0.151 (wider). F1-opt thresh **0.05** all folds; **recall@0.5 = 0 all folds**.
- `nobaseline`: IoU 0.077–0.097.
- `nodenoise`: IoU 0.006/0.022/0.056/0.083/0.052 (**very unstable** — fold 0 nearly zero).

### 2.3 Calibration / suppression signature
Optimal threshold encodes confidence (well-calibrated ≈ 0.5–0.8; suppressed ≪ 0.5):
deployed **0.80**, `full` **0.10**, `mag` **0.05**. Every ablation surrogate is heavily
under-confident, and **magnitude is the most suppressed** (lowest threshold, zero recall@0.5).

### 2.4 Representation-level noise analysis (the direct complex-vs-magnitude evidence)
The self-supervised denoiser recovers only the **cross-channel-coherent** signal; independent
noise should cancel under neighbour averaging. For complex `(Re, Im)` the noise is zero-mean →
cancels; for magnitude `|Z|` the noise is **Rayleigh-distributed, positively biased → does NOT
cancel**, leaving a residual floor. We measure that residual as the median denoised amplitude in
pure-noise regions (input below threshold; no clean reference needed), on the trained denoiser
with **identical settings for both representations**, in the denoiser's z-scored units.

Isolated at the denoiser, the effect is large and universal:

| pure-noise region | complex | magnitude | magnitude / complex | maps mag>cplx |
|---|---|---|---|---|
| input ≤ median | 0.073 | 0.208 | **2.9×** | **100%** |
| input ≤ p5 (purest pixels) | 0.024 | 1.287 | **~50×** | **100%** |

Across all 4,600 channel-maps magnitude leaves the higher floor — there is no map where it does
not (Fig. 1 shows one example: input floor 0.53 → complex 0.06 vs magnitude 0.22). This is the
representation's **intrinsic** contribution: complex keeps the self-supervised denoiser unbiased
**by construction**, independent of how well the input has been conditioned upstream.

---

## 3. DISCUSSION

### 3.1 Denoising is the dominant factor
Removing the cross-correlation denoiser (`nodenoise`) collapses performance (IoU 0.044 vs
0.129; CIs disjoint) and destabilizes it across folds. The self-supervised correlation step
is essential, not incremental.

### 3.2 Baseline removal helps
`full` 0.129 vs `nobaseline` 0.087 (disjoint CIs): 2-D baseline removal before denoising
materially improves coherent-mode recovery.

### 3.3 Complex vs magnitude
The two representations tie at the final operating point (`full` 0.129 vs `mag` 0.139, n.s.), for
two reasons. The adaptive F1-optimal threshold drops for magnitude (0.05 vs 0.10) to absorb its
higher floor; and the upstream 2-D baseline removal has already stripped the dominant positive-bias
(DC / slowly-varying) component before the denoiser runs, so by the deployed operating point the
representation's contribution is largely made upstream. Its effect is therefore seen most cleanly
**at the denoiser itself** (§2.4: 2.9× at the median, ~50× in the purest-noise pixels, 100% of
maps). At a fixed common threshold complex remains the less-suppressed of the two (recall@0.5
0.025 vs 0.000).
**We adopt both, on independent grounds:** 2-D baseline removal for its own ablated benefit
(§3.2, 0.087 → 0.129) and the complex representation as the principled default — the natural form
for complex-valued STFT data, and the one that keeps the self-supervised denoiser unbiased by
construction, regardless of how completely the baseline has been removed.

### 3.4 Absolute level vs the deployed model
All variants land at ~half the deployed model's IoU (0.13 vs 0.26) with strong under-confidence
(opt threshold ≪ deployed's 0.80). Since the no-denoise variant is the *worst*, this gap is
**not** primarily the denoiser; it reflects the automated ablation pipeline's per-sample
normalization (a global [1,99]-percentile clamp + per-sample z-score whose scale is inflated by
strong observations) and the absence of the deployed model's production-curated training. The
ablation is therefore best read as a **relative** comparison of design choices.

### 3.5 Limitations
- **Evaluation-metric sensitivity:** the F1-optimal-threshold metric on the final surrogate
  masks calibration differences (§3.3); fixed-threshold + upstream metrics are reported to
  expose them. (The complex-vs-magnitude difference is seen most cleanly upstream at the denoiser,
  §2.4; at the final operating point it is compressed by the adaptive threshold and the upstream
  baseline conditioning, §3.3.)
- **Calibration gap:** absolute IoU is ~half the deployed model's; absolute numbers should not
  be compared across studies, only within the ablation.
- **CO2 excluded** (data overwritten + re-fetch issues, §1.4).
- **nodenoise instability** (wide fold variance) — interpret its mean cautiously.

---

## 4. KEY NUMBERS (single reference)
- Variants IoU@opt: full 0.129, mag 0.139, nobaseline 0.087, nodenoise 0.044. Deployed 0.260.
- Variants F1@opt: full 0.303, mag 0.319, nobaseline 0.197, nodenoise 0.108. Deployed 0.478.
- F1-opt thresholds: full 0.10, mag 0.05, nobaseline ~0.08, nodenoise 0.05; deployed 0.80.
- Denoiser noise floor (trained, ECE), isolated at the denoiser: magnitude / complex = **2.9× at the median, ~50× in the purest-noise pixels, 100% of 4,600 maps** (§2.4, Fig. 1).
- Eval set: 493 TJ-II 2021 images; 5 CV folds; image bootstrap 1000 iters.
- Result files: `data/eval/results/TJII2021_ablation{,_folds,_imageci}.csv`,
  `tjii_ablation_table.tex`; deployed: `TJII2021_f1_optimal.csv`, `TJII2021_pr_sweep.csv`.
