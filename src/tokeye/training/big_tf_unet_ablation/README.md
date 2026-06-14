# big_tf_unet_ablation

Fully-automated ablation of the TokEye self-supervised recipe, with TJ-II
uncertainty quantification. Answers the two reviewer asks: (1) ablate the
central claim (complex vs magnitude STFT representation; ± baseline removal;
± multichannel denoising) and (2) report 95% CI / std across 5 CV folds and
across images on recall/F1/IoU.

Design + plan: `docs/superpowers/specs/2026-05-28-tokeye-ablation-design.md`,
`docs/superpowers/plans/2026-05-28-tokeye-ablation.md`.

## Variants (leave-one-out from the full recipe)

| id           | denoiser repr   | baseline removal | multichannel denoise |
|--------------|-----------------|------------------|----------------------|
| `full`       | complex (Re,Im) | yes              | yes                  |
| `mag`        | magnitude \|Z\| | yes              | yes (central claim)  |
| `nobaseline` | complex         | no               | yes                  |
| `nodenoise`  | —               | yes              | no                   |

Each variant trains the final surrogate as **5 CV folds** for fold-level
uncertainty. Eval reports mean ± std / 95% CI across folds + image-level
bootstrap on TJ-II (n=493 images).

## Dataflow (verified against the original big_tf_unet)

```
SHARED (once, per modality):  step_0a extract -> step_0b preemphasis ->
  step_0c window -> step_2a complex STFT -> step_2f model-based window filter
PER VARIANT (per modality 2b-4a, then combined 6a-6d):
  step_2b baseline (toggle) -> [step_3a/3b denoiser (toggle, repr)] ->
  step_4a coherent  threshold( denoised step_3b | step_2b if nodenoise )
  step_4a transient threshold( step_2b_baseline )
  step_6a -> dual-mask TIF [ch0=coherent, ch1=transient], per-modality stats
  step_6b refiner (n-fold) -> step_6c refined labels -> step_6d 5-fold final
EVAL: scripts/eval/TJII2021_ablation.py -> across-fold + image-bootstrap CSV
      scripts/eval/tjii_ablation_figure.py -> figure + LaTeX table
```

## Launch (SLURM)

```bash
cd $SCRATCH/tokeye
# 1. shared prefix (extract -> window -> STFT -> activity filter), once
JID_SHARED=$(sbatch --parsable scripts/commands/ablation/shared.sh)
# 2. one variant per array task (0-3), <=2 concurrent A100s
JID_VAR=$(sbatch --parsable --dependency=afterok:$JID_SHARED scripts/commands/ablation/variant.sh)
# 3. evaluate every variant x fold on TJ-II + build figure/table
sbatch --dependency=afterok:$JID_VAR scripts/commands/ablation/eval.sh
# progress / resume (re-running skips completed steps via task_matrix.json)
python -m tokeye.training.big_tf_unet_ablation.orchestrator --status
```

Outputs: `model/ablation/<variant>/fold_<k>/`,
`data/eval/results/TJII2021_ablation{,_folds,_imageci}.csv`,
`data/eval/results/figures/tjii_ablation.png`,
`data/eval/results/tjii_ablation_table.tex`.

## Training data (IMPORTANT — read before launching the real run)

The paper's original ~23-shot training set is in
`data/autoprocess/settings/shots.txt`, but the **raw H5s for 21 of those 23
shots are no longer present** in `/scratch/gpfs/EKOLEMEN/d3d_fusion_data`
(which only holds shots 160904–163119; the originals are 170k–193k).

- `d3d_fusion_data` (compatible with `step_0a`, all 4 diagnostics): only
  `161172`, `161403` of the originals; plus 936 other shots in 160904–163119.
- `/scratch/gpfs/EKOLEMEN/foundation_model` (`{shot}_processed.h5`, different
  layout — each modality is a group with `xdata`/`ydata (C,N)`): has originals
  `190904` (bes/ece/mhr) and `193273/193277/193280/193281` (ece/mhr only).
  Broad ece+mhr coverage, but co2/bes are full in only ~25% of shots.

Current `paths.shots_path` = `shots_ablation.txt` (10 shots from
`d3d_fusion_data` with all 4 diagnostics) as a stand-in. **To run on the
original shots once their raw data is re-fetched:** repoint `paths.shots_path`
to a shots file in `d3d_fusion_data` format and re-launch. If the shots arrive
in `foundation_model` format instead, a small direct loader is needed (each
modality is `group/ydata (C,N)`) — not yet written; ask if needed.

Missing originals to re-fetch: 163518, 166419, 170660, 170670, 170672, 170677,
170679, 170796, 175987, 176054, 178631, 180634, 184847, 184859, 184902, 184964.
