#!/bin/bash
#SBATCH --job-name=step_2_baseline
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --chdir=/scratch/gpfs/nc1514/tokeye
#SBATCH --time=08:00:00
#SBATCH --mem=43G
#SBATCH --cpus-per-task=42
#SBATCH --array=0-3   # one task per modality

source .venv/bin/activate
MODS=(ece mhr bes co2)
python -m tokeye.training.big_tf_unet_2.runner \
    --config "$SLURM_SUBMIT_DIR/step_2_baseline.yml" \
    --modalities "${MODS[$SLURM_ARRAY_TASK_ID]}"
