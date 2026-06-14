#!/bin/bash
#SBATCH --job-name=ms_window
#SBATCH --output=logs/multiscale/step0c_%A_%a.out
#SBATCH --error=logs/multiscale/step0c_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --array=0-3

cd $SCRATCH/tokeye
mkdir -p logs/multiscale

srun uv run --group train python -m tokeye.training.big_tf_unet_multiscale.orchestrator \
    --modality-index $SLURM_ARRAY_TASK_ID \
    --steps step_0c,step_1a
