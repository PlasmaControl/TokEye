#!/bin/bash
#SBATCH --job-name=ms_stft
#SBATCH --output=logs/multiscale/step2_%A_%a.out
#SBATCH --error=logs/multiscale/step2_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=64
#SBATCH --array=0-43

cd $SCRATCH/tokeye
mkdir -p logs/multiscale

srun uv run --group train python -m tokeye.training.big_tf_unet_multiscale.orchestrator \
    --combo-index $SLURM_ARRAY_TASK_ID \
    --steps step_2a,step_2b
