#!/bin/bash
#SBATCH --job-name=ms_corr
#SBATCH --output=logs/multiscale/step3_%A_%a.out
#SBATCH --error=logs/multiscale/step3_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-43

cd $SCRATCH/tokeye
mkdir -p logs/multiscale

srun uv run --group train python -m tokeye.training.big_tf_unet_multiscale.orchestrator \
    --combo-index $SLURM_ARRAY_TASK_ID \
    --steps step_3a,step_3b
