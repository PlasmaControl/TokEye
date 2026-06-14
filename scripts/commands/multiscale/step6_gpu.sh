#!/bin/bash
#SBATCH --job-name=ms_train
#SBATCH --output=logs/multiscale/step6_%A_%a.out
#SBATCH --error=logs/multiscale/step6_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-43

cd $SCRATCH/tokeye
mkdir -p logs/multiscale

srun uv run --group train python -m tokeye.training.big_tf_unet_multiscale.orchestrator \
    --combo-index $SLURM_ARRAY_TASK_ID \
    --steps step_6b,step_6c,step_6d
