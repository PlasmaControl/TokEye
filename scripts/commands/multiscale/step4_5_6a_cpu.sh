#!/bin/bash
#SBATCH --job-name=ms_thresh
#SBATCH --output=logs/multiscale/step456a_%A_%a.out
#SBATCH --error=logs/multiscale/step456a_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=64
#SBATCH --array=0-43

cd $SCRATCH/tokeye
mkdir -p logs/multiscale

srun uv run --group train python -m tokeye.training.big_tf_unet_multiscale.orchestrator \
    --combo-index $SLURM_ARRAY_TASK_ID \
    --steps step_4a,step_5a,step_6a
