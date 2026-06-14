#!/bin/bash
#SBATCH --job-name=ms_extract
#SBATCH --output=logs/multiscale/step0_%j.out
#SBATCH --error=logs/multiscale/step0_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

cd $SCRATCH/tokeye
mkdir -p logs/multiscale

srun uv run --group train python -m tokeye.training.big_tf_unet_multiscale.orchestrator \
    --shared-steps step_0a,step_0b
