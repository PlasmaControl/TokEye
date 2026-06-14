#!/bin/bash
#SBATCH --job-name=ms_final
#SBATCH --output=logs/multiscale/final_%j.out
#SBATCH --error=logs/multiscale/final_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

cd $SCRATCH/tokeye
mkdir -p logs/multiscale

srun uv run --group train python -m tokeye.training.big_tf_unet_multiscale.orchestrator \
    --combine-and-train
