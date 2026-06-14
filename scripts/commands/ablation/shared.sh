#!/bin/bash
#SBATCH --job-name=abl_shared
#SBATCH --output=logs/abl_shared_%j.out
#SBATCH --error=logs/abl_shared_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

# Shared prefix: extract all shots -> preemphasis -> window -> STFT ->
# model-based activity filter (0c/2a/2f loop over modalities). step_0a reads the
# raw per-shot H5s from raw_data_dir in faith_dataset_multiscale.yaml. Once
# shared/step_0b exists, step_0c uses it (the full shot set) rather than the
# 2-shot source cache.
cd "$SCRATCH/tokeye" && source .venv/bin/activate
srun python -m tokeye.training.big_tf_unet_ablation.orchestrator \
  --shared-steps step_0a,step_0b,step_0c,step_2a,step_2f
