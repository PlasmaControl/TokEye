#!/bin/bash
#SBATCH --job-name=abl_shared_fm
#SBATCH --output=logs/abl_shared_fm_%j.out
#SBATCH --error=logs/abl_shared_fm_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

# Shared prefix for foundation_model-sourced shots ({shot}_processed.h5):
# step_0f loads+preemphasis+windows directly (replacing 0a/0b/0c), then STFT +
# model-based activity filter. Set paths.shots_path to a foundation shots file.
cd "$SCRATCH/tokeye" && source .venv/bin/activate
srun python -m tokeye.training.big_tf_unet_ablation.orchestrator \
  --shared-steps step_0f,step_2a,step_2f
