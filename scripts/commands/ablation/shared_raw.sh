#!/bin/bash
#SBATCH --job-name=abl_shared
#SBATCH --output=logs/abl_shared_%j.out
#SBATCH --error=logs/abl_shared_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

# Shared prefix for the locally-preserved raw_fast shots
# (data/autoprocess/raw_fast/{shot}.h5, all 4 modalities). step_0g loads + resamples
# each modality to 500 kHz + preemphasis + windows (replacing 0a/0b/0c/0f), then STFT
# + model-based activity filter, looped over ece/mhr/bes/co2.
cd "$SCRATCH/tokeye" && source .venv/bin/activate
srun python -m tokeye.training.big_tf_unet_ablation.orchestrator \
  --shared-steps step_0g,step_2a,step_2f
