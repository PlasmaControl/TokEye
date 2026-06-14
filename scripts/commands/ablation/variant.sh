#!/bin/bash
#SBATCH --job-name=abl_var
#SBATCH --output=logs/abl_var_%A_%a.out
#SBATCH --error=logs/abl_var_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3%2

# One ablation variant's GPU steps end-to-end (correlation denoise for all
# modalities -> combine -> refiner 5-fold -> 5-fold final surrogate). At most 2
# concurrent (2 A100s). step_2b (pure-CPU baseline) is run beforehand on a CPU
# node (step2b_cpu.sh) so the A100 is held only for actual GPU work; this job
# therefore starts at step_3a. The final cleanup runs at the end of step_6d.
cd "$SCRATCH/tokeye" && source .venv/bin/activate
# Reduce CUDA allocator fragmentation (the OOM message recommends this); harmless otherwise.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun python -m tokeye.training.big_tf_unet_ablation.orchestrator \
  --variant-index "$SLURM_ARRAY_TASK_ID" \
  --steps step_3a,step_3b,step_4a_coh,step_4a_tra,step_6a,step_6b,step_6c,step_6d ${ABL_CONFIG:+--config "$ABL_CONFIG"}
