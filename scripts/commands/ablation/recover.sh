#!/bin/bash
#SBATCH --job-name=abl_recov
#SBATCH --output=logs/abl_recov_%A_%a.out
#SBATCH --error=logs/abl_recov_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16

# Recovery for the three ablation variants that failed in run 2807966 (the
# fourth -- nodenoise / index 3 -- completed on its own). Array index == variant
# index, and the step list is chosen per index:
#
#   0 full, 1 mag : the 5-fold step_6b refiner predictions (~280GB) were fully
#       written before the step_6c HOST-RAM OOM. That OOM is fixed in
#       utils/parmap.py (workers now default to the SLURM allocation, not all
#       128 node cores). Reuse the saved predictions -> rerun only step_6c (it
#       rmtree's its partial tif dir and regenerates) + step_6d. step_6a tifs
#       and the step_6b h5 are both intact and verified.
#
#   2 nobaseline : crashed at step_3a startup on the old new-API TF32 setting
#       (now fixed to the legacy API torch.set_float32_matmul_precision("high")
#       + cudnn.allow_tf32). Nothing salvageable -> full step_3a -> step_6d
#       re-run; its per-modality step_2b inputs from the CPU job are intact.
#
# Disk-safe submission (the 280GB step_6b h5 files overflowed the 3TB scratch
# quota and crashed step_6c on the first attempt; cleanup now deletes step_6b
# after step_6d): submit full+mag (--array=0-1) FIRST -- each frees its ~280GB
# refiner h5 when it finishes -- then submit nobaseline (--array=2) with
# --dependency=afterok on the full+mag job so there's room for its 280GB write.
# final.batch_size raised to 28 (GPU-mem fill); correlation 8 / refiner 16 kept.
cd "$SCRATCH/tokeye" && source .venv/bin/activate
# Reduce CUDA allocator fragmentation (the OOM message recommends this).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$SLURM_ARRAY_TASK_ID" in
  0|1) STEPS="step_6c,step_6d" ;;
  2)   STEPS="step_3a,step_3b,step_4a_coh,step_4a_tra,step_6a,step_6b,step_6c,step_6d" ;;
  *)   echo "recover.sh: unexpected array index $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

srun python -m tokeye.training.big_tf_unet_ablation.orchestrator \
  --variant-index "$SLURM_ARRAY_TASK_ID" \
  --steps "$STEPS"
