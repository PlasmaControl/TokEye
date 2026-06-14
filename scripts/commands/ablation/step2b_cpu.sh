#!/bin/bash
#SBATCH --job-name=abl_2b
#SBATCH --output=logs/abl_2b_%j.out
#SBATCH --error=logs/abl_2b_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42
#SBATCH --mem=43G
# No --partition / no --gres: this is CPU-only work, so it must not request a
# GPU (an idle GPU allocation tanks fairshare priority). SLURM routes it to the
# default partition and schedules it on a CPU node. Cores/mem are right-sized to
# observed usage: 64c/128G ran at ~55% CPU (one writer thread can't feed 64
# workers) and ~25% mem, so -> 42 cores (2/3) and 43G (1/3).

# step_2b (FABC 2D baseline) is pure CPU and was stranding the A100 at 0% for
# hours when bundled into the GPU variant job. Run it here on a CPU node (NO GPU
# allocated) for every variant, parallelized across all allocated cores: step_2b
# spawns SLURM_CPUS_PER_TASK workers, so the cores stay ~100% busy and no GPU is
# held. --no-cleanup preserves the step_2b / step_2b_baseline outputs that the
# downstream GPU job (step_3a+) consumes; that GPU job runs the final cleanup.
# Reads the shared cache (step_0g/2a/2f), which is already built.
cd "$SCRATCH/tokeye" && source .venv/bin/activate
for v in 0 1 2 3; do
  echo "=== step_2b: variant $v ==="
  srun python -m tokeye.training.big_tf_unet_ablation.orchestrator \
    --variant-index "$v" --steps step_2b --no-cleanup ${ABL_CONFIG:+--config "$ABL_CONFIG"}
done
