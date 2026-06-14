#!/bin/bash
# Submit all pipeline phases with SLURM dependency chains.
# Re-submitting is safe — the task matrix skips completed combos.

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting multiscale pipeline..."

JOB0=$(sbatch --parsable "$DIR/step0_shared.sh")
echo "  step0_shared: $JOB0"

JOB0C=$(sbatch --parsable --dependency=afterok:$JOB0 "$DIR/step0c_modalities.sh")
echo "  step0c_modalities: $JOB0C"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB0C "$DIR/step2_cpu.sh")
echo "  step2_cpu: $JOB2"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 "$DIR/step3_gpu.sh")
echo "  step3_gpu: $JOB3"

JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 "$DIR/step4_5_6a_cpu.sh")
echo "  step4_5_6a_cpu: $JOB4"

JOB6=$(sbatch --parsable --dependency=afterok:$JOB4 "$DIR/step6_gpu.sh")
echo "  step6_gpu: $JOB6"

FINAL=$(sbatch --parsable --dependency=afterok:$JOB6 "$DIR/final_combined.sh")
echo "  final_combined: $FINAL"

echo ""
echo "Pipeline submitted: $JOB0 -> $JOB0C -> $JOB2 -> $JOB3 -> $JOB4 -> $JOB6 -> $FINAL"
echo "Monitor with: python -m tokeye.training.big_tf_unet_multiscale.orchestrator --status"
