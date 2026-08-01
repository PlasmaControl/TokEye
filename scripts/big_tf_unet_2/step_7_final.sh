#!/bin/bash
#SBATCH --job-name=step_7_final
#SBATCH --output=logs/%j_%x.out
#SBATCH --chdir=/scratch/gpfs/nc1514/tokeye
#SBATCH --time=18:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

source .venv/bin/activate
python -m tokeye.training.big_tf_unet_2.runner \
    --config "$SLURM_SUBMIT_DIR/step_7_final.yml"
