#!/bin/bash
#SBATCH --job-name=step_5_dataset
#SBATCH --output=logs/%j_%x.out
#SBATCH --chdir=/scratch/gpfs/nc1514/tokeye
#SBATCH --time=02:00:00
#SBATCH --mem=43G
#SBATCH --cpus-per-task=42

source .venv/bin/activate
python -m tokeye.training.big_tf_unet_2.runner \
    --config "$SLURM_SUBMIT_DIR/step_5_dataset.yml"
