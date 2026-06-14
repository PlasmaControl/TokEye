#!/bin/bash
#SBATCH --job-name=abl_eval
#SBATCH --output=logs/abl_eval_%j.out
#SBATCH --error=logs/abl_eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Evaluate every variant x fold on TJ-II and build the figure + LaTeX table.
cd "$SCRATCH/tokeye" && source .venv/bin/activate
srun python scripts/eval/TJII2021_ablation.py ${ABL_CONFIG:+"$ABL_CONFIG"}
srun python scripts/eval/tjii_ablation_figure.py
