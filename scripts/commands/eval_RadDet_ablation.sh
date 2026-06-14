#!/bin/bash
#SBATCH --job-name=eval_RadDet_ablation
#SBATCH --output=logs/eval_RadDet_ablation_%j.out
#SBATCH --error=logs/eval_RadDet_ablation_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd $SCRATCH/tokeye
source .venv/bin/activate
python scripts/eval/RadDet_ablation.py
