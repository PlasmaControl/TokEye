#!/bin/bash
#SBATCH --job-name=eval_DCLDE
#SBATCH --output=logs/eval_DCLDE_%j.out
#SBATCH --error=logs/eval_DCLDE_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd $SCRATCH/tokeye
source .venv/bin/activate
python scripts/eval/DCLDE2011.py
