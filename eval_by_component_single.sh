#!/bin/bash
#SBATCH -J ift6164-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:80gb:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=2:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache
# export PYTHONDONTWRITEBYTECODE=1

source .venv/bin/activate

python eval_by_component.py 0
