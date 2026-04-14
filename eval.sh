#!/bin/bash
#SBATCH -J ift6164-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:80gb:4
#SBATCH --cpus-per-task=4
#SBATCH --constraint ampere
#SBATCH --mem=32G
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate
python eval.py