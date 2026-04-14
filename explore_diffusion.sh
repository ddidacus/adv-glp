#!/bin/bash
#SBATCH -J ift6164-explore-diffusion
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=4:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate
python explore_diffusion_anomaly_detection.py \
    --num_samples 1000 \
    --num_samples_train 10000 \
    --seed 42 \
    "$@"
