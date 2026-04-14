#!/bin/bash
#SBATCH -J ift6164-diffusion-probe
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=8:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate
python train_diffusion_probe_anomaly_detection.py \
    --num_samples_train 10_000 \
    --num_samples_eval 1_000 \
    --batch_size 1024 \
    --save_acts_batch_size 64 \
    --noise_level 0.5 \
    --num_timesteps 100 \
    --epochs 10 \
    --device "cuda:0" \
    --out_dir "results/diffusion_probe_10k" \
    "$@"
