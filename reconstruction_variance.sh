#!/bin/bash
#SBATCH -J ift6164-recon-variance
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=4:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate
python reconstruction_variance.py \
    --num_samples 64 \
    --num_samples_train 10000 \
    --k 32 \
    --noise_level 0.5 \
    --num_timesteps 100 \
    --seed 42 \
    --save_acts_batch_size 8 \
    --llm_model_id "unsloth/Meta-Llama-3.1-8B" \
    --glp_model_id "generative-latent-prior/glp-llama8b-d6" \
    --out_dir "results/iter_3/reconstruction_variance" \
    "$@"
