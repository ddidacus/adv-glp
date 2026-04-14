#!/bin/bash
#SBATCH -J ift6164-explore-diffusion-8b
#SBATCH --nodes=1
#SBATCH --gres=gpu:40gb:1
#SBATCH --constraint ampere
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=8:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate
python explore_diffusion_anomaly_detection.py \
    --num_samples 1000 \
    --seed 42 \
    --llm_model_id "unsloth/Meta-Llama-3.1-8B" \
    --glp_model_id "generative-latent-prior/glp-llama8b-d6" \
    --save_acts_batch_size 8 \
    --num_layers 32 \
    "$@"

source .venv/bin/activate
python explore_diffusion_anomaly_detection.py \
    --num_samples 16 \
    --skip_samples_idx 0 \
    --seed 42 \
    --llm_model_id "unsloth/Meta-Llama-3.1-8B" \
    --glp_model_id "generative-latent-prior/glp-llama8b-d6" \
    --save_acts_batch_size 8 \
    --num_layers 32 \
    "$@"
