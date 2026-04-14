#!/bin/bash
#SBATCH -J ift6164-eval-probe
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate
python eval_linear_probe_simplified.py \
    --num_samples 1_000 \
    --batch_size 256 \
    --save_acts_batch_size 32 \
    --device "cuda:0" \
    --checkpoint_dir "results/iter_4/linear_probe_8b_10k" \
    --out_dir "results/linear_probe_eval" \
    "$@"
