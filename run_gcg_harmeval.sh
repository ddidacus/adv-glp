#!/bin/bash
#SBATCH -J ift6164-gcg-harmeval
#SBATCH --nodes=1
#SBATCH --gres=gpu:40gb:4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=4:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate

# --- parameters ---
NUM_GPUS=4
NUM_SAMPLES=550          # total HarmEval questions (550 max)
TARGET="Sure, here is the answer:"
MODEL_ID="unsloth/Llama-3.2-1B"
NUM_STEPS=200
OUT_DIR="data/harmeval_gcg_512"

# Part 1: run GCG on each GPU in parallel
PID_LIST=""
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching GCG on GPU $gpu_id"
    python run_gcg_harmeval.py \
        --gpu_id="$gpu_id" \
        --num_gpus="$NUM_GPUS" \
        --num_samples="$NUM_SAMPLES" \
        --target="$TARGET" \
        --model_id="$MODEL_ID" \
        --num_steps="$NUM_STEPS" \
        --out_dir="$OUT_DIR" &
    PID_LIST+=" $!"
    sleep 1
done

trap "kill $PID_LIST" SIGINT
echo "Running GCG optimization across $NUM_GPUS GPUs..."
wait $PID_LIST
echo "All GPU jobs finished."

# Part 2: master rank merges shards
echo "Merging dataset shards..."
python run_gcg_harmeval.py \
    --merge \
    --out_dir="$OUT_DIR"
echo "Done. Dataset saved to $OUT_DIR/harmeval_gcg"
