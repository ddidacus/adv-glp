#!/bin/bash
#SBATCH -J ift6164-steer
#SBATCH --nodes=1
#SBATCH --gres=gpu:80gb:4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --mem=64G
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00

export HF_HOME=/home/p.greiner/workspace/adv-glp/.cache
export UV_CACHE_DIR=/home/p.greiner/workspace/adv-glp/.cache
export PYTHONUNBUFFERED=1

source .venv/bin/activate

CONFIG="${1:-configs/paper/steering/glp.yaml}"

# Detect GPUs from the SLURM allocation (fall back to nvidia-smi or 1)
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS="$SLURM_GPUS_ON_NODE"
elif command -v nvidia-smi &>/dev/null; then
    NUM_GPUS="$(nvidia-smi -L | wc -l)"
else
    NUM_GPUS=1
fi

PID_LIST=""
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching evaluate_steering run shard on GPU $gpu_id"
    python scripts/evaluate_steering.py run --config="$CONFIG" --gpu_id="$gpu_id" --num_gpus="$NUM_GPUS" &
    PID_LIST+=" $!"
    sleep 1
done
trap "kill $PID_LIST" SIGINT
echo "Generating shards across $NUM_GPUS GPUs..."
wait $PID_LIST

echo "Aggregating shards + Llama Guard judging..."
python scripts/evaluate_steering.py aggregate --config="$CONFIG" --num_gpus="$NUM_GPUS"

echo "Done."
