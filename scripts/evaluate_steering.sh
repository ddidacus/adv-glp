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

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache
export PYTHONUNBUFFERED=1

source .venv/bin/activate

CONFIG="${1:-configs/paper/steering/glp.yaml}"
NUM_GPUS="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('num_gpus', 4))")"

PID_LIST=""
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching evaluate_steering run shard on GPU $gpu_id"
    python scripts/evaluate_steering.py run --config="$CONFIG" --gpu_id="$gpu_id" &
    PID_LIST+=" $!"
    sleep 5
done
trap "kill $PID_LIST" SIGINT
echo "Generating shards across $NUM_GPUS GPUs..."
wait $PID_LIST

echo "Aggregating shards + Llama Guard judging..."
python scripts/evaluate_steering.py aggregate --config="$CONFIG"

echo "Done."
