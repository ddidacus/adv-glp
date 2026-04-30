#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate

NUM_GPUS="${NUM_GPUS:-8}"
RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR to the path containing *responses*.json files}"

PID_LIST=""
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching rejudge on GPU $gpu_id"
    python scripts/rejudge_responses.py run \
        --results_dir="$RESULTS_DIR" \
        --gpu_id="$gpu_id" \
        --num_gpus="$NUM_GPUS" &
    PID_LIST+=" $!"
    sleep 0.5
done
trap "kill $PID_LIST" SIGINT
echo "Rejudging..."
wait $PID_LIST

echo "Aggregating results..."
python scripts/rejudge_responses.py aggregate --results_dir="$RESULTS_DIR"

echo "Done."
