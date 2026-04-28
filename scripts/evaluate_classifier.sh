#!/bin/bash
#SBATCH -J ift6164-classifier
#SBATCH --nodes=1
#SBATCH --gres=gpu:80gb:4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --mem=32G
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate

CONFIG="${1:-configs/paper/eval_pi.yaml}"

PID_LIST=""
for gpu_id in 0 1 2 3; do
    echo "Launching eval_classifier on GPU $gpu_id"
    python scripts/evaluate_classifier.py run --config="$CONFIG" --gpu_id="$gpu_id" &
    PID_LIST+=" $!"
    sleep 5
done
trap "kill $PID_LIST" SIGINT
echo "Computing log-probs..."
wait $PID_LIST

echo "Aggregating results..."
python scripts/evaluate_classifier.py aggregate --out_dir="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['out_dir'])")"

echo "Done."
