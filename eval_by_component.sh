#!/bin/bash
#SBATCH -J ift6164-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:80gb:4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=2:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate

# --- parameters ---
NOISE_LEVEL=0.5
NUM_TIMESTEPS=500
LAYERS="[1, 3, 5, 7, 9, 11, 13, 15]"

OUT_DIR="results/sweep-layers${NOISE_LEVEL}-t${NUM_TIMESTEPS}"

# Part 1: extraction
PID_LIST=""
for gpu_id in 0 1 2 3; do
    echo "Launching on GPU $gpu_id"
    python eval_by_component.py \
        --gpu_id="$gpu_id" \
        --noise_level="$NOISE_LEVEL" \
        --num_timesteps="$NUM_TIMESTEPS" \
        --layers="$LAYERS" \
        --out_dir="$OUT_DIR" &
    PID_LIST+=" $!"
done
trap "kill $PID_LIST" SIGINT
echo "Extracting tensors..."
wait $PID_LIST

# Part 2: results
echo "Extraction completed, plotting..."
python aggregate_plot_by_component.py "$OUT_DIR" "${LAYERS//[\[\] ]/}"
echo "Done."