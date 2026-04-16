#!/bin/bash
#SBATCH -J ift6164-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:80gb:4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=3:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate

CONFIG="eval_config.yaml"

# Read values from config for use in this script
_py() { python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print($1)"; }
OUT_DIR=$(_py "c['out_dir']")
METHOD=$(_py "c['method']")
LAYERS_COMPACT=$(_py "','.join(str(x) for x in c['layers'])")

# Part 1: extraction
PID_LIST=""
for gpu_id in 0 1 2 3; do
    echo "Launching on GPU $gpu_id"
    python eval_by_component.py --config="$CONFIG" --gpu_id="$gpu_id" &
    PID_LIST+=" $!"
    sleep 1
done
trap "kill $PID_LIST" SIGINT
echo "Extracting tensors..."
wait $PID_LIST

# Part 2: results
echo "Extraction completed, plotting..."
python aggregate_plot_by_component.py --results_dir="$OUT_DIR" --layers="$LAYERS_COMPACT" --method="$METHOD"
echo "Done."
