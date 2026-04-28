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

# evaluate against method = classic

METHOD="${1:-glp}"
DATA_PATH="${DATA_PATH:-results/wildjailbreak_eval.json}"
OUT_DIR="${OUT_DIR:-results/test_steering_${METHOD}_wjb_large/}"
ALPHAS="${ALPHAS:-0.01 0.1 0.5 1.0}"
N_BENIGN="${N_BENIGN:-1024}"
N_ADVERSARIAL="${N_ADVERSARIAL:-1024}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GUARD_BATCH_SIZE="${GUARD_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_GPUS="${NUM_GPUS:-4}"

mkdir -p "$OUT_DIR"

COMMON_ARGS=(
    --method "$METHOD"
    --data_path "$DATA_PATH"
    --out_dir "$OUT_DIR"
    --alphas $ALPHAS
    --batch_size "$BATCH_SIZE"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --num_gpus "$NUM_GPUS"
)
if [[ -n "$N_BENIGN" ]]; then
    COMMON_ARGS+=(--n_benign "$N_BENIGN")
fi
if [[ -n "$N_ADVERSARIAL" ]]; then
    COMMON_ARGS+=(--n_adversarial "$N_ADVERSARIAL")
fi

PID_LIST=""
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching eval_steering run shard on GPU $gpu_id"
    python eval_steering.py \
        --mode run \
        --gpu_id "$gpu_id" \
        --gen_device "cuda:$gpu_id" \
        "${COMMON_ARGS[@]}" &
    PID_LIST+=" $!"
    sleep 5
done
trap "kill $PID_LIST" SIGINT
echo "Generating shards across $NUM_GPUS GPUs..."
wait $PID_LIST

echo "Aggregating shards + Llama Guard judging..."
python eval_steering.py \
    --mode aggregate \
    --method "$METHOD" \
    --out_dir "$OUT_DIR" \
    --guard_device "cuda:0" \
    --guard_batch_size "$GUARD_BATCH_SIZE"

echo "Done."
