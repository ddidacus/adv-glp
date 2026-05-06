#!/bin/bash
#SBATCH -J adv-prompts
#SBATCH --nodes=1
#SBATCH --gres=gpu:80gb:4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --mem=128G
#SBATCH --partition=long
#SBATCH --time=2:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache
export PYTHONUNBUFFERED=1

source .venv/bin/activate

NUM_GPUS=4
OUT_DIR="results/adv_prompts_shards"
FINAL_OUT="results/adv_prompts_eval.json"
N_SAMPLES="${N_SAMPLES:-}"   # optionally override via env: N_SAMPLES=200 sbatch ...

mkdir -p "$OUT_DIR"

# Warm HF cache serially to avoid parallel file-lock races
echo "Warming dataset cache..."
python -c "
from datasets import load_dataset
load_dataset('centrepourlasecuriteia/jailbreak-dataset')
"

# Launch one worker per GPU
PID_LIST=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD_OUT="$OUT_DIR/shard_${gpu_id}.json"
    extra=""
    if [ -n "$N_SAMPLES" ]; then
        extra="--n_samples $N_SAMPLES"
    fi
    python test_adv_prompts.py \
        --shard_id     "$gpu_id" \
        --num_shards   "$NUM_GPUS" \
        --gen_device   "cuda:${gpu_id}" \
        --guard_device "cuda:${gpu_id}" \
        --batch_size        256 \
        --guard_batch_size   64 \
        --max_new_tokens    200 \
        --out "$SHARD_OUT" \
        $extra \
        > "$OUT_DIR/shard_${gpu_id}.log" 2>&1 &
    PID_LIST+=($!)
    echo "  GPU $gpu_id -> PID ${PID_LIST[-1]}, out: $SHARD_OUT"
done

trap "echo 'Interrupted, killing workers...'; kill ${PID_LIST[*]}" SIGINT SIGTERM

echo "Waiting for $NUM_GPUS workers..."
FAILED=0
for pid in "${PID_LIST[@]}"; do
    if ! wait "$pid"; then
        echo "  Worker PID $pid failed (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED worker(s) failed. Check logs in $OUT_DIR/"
fi

# Aggregate shard results
echo "Aggregating results..."
python - <<'PYEOF'
import json, glob, sys, os

out_dir  = os.environ.get("OUT_DIR",   "results/adv_prompts_shards")
final    = os.environ.get("FINAL_OUT", "results/adv_prompts_eval.json")

shards = sorted(glob.glob(f"{out_dir}/shard_*.json"))
if not shards:
    print("No shard files found — check worker logs.", file=sys.stderr)
    sys.exit(1)

records = []
for path in shards:
    with open(path) as f:
        records.extend(json.load(f))

with open(final, "w") as f:
    json.dump(records, f, indent=2)

total    = len(records)
unsafe   = sum(r["safety_label"] == "unsafe" for r in records)
unknown  = sum(r["safety_label"] == "unknown" for r in records)
asr      = unsafe / total if total else 0.0

print(f"Saved {total} records to {final}")
print(f"  Attack Success Rate (ASR): {unsafe}/{total} = {100*asr:.1f}%")
if unknown:
    print(f"  Skipped (OOM during judging): {unknown}")
PYEOF

echo "Done."
