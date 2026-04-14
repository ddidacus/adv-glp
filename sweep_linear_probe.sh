#!/bin/bash
# Random search over lr in [1e-4, 5e-3] and train_batch_size in [32, 256].
# 8 random (lr, bs) combinations x 4 seeds = 32 runs total.

SEEDS=(42 137 2718 9999)
SWEEP_DIR="results/linear_probe_sweep"
mkdir -p "$SWEEP_DIR"

# Generate 8 random (lr, bs) pairs; sweep_seed=0 for reproducibility.
mapfile -t COMBOS < <(python3 - <<'EOF'
import numpy as np
rng = np.random.default_rng(0)
n = 8
lrs = np.exp(rng.uniform(np.log(1e-4), np.log(5e-3), n))
# batch size: power of 2 between 32 (2^5) and 256 (2^8)
bss = 2 ** rng.integers(5, 9, n)
for lr, bs in zip(lrs, bss):
    print(f"{lr:.2e} {bs}")
EOF
)

for combo in "${COMBOS[@]}"; do
    lr=$(echo "$combo" | awk '{print $1}')
    bs=$(echo "$combo" | awk '{print $2}')
    for seed in "${SEEDS[@]}"; do
        run_name="lr${lr}_bs${bs}_seed${seed}"
        out_dir="${SWEEP_DIR}/${run_name}"
        mkdir -p "$out_dir"
        echo "Submitting: $run_name"
        sbatch \
            --job-name="lp_${run_name}" \
            --output="${out_dir}/slurm-%j.out" \
            --error="${out_dir}/slurm-%j.err" \
            train_linear_probe.sh \
                --lr "$lr" \
                --train_batch_size "$bs" \
                --seed "$seed" \
                --out_dir "$out_dir"
    done
done

echo "All 32 jobs submitted. Results under ${SWEEP_DIR}/"
