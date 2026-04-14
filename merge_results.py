"""Merge activations_good/reconstructed_good from one results dir
with activations_bad/reconstructed_bad from another.

Usage:
    python merge_results.py <good_dir> <bad_dir> <out_dir>

Example:
    python merge_results.py \
        results/fineweb-vs-easy-good-llama8b \
        results/fineweb-vs-easy-bad-llama8b \
        results/merged-easy
"""
import os
import sys
import glob
import torch
from pathlib import Path


def load_and_concat(results_dir, keys):
    files = sorted(glob.glob(os.path.join(results_dir, "results_*.th")))
    if not files:
        raise FileNotFoundError(f"No results_*.th files in {results_dir}")
    accum = {k: [] for k in keys}
    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=True)
        for k in keys:
            accum[k].append(data[k])
    return {k: torch.cat(v, dim=0) for k, v in accum.items()}


if __name__ == "__main__":
    dir_a = sys.argv[1]
    dir_b = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "results/merged-easy"

    print(f"Dir A: {dir_a}")
    print(f"Dir B: {dir_b}")
    print(f"Output: {out_dir}")

    # merge over gpu batches
    data_a = load_and_concat(dir_a, ["activations_good_set", "reconstructed_good_set"])
    data_b = load_and_concat(dir_b, ["activations_bad_set", "reconstructed_bad_set"])

    # merge over folders
    merged = {
        "activations_good_set": torch.cat([data_a["activations_good_set"], data_b["activations_bad_set"]], dim=0),
        "reconstructed_good_set": torch.cat([data_a["reconstructed_good_set"], data_b["reconstructed_bad_set"]], dim=0),
    }

    print(f"Merged good: {merged['activations_good_set'].shape}")
    print(f"Merged recon: {merged['reconstructed_good_set'].shape}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, "results_0.th")
    torch.save(merged, out_path)
    print(f"Saved to {out_path}")
