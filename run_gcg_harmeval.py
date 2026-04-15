"""
Generate GCG adversarial suffixes for every HarmEval question and store the
results as a HuggingFace dataset shard, one shard per GPU.

Usage (single GPU):
    python run_gcg_harmeval.py --gpu_id=0 --num_gpus=1 --num_samples=550

Usage (merge shards after all GPU jobs finish):
    python run_gcg_harmeval.py --merge --out_dir=results/harmeval_gcg
"""

import os
import glob
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import nanogcg
from nanogcg import GCGConfig, ProbeSamplingConfig

# reuse harmeval_batches from eval_by_component
from eval_by_component import harmeval_batches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_models(model_id: str, device: str):
    draft_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device)
    draft_tokenizer = AutoTokenizer.from_pretrained(model_id)

    target_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    target_tokenizer = AutoTokenizer.from_pretrained(model_id)
    return draft_model, draft_tokenizer, target_model, target_tokenizer


def _build_config(draft_model, draft_tokenizer, num_steps: int) -> GCGConfig:
    probe_sampling_config = ProbeSamplingConfig(
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        r=64,
        sampling_factor=16,
    )
    return GCGConfig(
        probe_sampling_config=probe_sampling_config,
        num_steps=num_steps,
        early_stop=True,
    )


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

def run(
    gpu_id: int = 0,
    num_gpus: int = 1,
    num_samples: int = 550,
    target: str = "Sure, here is the answer:",
    model_id: str = "unsloth/Llama-3.2-1B",
    num_steps: int = 100,
    out_dir: str = "results/harmeval_gcg",
    merge: bool = False,
):
    if merge:
        _merge_shards(out_dir)
        return

    device = f"cuda:{gpu_id}"
    per_gpu = num_samples // num_gpus
    # last GPU picks up any remainder
    if gpu_id == num_gpus - 1:
        per_gpu = num_samples - gpu_id * (num_samples // num_gpus)
    skip = gpu_id * (num_samples // num_gpus)

    shard_dir = Path(out_dir) / f"shard_{gpu_id}"

    print(f"[GPU {gpu_id}] skip={skip}, per_gpu_samples={per_gpu}, target='{target}'")

    draft_model, draft_tokenizer, target_model, target_tokenizer = _load_models(
        model_id, device
    )
    config = _build_config(draft_model, draft_tokenizer, num_steps)

    # collect all questions for this GPU (batch_size=1 so we can process individually)
    questions: list[str] = []
    for batch in harmeval_batches(batch_size=1, num_samples=per_gpu, skip=skip):
        questions.extend(batch)

    print(f"[GPU {gpu_id}] Running GCG on {len(questions)} questions...")

    records = []
    for question in tqdm(questions, desc=f"GPU {gpu_id}"):
        result = nanogcg.run(target_model, target_tokenizer, question, target, config)
        records.append(
            {
                "question": question,
                "target": target,
                "attack": result.best_string,
                "adv_prompt": question + result.best_string,
                "best_loss": float(result.best_loss),
            }
        )

    dataset = Dataset.from_list(records)
    shard_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(shard_dir))
    print(f"[GPU {gpu_id}] Saved {len(dataset)} records to {shard_dir}")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _merge_shards(out_dir: str):
    shard_paths = sorted(glob.glob(str(Path(out_dir) / "shard_*")))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found under {out_dir}")

    print(f"Merging {len(shard_paths)} shards: {shard_paths}")
    shards = [load_from_disk(p) for p in shard_paths]
    merged = concatenate_datasets(shards)
    merged = merged.shuffle(seed=42)

    final_path = str(Path(out_dir) / "harmeval_gcg")
    merged.save_to_disk(final_path)
    print(f"Merged dataset ({len(merged)} records) saved to {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import fire
    fire.Fire(run)
