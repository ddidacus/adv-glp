"""
For each prompt, run diffusion reconstruction K times and compute the variance
of the reconstruction vector across runs. This measures how stochastic the
denoising is for benign vs. harmful prompts.
"""
import json
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Iterator
import fire

from glp.denoiser import load_glp
from glp.utils_acts import save_acts
from glp import flow_matching

############ Config

BATCH_SIZE = 8  # safe for 8B model on L40S

############ Dataset iterators (same permutation logic as training scripts)


def vanilla_wildjailbreak_batches(batch_size: int = BATCH_SIZE, num_samples: int = 64,
                                   benign=False, seed: int = 42, skip: int = 0) -> Iterator[list[str]]:
    label = "benign" if benign else "harmful"
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    filtered = dataset.filter(lambda x: x["data_type"] == f"vanilla_{label}")
    all_prompts = [sample["vanilla"] for sample in filtered["train"]]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_prompts))
    prompts = [all_prompts[i] for i in indices[skip : skip + num_samples]]
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]


############ Core: K-run reconstruction variance


@torch.no_grad()
def compute_reconstruction_variance(
    texts: list[str],
    k: int,
    noise_level: float,
    num_timesteps: int,
    llm_model,
    llm_tokenizer,
    diffusion_model,
    save_acts_batch_size: int = BATCH_SIZE,
    device: str = "cuda:0",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each text:
      1. Extract LLM activations once (deterministic).
      2. Run noise → denoise K times (each with fresh noise).
      3. Compute per-prompt variance across the K reconstruction vectors.

    Returns:
      scalar_var : (N,)  mean per-dimension variance per prompt
      per_dim_var: (N, D) full per-dimension variance per prompt
    """
    # 1. Extract activations — done once, deterministic
    activations = save_acts(
        hf_model=llm_model,
        hf_tokenizer=llm_tokenizer,
        text=texts,
        tracedict_config=diffusion_model.tracedict_config,
        token_idx="last",
        batch_size=save_acts_batch_size,
    )
    activations = activations.to(device=device, dtype=torch.bfloat16)
    normalized_acts = diffusion_model.normalizer.normalize(activations)

    # 2. K independent denoising runs (different noise each time)
    reconstructions = []
    for run in range(k):
        noise = torch.randn_like(normalized_acts)
        noisy_acts, _, timesteps, _ = flow_matching.fm_prepare(
            diffusion_model.scheduler,
            normalized_acts,
            noise,
            u=torch.ones(normalized_acts.shape[0]) * noise_level,
        )
        reconstructed = flow_matching.sample_on_manifold(
            diffusion_model,
            noisy_acts,
            start_timestep=timesteps[0].item(),
            num_timesteps=num_timesteps,
        )
        reconstructed = diffusion_model.normalizer.denormalize(reconstructed)
        flat = reconstructed.reshape(reconstructed.shape[0], -1).float().cpu()
        reconstructions.append(flat)
        if (run + 1) % 8 == 0:
            print(f"    run {run + 1}/{k}")

    # 3. Stack → (K, N, D) → variance over K
    stacked = torch.stack(reconstructions, dim=0)   # (K, N, D)
    per_dim_var = stacked.var(dim=0)                # (N, D)
    scalar_var = per_dim_var.mean(dim=1)            # (N,)  mean variance per prompt
    return scalar_var, per_dim_var


def distribution_stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "var":  float(arr.var()),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
    }


############ Main


def main(
    num_samples: int = 64,
    num_samples_train: int = 10000,
    k: int = 32,
    noise_level: float = 0.5,
    num_timesteps: int = 100,
    seed: int = 42,
    save_acts_batch_size: int = BATCH_SIZE,
    device: str = "cuda:0",
    llm_model_id: str = "unsloth/Meta-Llama-3.1-8B",
    glp_model_id: str = "generative-latent-prior/glp-llama8b-d6",
    out_dir: str = "results/reconstruction_variance",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading LLM: {llm_model_id}")
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)

    print(f"Loading GLP: {glp_model_id}")
    diffusion_model = load_glp(glp_model_id, device=device, checkpoint="final")

    sources = {
        "vanilla_benign":  vanilla_wildjailbreak_batches(save_acts_batch_size, num_samples, benign=True,  seed=seed, skip=num_samples_train),
        "vanilla_harmful": vanilla_wildjailbreak_batches(save_acts_batch_size, num_samples, benign=False, seed=seed, skip=num_samples_train),
    }

    results = {}

    for src_name, batch_iter in sources.items():
        print(f"\n=== {src_name} ===")
        all_scalar_vars: list[float] = []

        for i, texts in enumerate(batch_iter):
            print(f"  batch {i + 1}: {len(texts)} prompts  (K={k} runs each)")
            scalar_var, _ = compute_reconstruction_variance(
                texts=texts,
                k=k,
                noise_level=noise_level,
                num_timesteps=num_timesteps,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                diffusion_model=diffusion_model,
                save_acts_batch_size=save_acts_batch_size,
                device=device,
            )
            all_scalar_vars.extend(scalar_var.tolist())
            torch.cuda.empty_cache()

        results[src_name] = {
            "variances": all_scalar_vars,
            "distribution": distribution_stats(all_scalar_vars),
            "num_samples": len(all_scalar_vars),
        }
        print(f"  distribution: {results[src_name]['distribution']}")

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
