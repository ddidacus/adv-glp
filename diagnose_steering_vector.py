"""
diagnose_steering_vector.py

Checks whether the steering vector (benign_mean - malicious_mean) captures a
meaningful direction or is essentially random noise.

Metrics reported per layer:
  - Fisher ratio:  (mu_b - mu_m)^2 / (sigma_b^2 + sigma_m^2)  along the sv
  - Projected distributions: mean and std of benign/malicious dot products with sv
  - Cosine similarity between sv and 100 random unit vectors (should be ~0)
  - Linear separability AUC along the sv projection

Usage:
  python diagnose_steering_vector.py [--layers 7 15] [--n_samples 200]
"""

import argparse

import numpy as np
import torch
from baukit import TraceDict
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_DATASET_PATH = "data/centreia_llama1b_prompts"
LLM_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


@torch.no_grad()
def collect_all_acts(
    prompts: list[str],
    llm,
    tokenizer,
    layer_indices: list[int],
    device: str,
    batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """Mean-pool over tokens, collect per-sample activations. Returns {li: (N, D)}."""
    all_acts = {li: [] for li in layer_indices}
    layer_names = [f"model.layers.{li}" for li in layer_indices]

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in batch
        ]
        enc = tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(device)
        mask = enc["attention_mask"].float()
        lengths = mask.sum(dim=1, keepdim=True)

        with TraceDict(llm, layers=layer_names, retain_output=True) as td:
            getattr(llm, "model", llm)(**enc)

        for li in layer_indices:
            out = td[f"model.layers.{li}"].output
            act = out[0] if isinstance(out, tuple) else out       # (B, S, D)
            act_mean = (act * mask[:, :, None]).sum(1) / lengths  # (B, D)
            all_acts[li].append(act_mean.float().cpu())

    return {li: torch.cat(all_acts[li], dim=0) for li in layer_indices}


def analyse_layer(
    layer_idx: int,
    benign_acts: torch.Tensor,   # (N_b, D)
    malicious_acts: torch.Tensor, # (N_m, D)
):
    sv = (benign_acts.mean(0) - malicious_acts.mean(0))
    sv_norm = sv.norm().item()
    sv_unit = sv / (sv_norm + 1e-8)

    # projections onto the steering vector
    proj_b = (benign_acts   @ sv_unit).numpy()   # (N_b,)
    proj_m = (malicious_acts @ sv_unit).numpy()  # (N_m,)

    mu_b, sigma_b = proj_b.mean(), proj_b.std()
    mu_m, sigma_m = proj_m.mean(), proj_m.std()

    fisher = (mu_b - mu_m) ** 2 / (sigma_b ** 2 + sigma_m ** 2 + 1e-8)

    labels = np.concatenate([np.ones(len(proj_b)), np.zeros(len(proj_m))])
    scores = np.concatenate([proj_b, proj_m])
    auc = roc_auc_score(labels, scores)

    # cosine sim of sv to 100 random unit vectors — expected ~0 if sv is meaningful
    rng = torch.Generator().manual_seed(0)
    rand_vecs = torch.randn(100, sv.shape[0], generator=rng)
    rand_vecs = rand_vecs / rand_vecs.norm(dim=1, keepdim=True)
    cos_rand = (rand_vecs @ sv_unit).abs().mean().item()

    # within-class variance along sv vs total variance
    all_proj = np.concatenate([proj_b, proj_m])
    var_total = all_proj.var()
    var_within = (len(proj_b) * proj_b.var() + len(proj_m) * proj_m.var()) / (len(proj_b) + len(proj_m))
    var_between = var_total - var_within

    print(f"\n{'='*60}")
    print(f"Layer {layer_idx}")
    print(f"{'='*60}")
    print(f"  Steering vector norm (unnormalised): {sv_norm:.4f}")
    print(f"  Projection μ (benign):    {mu_b:+.4f}  σ={sigma_b:.4f}")
    print(f"  Projection μ (malicious): {mu_m:+.4f}  σ={sigma_m:.4f}")
    print(f"  Gap (μ_b - μ_m):          {mu_b - mu_m:+.4f}")
    print(f"  Fisher ratio:             {fisher:.4f}  (>1 = good separation)")
    print(f"  AUC (sv as classifier):   {auc:.4f}  (0.5 = random, 1.0 = perfect)")
    print(f"  Var explained by sv:      {100*var_between/var_total:.1f}%")
    print(f"  Mean |cos(sv, random)|:   {cos_rand:.4f}  (expected ~{1/sv.shape[0]**0.5:.4f} for true random)")

    return {"fisher": fisher, "auc": auc, "gap": mu_b - mu_m,
            "sigma_b": sigma_b, "sigma_m": sigma_m,
            "var_explained_pct": 100 * var_between / var_total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=[7, 15])
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    print(f"Loading {LLM_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID, torch_dtype=torch.bfloat16
    ).to(args.device).eval()

    print(f"Loading dataset from {LOCAL_DATASET_PATH}...")
    ds = load_from_disk(LOCAL_DATASET_PATH)
    benign_prompts    = [r["prompt"] for r in ds if r["label"] == "benign"]
    malicious_prompts = [r["prompt"] for r in ds if r["label"] == "adversarial_successful"]

    if args.n_samples:
        benign_prompts    = benign_prompts[:args.n_samples]
        malicious_prompts = malicious_prompts[:args.n_samples]

    print(f"  {len(benign_prompts)} benign, {len(malicious_prompts)} malicious")

    print("\nCollecting benign activations...")
    benign_acts = collect_all_acts(
        benign_prompts, llm, tokenizer, args.layers, args.device, args.batch_size
    )
    print("Collecting malicious activations...")
    malicious_acts = collect_all_acts(
        malicious_prompts, llm, tokenizer, args.layers, args.device, args.batch_size
    )

    for li in args.layers:
        analyse_layer(li, benign_acts[li], malicious_acts[li])


if __name__ == "__main__":
    main()
