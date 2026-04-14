import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Iterator
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import fire

from glp.denoiser import load_glp
from glp.utils_acts import save_acts
from glp import flow_matching

############ Config

BATCH_SIZE = 32

############ Dataset iterators (identical to eval_linear_probe_simplified.py)


def eval_fineweb_batches(batch_size: int = BATCH_SIZE, num_samples: int | None = None, skip: int = 0) -> Iterator[list[str]]:
    """Yield batches of strings from HuggingFaceFW/fineweb, skipping the first `skip` samples.
    If num_samples is None, yield all remaining samples."""
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    batch: list[str] = []
    skipped = 0
    total = 0
    for sample in dataset:
        if skipped < skip:
            skipped += 1
            continue
        if num_samples is not None and total >= num_samples:
            break
        batch.append(sample["text"])
        total += 1
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def eval_wildjailbreak_batches(batch_size: int = BATCH_SIZE, num_samples: int | None = None, benign=False, data_type_prefix="vanilla", skip: int = 0, seed: int = 42) -> Iterator[list[str]]:
    """Yield batches of prompts from allenai/wildjailbreak (randomly sampled, non-overlapping with training).
    If num_samples is None, yield all remaining samples after skip."""
    label = "benign" if benign else "harmful"
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    filtered = dataset.filter(lambda x: x["data_type"] == f"{data_type_prefix}_{label}")
    col = "vanilla" if data_type_prefix == "vanilla" else "adversarial"
    all_prompts = [sample[col] for sample in filtered["train"]]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_prompts))
    selected = indices[skip:] if num_samples is None else indices[skip : skip + num_samples]
    prompts = [all_prompts[i] for i in selected]
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]


############ Activation extraction (via diffusion reconstruction)


@torch.no_grad()
def collect_reconstruction_errors(
    batch_iter: Iterator[list[str]],
    llm_model,
    llm_tokenizer,
    diffusion_model,
    label: int,
    noise_level: float,
    num_timesteps: int,
    save_acts_batch_size: int = 32,
    device: str = "cuda:0",
    name: str = "",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute per-sample L2 reconstruction error from the diffusion model and pair with labels."""
    acts_list, labels_list = [], []
    n_samples = 0
    for i, texts in enumerate(batch_iter):
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
        noise = torch.randn_like(normalized_acts)
        noisy_acts, _, timesteps, _ = flow_matching.fm_prepare(
            diffusion_model.scheduler,
            normalized_acts,
            noise,
            u=torch.ones(normalized_acts.shape[0]) * noise_level,
        )
        reconstructed_acts = flow_matching.sample_on_manifold(
            diffusion_model,
            noisy_acts,
            start_timestep=timesteps[0].item(),
            num_timesteps=num_timesteps,
        )
        reconstructed_acts = diffusion_model.normalizer.denormalize(reconstructed_acts)

        # Per-sample L2 reconstruction error: shape (N, 1)
        errors = torch.norm(
            (activations - reconstructed_acts.to(activations.dtype)).reshape(activations.shape[0], -1),
            p=2, dim=1,
        ).float().cpu().unsqueeze(-1)
        acts_list.append(errors)
        labels_list.append(torch.full((errors.shape[0],), label, dtype=torch.float32))
        n_samples += errors.shape[0]
        print(f"  [{name}] batch {i+1}: {n_samples} samples so far")
        torch.cuda.empty_cache()
    return acts_list, labels_list


############ Metrics


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "num_samples": int(len(y_true)),
    }


############ Main


def main(
    num_samples_train: int = 1024,
    batch_size: int = BATCH_SIZE,
    save_acts_batch_size: int = 32,
    device: str = "cuda:0",
    seed: int = 42,
    checkpoint_dir: str = "results/diffusion_probe_baseline",
    out_dir: str = "results/diffusion_probe_eval",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load checkpoint ----
    print(f"Loading diffusion probe checkpoint from {checkpoint_dir}...")
    ckpt = torch.load(os.path.join(checkpoint_dir, "linear_probe.pt"), map_location=device)
    input_dim = ckpt["input_dim"]
    mean = ckpt["mean"].to(device)
    std = ckpt["std"].to(device)
    noise_level = ckpt["noise_level"]
    num_timesteps = ckpt["num_timesteps"]
    probe = torch.nn.Sequential(
        torch.nn.Linear(input_dim, input_dim * 2),
        torch.nn.ReLU(),
        torch.nn.Linear(input_dim * 2, 1),
    ).to(device)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()
    print(f"  noise_level={noise_level}  num_timesteps={num_timesteps}")

    # ---- Load LLM ----
    print("Loading LLM...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map=device
    )
    llm_tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")

    # ---- Load GLP ----
    print("Loading GLP diffusion model...")
    diffusion_model = load_glp("generative-latent-prior/glp-llama1b-d6", device=device, checkpoint="final")

    ckwargs = dict(
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        diffusion_model=diffusion_model,
        noise_level=noise_level,
        num_timesteps=num_timesteps,
        save_acts_batch_size=save_acts_batch_size,
        device=device,
    )

    # ---- Define data sources (same splits as eval_linear_probe_simplified.py) ----
    sources = {
        "fineweb_holdout": (
            eval_fineweb_batches(batch_size, skip=num_samples_train),
            0,
        ),
        "vanilla_benign": (
            eval_wildjailbreak_batches(batch_size, benign=True, data_type_prefix="vanilla", skip=num_samples_train, seed=seed),
            0,
        ),
        "vanilla_harmful": (
            eval_wildjailbreak_batches(batch_size, benign=False, data_type_prefix="vanilla", skip=num_samples_train, seed=seed),
            1,
        ),
        "adversarial_benign": (
            eval_wildjailbreak_batches(batch_size, benign=True, data_type_prefix="adversarial", seed=seed),
            0,
        ),
        "adversarial_harmful": (
            eval_wildjailbreak_batches(batch_size, benign=False, data_type_prefix="adversarial", seed=seed),
            1,
        ),
    }

    # ---- Collect reconstructed activations & predictions per source ----
    print("\n=== Extracting eval activations (diffusion-reconstructed) ===")
    source_preds = {}
    source_labels = {}

    for src_name, (batch_iter, label) in sources.items():
        print(f"\n--- Source: {src_name} (label={label}) ---")
        acts, labels = collect_reconstruction_errors(
            batch_iter, label=label, name=src_name, **ckwargs,
        )
        if not acts:
            print(f"  WARNING: no samples collected for {src_name}, skipping")
            continue

        X = torch.cat(acts, dim=0).to(device)
        y = torch.cat(labels, dim=0)

        X = (X - mean) / std.clamp(min=1e-8)

        with torch.no_grad():
            logits = probe(X).squeeze(-1)
            preds = (logits > 0).cpu().numpy().astype(int)
        y_np = y.numpy().astype(int)

        source_preds[src_name] = preds
        source_labels[src_name] = y_np
        print(f"  collected {len(preds)} samples")
        torch.cuda.empty_cache()

    # ---- Compute metrics on mixed benign+harmful eval groups ----
    print("\n=== Evaluation (mixed benign+harmful per group) ===")
    eval_groups = {
        "vanilla": ["vanilla_benign", "vanilla_harmful"],
        "adversarial": ["adversarial_benign", "adversarial_harmful"],
    }

    results = {}
    all_preds, all_labels_list = [], []

    for group_name, src_names in eval_groups.items():
        group_preds = np.concatenate([source_preds[s] for s in src_names if s in source_preds])
        group_labels = np.concatenate([source_labels[s] for s in src_names if s in source_labels])
        group_metrics = compute_metrics(group_labels, group_preds)
        results[group_name] = group_metrics
        all_preds.append(group_preds)
        all_labels_list.append(group_labels)
        print(f"\n  {group_name}: accuracy={group_metrics['accuracy']:.4f}  precision={group_metrics['precision']:.4f}  recall={group_metrics['recall']:.4f}  f1={group_metrics['f1']:.4f}  n={group_metrics['num_samples']}")

    all_preds = np.concatenate(all_preds)
    all_labels_arr = np.concatenate(all_labels_list)
    results["overall"] = compute_metrics(all_labels_arr, all_preds)

    print("\n=== Overall Results ===")
    for k, v in results["overall"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # ---- Save results ----
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    fire.Fire(main)
