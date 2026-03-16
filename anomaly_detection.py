import json
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Iterator
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glp.denoiser import load_glp
from glp.utils_acts import save_acts
from glp import flow_matching

# (1) Load the LLM
llm_model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map="cuda:0")
llm_tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")

# (2) Load the diffusion model
diffusion_model = load_glp("generative-latent-prior/glp-llama1b-d6", device="cuda:0", checkpoint="final")

############ Batch reconstruction error

BATCH_SIZE = 32
NOISE_LEVEL = 0.5
NUM_TIMESTEPS = 100


@torch.no_grad()
def compute_reconstruction_error(texts: list[str], batch_size: int = BATCH_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a list of strings, extract LLM activations, noise-denoise via GLP,
    and return (per-sample L2 errors, flattened activations).
    """
    # Extract activations for the full list (save_acts handles internal batching)
    activations = save_acts(
        hf_model=llm_model,
        hf_tokenizer=llm_tokenizer,
        text=texts,
        tracedict_config=diffusion_model.tracedict_config,
        token_idx="last",
        batch_size=batch_size,
    )
    # activations shape: (N, num_layers, dim)
    activations = activations.to(device="cuda:0", dtype=torch.bfloat16)

    # Normalize
    normalized_acts = diffusion_model.normalizer.normalize(activations)

    # Forward noising
    u = NOISE_LEVEL
    noise = torch.randn_like(normalized_acts)
    noisy_acts, _, timesteps, _ = flow_matching.fm_prepare(
        diffusion_model.scheduler,
        normalized_acts,
        noise,
        u=torch.ones(normalized_acts.shape[0]) * u,
    )

    # Backward denoising
    reconstructed_acts = flow_matching.sample_on_manifold(
        diffusion_model,
        noisy_acts,
        start_timestep=timesteps[0].item(),
        num_timesteps=NUM_TIMESTEPS,
    )

    # Denormalize
    reconstructed_acts = diffusion_model.normalizer.denormalize(reconstructed_acts)
    reconstructed_acts = reconstructed_acts.to(device=activations.device, dtype=activations.dtype)

    # Flatten activations to (N, num_layers * dim) for downstream PCA
    flat_acts = activations.reshape(activations.shape[0], -1).cpu().float()

    # Per-sample L2 norm over all dimensions except batch
    errors = torch.norm(
        (activations - reconstructed_acts).reshape(activations.shape[0], -1),
        p=2,
        dim=1,
    )
    return errors, flat_acts


############ Dataset iterators

def fineweb_batches(batch_size: int = BATCH_SIZE, num_samples: int = 1024) -> Iterator[list[str]]:
    """Yield batches of strings from HuggingFaceFW/fineweb (sample-10BT, train split)."""
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    batch: list[str] = []
    total = 0
    for sample in dataset:
        if total >= num_samples:
            break
        batch.append(sample["text"])
        total += 1
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def wildjailbreak_batches(batch_size: int = BATCH_SIZE, num_samples: int = 1024) -> Iterator[list[str]]:
    """Yield batches of adversarial prompts from allenai/wildjailbreak."""
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    adv_harmful = dataset.filter(lambda x: x["data_type"] == "adversarial_harmful")
    adversarial_prompts = [sample["adversarial"] for sample in adv_harmful["train"]]
    adversarial_prompts = adversarial_prompts[:num_samples]
    for i in range(0, len(adversarial_prompts), batch_size):
        yield adversarial_prompts[i : i + batch_size]


############ Aggregation

def aggregate_errors(errors: list[float]) -> dict:
    """Return min, max, median, mean of a flat list of reconstruction errors."""
    arr = np.array(errors, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
    }


############ Main: aggregate reconstruction errors over datasets

NUM_SAMPLES = 1024


def run_evaluation(name: str, batch_iter: Iterator[list[str]]) -> tuple[list[float], np.ndarray, dict]:
    """Returns (pointwise_errors, activations_matrix, aggregate_stats)."""
    all_errors: list[float] = []
    all_acts: list[torch.Tensor] = []
    for i, batch in enumerate(batch_iter):
        errors, flat_acts = compute_reconstruction_error(batch)
        all_errors.extend(errors.cpu().float().tolist())
        all_acts.append(flat_acts)
        torch.cuda.empty_cache()
        print(f"[{name}] batch {i+1}: {len(all_errors)} samples so far")
    stats = aggregate_errors(all_errors)
    print(f"\n[{name}] aggregated over {len(all_errors)} samples: {stats}\n")
    acts_matrix = torch.cat(all_acts, dim=0).numpy()
    return all_errors, acts_matrix, stats


############ PCA

def pca_encode(acts_dict: dict[str, np.ndarray], n_components: int = 2) -> dict[str, np.ndarray]:
    """Fit PCA on concatenated activations, return per-dataset projections."""
    all_acts = np.concatenate(list(acts_dict.values()), axis=0)
    pca = PCA(n_components=n_components)
    pca.fit(all_acts)
    return {name: pca.transform(acts) for name, acts in acts_dict.items()}


############ Plotting

COLORS = {"fineweb": "#4C72B0", "wildjailbreak": "#DD5555"}
LABELS = {"fineweb": "FineWeb (benign)", "wildjailbreak": "WildJailbreak (adversarial)"}


def plot_pca_scatter(pca_dict: dict[str, np.ndarray], path: str = "pca_scatter.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, coords in pca_dict.items():
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.5,
                   color=COLORS[name], label=LABELS[name])
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("PCA of LLM Activations")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved PCA scatter plot to {path}")


def plot_error_distributions(errors_dict: dict[str, list[float]], path: str = "error_distributions.png"):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, name in zip(axes, ["fineweb", "wildjailbreak"]):
        arr = np.array(errors_dict[name])
        ax.hist(arr, bins=50, density=True, alpha=0.7, color=COLORS[name], edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Density")
        ax.set_title(LABELS[name])
        ax.axvline(arr.mean(), color="black", linestyle="--", linewidth=1, label=f"mean={arr.mean():.3f}")
        ax.legend()
    axes[-1].set_xlabel("Reconstruction Error (L2)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved error distribution plot to {path}")


############ JSON export

def export_results_json(errors_dict: dict[str, list[float]], stats_dict: dict[str, dict], path: str = "results.json"):
    payload = {}
    for name in errors_dict:
        payload[name] = {
            "reconstruction_errors": errors_dict[name],
            "aggregate": stats_dict[name],
        }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results JSON to {path}")


############ Main

if __name__ == "__main__":
    errors_dict: dict[str, list[float]] = {}
    acts_dict: dict[str, np.ndarray] = {}
    stats_dict: dict[str, dict] = {}

    print("=== FineWeb (benign) ===")
    errors_dict["fineweb"], acts_dict["fineweb"], stats_dict["fineweb"] = \
        run_evaluation("fineweb", fineweb_batches(BATCH_SIZE, NUM_SAMPLES))

    print("=== WildJailbreak (adversarial) ===")
    errors_dict["wildjailbreak"], acts_dict["wildjailbreak"], stats_dict["wildjailbreak"] = \
        run_evaluation("wildjailbreak", wildjailbreak_batches(BATCH_SIZE, NUM_SAMPLES))

    # PCA + scatter
    pca_dict = pca_encode(acts_dict)
    plot_pca_scatter(pca_dict)

    # Error distribution plots
    plot_error_distributions(errors_dict)

    # JSON export
    export_results_json(errors_dict, stats_dict)
