import os
import torch
import numpy as np
from pathlib import Path
from typing import Iterator
from glp import flow_matching
import matplotlib.pyplot as plt
from datasets import load_dataset
from glp.denoiser import load_glp
from glp.utils_acts import save_acts
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

# - for each principal component, compute the mean and compare the differences
# - recompute errors 

@torch.no_grad()
def extract_activations(
    texts: list[str], 
    noise_level: float, 
    num_timesteps: int,
    llm_model, 
    llm_tokenizer, 
    diffusion_model, 
    device: str = "cuda:0",
    batch_size: int = 8):

    # extract llm activations
    
    activations = save_acts(
        hf_model=llm_model,
        hf_tokenizer=llm_tokenizer,
        text=texts,
        tracedict_config=diffusion_model.tracedict_config,
        token_idx="last",
        batch_size=batch_size,
    )  # (N, L, D)

    activations = activations.to(device=device, dtype=torch.bfloat16)
    raw_acts = activations.cpu().float()

    # extract bwd diffusion reconstructions
    # extract fwd diffusion noised samples

    reconstructed_layers = []
    noisy_denorm_layers = []

    for li in range(activations.shape[1]):
        layer_acts = activations[:, li:li+1, :]  # (N, 1, D)
        normalized = diffusion_model.normalizer.normalize(layer_acts)
        noise = torch.randn_like(normalized)
        noisy, _, timesteps, _ = flow_matching.fm_prepare(
            diffusion_model.scheduler,
            normalized,
            noise,
            u=torch.ones(normalized.shape[0]) * noise_level,
        )
        reconstructed = flow_matching.sample_on_manifold(
            diffusion_model,
            noisy,
            start_timestep=timesteps[0].item(),
            num_timesteps=num_timesteps,
        )
        reconstructed_layers.append(diffusion_model.normalizer.denormalize(reconstructed))
        noisy_denorm_layers.append(diffusion_model.normalizer.denormalize(noisy))

    reconstructed_acts = torch.cat(reconstructed_layers, dim=1)  # (N, L, D)
    noisy_acts = torch.cat(noisy_denorm_layers, dim=1)            # (N, L, D)
    reconstructed_acts = reconstructed_acts.to(device=activations.device, dtype=activations.dtype)
    noisy_acts = noisy_acts.to(device=activations.device, dtype=activations.dtype)

    # compute delta vectors for visualization

    delta_orig_recon = torch.abs(activations - reconstructed_acts)
    delta_noisy_recon = torch.abs(noisy_acts - reconstructed_acts)

    errors = torch.norm(
        (activations - reconstructed_acts).float().reshape(activations.shape[0], -1),
        p=2,
        dim=1,
    )

    return {
        "activations": raw_acts,
        "noised_activations": noisy_acts,
        "reconstructed_activations": reconstructed_acts,
        "delta_original_reconstructed": delta_orig_recon,
        "reconstruction_errors": errors.cpu(),
    }

def vanilla_wildjailbreak_batches(batch_size: int = 8, num_samples: int = 1024,
                                   benign=False, seed: int = 42, skip: int = 0) -> Iterator[list[str]]:
    """Yield batches of vanilla prompts (random permutation, non-overlapping with training via skip)."""
    label = "benign" if benign else "harmful"
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    filtered = dataset.filter(lambda x: x["data_type"] == f"vanilla_{label}")
    all_prompts = [sample["vanilla"] for sample in filtered["train"]]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_prompts))
    prompts = [all_prompts[i] for i in indices[skip : skip + num_samples]]
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]

def adversarial_wildjailbreak_batches(batch_size: int = 8, num_samples: int = 1024,
                                       benign=False, seed: int = 42) -> Iterator[list[str]]:
    """Yield batches of adversarial prompts (random permutation)."""
    label = "benign" if benign else "harmful"
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    filtered = dataset.filter(lambda x: x["data_type"] == f"adversarial_{label}")
    all_prompts = [sample["adversarial"] for sample in filtered["train"]]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_prompts))
    prompts = [all_prompts[i] for i in indices[:num_samples]]
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]

def fineweb_batches(batch_size: int, num_samples: int | None = None, skip: int = 0) -> Iterator[list[str]]:
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

def pca_encode(activations: np.ndarray, n_components: int = 2) -> dict[str, np.ndarray]:
    pca = PCA(n_components=n_components)
    pca.fit(activations) # N, D
    return pca.transform(activations) # N, n_components

def plot_pca_layerwise(
    activations_1: torch.tensor, 
    activations_2: torch.tensor, 
    layer_indices: list[int],
    out_dir: str, 
    prefix: str = "pca_layer",
    num_dimensions: int = 2):
    for idx, layer_num in enumerate(layer_indices):

        # slice

        layer_activations_1 = activations_1[:, idx, :].float().cpu().numpy() # N, D
        layer_activations_2 = activations_2[:, idx, :].float().cpu().numpy() # N, D
        pca_layer_activations_1 = pca_encode(layer_activations_1, n_components=num_dimensions) # N, n_components
        pca_layer_activations_2 = pca_encode(layer_activations_2, n_components=num_dimensions) # N, n_components
        
        # plot

        path = os.path.join(out_dir, f"{prefix}_{layer_num-1}.png")
        generate_pca_plot(
            points_1 = pca_layer_activations_1, 
            points_2 = pca_layer_activations_2,
            labels=["Good prompts", "Bad prompts"],
            path=path, 
            title=f"PCA of Activations — Layer {layer_num}",
            num_dimensions=num_dimensions
        )

def generate_pca_plot(
    points_1: np.ndarray, 
    points_2: np.ndarray, 
    labels: list, 
    num_dimensions: int,
    path: str = "pca_scatter.png", 
    title: str = "PCA of LLM Activations", 
    colors: list = ["tab:blue", "tab:orange"]):

    if num_dimensions == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        # scater plots
        ax.scatter(points_1[:, 0], points_1[:, 1], s=8, alpha=0.5, color=colors[0], label=labels[0])
        ax.scatter(points_2[:, 0], points_2[:, 1], s=8, alpha=0.5, color=colors[1], label=labels[1])
        # format plot
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(title)
        ax.legend(markerscale=3)

    elif num_dimensions == 3:
        # Initialize a 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # 3D scatter plots
        ax.scatter(points_1[:, 0], points_1[:, 1], points_1[:, 2], 
                s=8, alpha=0.5, color=colors[0], label=labels[0])
        ax.scatter(points_2[:, 0], points_2[:, 1], points_2[:, 2], 
                s=8, alpha=0.5, color=colors[1], label=labels[1])
        # Set Isometric View
        # Elevation of 35.264 degrees and Azimuth of 45 degrees creates a true isometric perspective
        ax.view_init(elev=35.264, azim=45)
        # Format plot
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        ax.set_title(title)
        # Adjust legend and layout
        ax.legend(markerscale=3, loc='upper left')

    # save
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_error_distributions(
    errors_good: np.ndarray,
    errors_bad: np.ndarray,
    path: str = "error_distributions.png",
    title: str = "Reconstruction Error Distribution"):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.hist(errors_good, bins=50, alpha=0.7, color="tab:blue", density=True)
    ax1.set_ylabel("Density")
    ax1.set_title(f"{title} — Good prompts")
    ax2.hist(errors_bad, bins=50, alpha=0.7, color="tab:orange", density=True)
    ax2.set_ylabel("Density")
    ax2.set_xlabel("L2 Error")
    ax2.set_title(f"{title} — Bad prompts")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def encode_prompts(prompts, batch_size, llm_model, llm_tokenizer, diffusion_model) -> dict:
    # returns only the layers set in the config
    return extract_activations(
        texts=prompts, 
        noise_level = 0.25, 
        num_timesteps = 100,
        llm_model = llm_model, 
        llm_tokenizer = llm_tokenizer, 
        diffusion_model = diffusion_model, 
        batch_size = batch_size,
        device = "cuda:0"
    )

if __name__ == "__main__":

    batch_size = 64
    num_samples = 128
    # llm_model_id = "unsloth/Llama-3.2-1B"
    # glp_model_id = "generative-latent-prior/glp-llama1b-d6"
    llm_model_id = "unsloth/Meta-Llama-3.1-8B"
    # llm_model_id = "meta-llama/Llama-Guard-3-8B"
    glp_model_id = "generative-latent-prior/glp-llama8b-d6"
    # out_dir = "pca_plots_llama_8b"
    out_dir = "results/fineweb-vs-hardharmful-llama8b"
    layers = [7, 15, 31]

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    diffusion_model = load_glp(
        glp_model_id, device="cuda:0", checkpoint="final"
    )
    # llama-3-1B: diffusion_model.tracedict_config.layers = [7]
    # llama-3-8B: diffusion_model.tracedict_config.layers = [15]
    diffusion_model.tracedict_config.layers = layers

    print("================================================")
    print(f"[+] LLM: \t\t{llm_model_id}")
    print(f"[+] GLP: \t\t{glp_model_id}")
    print(f"[+] batch_size: \t{batch_size}")
    print(f"[+] num_samples: \t{num_samples}")
    print(f"[+] layers: \t\t{diffusion_model.tracedict_config.layers}")
    print("================================================")

    # destination dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # fineweb
    indistribution_prompts = fineweb_batches(batch_size=batch_size, num_samples = num_samples)

    # wildjailbreak (vanilla)
    # easy_good_prompts = vanilla_wildjailbreak_batches(batch_size=batch_size, benign=True, num_samples = num_samples)
    # easy_bad_prompts = vanilla_wildjailbreak_batches(batch_size=batch_size, benign=False, num_samples = num_samples)
    
    # wildjailbreak (adversarial)
    # hard_good_prompts = adversarial_wildjailbreak_batches(batch_size=batch_size, benign=True, num_samples = num_samples)
    hard_bad_prompts = adversarial_wildjailbreak_batches(batch_size=batch_size, benign=False, num_samples = num_samples)

    # return {
    #     "activations": raw_acts,
    #     "noised_activations": noisy_acts,
    #     "reconstructed_activations": reconstructed_acts,
    #     "delta_original_reconstructed": delta_orig_recon,
    # }
    
    # easy good prompts
    good_set = indistribution_prompts
    bad_set = hard_bad_prompts

    print("======= Encoding the good set =======")
    batch_good_set = []
    for batch in good_set:
        batch_good_set.append(
            encode_prompts(batch, batch_size, llm_model, llm_tokenizer, diffusion_model)
        )
    delta_orig_recon_good_set = torch.cat([x["delta_original_reconstructed"] for x in batch_good_set], dim=0)
    activations_good_set = torch.cat([x["activations"] for x in batch_good_set], dim=0)
    reconstructed_good_set = torch.cat([x["reconstructed_activations"] for x in batch_good_set], dim=0)
    errors_good_set = torch.cat([x["reconstruction_errors"] for x in batch_good_set], dim=0)

    # easy bad prompts

    print("======= Encoding the bad set =======")
    batch_bad_set = []
    for batch in bad_set:
        batch_bad_set.append(
            encode_prompts(batch, batch_size, llm_model, llm_tokenizer, diffusion_model)
        )
    delta_orig_recon_bad_set = torch.cat([x["delta_original_reconstructed"] for x in batch_bad_set], dim=0)
    activations_bad_set = torch.cat([x["activations"] for x in batch_bad_set], dim=0)
    reconstructed_bad_set = torch.cat([x["reconstructed_activations"] for x in batch_bad_set], dim=0)
    errors_bad_set = torch.cat([x["reconstruction_errors"] for x in batch_bad_set], dim=0)

    # plot

    plot_error_distributions(
        errors_good = errors_good_set.numpy(),
        errors_bad = errors_bad_set.numpy(),
        path = os.path.join(out_dir, "error_distributions.png"),
    )

    plot_pca_layerwise(
        activations_1 = delta_orig_recon_good_set,
        activations_2 = delta_orig_recon_bad_set, 
        layer_indices = layers,
        out_dir = out_dir, 
        prefix = "pca_delta_orig_recon",
        num_dimensions = 2,
    )

    plot_pca_layerwise(
        activations_1 = activations_good_set, 
        activations_2 = activations_bad_set, 
        layer_indices = layers,
        out_dir = out_dir, 
        prefix = "pca_activations",
        num_dimensions = 2,
    )

    plot_pca_layerwise(
        activations_1 = reconstructed_good_set, 
        activations_2 = reconstructed_bad_set, 
        layer_indices = layers,
        out_dir = out_dir, 
        prefix = "pca_reconstructions",
        num_dimensions = 2,
    )