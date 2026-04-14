import json
import os
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
import fire

from glp.denoiser import load_glp
from glp.utils_acts import save_acts
from glp import flow_matching

import time

############ Batch reconstruction error

BATCH_SIZE = 32
LAYERS = list(range(32))

@torch.no_grad()
def compute_reconstruction_error(texts: list[str], noise_level: float, num_timesteps: int,
                                 llm_model, llm_tokenizer, diffusion_model, device: str = "cuda:0",
                                 batch_size: int = BATCH_SIZE):
    """
    Given a list of strings, extract LLM activations, noise-denoise via GLP,
    and return per-sample L2 errors, flattened activation views, and raw (N, L, D) activations.
    """
    # Extract activations for the full list (save_acts handles internal batching)
    start = time.time()
    activations = save_acts(
        hf_model=llm_model,
        hf_tokenizer=llm_tokenizer,
        text=texts,
        tracedict_config=diffusion_model.tracedict_config,
        token_idx="last",
        batch_size=batch_size,
    )  # (N, L, D)
    llm_encode_time = time.time() - start

    activations = activations.to(device=device, dtype=torch.bfloat16)

    # Denoiser processes one layer at a time — loop over L and stack results
    reconstructed_layers = []
    noisy_denorm_layers = []
    start = time.time()
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
    diffusion_reconstruct_time = time.time() - start

    reconstructed_acts = torch.cat(reconstructed_layers, dim=1)  # (N, L, D)
    noisy_acts = torch.cat(noisy_denorm_layers, dim=1)            # (N, L, D)
    reconstructed_acts = reconstructed_acts.to(device=activations.device, dtype=activations.dtype)
    noisy_acts = noisy_acts.to(device=activations.device, dtype=activations.dtype)

    # Raw per-layer activations (N, L, D) — used for per-layer PCA
    raw_acts = activations.cpu().float()

    # Flatten all activation variants to (N, num_layers * dim) for downstream PCA
    # flat_acts = activations.reshape(activations.shape[0], -1).cpu().float()
    # flat_noisy = noisy_acts.reshape(activations.shape[0], -1).cpu().float()
    # flat_reconstructed = reconstructed_acts.reshape(activations.shape[0], -1).cpu().float()
    # flat_diff_orig_recon = flat_acts - flat_reconstructed
    # flat_diff_noisy_recon = flat_noisy - flat_reconstructed
    
    delta_orig_recon = activations - reconstructed_acts
    delta_noisy_recon = noisy_acts - reconstructed_acts

    # Per-sample L2 norm over all dimensions except batch
    errors = torch.norm(
        (activations - reconstructed_acts).reshape(activations.shape[0], -1),
        p=2,
        dim=1,
    )
    # return (errors, raw_acts, flat_acts, flat_noisy, flat_reconstructed,
    #         flat_diff_orig_recon, flat_diff_noisy_recon, llm_encode_time, diffusion_reconstruct_time)
    # N, L, D
    return (errors, raw_acts, activations, noisy_acts, reconstructed_acts,
            delta_orig_recon, delta_noisy_recon, llm_encode_time, diffusion_reconstruct_time)


############ Dataset iterators (aligned with train_linear_probe_anomaly_detection.py eval splits)

def vanilla_wildjailbreak_batches(batch_size: int = BATCH_SIZE, num_samples: int = 1024,
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


def adversarial_wildjailbreak_batches(batch_size: int = BATCH_SIZE, num_samples: int = 1024,
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


def run_evaluation(name: str, batch_iter: Iterator[list[str]], noise_level: float, num_timesteps: int,
                   llm_model, llm_tokenizer, diffusion_model, device: str = "cuda:0",
                   save_acts_batch_size: int = BATCH_SIZE):
    """Returns (errors, raw_acts, acts, noisy_acts, reconstructed_acts, diff_orig_recon, diff_noisy_recon, stats, t_llama, t_diff).

    raw_acts has shape (N, L, D) and is used for per-layer PCA.
    """
    all_errors: list[float] = []
    all_raw_acts: list[torch.Tensor] = []
    all_acts: list[torch.Tensor] = []
    all_noisy: list[torch.Tensor] = []
    all_reconstructed: list[torch.Tensor] = []
    all_diff_orig_recon: list[torch.Tensor] = []
    all_diff_noisy_recon: list[torch.Tensor] = []
    time_llama_enc: list[float] = []
    time_diffusion_recon: list[float] = []
    for i, batch in enumerate(batch_iter):
        # (errors, raw_acts, flat_acts, flat_noisy, flat_reconstructed,
        #  flat_diff_orig_recon, flat_diff_noisy_recon,
        #  llama_encode_time, diffusion_reconstruct_time) = \
        #     compute_reconstruction_error(batch, noise_level=noise_level, num_timesteps=num_timesteps,
        #                                  llm_model=llm_model, llm_tokenizer=llm_tokenizer,
        #                                  diffusion_model=diffusion_model, device=device,
        #                                  batch_size=save_acts_batch_size)
        (errors, raw_acts, acts, noisy, reconstructed,
         diff_orig_recon, diff_noisy_recon,
         llama_encode_time, diffusion_reconstruct_time) = \
            compute_reconstruction_error(batch, noise_level=noise_level, num_timesteps=num_timesteps,
                                         llm_model=llm_model, llm_tokenizer=llm_tokenizer,
                                         diffusion_model=diffusion_model, device=device,
                                         batch_size=save_acts_batch_size)
        # N_B, B, L, D
        all_errors.extend(errors.cpu().float().tolist())
        all_raw_acts.append(raw_acts)
        all_acts.append(acts)
        all_noisy.append(noisy)
        all_reconstructed.append(reconstructed)
        all_diff_orig_recon.append(diff_orig_recon)
        all_diff_noisy_recon.append(diff_noisy_recon)
        torch.cuda.empty_cache()
        print(f"[{name}] batch {i+1}: {len(all_errors)} samples so far")
        time_llama_enc.append(llama_encode_time)
        time_diffusion_recon.append(diffusion_reconstruct_time)
    stats = aggregate_errors(all_errors)
    print(f"\n[{name}] aggregated over {len(all_errors)} samples: {stats}\n")
    return (
        all_errors,
        torch.cat(all_raw_acts, dim=0).cpu().numpy(),   # (N, L, D)
        torch.cat(all_acts, dim=0).cpu().numpy(), # (N, L, D)
        torch.cat(all_noisy, dim=0).cpu().numpy(), # (N, L, D)
        torch.cat(all_reconstructed, dim=0).cpu().numpy(), # (N, L, D)
        torch.cat(all_diff_orig_recon, dim=0).cpu().numpy(), # (N, L, D)
        torch.cat(all_diff_noisy_recon, dim=0).cpu().numpy(), # (N, L, D)
        stats,
        torch.tensor(time_llama_enc).mean(),
        torch.tensor(time_diffusion_recon).mean(),
    )


############ PCA

def pca_encode(acts_dict: dict[str, np.ndarray], n_components: int = 2) -> dict[str, np.ndarray]:
    """Fit PCA on concatenated activations, return per-dataset projections."""
    all_acts = np.concatenate(list(acts_dict.values()), axis=0)
    pca = PCA(n_components=n_components)
    pca.fit(all_acts)
    return {name: pca.transform(acts) for name, acts in acts_dict.items()}


############ Plotting

COLORS = {
    "vanilla_benign_holdout": "#55AA55",
    "vanilla_harmful_holdout": "#DD5555",
    "adversarial_benign": "#4C72B0",
    "adversarial_harmful": "#E07020",
}
LABELS = {
    "vanilla_benign_holdout": "Vanilla benign (holdout)",
    "vanilla_harmful_holdout": "Vanilla harmful (holdout)",
    "adversarial_benign": "Adversarial benign",
    "adversarial_harmful": "Adversarial harmful",
}


def plot_pca_scatter(pca_dict: dict[str, np.ndarray], path: str = "pca_scatter.png", title: str = "PCA of LLM Activations"):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, coords in pca_dict.items():
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.5,
                   color=COLORS[name], label=LABELS[name])
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved PCA scatter plot to {path}")


def plot_pca_per_layer(raw_acts_dict: dict[str, np.ndarray], layer_indices: list[int],
                       out_dir: str, prefix: str = "pca_layer"):
    """One PCA scatter plot per layer. raw_acts_dict values have shape (N, L, D)."""
    for li, layer_num in enumerate(layer_indices):
        layer_slice = {name: acts[:, li, :] for name, acts in raw_acts_dict.items()}
        pca_dict = pca_encode(layer_slice)
        path = os.path.join(out_dir, f"{prefix}_{layer_num}.png")
        plot_pca_scatter(pca_dict, path=path, title=f"PCA of Activations — Layer {layer_num}")


def plot_error_distributions(errors_dict: dict[str, list[float]], path: str = "error_distributions.png"):
    keys = list(errors_dict.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 2 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, name in zip(axes, keys):
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

def main(noise_levels: list[float] = (0.25,), n_denoising_steps: list[int] = (100,),
         num_samples: int = 1024, skip_samples_idx: int = 0, seed: int = 42,
         out_dir_prefix: str = "results_explore",
         llm_model_id: str = "unsloth/Llama-3.2-1B",
         glp_model_id: str = "generative-latent-prior/glp-llama1b-d6",
         save_acts_batch_size: int = BATCH_SIZE, num_layers=16):

    # LAYERS = list(range(num_layers))
    LAYERS = [14]

    print(f"[+] LLM: {llm_model_id}")
    print(f"[+] GLP: {glp_model_id}")
    print(f"[+] Layers: {LAYERS}")

    # Exact same eval splits as train_linear_probe_anomaly_detection.py
    datasets_spec = [
        ("vanilla_benign_holdout",  "Vanilla benign (holdout)",  "vanilla",     True,  skip_samples_idx),
        ("vanilla_harmful_holdout", "Vanilla harmful (holdout)", "vanilla",     False, skip_samples_idx),
        ("adversarial_benign",      "Adversarial benign",        "adversarial", True,  0),
        ("adversarial_harmful",     "Adversarial harmful",       "adversarial", False, 0),
    ]

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    diffusion_model = load_glp(
        glp_model_id, device="cuda:0", checkpoint="final"
    )
    diffusion_model.tracedict_config.layers = LAYERS

    for nl in noise_levels:
        for nds in n_denoising_steps:
            out_dir = f"{out_dir_prefix}_noise_{nl}_denoise_{nds}"
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"  Noise level: {nl}  |  Denoising steps: {nds}")
            print(f"  Output directory: {out_dir}")
            print(f"{'='*60}\n")

            errors_dict: dict[str, list[float]] = {}
            raw_acts_dict: dict[str, np.ndarray] = {}
            acts_dict: dict[str, np.ndarray] = {}
            noisy_dict: dict[str, np.ndarray] = {}
            reconstructed_dict: dict[str, np.ndarray] = {}
            diff_orig_recon_dict: dict[str, np.ndarray] = {}
            diff_noisy_recon_dict: dict[str, np.ndarray] = {}
            stats_dict: dict[str, dict] = {}

            for ds_key, ds_label, data_type, benign, skip in datasets_spec:
                if data_type == "vanilla":
                    ds_iter = vanilla_wildjailbreak_batches(save_acts_batch_size, num_samples, benign=benign, seed=seed, skip=skip)
                else:
                    ds_iter = adversarial_wildjailbreak_batches(save_acts_batch_size, num_samples, benign=benign, seed=seed)

                print(f"=== {ds_label} ===")
                (errors_dict[ds_key], raw_acts_dict[ds_key], acts_dict[ds_key], noisy_dict[ds_key],
                 reconstructed_dict[ds_key], diff_orig_recon_dict[ds_key],
                 diff_noisy_recon_dict[ds_key], stats_dict[ds_key],
                 time_llama, time_diffusion) = \
                    run_evaluation(ds_key, ds_iter, noise_level=nl, num_timesteps=nds,
                                   llm_model=llm_model, llm_tokenizer=llm_tokenizer,
                                   diffusion_model=diffusion_model, device="cuda:0",
                                   save_acts_batch_size=save_acts_batch_size)
                print(f"avg. llama encode time: {time_llama:.4f}")
                print(f"avg. diffusion reconstr. time: {time_diffusion:.4f}")

            # Per-layer PCA scatter plots (one plot per layer)
            plot_pca_per_layer(raw_acts_dict, LAYERS, out_dir, prefix="pca_layer")
            plot_pca_per_layer(diff_orig_recon_dict, LAYERS, out_dir, prefix="pca_diff_origin_recon")
            plot_pca_per_layer(diff_noisy_recon_dict, LAYERS, out_dir, prefix="pca_diff_noisy_recon")

            # PCA plots for each activation view (all layers flattened)
            # pca_views = [
            #     (acts_dict, "pca_original.png", "PCA of Original LLM Activations"),
            #     (noisy_dict, "pca_noisy.png", "PCA of Noisy Activations (after forward diffusion)"),
            #     (reconstructed_dict, "pca_reconstructed.png", "PCA of Reconstructed Activations (after fwd+bwd diffusion)"),
            #     (diff_orig_recon_dict, "pca_diff_orig_recon.png", "PCA of Difference (original - reconstructed)"),
            #     (diff_noisy_recon_dict, "pca_diff_noisy_recon.png", "PCA of Difference (noisy - reconstructed)"),
            # ]
            # for view_dict, filename, title in pca_views:
            #     pca_dict = pca_encode(view_dict)
            #     plot_pca_scatter(pca_dict, path=os.path.join(out_dir, filename), title=title)

            # Error distribution plots
            plot_error_distributions(errors_dict, path=os.path.join(out_dir, "error_distributions.png"))

            # JSON export
            export_results_json(errors_dict, stats_dict, path=os.path.join(out_dir, "results.json"))


if __name__ == "__main__":
    fire.Fire(main)
