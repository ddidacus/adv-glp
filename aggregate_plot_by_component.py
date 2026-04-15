import os
import sys
import glob
import json
import torch
import tqdm
import fire
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from glp.script_eval import compute_pca

def _pca_fit_transform(arrays: list, k: int | None = None):
    """Fit PCA on the concatenation of all arrays, then project each array
    centred by its own mean — matching the convention in script_eval.plot_pca."""
    tensors = [torch.from_numpy(a.copy()).float() for a in arrays]
    combined = torch.cat(tensors, dim=0)
    W, _ = compute_pca(combined, k=k)   # W: D x k  (modifies combined in-place)
    return W, [((t - t.mean(0, keepdim=True)) @ W).numpy() for t in tensors]

def _tsne_fit_transform(arrays: list, k: int = 2, perplexity: float = 30.0):
    """Fit t-SNE on the concatenation of all arrays and split back."""
    sizes = [a.shape[0] for a in arrays]
    combined = np.concatenate(arrays, axis=0)
    embedded = TSNE(n_components=k, perplexity=perplexity, random_state=42).fit_transform(combined)
    result = []
    offset = 0
    for s in sizes:
        result.append(embedded[offset:offset + s])
        offset += s
    return result

def plot_pca_distributions_layerwise(
    activations_good: torch.Tensor,
    activations_bad: torch.Tensor,
    reconstructed_good: torch.Tensor,
    reconstructed_bad: torch.Tensor,
    layer_indices: list[int],
    out_dir: str,
    prefix: str = "pca_components",
    n_components: int = 10,
    method: str = "pca"):

    method = method.lower()
    assert method in ("pca", "tsne"), f"Unknown method: {method}"
    method_label = "PCA" if method == "pca" else "t-SNE"

    # first pass: compute projection per layer, save 1D hist plots, collect scatter data
    scatter_data = []

    for idx, layer_num in enumerate(layer_indices):

        good = activations_good[:, idx, :].float().cpu().numpy()          # N, D
        bad = activations_bad[:, idx, :].float().cpu().numpy()            # N, D
        recon_good = reconstructed_good[:, idx, :].float().cpu().numpy()  # N, D
        recon_bad = reconstructed_bad[:, idx, :].float().cpu().numpy()    # N, D

        good_err = np.abs(good - recon_good)  # N, D
        bad_err  = np.abs(bad  - recon_bad)   # N, D

        if method == "pca":
            # fit PCA on original activations only, then transform reconstructions
            W, (good_proj, bad_proj) = _pca_fit_transform([good, bad], k=n_components)
            recon_good_t = torch.from_numpy(recon_good.copy()).float()
            recon_bad_t  = torch.from_numpy(recon_bad.copy()).float()
            recon_good_proj = ((recon_good_t - recon_good_t.mean(0, keepdim=True)) @ W).numpy()
            recon_bad_proj  = ((recon_bad_t  - recon_bad_t.mean(0,  keepdim=True)) @ W).numpy()

            good_err_t = torch.from_numpy(good_err).float()
            bad_err_t  = torch.from_numpy(bad_err).float()
            good_err_proj = ((good_err_t - good_err_t.mean(0, keepdim=True)) @ W).numpy()
            bad_err_proj  = ((bad_err_t  - bad_err_t.mean(0,  keepdim=True)) @ W).numpy()
        else:  # tsne
            n_components_tsne = min(n_components, 2)
            all_proj = _tsne_fit_transform(
                [good, bad, recon_good, recon_bad, good_err, bad_err],
                k=n_components_tsne,
            )
            good_proj, bad_proj, recon_good_proj, recon_bad_proj, good_err_proj, bad_err_proj = all_proj

        n_comp_actual = good_proj.shape[1]

        scatter_data.append((layer_num, good_proj, bad_proj, recon_good_proj, recon_bad_proj, good_err_proj, bad_err_proj))

        # ===== 1D histogram plots (per layer)
        comp_label = lambda c: f"PC {c+1}" if method == "pca" else f"t-SNE {c+1}"

        # original activations
        fig, axes = plt.subplots(n_comp_actual, 1, figsize=(10, 2.5 * n_comp_actual), sharex=False)
        if n_comp_actual == 1:
            axes = [axes]
        for c in range(n_comp_actual):
            ax = axes[c]
            ax.hist(good_proj[:, c], bins=50, alpha=0.5, color="tab:blue", density=True, label="Good")
            ax.hist(bad_proj[:, c], bins=50, alpha=0.5, color="tab:orange", density=True, label="Bad")
            ax.set_ylabel("Density")
            ax.set_title(comp_label(c))
            if c == 0:
                ax.legend()
        axes[-1].set_xlabel("Value")
        fig.suptitle(f"{method_label} Component Distributions — Layer {layer_num}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_{layer_num}.png"), dpi=200)
        plt.close(fig)

        # reconstructed activations
        fig, axes = plt.subplots(n_comp_actual, 1, figsize=(10, 2.5 * n_comp_actual), sharex=False)
        if n_comp_actual == 1:
            axes = [axes]
        for c in range(n_comp_actual):
            ax = axes[c]
            ax.hist(recon_good_proj[:, c], bins=50, alpha=0.5, color="tab:blue", density=True, label="Good")
            ax.hist(recon_bad_proj[:, c], bins=50, alpha=0.5, color="tab:orange", density=True, label="Bad")
            ax.set_ylabel("Density")
            ax.set_title(comp_label(c))
            if c == 0:
                ax.legend()
        axes[-1].set_xlabel("Value")
        fig.suptitle(f"Reconstructed {method_label} Component Distributions — Layer {layer_num}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_reconstructions_{layer_num}.png"), dpi=200)
        plt.close(fig)

        # reconstruction error
        fig, axes = plt.subplots(n_comp_actual, 1, figsize=(10, 2.5 * n_comp_actual), sharex=False)
        if n_comp_actual == 1:
            axes = [axes]
        for c in range(n_comp_actual):
            ax = axes[c]
            ax.hist(good_err_proj[:, c], bins=50, alpha=0.5, color="tab:blue", density=True, label="Good")
            ax.hist(bad_err_proj[:, c], bins=50, alpha=0.5, color="tab:orange", density=True, label="Bad")
            ax.set_ylabel("Density")
            ax.set_title(f"{comp_label(c)} — Reconstruction Error")
            if c == 0:
                ax.legend()
        axes[-1].set_xlabel("|Original - Reconstructed|")
        fig.suptitle(f"Per-Component Reconstruction Error — Layer {layer_num}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_errors_{layer_num}.png"), dpi=200)
        plt.close(fig)

    # second pass: combined scatter — one row per layer, three columns
    n_layers = len(layer_indices)
    fig, axes = plt.subplots(n_layers, 3, figsize=(15, 5 * n_layers), squeeze=False)

    dim_label = "PC" if method == "pca" else "t-SNE"
    col_titles = ["Original activations", "Reconstructed activations", "Reconstruction error"]
    col_xlabels = [f"{dim_label} 1", f"{dim_label} 1", f"|error| {dim_label} 1"]
    col_ylabels = [f"{dim_label} 2", f"{dim_label} 2", f"|error| {dim_label} 2"]

    for row, (layer_num, good_proj, bad_proj, recon_good_proj, recon_bad_proj, good_err_proj, bad_err_proj) in enumerate(scatter_data):
        panels = [
            (good_proj,     bad_proj),
            (recon_good_proj, recon_bad_proj),
            (good_err_proj,  bad_err_proj),
        ]
        for col, (gp, bp) in enumerate(panels):
            ax = axes[row, col]
            ax.scatter(gp[:, 0], gp[:, 1], s=4, alpha=0.3, color="tab:blue",   label="Good", rasterized=True)
            ax.scatter(bp[:, 0], bp[:, 1], s=4, alpha=0.3, color="tab:orange", label="Bad",  rasterized=True)
            ax.set_xlabel(col_xlabels[col])
            ax.set_ylabel(f"Layer {layer_num}\n{col_ylabels[col]}" if col == 0 else col_ylabels[col])
            if row == 0:
                ax.set_title(col_titles[col])
            if row == 0 and col == 0:
                ax.legend(markerscale=2)

    fig.suptitle(f"{method_label} Scatter ({dim_label}1 vs {dim_label}2) — All Layers", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_scatter2d_all.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_error_comparison(
    activations_good: torch.Tensor,
    activations_bad: torch.Tensor,
    reconstructed_good: torch.Tensor,
    reconstructed_bad: torch.Tensor,
    layer_indices: list[int],
    out_dir: str, prefix:str):

    layerwise_all_good_errors = []
    layerwise_all_bad_errors = []

    fig, axes = plt.subplots(len(layer_indices), 1, figsize=(10, 2.5 * len(layer_indices)), sharex=False)

    good = activations_good.float().cpu().numpy() # N, L, D
    bad = activations_bad.float().cpu().numpy()
    recon_good = reconstructed_good.float().cpu().numpy()
    recon_bad = reconstructed_bad.float().cpu().numpy()

    good_err = np.abs(good - recon_good) # N, L, D
    bad_err = np.abs(bad - recon_bad)

    for idx, layer_num in enumerate(layer_indices):

        good_err_layer = np.abs(good[:, idx, :] - recon_good[:, idx, :])  # N, D
        bad_err_layer = np.abs(bad[:, idx, :] - recon_bad[:, idx, :])

        good_err_layer = np.linalg.norm(good_err_layer, ord=2, axis=1)  # (N,)
        bad_err_layer = np.linalg.norm(bad_err_layer, ord=2, axis=1)

        ax = axes[idx]
        ax.hist(good_err_layer, bins=50, alpha=0.5, color="tab:blue", density=True, label="Good")
        ax.hist(bad_err_layer, bins=50, alpha=0.5, color="tab:orange", density=True, label="Bad")
        ax.set_ylabel("Density")
        ax.set_title(f"Layer {layer_num} — Reconstruction Error")
        if idx == 0:
            ax.legend()

    axes[-1].set_xlabel("Value")
    fig.suptitle("Error distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_errors.png"), dpi=200)
    plt.close(fig)

    # # average across all layers: (n_layers, n_components) -> (n_components,)
    # mean_good = np.mean(all_good_errors, axis=0)
    # mean_bad = np.mean(all_bad_errors, axis=0)

    # x = np.arange(n_components)
    # width = 0.35
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.bar(x - width / 2, mean_good, width, label="Benign", color="tab:blue", alpha=0.8)
    # ax.bar(x + width / 2, mean_bad, width, label="Harmful", color="tab:orange", alpha=0.8)
    # ax.set_xlabel("PCA Component")
    # ax.set_ylabel("Mean Absolute Reconstruction Error")
    # ax.set_title("Average Reconstruction Error by Component (Benign vs Harmful)")
    # ax.set_xticks(x)
    # ax.set_xticklabels([f"PC {i+1}" for i in x])
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig(os.path.join(out_dir, "mean_error_comparison.png"), dpi=200)
    # plt.close(fig)

def plot_mean_error_comparison(
    activations_good: torch.Tensor,
    activations_bad: torch.Tensor,
    reconstructed_good: torch.Tensor,
    reconstructed_bad: torch.Tensor,
    layer_indices: list[int],
    out_dir: str,
    n_components: int = 10):

    all_good_errors = []
    all_bad_errors = []

    for idx, layer_num in enumerate(layer_indices):
        good = activations_good[:, idx, :].float().cpu().numpy()
        bad = activations_bad[:, idx, :].float().cpu().numpy()
        recon_good = reconstructed_good[:, idx, :].float().cpu().numpy()
        recon_bad = reconstructed_bad[:, idx, :].float().cpu().numpy()

        W, (good_pca, bad_pca) = _pca_fit_transform([good, bad], k=n_components)
        recon_good_t = torch.from_numpy(recon_good.copy()).float()
        recon_bad_t  = torch.from_numpy(recon_bad.copy()).float()
        recon_good_pca = ((recon_good_t - recon_good_t.mean(0, keepdim=True)) @ W).numpy()
        recon_bad_pca  = ((recon_bad_t  - recon_bad_t.mean(0,  keepdim=True)) @ W).numpy()

        # mean absolute error per component, averaged across samples
        good_err = np.abs(good_pca - recon_good_pca).mean(axis=0)  # (n_components,)
        bad_err = np.abs(bad_pca - recon_bad_pca).mean(axis=0)      # (n_components,)
        all_good_errors.append(good_err)
        all_bad_errors.append(bad_err)

    # average across all layers: (n_layers, n_components) -> (n_components,)
    mean_good = np.mean(all_good_errors, axis=0)
    mean_bad = np.mean(all_bad_errors, axis=0)

    x = np.arange(n_components)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, mean_good, width, label="Benign", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, mean_bad, width, label="Harmful", color="tab:orange", alpha=0.8)
    ax.set_xlabel("PCA Component")
    ax.set_ylabel("Mean Absolute Reconstruction Error")
    ax.set_title("Average Reconstruction Error by Component (Benign vs Harmful)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC {i+1}" for i in x])
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mean_error_comparison.png"), dpi=200)
    plt.close(fig)

def plot_error_comparison(
    activations_good: torch.Tensor,
    activations_bad: torch.Tensor,
    reconstructed_good: torch.Tensor,
    reconstructed_bad: torch.Tensor,
    layer_indices: list[int],
    out_dir: str, prefix: str):

    fig, axes = plt.subplots(len(layer_indices), 1, figsize=(10, 2.5 * len(layer_indices)), sharex=False)
    if len(layer_indices) == 1:
        axes = [axes]

    good = activations_good.float().cpu().numpy()
    bad = activations_bad.float().cpu().numpy()
    recon_good = reconstructed_good.float().cpu().numpy()
    recon_bad = reconstructed_bad.float().cpu().numpy()

    for idx, layer_num in enumerate(layer_indices):
        good_err_layer = np.abs(good[:, idx, :] - recon_good[:, idx, :])
        bad_err_layer = np.abs(bad[:, idx, :] - recon_bad[:, idx, :])

        good_err_layer = np.linalg.norm(good_err_layer, ord=2, axis=1)
        bad_err_layer = np.linalg.norm(bad_err_layer, ord=2, axis=1)

        clip = np.percentile(np.concatenate([good_err_layer, bad_err_layer]), 99.9)
        ax = axes[idx]
        ax.hist(good_err_layer, bins=50, alpha=0.5, color="tab:blue", density=True, label="Good", range=(0, clip))
        ax.hist(bad_err_layer, bins=50, alpha=0.5, color="tab:orange", density=True, label="Bad", range=(0, clip))
        ax.set_xlim(0, clip)
        ax.set_ylabel("Density")
        ax.set_title(f"Layer {layer_num} — Reconstruction Error")
        if idx == 0:
            ax.legend()

    axes[-1].set_xlabel("Value")
    fig.suptitle("Error distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_errors.png"), dpi=200)
    plt.close(fig)

def plot_error_by_layer(error_gap_stats: dict, out_dir: str, prefix: str = "error_by_layer"):
    """Line plot: x = layer index, y = mean reconstruction error norm (good vs bad)."""
    layer_nums = [int(k) for k in error_gap_stats]
    good_means = [error_gap_stats[k]["good_mean"] for k in error_gap_stats]
    good_stds  = [error_gap_stats[k]["good_std"]  for k in error_gap_stats]
    bad_means  = [error_gap_stats[k]["bad_mean"]  for k in error_gap_stats]
    bad_stds   = [error_gap_stats[k]["bad_std"]   for k in error_gap_stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(layer_nums, good_means, yerr=good_stds, marker="o", capsize=4,
                color="tab:blue", label="Good")
    ax.errorbar(layer_nums, bad_means, yerr=bad_stds, marker="s", capsize=4,
                color="tab:orange", label="Bad")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Reconstruction Error (L2 norm)")
    ax.set_title("Reconstruction Error by Layer — Good vs Bad")
    ax.set_xticks(layer_nums)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}.png"), dpi=200)
    plt.close(fig)

def equivalence_test(acts_good, acts_bad, recon_good, recon_bad, alpha=0.05):
    
    good_err = np.linalg.norm(
        np.abs(acts_good[:, -1, :] - recon_good[:, -1, :]), ord=2, axis=1) # N, 1
    
    bad_err = np.linalg.norm(
        np.abs(acts_bad[:, -1, :] - recon_bad[:, -1, :]), ord=2, axis=1) # N, 1

    rng = np.random.default_rng()
    noise = rng.normal(0, 0.1, good_err.shape)
    
    a = good_err
    b = bad_err

    print(a.mean(), a.std())
    print(b.mean(), b.std())

    # t-test
    # result = stats.ttest_ind(a, b, equal_var=False)
    result = stats.ttest_1samp(a, b.mean())
    pvalue = float(result.pvalue)
    statistic = float(result.statistic)
    equivalence = (pvalue > alpha)
    print(result)

    return equivalence

class StatisticalTestClassifier:

    def __init__(self, reference_population: np.ndarray, significance_level: float = 0.05):
        self._alpha = significance_level
        self._ref_pop = reference_population

    def eval(self, query: np.ndarray, population: bool = False):
        if population:
            result = stats.ttest_ind(self._ref_pop, query, equal_var=False)
        else:
            result = stats.ttest_1samp(self._ref_pop, query)
        pvalue = float(result.pvalue)
        return pvalue > self._alpha # accept H0 (equivalence)

class VarianceClassifier:
    def __init__(self, mu: float, std: float, distance: float):
        self._mu = mu
        self._std = std
        self._dist = distance

    def eval(self, query: np.ndarray):
        return (np.abs(self._mu - query) <= (self._std * self._dist)).item()

def main(results_dir: str, layers: str = "1,7,15", method: str = "pca"):
    """Aggregate results and plot component-wise analysis.

    Args:
        results_dir: Directory containing results_*.th files.
        layers: Comma-separated layer indices.
        method: Dimensionality reduction method ("pca" or "tsne").
    """
    if isinstance(layers, str):
        layers = [int(x) for x in layers.split(",")]
    elif isinstance(layers, int):
        layers = [layers]
    else:
        layers = [int(x) for x in layers]

    result_files = sorted(glob.glob(os.path.join(results_dir, "results_*.th")))
    if not result_files:
        print(f"No results_*.th files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(result_files)} result files: {result_files}")

    all_activations_good = []
    all_reconstructed_good = []
    all_activations_bad = []
    all_reconstructed_bad = []

    for path in result_files:
        data = torch.load(path, map_location="cpu", weights_only=True)
        all_activations_good.append(data["activations_good_set"])
        all_reconstructed_good.append(data["reconstructed_good_set"])
        all_activations_bad.append(data["activations_bad_set"])
        all_reconstructed_bad.append(data["reconstructed_bad_set"])

    activations_good = torch.cat(all_activations_good, dim=0)
    reconstructed_good = torch.cat(all_reconstructed_good, dim=0)
    activations_bad = torch.cat(all_activations_bad, dim=0)
    reconstructed_bad = torch.cat(all_reconstructed_bad, dim=0)

    print(f"Aggregated good: {activations_good.shape[0]} samples")
    print(f"Aggregated bad:  {activations_bad.shape[0]} samples")

    # plotting

    plot_error_comparison(
        activations_good=activations_good,
        activations_bad=activations_bad,
        reconstructed_good=reconstructed_good,
        reconstructed_bad=reconstructed_bad,
        layer_indices=layers,
        out_dir=results_dir,
        prefix="aggregated",
    )

    print(f"Plot saved to {results_dir}/aggregated_errors.png")

    # mean reconstruction error gap per layer
    print("\n--- Reconstruction error gap (mean bad - mean good) ---")
    acts_good_np = activations_good.float().numpy()
    acts_bad_np  = activations_bad.float().numpy()
    recon_good_np = reconstructed_good.float().numpy()
    recon_bad_np  = reconstructed_bad.float().numpy()
    error_gap_stats = {}
    for idx, layer_num in enumerate(layers):
        good_err = np.linalg.norm(np.abs(acts_good_np[:, idx, :] - recon_good_np[:, idx, :]), ord=2, axis=1)
        bad_err  = np.linalg.norm(np.abs(acts_bad_np[:, idx, :]  - recon_bad_np[:, idx, :]),  ord=2, axis=1)
        print(f"  Layer {layer_num:2d}: good={good_err.mean():.4f} ± {good_err.std():.4f}  "
              f"bad={bad_err.mean():.4f} ± {bad_err.std():.4f}  "
              f"gap={bad_err.mean() - good_err.mean():.4f}")
        error_gap_stats[str(layer_num)] = {
            "good_mean": float(good_err.mean()),
            "good_std":  float(good_err.std()),
            "bad_mean":  float(bad_err.mean()),
            "bad_std":   float(bad_err.std()),
            "gap":       float(bad_err.mean() - good_err.mean()),
        }
    print()
    gap_json_path = os.path.join(results_dir, "reconstruction_error_gap.json")
    with open(gap_json_path, "w") as f:
        json.dump(error_gap_stats, f, indent=2)
    print(f"Reconstruction error gap saved to {gap_json_path}")

    plot_error_by_layer(error_gap_stats, results_dir, prefix="error_by_layer")
    print(f"Error-by-layer line plot saved to {results_dir}/error_by_layer.png")

    plot_pca_distributions_layerwise(
        activations_good=activations_good,
        activations_bad=activations_bad,
        reconstructed_good=reconstructed_good,
        reconstructed_bad=reconstructed_bad,
        layer_indices=layers,
        out_dir=results_dir,
        prefix=method,
        n_components=5,
        method=method,
    )
    print(f"{method.upper()} scatter plot saved to {results_dir}/{method}_scatter2d_all.png")


if __name__ == "__main__":
    fire.Fire(main)

