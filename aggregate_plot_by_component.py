import os
import sys
import glob
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def plot_pca_distributions_layerwise(
    activations_good: torch.Tensor,
    activations_bad: torch.Tensor,
    reconstructed_good: torch.Tensor,
    reconstructed_bad: torch.Tensor,
    layer_indices: list[int],
    out_dir: str,
    prefix: str = "pca_components",
    n_components: int = 10):

    for idx, layer_num in enumerate(layer_indices):

        good = activations_good[:, idx, :].float().cpu().numpy()          # N, D
        bad = activations_bad[:, idx, :].float().cpu().numpy()            # N, D
        recon_good = reconstructed_good[:, idx, :].float().cpu().numpy()  # N, D
        recon_bad = reconstructed_bad[:, idx, :].float().cpu().numpy()    # N, D

        # fit PCA on combined original activations

        pca = PCA(n_components=n_components)
        pca.fit(np.concatenate([good, bad], axis=0))
        good_pca = pca.transform(good)              # N, n_components
        bad_pca = pca.transform(bad)                # N, n_components
        
        recon_pca = PCA(n_components=n_components)
        recon_pca.fit(np.concatenate([recon_good, recon_bad], axis=0))

        # ===== Scatter plots

        # original activations
        # per single PCA component 

        fig, axes = plt.subplots(n_components, 1, figsize=(10, 2.5 * n_components), sharex=False)
        for c in range(n_components):
            ax = axes[c]
            ax.hist(good_pca[:, c], bins=50, alpha=0.5, color="tab:blue", density=True, label="Good")
            ax.hist(bad_pca[:, c], bins=50, alpha=0.5, color="tab:orange", density=True, label="Bad")
            ax.set_ylabel("Density")
            ax.set_title(f"PC {c+1}")
            if c == 0:
                ax.legend()
        axes[-1].set_xlabel("Value")
        fig.suptitle(f"PCA Component Distributions — Layer {layer_num}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_{layer_num}.png"), dpi=200)
        plt.close(fig)

        # reconstructed activations
        # per single PCA component 

        recon_good_pca = recon_pca.transform(recon_good)  # N, n_components
        recon_bad_pca = recon_pca.transform(recon_bad)    # N, n_components

        fig, axes = plt.subplots(n_components, 1, figsize=(10, 2.5 * n_components), sharex=False)
        for c in range(n_components):
            ax = axes[c]
            ax.hist(recon_good_pca[:, c], bins=50, alpha=0.5, color="tab:blue", density=True, label="Good")
            ax.hist(recon_bad_pca[:, c], bins=50, alpha=0.5, color="tab:orange", density=True, label="Bad")
            ax.set_ylabel("Density")
            ax.set_title(f"PC {c+1}")
            if c == 0:
                ax.legend()
        axes[-1].set_xlabel("Value")
        fig.suptitle(f"PCA Component Distributions — Layer {layer_num}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_reconstructions_{layer_num}.png"), dpi=200)
        plt.close(fig)

        # reconstructed activations
        # 2D scatter plot

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (proj, title) in zip(axes, [
            ((good_pca, bad_pca), "Original activations"),
            ((recon_good_pca, recon_bad_pca), "Reconstructed activations"),
        ]):
            gp, bp = proj
            ax.scatter(gp[:, 0], gp[:, 1], s=6, alpha=0.4, color="tab:blue", label="Good", rasterized=True)
            ax.scatter(bp[:, 0], bp[:, 1], s=6, alpha=0.4, color="tab:orange", label="Bad", rasterized=True)
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.set_title(title)
            ax.legend(markerscale=2)
        fig.suptitle(f"PCA Scatter (PC1 vs PC2) — Layer {layer_num}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_scatter2d_{layer_num}.png"), dpi=200)
        plt.close(fig)

        # ===== Error vectors

        # 1D layer wise plot

        recon_good_pca = pca.transform(recon_good)  # N, n_components
        recon_bad_pca = pca.transform(recon_bad)    # N, n_components

        good_err = np.abs(good_pca - recon_good_pca)  # N, n_components
        bad_err = np.abs(bad_pca - recon_bad_pca)      # N, n_components

        fig, axes = plt.subplots(n_components, 1, figsize=(10, 2.5 * n_components), sharex=False)
        for c in range(n_components):
            ax = axes[c]
            ax.hist(good_err[:, c], bins=50, alpha=0.5, color="tab:blue", density=True, label="Good")
            ax.hist(bad_err[:, c], bins=50, alpha=0.5, color="tab:orange", density=True, label="Bad")
            ax.set_ylabel("Density")
            ax.set_title(f"PC {c+1} — Reconstruction Error")
            if c == 0:
                ax.legend()
        axes[-1].set_xlabel("|Original - Reconstructed|")
        fig.suptitle(f"Per-Component Reconstruction Error — Layer {layer_num}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_errors_{layer_num}.png"), dpi=200)
        plt.close(fig)

        # 2D scatter plot

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(good_err[:, 0], good_err[:, 1], s=6, alpha=0.4, color="tab:blue", label="Good", rasterized=True)
        ax.scatter(bad_err[:, 0], bad_err[:, 1], s=6, alpha=0.4, color="tab:orange", label="Bad", rasterized=True)
        ax.set_xlabel("|error| PC 1")
        ax.set_ylabel("|error| PC 2")
        ax.set_title(f"Reconstruction Error Vectors — Layer {layer_num}")
        ax.legend(markerscale=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_error_scatter2d_{layer_num}.png"), dpi=200)
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

        pca = PCA(n_components=n_components)
        pca.fit(np.concatenate([good, bad], axis=0))
        good_pca = pca.transform(good)
        bad_pca = pca.transform(bad)
        recon_good_pca = pca.transform(recon_good)
        recon_bad_pca = pca.transform(recon_bad)

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

if __name__ == "__main__":
    results_dir = sys.argv[1]
    layers = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 7, 15]

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

    # statistical testing

    layer = 0
    good_err = np.linalg.norm(
        np.abs(activations_good[:, layer, :] - reconstructed_good[:, layer, :]), ord=2, axis=1) # N, 1
    
    bad_err = np.linalg.norm(
        np.abs(activations_bad[:, layer, :] - reconstructed_bad[:, layer, :]), ord=2, axis=1) # N, 1
    
    # welch's t-test
    # classifier = StatisticalTestClassifier(
    #     reference_population=good_err,
    #     significance_level=0.05
    # )

    classifier = VarianceClassifier(
        mu = good_err.mean(),
        std = good_err.std(),
        distance = 0.01
    )

    # is_pop_equivalent = classifier.eval(bad_err, population=True)
    # print(f"Population: {is_pop_equivalent}")

    # print("===== good =====")
    # negatives = 0
    # positives = 0
    # for sample in tqdm.tqdm(good_err):
    #     label = classifier.eval(np.array([sample]))
    #     negatives += int(not label)
    #     positives += int(label)
    # negatives = negatives / good_err.shape[0]
    # positives = positives / good_err.shape[0]

    # print(f"negatives: {negatives}")
    # print(f"positives: {positives}")

    # print("===== bad =====")
    # negatives = 0
    # positives = 0
    # for sample in tqdm.tqdm(bad_err):
    #     label = classifier.eval(np.array([sample]))
    #     negatives += int(not label)
    #     positives += int(label)
    # negatives = negatives / bad_err.shape[0]
    # positives = positives / bad_err.shape[0]

    # print(f"negatives: {negatives}")
    # print(f"positives: {positives}")

    # plot

    # plot_error_comparison(
    #     activations_good = activations_good_set,
    #     activations_bad = activations_bad_set,
    #     reconstructed_good = reconstructed_good_set,
    #     reconstructed_bad = reconstructed_bad_set,
    #     layer_indices = layers,
    #     out_dir = out_dir,
    #     prefix = f"errors_gpu{gpu_id}"
    # )

    last_idx = len(layers) - 1
    plot_pca_distributions_layerwise(
        activations_good=activations_good[:, last_idx:last_idx+1, :],
        activations_bad=activations_bad[:, last_idx:last_idx+1, :],
        reconstructed_good=reconstructed_good[:, last_idx:last_idx+1, :],
        reconstructed_bad=reconstructed_bad[:, last_idx:last_idx+1, :],
        layer_indices=[layers[-1]],
        out_dir=results_dir,
        prefix="pca_last_layer",
        n_components=5,
    )
    print(f"PCA separation plot saved to {results_dir}/pca_last_layer_{layers[-1]}.png")

    # plot_mean_error_comparison(
    #     activations_good = activations_good_set,
    #     activations_bad = activations_bad_set,
    #     reconstructed_good = reconstructed_good_set,
    #     reconstructed_bad = reconstructed_bad_set,
    #     layer_indices = layers,
    #     out_dir = out_dir,
    #     n_components = n_components
    # )

