import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Iterator
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fire

from glp.denoiser import load_glp
from glp.utils_acts import save_acts

############ Config

BATCH_SIZE = 32

############ Dataset iterators

def eval_wildjailbreak_batches(batch_size: int = BATCH_SIZE, num_samples: int | None = None, benign=False, data_type_prefix="vanilla", seed: int = 42) -> Iterator[list[str]]:
    """Yield batches of prompts from allenai/wildjailbreak (randomly sampled).
    If num_samples is set, randomly sample that many."""
    label = "benign" if benign else "harmful"
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    filtered = dataset.filter(lambda x: x["data_type"] == f"{data_type_prefix}_{label}")
    col = "vanilla" if data_type_prefix == "vanilla" else "adversarial"
    all_prompts = [sample[col] for sample in filtered["train"]]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_prompts))
    if num_samples is not None:
        indices = indices[:num_samples]
    prompts = [all_prompts[i] for i in indices]
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]


def _subsample(prompts: list[str], labels: list[int], num_samples: int | None, seed: int) -> tuple[list[str], list[int]]:
    """Randomly subsample prompts and labels if num_samples is set."""
    if num_samples is not None and num_samples < len(prompts):
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(prompts))[:num_samples]
        prompts = [prompts[i] for i in indices]
        labels = [labels[i] for i in indices]
    return prompts, labels


def eval_jbb_behaviors_batches(batch_size: int = BATCH_SIZE, num_samples: int | None = None, seed: int = 42) -> Iterator[tuple[list[str], list[int]]]:
    """Yield batches of (prompts, labels) from JailbreakBench/JBB-Behaviors.
    Labels derived from splits: harmful=1, benign=0."""
    dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    prompts, labels = [], []
    for sample in dataset["harmful"]:
        prompts.append(sample["Goal"])
        labels.append(1)
    for sample in dataset["benign"]:
        prompts.append(sample["Goal"])
        labels.append(0)
    prompts, labels = _subsample(prompts, labels, num_samples, seed)
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size], labels[i : i + batch_size]


def eval_xstest_batches(batch_size: int = BATCH_SIZE, num_samples: int | None = None, seed: int = 42) -> Iterator[tuple[list[str], list[int]]]:
    """Yield batches of (prompts, labels) from walledai/XSTest.
    Label field: 'label' (safe=0, unsafe=1)."""
    dataset = load_dataset("walledai/XSTest", split="test")
    prompts, labels = [], []
    for sample in dataset:
        prompts.append(sample["prompt"])
        labels.append(1 if sample["label"] == "unsafe" else 0)
    prompts, labels = _subsample(prompts, labels, num_samples, seed)
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size], labels[i : i + batch_size]


def eval_wildguardtest_batches(batch_size: int = BATCH_SIZE, num_samples: int | None = None, seed: int = 42) -> Iterator[tuple[list[str], list[int]]]:
    """Yield batches of (prompts, labels) from allenai/wildguardmix (wildguardtest subset).
    Label field: 'prompt_harm_label' (harmful=1, unharmful=0)."""
    dataset = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
    prompts, labels = [], []
    for sample in dataset:
        prompts.append(sample["prompt"])
        labels.append(1 if sample["prompt_harm_label"] == "harmful" else 0)
    prompts, labels = _subsample(prompts, labels, num_samples, seed)
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size], labels[i : i + batch_size]


def eval_expguardtest_batches(batch_size: int = BATCH_SIZE, num_samples: int | None = None, seed: int = 42) -> Iterator[tuple[list[str], list[int]]]:
    """Yield batches of (prompts, labels) from 6rightjade/expguardmix (expguardtest subset).
    Label field: 'prompt_label' (unsafe=1, safe=0)."""
    dataset = load_dataset("6rightjade/expguardmix", "expguardtest", split="test")
    prompts, labels = [], []
    for sample in dataset:
        prompts.append(sample["prompt"])
        labels.append(1 if sample["prompt_label"] == "unsafe" else 0)
    prompts, labels = _subsample(prompts, labels, num_samples, seed)
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size], labels[i : i + batch_size]


############ Activation extraction


@torch.no_grad()
def collect_activations(
    batch_iter: Iterator[list[str]],
    llm_model,
    llm_tokenizer,
    tracedict_config: dict,
    label: int,
    save_acts_batch_size: int = 32,
    name: str = "",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Extract flattened LLM activations from a batch iterator and pair with labels."""
    acts_list, labels_list = [], []
    n_samples = 0
    for i, texts in enumerate(batch_iter):
        activations = save_acts(
            hf_model=llm_model,
            hf_tokenizer=llm_tokenizer,
            text=texts,
            tracedict_config=tracedict_config,
            token_idx="last",
            batch_size=save_acts_batch_size,
        )
        # activations: (N, num_layers, dim) -> flatten to (N, num_layers*dim)
        flat = activations.reshape(activations.shape[0], -1).float()
        acts_list.append(flat)
        labels_list.append(torch.full((flat.shape[0],), label, dtype=torch.float32))
        n_samples += flat.shape[0]
        print(f"  [{name}] batch {i+1}: {n_samples} samples so far")
        torch.cuda.empty_cache()
    return acts_list, labels_list


@torch.no_grad()
def collect_activations_labeled(
    batch_iter: Iterator[tuple[list[str], list[int]]],
    llm_model,
    llm_tokenizer,
    tracedict_config: dict,
    save_acts_batch_size: int = 32,
    name: str = "",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Extract flattened LLM activations from a batch iterator that yields (texts, labels) tuples."""
    acts_list, labels_list = [], []
    n_samples = 0
    for i, (texts, labels) in enumerate(batch_iter):
        activations = save_acts(
            hf_model=llm_model,
            hf_tokenizer=llm_tokenizer,
            text=texts,
            tracedict_config=tracedict_config,
            token_idx="last",
            batch_size=save_acts_batch_size,
        )
        flat = activations.reshape(activations.shape[0], -1).float()
        acts_list.append(flat)
        labels_list.append(torch.tensor(labels, dtype=torch.float32))
        n_samples += flat.shape[0]
        print(f"  [{name}] batch {i+1}: {n_samples} samples so far")
        torch.cuda.empty_cache()
    return acts_list, labels_list


############ Main


def compute_metrics(y_true, y_pred):
    """Compute classification metrics. Returns dict."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "num_samples": int(len(y_true)),
    }


def main(
    num_samples: int = 1024,
    batch_size: int = BATCH_SIZE,
    save_acts_batch_size: int = 32,
    device: str = "cuda:0",
    seed: int = 42,
    checkpoint_dir: str = "results/linear_probe_baseline",
    out_dir: str = "results/linear_probe_eval",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load checkpoint ----
    print(f"Loading linear probe checkpoint from {checkpoint_dir}...")
    ckpt = torch.load(os.path.join(checkpoint_dir, "linear_probe.pt"), map_location=device)
    input_dim = ckpt["input_dim"]
    mean = ckpt["mean"].to(device)
    std = ckpt["std"].to(device)
    probe = torch.nn.Sequential(
        torch.nn.Linear(input_dim, input_dim * 2),
        torch.nn.ReLU(),
        torch.nn.Linear(input_dim * 2, 1),
    ).to(device)
    # probe = torch.nn.Linear(input_dim, 1).to(device)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()

    # ---- Load LLM ----
    print("Loading LLM...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map=device
    )
    llm_tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

    # ---- Get tracedict_config from GLP (load to CPU, then discard) ----
    print("Loading GLP model to obtain tracedict_config...")
    _glp = load_glp("generative-latent-prior/glp-llama8b-d6", device="cpu", checkpoint="final")
    tracedict_config = _glp.tracedict_config
    del _glp

    # ---- Define data sources ----
    # Fixed-label sources (all samples share a single label)
    fixed_label_sources = {
        "vanilla_benign": (
            eval_wildjailbreak_batches(batch_size, num_samples=num_samples, benign=True, data_type_prefix="vanilla", seed=seed),
            0,
        ),
        "vanilla_harmful": (
            eval_wildjailbreak_batches(batch_size, num_samples=num_samples, benign=False, data_type_prefix="vanilla", seed=seed),
            1,
        ),
    }

    # Per-sample-label sources (each sample has its own label)
    labeled_sources = {
        "jbb_behaviors": eval_jbb_behaviors_batches(batch_size, num_samples=num_samples, seed=seed),
        "xstest": eval_xstest_batches(batch_size, num_samples=num_samples, seed=seed),
        "wildguardtest": eval_wildguardtest_batches(batch_size, num_samples=num_samples, seed=seed),
        "expguardtest": eval_expguardtest_batches(batch_size, num_samples=num_samples, seed=seed),
    }

    # ---- Collect activations & predictions per source ----
    print("\n=== Extracting eval activations ===")
    source_preds = {}
    source_labels = {}
    source_raw_acts = {}  # store raw (unnormalized) activations for PCA

    for src_name, (batch_iter, label) in fixed_label_sources.items():
        print(f"\n--- Source: {src_name} (label={label}) ---")
        acts, labels = collect_activations(
            batch_iter, llm_model, llm_tokenizer, tracedict_config,
            label=label, save_acts_batch_size=save_acts_batch_size, name=src_name,
        )
        if not acts:
            print(f"  WARNING: no samples collected for {src_name}, skipping")
            continue

        X_raw = torch.cat(acts, dim=0)
        y = torch.cat(labels, dim=0)
        source_raw_acts[src_name] = X_raw.numpy()

        X = X_raw.to(device)
        X = (X - mean) / std.clamp(min=1e-8)

        with torch.no_grad():
            logits = probe(X).squeeze(-1)
            preds = (logits > 0).cpu().numpy().astype(int)
        y_np = y.numpy().astype(int)

        source_preds[src_name] = preds
        source_labels[src_name] = y_np
        print(f"  collected {len(preds)} samples")
        torch.cuda.empty_cache()

    for src_name, batch_iter in labeled_sources.items():
        print(f"\n--- Source: {src_name} (per-sample labels) ---")
        acts, labels = collect_activations_labeled(
            batch_iter, llm_model, llm_tokenizer, tracedict_config,
            save_acts_batch_size=save_acts_batch_size, name=src_name,
        )
        if not acts:
            print(f"  WARNING: no samples collected for {src_name}, skipping")
            continue

        X_raw = torch.cat(acts, dim=0)
        y = torch.cat(labels, dim=0)
        source_raw_acts[src_name] = X_raw.numpy()

        X = X_raw.to(device)
        X = (X - mean) / std.clamp(min=1e-8)

        with torch.no_grad():
            logits = probe(X).squeeze(-1)
            preds = (logits > 0).cpu().numpy().astype(int)
        y_np = y.numpy().astype(int)

        source_preds[src_name] = preds
        source_labels[src_name] = y_np
        print(f"  collected {len(preds)} samples")
        torch.cuda.empty_cache()

    # ---- Compute metrics per eval group ----
    print("\n=== Evaluation ===")
    eval_groups = {
        "vanilla": ["vanilla_benign", "vanilla_harmful"],
        "jbb_behaviors": ["jbb_behaviors"],
        "xstest": ["xstest"],
        "wildguardtest": ["wildguardtest"],
        "expguardtest": ["expguardtest"],
    }

    results = {}
    all_preds, all_labels = [], []

    for group_name, src_names in eval_groups.items():
        group_preds = np.concatenate([source_preds[s] for s in src_names if s in source_preds])
        group_labels = np.concatenate([source_labels[s] for s in src_names if s in source_labels])
        group_metrics = compute_metrics(group_labels, group_preds)
        results[group_name] = group_metrics
        all_preds.append(group_preds)
        all_labels.append(group_labels)
        print(f"\n  {group_name}: accuracy={group_metrics['accuracy']:.4f}  precision={group_metrics['precision']:.4f}  recall={group_metrics['recall']:.4f}  f1={group_metrics['f1']:.4f}  n={group_metrics['num_samples']}")

    # Overall across all groups
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    results["overall"] = compute_metrics(all_labels, all_preds)

    print("\n=== Overall Results ===")
    for k, v in results["overall"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # ---- PCA scatter plots of original LLM activations per dataset ----
    print("\n=== PCA plots ===")
    # Fit PCA on all activations combined
    all_acts_np = np.concatenate(list(source_raw_acts.values()), axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_acts_np)

    # Group sources for PCA plots: vanilla_benign + vanilla_harmful share one plot
    pca_groups = {
        "vanilla": ["vanilla_benign", "vanilla_harmful"],
        "jbb_behaviors": ["jbb_behaviors"],
        "xstest": ["xstest"],
        "wildguardtest": ["wildguardtest"],
        "expguardtest": ["expguardtest"],
    }

    # Per-group PCA scatter, colored by label
    for group_name, src_names in pca_groups.items():
        group_acts = [source_raw_acts[s] for s in src_names if s in source_raw_acts]
        group_lbls = [source_labels[s] for s in src_names if s in source_labels]
        if not group_acts:
            continue
        acts_2d = pca.transform(np.concatenate(group_acts, axis=0))
        labels_np = np.concatenate(group_lbls)
        fig, ax = plt.subplots(figsize=(8, 6))
        for lbl, color, lbl_name in [(0, "#55AA55", "benign (0)"), (1, "#DD5555", "harmful (1)")]:
            mask = labels_np == lbl
            if mask.any():
                ax.scatter(acts_2d[mask, 0], acts_2d[mask, 1], s=8, alpha=0.5,
                           color=color, label=lbl_name)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(f"PCA of LLM Activations — {group_name}")
        ax.legend(markerscale=3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"pca_{group_name}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"  Saved {path}")

    # Combined PCA scatter (all datasets, colored by label)
    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, color, lbl_name in [(0, "#55AA55", "benign (0)"), (1, "#DD5555", "harmful (1)")]:
        coords_list = []
        for src_name in source_raw_acts:
            acts_2d = pca.transform(source_raw_acts[src_name])
            mask = source_labels[src_name] == lbl
            if mask.any():
                coords_list.append(acts_2d[mask])
        if coords_list:
            coords = np.concatenate(coords_list, axis=0)
            ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.4,
                       color=color, label=lbl_name)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("PCA of LLM Activations — All Datasets")
    ax.legend(markerscale=3)
    fig.tight_layout()
    path = os.path.join(out_dir, "pca_all.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")

    # ---- Save results ----
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    fire.Fire(main)
