import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Iterator
import numpy as np
import fire

from glp.denoiser import load_glp
from glp.utils_acts import save_acts
from glp import flow_matching

############ Config

BATCH_SIZE = 32

############ Dataset iterators (identical to train_linear_probe_anomaly_detection.py)


def train_fineweb_batches(batch_size: int = BATCH_SIZE, num_samples: int = 1024) -> Iterator[list[str]]:
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


def train_wildjailbreak_batches(batch_size: int = BATCH_SIZE, num_samples: int = 1024, benign=False, seed: int = 42, skip: int = 0) -> Iterator[list[str]]:
    """Yield batches of vanilla prompts from allenai/wildjailbreak (randomly sampled)."""
    label = "harmful" if not benign else "benign"
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    adv_harmful = dataset.filter(lambda x: x["data_type"] == f"vanilla_{label}")
    all_prompts = [sample["vanilla"] for sample in adv_harmful["train"]]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_prompts))
    prompts = [all_prompts[i] for i in indices[skip : skip + num_samples]]
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]


############ Activation extraction (via diffusion reconstruction)


@torch.no_grad()
def collect_deltas(
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
    """Compute per-sample reconstruction deltas from the diffusion model and pair with labels."""
    acts_list, labels_list = [], []
    n_samples = 0
    for i, texts in enumerate(batch_iter):
        # 1. Raw LLM activations
        activations = save_acts(
            hf_model=llm_model,
            hf_tokenizer=llm_tokenizer,
            text=texts,
            tracedict_config=diffusion_model.tracedict_config,
            token_idx="last",
            batch_size=save_acts_batch_size,
        )
        activations = activations.to(device=device, dtype=torch.bfloat16)

        # 2. Normalize → noise → denoise → denormalize
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

        # 3. Per-sample delta vector: shape (N, D)
        deltas = (activations - reconstructed_acts.to(activations.dtype)).reshape(activations.shape[0], -1).float().cpu()
        acts_list.append(deltas)
        labels_list.append(torch.full((deltas.shape[0],), label, dtype=torch.float32))
        n_samples += deltas.shape[0]
        print(f"  [{name}] batch {i+1}: {n_samples} samples so far")
        torch.cuda.empty_cache()
    return acts_list, labels_list


############ Main


def main(
    num_samples_train: int = 1024,
    batch_size: int = BATCH_SIZE,
    save_acts_batch_size: int = 32,
    noise_level: float = 0.5,
    num_timesteps: int = 100,
    lr: float = 1e-3,
    epochs: int = 10,
    train_batch_size: int = 256,
    device: str = "cuda:0",
    seed: int = 42,
    out_dir: str = "results/diffusion_probe_baseline",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load LLM ----
    print("Loading LLM...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Meta-Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map=device
    )
    llm_tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

    # ---- Load GLP (kept in memory for reconstruction) ----
    print("Loading GLP diffusion model...")
    diffusion_model = load_glp("generative-latent-prior/glp-llama8b-d6", device=device, checkpoint="final")

    ckwargs = dict(
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        diffusion_model=diffusion_model,
        noise_level=noise_level,
        num_timesteps=num_timesteps,
        save_acts_batch_size=save_acts_batch_size,
        device=device,
    )

    # ---- Collect training activations & labels ----
    print("\n=== Extracting training activations (diffusion-reconstructed) ===")
    all_acts, all_labels = [], []

    # WildJailbreak vanilla benign (label 0)
    a, l = collect_deltas(
        train_wildjailbreak_batches(batch_size, num_samples_train, benign=True, seed=seed),
        label=0, name="wildjailbreak_benign", **ckwargs,
    )
    all_acts.extend(a); all_labels.extend(l)

    # WildJailbreak vanilla harmful (label 1)
    a, l = collect_deltas(
        train_wildjailbreak_batches(batch_size, num_samples_train, benign=False, seed=seed),
        label=1, name="wildjailbreak_harmful", **ckwargs,
    )
    all_acts.extend(a); all_labels.extend(l)

    X_train = torch.cat(all_acts, dim=0)
    y_train = torch.cat(all_labels, dim=0)
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  benign={int((y_train == 0).sum())}  harmful={int((y_train == 1).sum())}")

    # Free models from GPU
    del llm_model, diffusion_model
    torch.cuda.empty_cache()

    # ---- Standardize features (fit on train) ----
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0).clamp(min=1e-8)
    X_train = (X_train - mean) / std

    # ---- Train linear probe ----
    print(f"\n=== Training linear probe ({epochs} epochs, lr={lr}) ===")
    input_dim = X_train.shape[1]
    probe = torch.nn.Sequential(
        torch.nn.Linear(input_dim, input_dim * 2),
        torch.nn.ReLU(),
        torch.nn.Linear(input_dim * 2, 1),
    ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    criterion = torch.nn.BCEWithLogitsLoss()

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)

    for epoch in range(epochs):
        probe.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = probe(xb).squeeze(-1)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        print(f"  Epoch {epoch + 1}/{epochs}  loss={total_loss / n:.4f}")

    # ---- Save weights ----
    weights_path = os.path.join(out_dir, "linear_probe.pt")
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "mean": mean,
            "std": std,
            "input_dim": input_dim,
            "noise_level": noise_level,
            "num_timesteps": num_timesteps,
        },
        weights_path,
    )
    print(f"Saved probe weights to {weights_path}")


if __name__ == "__main__":
    fire.Fire(main)
