import os
import json
import urllib.request
import torch
import random
import numpy as np
from pathlib import Path
from typing import Iterator, Literal
from glp import flow_matching
# import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk
from glp.denoiser import load_glp
from glp.utils_acts import save_acts
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def extract_activations(
    texts: list[str],
    noise_level: float,
    num_timesteps: int,
    llm_model,
    llm_tokenizer,
    diffusion_model,
    layers: list,
    device: str = "cuda:0",
    batch_size: int = 8) -> dict:

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

    for li, actual_layer in enumerate(layers):
        layer_acts = activations[:, li:li+1, :]  # (N, 1, D)
        normalized = diffusion_model.normalizer.normalize(layer_acts, layer_idx=actual_layer)
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
            layer_idx=actual_layer,
        )
        reconstructed_layers.append(diffusion_model.normalizer.denormalize(reconstructed, layer_idx=actual_layer))
        noisy_denorm_layers.append(diffusion_model.normalizer.denormalize(noisy, layer_idx=actual_layer))

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
    all_prompts = [sample["vanilla"] for sample in filtered["train"] if sample["vanilla"] is not None]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_prompts))
    prompts = [all_prompts[i] for i in indices[skip : skip + num_samples]]
    for i in range(0, len(prompts), batch_size):
        batch = [p for p in prompts[i : i + batch_size] if p is not None]
        if batch:
            yield batch

def sg_bench_batches(batch_size: int = 8, num_samples: int = 1024,
                                   seed: int = 42, skip: int = 0,
                                   repo_id: str = "ddidacus/SG-Bench-malicious-instructions") -> Iterator[list[str]]:
    """Yield batches of malicious instruction prompts from the custom dataset."""
    dataset = load_dataset(repo_id, split="train")
    all_prompts = [sample["query"] for sample in dataset if sample["query"] is not None]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_prompts))
    prompts = [all_prompts[i] for i in indices[skip : skip + num_samples]]
    for i in range(0, len(prompts), batch_size):
        batch = [p for p in prompts[i : i + batch_size] if p is not None]
        if batch:
            yield batch

def jailbreakbench_batches(
    batch_size: int = 8,
    num_samples: int = 8,
    method: str = "PAIR",
    model_name: str = "vicuna-13b-v1.5",
    attack_type: Literal["black_box", "white_box"] = "black_box",
    field: Literal["prompt", "goal"] = "prompt",
    only_jailbroken: bool = False,
    seed: int = 42, skip: int = 0,
) -> Iterator[list[str]]:
    """Yield batches of prompts from the JailbreakBench artifacts.

    Fetches directly from raw.githubusercontent.com — no extra package needed.

    Args:
        method:         attack method, e.g. "PAIR", "GCG", "JBC", "DSN".
        model_name:     target model, e.g. "vicuna-13b-v1.5", "llama-2-7b-chat-hf".
        attack_type:    "black_box" or "white_box" (GCG uses white_box).
        field:          "prompt" for the adversarial jailbreak text,
                        "goal" for the raw harmful instruction.
        only_jailbroken: if True, keep only entries where jailbroken=True.
    """
    url = (
        f"https://raw.githubusercontent.com/JailbreakBench/artifacts/main"
        f"/attack-artifacts/{method}/{attack_type}/{model_name}.json"
    )
    with urllib.request.urlopen(url) as response:
        data = json.load(response)

    entries = data["jailbreaks"]
    if only_jailbroken:
        entries = [e for e in entries if e.get("jailbroken")]
    all_prompts = [e[field] for e in entries if e.get(field) is not None]

    rng = np.random.RandomState(seed)
    all_prompts = [all_prompts[i] for i in rng.permutation(len(all_prompts))]
    all_prompts = all_prompts[skip : skip + num_samples]

    for i in range(0, len(all_prompts), batch_size):
        batch = all_prompts[i : i + batch_size]
        if batch:
            yield batch

def harmeval_batches(batch_size: int = 8, num_samples: int = 512,
                     seed: int = 42, skip: int = 0,
                     topic: str | None = None) -> Iterator[list[str]]:
    """Yield batches of harmful prompts from SoftMINER-Group/HarmEval (550 total).

    Args:
        topic: optional filter, e.g. "Illegal Activity". If None, use all topics.
    """
    dataset = load_dataset("SoftMINER-Group/HarmEval", split="train")
    if topic is not None:
        dataset = dataset.filter(lambda x: x["Topic"] == topic)
    all_prompts = [sample["Question"] for sample in dataset if sample["Question"] is not None]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_prompts))
    prompts = [all_prompts[i] for i in indices[skip : skip + num_samples]]
    for i in range(0, len(prompts), batch_size):
        batch = [p for p in prompts[i : i + batch_size] if p is not None]
        if batch:
            yield batch

def harmeval_gcg_batches(batch_size: int = 8, num_samples: int = 512,
                         seed: int = 42, skip: int = 0,
                         data_dir: str = "data/harmeval_gcg/harmeval_gcg",
                         field: Literal["adv_prompt", "question"] = "adv_prompt") -> Iterator[list[str]]:
    """Yield batches of GCG-adversarial prompts from the local harmeval_gcg dataset.

    Args:
        data_dir: path to the dataset directory (Arrow format).
        field:    "adv_prompt" for the GCG suffix-appended prompt,
                  "question" for the original harmful question.
    """
    abs_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)
    dataset = load_from_disk(abs_data_dir)
    all_prompts = [sample[field] for sample in dataset if sample[field] is not None]
    rng = np.random.RandomState(seed)
    all_prompts = [all_prompts[i] for i in rng.permutation(len(all_prompts))]
    all_prompts = all_prompts[skip : skip + num_samples]
    for i in range(0, len(all_prompts), batch_size):
        batch = all_prompts[i : i + batch_size]
        if batch:
            yield batch

def adversarial_wildjailbreak_batches(batch_size: int = 8, num_samples: int = 1024,
                                       benign=False, seed: int = 42, skip:int = 0) -> Iterator[list[str]]:
    """Yield batches of adversarial prompts (random permutation)."""
    label = "benign" if benign else "harmful"
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    filtered = dataset.filter(lambda x: x["data_type"] == f"adversarial_{label}")
    all_prompts = [sample["adversarial"] for sample in filtered["train"] if sample["adversarial"] is not None]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_prompts))
    prompts = [all_prompts[i] for i in indices[skip: skip + num_samples]]
    for i in range(0, len(prompts), batch_size):
        batch = [p for p in prompts[i : i + batch_size] if p is not None]
        if batch:
            yield batch

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
        coin = random.choice([True, False])
        if not coin: continue
        text = sample["text"]
        if text is None:
            continue
        batch.append(text)
        total += 1
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def encode_prompts(prompts, batch_size, llm_model, llm_tokenizer, diffusion_model, layers:list,
                   noise_level: float, num_timesteps: int, device="cuda:0") -> dict:
    # returns only the layers set in the config
    return extract_activations(
        texts=prompts,
        noise_level=noise_level,
        num_timesteps=num_timesteps,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        diffusion_model=diffusion_model,
        batch_size=batch_size,
        device=device,
        layers=layers,
    )

def main(
    gpu_id: int,
    noise_level: float,
    num_timesteps: int,
    layers: list[int],
    out_dir: str,
    model: str = "1b",
    data_selection: str = "all",
    num_samples: int = 128,
):
    torch.manual_seed(42)
    random.seed(42)

    device = f"cuda:{gpu_id}"

    # parameters
    if model == "1b":
        batch_size = 128
        llm_model_id = "unsloth/Llama-3.2-1B"
        glp_model_id = "generative-latent-prior/glp-llama1b-d12-multi"

    elif model == "8b":
        batch_size = 32
        llm_model_id = "meta-llama/Llama-3.1-8B"
        glp_model_id = "generative-latent-prior/glp-llama8b-d6"
    
    else:
        raise NotImplementedError()

    # model
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    diffusion_model = load_glp(
        glp_model_id, device=device, checkpoint="final"
    )
    diffusion_model.tracedict_config.layers = layers

    print("================================================")
    print(f"[+] LLM: \t\t{llm_model_id}")
    print(f"[+] GLP: \t\t{glp_model_id}")
    print(f"[+] batch_size: \t{batch_size}")
    print(f"[+] num_samples: \t{num_samples}")
    print(f"[+] layers: \t\t{diffusion_model.tracedict_config.layers}")
    print(f"[+] noise_level: \t{noise_level}")
    print(f"[+] num_timesteps: \t{num_timesteps}")
    print(f"[+] out_dir: \t\t{out_dir}")
    print("================================================")

    # destination dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # good prompts
    indistribution_prompts = fineweb_batches(skip=gpu_id*num_samples, batch_size=batch_size, num_samples=num_samples)
    
    # bad prompts
    kwargs = {
        "skip": gpu_id*num_samples, 
        "batch_size": batch_size, 
        "num_samples": num_samples
    }
    dataset_selections = {
        "all_semantic": [
            harmeval_batches,
            jailbreakbench_batches,
            sg_bench_batches,
            vanilla_wildjailbreak_batches,
        ],
        "harmeval": [harmeval_batches],
        "harmeval_gcg": [harmeval_gcg_batches],
        "jailbreakbench": [jailbreakbench_batches],
        "sgbench": [sg_bench_batches],
        "vanilla_wildjailbreak": [vanilla_wildjailbreak_batches],
    }
    assert data_selection in dataset_selections, f"Invalid dataset selection, choose: {list(dataset_selections.keys())}"
    selection = dataset_selections[data_selection]

    # extract bad prompts
    bad_prompts_batches = []
    for batch_fn in selection:
        bad_prompts_batches.extend(batch_fn(**kwargs))

    # loaded datasets (good, bad)
    good_set = list(indistribution_prompts)
    bad_set = list(bad_prompts_batches)

    print(f"Good prompts: {sum(len(b) for b in good_set)}")
    print(f"Bad prompts: {sum(len(b) for b in bad_set)}")

    # encoding
    print(f"======= Encoding the good set (GPU {gpu_id}) =======")
    batch_good_set = []
    for batch in good_set:
        batch_good_set.append(
            encode_prompts(batch, batch_size, llm_model, llm_tokenizer, diffusion_model,
                           layers=layers, noise_level=noise_level, num_timesteps=num_timesteps, device=device)
        )
    activations_good_set = torch.cat([x["activations"] for x in batch_good_set], dim=0)
    reconstructed_good_set = torch.cat([x["reconstructed_activations"] for x in batch_good_set], dim=0)

    print(f"======= Encoding the bad set (GPU {gpu_id}) =======")
    batch_bad_set = []
    for batch in bad_set:
        batch_bad_set.append(
            encode_prompts(batch, batch_size, llm_model, llm_tokenizer, diffusion_model,
                           layers=layers, noise_level=noise_level, num_timesteps=num_timesteps, device=device)
        )
    activations_bad_set = torch.cat([x["activations"] for x in batch_bad_set], dim=0)
    reconstructed_bad_set = torch.cat([x["reconstructed_activations"] for x in batch_bad_set], dim=0)

    # storing files
    out_file = out_dir + f"/results_{gpu_id}.th"
    torch.save({
        "activations_good_set": activations_good_set,
        "reconstructed_good_set": reconstructed_good_set,
        "activations_bad_set": activations_bad_set,
        "reconstructed_bad_set": reconstructed_bad_set,
    }, out_file)

if __name__ == "__main__":
    import sys
    import yaml
    import fire

    def run(config: str = "eval_config.yaml", gpu_id: int = 0):
        import shutil
        with open(config) as f:
            cfg = yaml.safe_load(f)
        if gpu_id == 0:
            out_dir = cfg["out_dir"]
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(config, Path(out_dir) / Path(config).name)
        main(
            gpu_id=gpu_id,
            noise_level=cfg["noise_level"],
            num_timesteps=cfg["num_timesteps"],
            layers=cfg["layers"],
            out_dir=cfg["out_dir"],
            model=cfg["model"],
            data_selection=cfg["data"],
            num_samples=cfg["num_samples"],
        )

    fire.Fire(run)