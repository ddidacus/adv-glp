import json
import os
import random
import numpy as np
import torch
from pathlib import Path

from tqdm import tqdm
from glp import flow_matching
from glp.denoiser import load_glp
from glp.utils_acts import save_acts
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_by_component import (
    fineweb_batches,
    harmeval_batches,
    harmeval_gcg_batches,
    jailbreakbench_batches,
    sg_bench_batches,
    vanilla_wildjailbreak_batches,
    adversarial_wildjailbreak_batches,
)


def _threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """ compute eval metrics given labels, scores, threshold """
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    tpr       = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return dict(threshold=threshold, precision=precision,
                recall=tpr, tpr=tpr, fpr=fpr, fnr=fnr,
                tp=tp, fp=fp, fn=fn, tn=tn)


def _find_youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """ compute the youden threshold that maximizes AUROC """
    from sklearn.metrics import roc_curve
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)
    return float(thresholds[(tpr_arr - fpr_arr).argmax()])


def _classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    score_name: str,
    verbose: bool = True,
    target_tprs: tuple = tuple(np.arange(0.60, 1.00, 0.05).round(2)),
    youden_threshold: float | None = None,
) -> dict:
    """ final scoring function: compute metrics on fixed youden threshold + some TPR thresholds """

    # AUROC
    from sklearn.metrics import roc_auc_score, roc_curve
    auroc = float(roc_auc_score(labels, scores))
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)

    if youden_threshold is None:
        youden_threshold = float(thresholds[(tpr_arr - fpr_arr).argmax()])

    # threshold at Youden's J (from calibration set or this set)
    youden = _threshold_metrics(labels, scores, youden_threshold)

    # thresholds at fixed TPR
    tpr_results = {}
    for target in target_tprs:
        idx = np.searchsorted(tpr_arr, target)
        idx = min(idx, len(thresholds) - 1)
        tpr_results[f"tpr{int(target*100)}"] = _threshold_metrics(
            labels, scores, float(thresholds[idx])
        )

    if verbose:
        print(f"\n  [{score_name}]")
        print(f"    AUROC: {auroc:.4f}")
        print(f"    @Youden(thr={youden['threshold']:.4f}): "
              f"P={youden['precision']:.3f}  TPR={youden['tpr']:.3f}  "
              f"FPR={youden['fpr']:.3f}  FNR={youden['fnr']:.3f}  "
              f"TP={youden['tp']}  FP={youden['fp']}  FN={youden['fn']}  TN={youden['tn']}")
        for key, m in tpr_results.items():
            print(f"    @{key}  (thr={m['threshold']:.4f}): "
                  f"P={m['precision']:.3f}  TPR={m['tpr']:.3f}  "
                  f"FPR={m['fpr']:.3f}  FNR={m['fnr']:.3f}  "
                  f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

    return dict(auroc=auroc, youden=youden, **tpr_results)


def _make_plots(out_dir: Path, named_scores: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    """ generate all plots: AUROC, TPR/FNR """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(named_scores), 1)))

    # ── ROC curve ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    for (name, labels, scores), color in zip(named_scores, colors):
        fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        ax.plot(fpr_arr, tpr_arr, label=f"{name}  (AUC={auroc:.3f})", color=color)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    # ── TPR and FNR vs threshold ───────────────────────────────────────────
    fig, axes = plt.subplots(1, len(named_scores), figsize=(5 * len(named_scores), 4),
                             squeeze=False)
    for idx, ((name, labels, scores), color) in enumerate(zip(named_scores, colors)):
        ax = axes[0, idx]
        fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)
        # roc_curve adds an extra point; align
        fnr_arr = 1.0 - tpr_arr
        # drop the sentinel threshold at +inf
        thr = thresholds[1:]
        tpr_plot = tpr_arr[1:]
        fnr_plot = fnr_arr[1:]
        ax.plot(thr, tpr_plot, label="TPR", color="steelblue")
        ax.plot(thr, fnr_plot, label="FNR", color="tomato")
        # mark Youden's J
        youden_idx = (tpr_arr - fpr_arr).argmax()
        if youden_idx > 0:
            ax.axvline(thresholds[youden_idx], color="gray", ls="--", lw=0.9, label="Youden")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Rate")
        ax.set_title(name, fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle("TPR and FNR vs Threshold", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "tpr_fnr_threshold.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved: {out_dir / 'roc_curves.png'}  |  {out_dir / 'tpr_fnr_threshold.png'}")


def _chunk(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def load_hf_splits(
    repo_id: str,
    gpu_id: int,
    num_gpus: int,
    num_samples: int | None,
    batch_size: int,
    reference_num_samples: int = 0,
) -> dict:
    """ load dataset gpu-shards, merge and organize splits logically """
    import math
    from datasets import load_dataset as _lds
    ds = _lds(repo_id)

    def _dedup(split: str) -> list[str]:
        prompts = ds[split]["prompt"]
        seen, unique = set(), []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        if len(unique) < len(prompts) and gpu_id == 0:
            print(f"  [load_hf_splits] dedup '{split}': {len(prompts)} → {len(unique)} prompts")
        return unique

    def _shard(split: str) -> list[str]:
        prompts = _dedup(split)
        n     = num_samples if num_samples is not None else math.ceil(len(prompts) / num_gpus)
        start = gpu_id * n
        end   = min(start + n, len(prompts))
        if start >= len(prompts):
            raise ValueError(
                f"GPU shard is empty for split '{split}': gpu_id={gpu_id}, n={n}, "
                f"start={start} >= split_size={len(prompts)}. "
                f"Reduce num_samples or num_gpus."
            )
        return prompts[start:end]

    ref_prompts: list[str] = []
    if reference_num_samples > 0:
        fit_prompts = _dedup("benign_train")
        n_fit = num_samples if num_samples is not None else math.ceil(len(fit_prompts) / num_gpus)
        ref_start = num_gpus * n_fit + gpu_id * reference_num_samples
        if ref_start >= len(fit_prompts):
            ref_start = gpu_id * n_fit
        ref_end = min(ref_start + reference_num_samples, len(fit_prompts))
        ref_prompts = fit_prompts[ref_start:ref_end]

    return dict(
        good_prompts      = _shard("benign_train"),
        good_eval_prompts = _shard("benign_test"),
        cal_bad_prompts   = _shard("adversarial_calibration"),
        test_bad_prompts  = _shard("adversarial_test"),
        ref_prompts       = ref_prompts,
    )


def extract_activations(
    texts: list[str],
    llm_model,
    llm_tokenizer,
    diffusion_model,
    device: str = "cuda:0",
    batch_size: int = 8,
    token_pooling: str = "last",
) -> torch.Tensor:
    """ run the LLM and return cached (N, L, D) activations on `device` in bfloat16 """
    activations = save_acts(
        hf_model=llm_model,
        hf_tokenizer=llm_tokenizer,
        text=texts,
        tracedict_config=diffusion_model.tracedict_config,
        token_idx=token_pooling,
        batch_size=batch_size,
    )
    return activations.to(device=device, dtype=torch.bfloat16)


def extract_log_probs(
    texts: list[str],
    llm_model,
    llm_tokenizer,
    diffusion_model,
    layers: list,
    num_steps: int,
    num_hutchinson_samples: int,
    device: str = "cuda:0",
    batch_size: int = 8,
    method: str = "hutchinson",
    reference_activations: torch.Tensor | None = None,
    K: int = 5,
    num_sigma_bins: int = 100,
    normalize: bool = False,
) -> dict:
    """ extract log probs depending on method (path integral / dte non-parametric posterior)"""
    activations = extract_activations(
        texts, llm_model, llm_tokenizer, diffusion_model,
        device=device, batch_size=batch_size,
    )

    layer_log_probs = []
    layer_probs = []
    layer_expected_sigma = []  # DTE only; None otherwise
    for li, actual_layer in enumerate(layers):
        layer_acts = activations[:, li : li + 1, :]  # (N, 1, D)
        kwargs = dict(
            method=method,
            layer_idx=actual_layer,
            normalize=normalize,
        )
        if method == "hutchinson":
            kwargs.update(
                num_steps=num_steps,
                num_hutchinson_samples=num_hutchinson_samples,
            )
        elif method == "dte":
            assert reference_activations is not None, \
                "method='dte' requires reference_activations"
            kwargs.update(
                reference_latents=reference_activations[:, li : li + 1, :],
                K=K,
                num_sigma_bins=num_sigma_bins,
            )
        result = flow_matching.log_prob(diffusion_model, layer_acts, **kwargs)
        layer_log_probs.append(result.log_prob)
        layer_probs.append(result.prob)
        if method == "dte":
            layer_expected_sigma.append(result.expected_sigma.cpu())

    log_probs = torch.stack(layer_log_probs, dim=1)  # (N, num_layers)
    probs = torch.stack(layer_probs, dim=1)          # (N, num_layers)
    out = {
        "log_probs": log_probs.cpu(),
        "probs": probs.cpu(),
        "activations": activations.cpu().float(),
    }
    if layer_expected_sigma:
        out["expected_sigma"] = torch.stack(layer_expected_sigma, dim=1).cpu()
    return out


@torch.no_grad()
def extract_reconstruction_errors(
    texts: list[str],
    llm_model,
    llm_tokenizer,
    diffusion_model,
    layers: list[int],
    noise_level: float,
    num_timesteps: int,
    device: str = "cuda:0",
    batch_size: int = 8,
) -> torch.Tensor:
    """Reconstruct activations via GLP and return per-layer L2 errors (N, L). Higher = more anomalous."""
    activations = save_acts(
        hf_model=llm_model,
        hf_tokenizer=llm_tokenizer,
        text=texts,
        tracedict_config=diffusion_model.tracedict_config,
        token_idx="last",
        batch_size=batch_size,
    ).to(device=device, dtype=torch.bfloat16)  # (N, L, D)

    errors_per_layer = []
    for li, actual_layer in enumerate(layers):
        layer_acts = activations[:, li : li + 1, :]  # (N, 1, D)
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
        reconstructed = diffusion_model.normalizer.denormalize(reconstructed, layer_idx=actual_layer)
        err = torch.norm(
            (layer_acts - reconstructed).float().reshape(layer_acts.shape[0], -1),
            p=2, dim=1,
        )  # (N,)
        errors_per_layer.append(err.cpu())

    return torch.stack(errors_per_layer, dim=1)  # (N, L)


def main(
    gpu_id: int,
    layers: list[int],
    out_dir: str,
    model: str = "1b",
    data_selection: str = "all_semantic",
    metric_data_selection: str | None = None,
    num_samples: int | None = None,
    num_steps: int = 100,
    num_hutchinson_samples: int = 1,
    method: str = "hutchinson",
    reference_num_samples: int = 512,
    glp_sample_steps: int = 100,
    dte_K: int = 5,
    dte_num_sigma_bins: int = 100,
    normalize: bool = False,
    noise_level: float = 0.5,
    rec_num_timesteps: int = 100,
    hf_dataset: str | None = None,
    num_gpus: int = 4,
    batch_size: int | None = None,
):
    torch.manual_seed(42)
    random.seed(42)

    device = f"cuda:{gpu_id}"

    # load models

    print("[+] Loading models...")

    if model == "1b":
        _default_batch_size = 64
        llm_model_id = "unsloth/Llama-3.2-1B"
        glp_model_id = "generative-latent-prior/glp-llama1b-d12-multi"
    elif model == "8b":
        _default_batch_size = 32
        llm_model_id = "meta-llama/Llama-3.1-8B"
        glp_model_id = "generative-latent-prior/glp-llama8b-d6"
    else:
        raise NotImplementedError()
    if batch_size is None:
        batch_size = _default_batch_size

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    diffusion_model = load_glp(glp_model_id, device=device, checkpoint="final")
    diffusion_model.tracedict_config.layers = layers

    print("================================================")
    print(f"[+] LLM:            {llm_model_id}")
    print(f"[+] GLP:            {glp_model_id}")
    print(f"[+] batch_size:     {batch_size}")
    print(f"[+] num_samples:    {num_samples}")
    print(f"[+] layers:         {layers}")
    print(f"[+] num_steps:      {num_steps}")
    print(f"[+] hutch_samples:  {num_hutchinson_samples}")
    print(f"[+] data_selection: {data_selection}")
    print(f"[+] metric_data:    {metric_data_selection or '(same as data_selection)'}")
    print(f"[+] method:         {method}")
    if method in ("dte", "dte_glp"):
        print(f"[+] dte_K:          {dte_K}")
        print(f"[+] dte_sigma_bins: {dte_num_sigma_bins}")
        print(f"[+] ref_samples:    {reference_num_samples}")
    if method == "dte_glp":
        print(f"[+] glp_sample_steps: {glp_sample_steps}")
    if method == "reconstruction_error":
        print(f"[+] noise_level:    {noise_level}")
        print(f"[+] rec_timesteps:  {rec_num_timesteps}")
    print(f"[+] out_dir:        {out_dir}")
    print("================================================")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # load hf dataset
    if hf_dataset:
        data_selection        = "hf_test"
        metric_data_selection = "hf_calibration"
        _hf = load_hf_splits(
            hf_dataset, gpu_id, num_gpus, num_samples, batch_size,
            reference_num_samples=(reference_num_samples if method == "dte" else 0),
        )

    # load good prompts: benign_train for scoring context, benign_test (held-out) for eval
    if hf_dataset:
        good_batches      = _chunk(_hf["good_prompts"],      batch_size)
        good_eval_batches = _chunk(_hf["good_eval_prompts"], batch_size)
    else:
        assert num_samples is not None, "num_samples is required when not using hf_dataset"
        good_batches = list(
            fineweb_batches(skip=gpu_id * num_samples, batch_size=batch_size, num_samples=num_samples)
        )
        good_eval_batches = good_batches  # no held-out split in legacy path

    # if dte, set reference samples for KNN (non-parametric dte)
    reference_activations = None

    # knn ref samples are from the fineweb dataset
    if method == "dte":
        print("[+] Loading reference samples for kNN-DTE...")
        if hf_dataset:
            ref_batches = _chunk(_hf["ref_prompts"], batch_size)
            print(f"[+] Building DTE reference set from HF train split ({len(_hf['ref_prompts'])} samples)")
        else:
            reference_skip = gpu_id * reference_num_samples
            print(f"[+] Building DTE reference set from fineweb (skip={reference_skip}, N={reference_num_samples})")
            ref_batches = list(fineweb_batches(
                skip=reference_skip, batch_size=batch_size, num_samples=reference_num_samples,
            ))
        ref_chunks = []
        for batch in ref_batches:
            ref_chunks.append(extract_activations(
                batch, llm_model, llm_tokenizer, diffusion_model,
                device=device, batch_size=batch_size,
            ))
        reference_activations = torch.cat(ref_chunks, dim=0)  # (N_ref, L, D) on device
        print(f"    reference_activations: {tuple(reference_activations.shape)}")

    # knn ref samples are generated by GLP (approx. fineweb, more logically aligned for classification here)
    elif method == "dte_glp":
        print("[+] Sampling reference data for DTE-GLP...")
        d_input = diffusion_model.denoiser.model.d_input
        print(f"[+] Building DTE reference set by sampling kNN-DTE "
              f"(N={reference_num_samples}, steps={glp_sample_steps}, d_input={d_input})")
        per_layer_refs = []
        for actual_layer in layers:
            noise = torch.randn(reference_num_samples, 1, d_input,
                                device=device, dtype=torch.bfloat16)
            sampled = flow_matching.sample(
                diffusion_model, noise,
                num_timesteps=glp_sample_steps,
                layer_idx=actual_layer,
            )  # (N_ref, 1, D)
            per_layer_refs.append(sampled.cpu())
        # stack into (N_ref, num_layers, D) — same layout as fineweb reference_activations
        reference_activations = torch.cat(per_layer_refs, dim=1).to(device)
        print(f"    reference_activations: {tuple(reference_activations.shape)}")

    # test samples (adversarial)
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

    # just load from HF dataset
    if hf_dataset:
        bad_batches = _chunk(_hf["test_bad_prompts"], batch_size)

    # or use the data selection functions from raw sources (not gonna be used since the HF version)
    else:
        assert data_selection in dataset_selections, (
            f"Invalid dataset selection, choose: {list(dataset_selections.keys())}"
        )
        kwargs = {
            "skip": gpu_id * num_samples,
            "batch_size": batch_size,
            "num_samples": int(num_samples / len(dataset_selections[data_selection])),
        }
        bad_batches = []
        for batch_fn in dataset_selections[data_selection]:
            bad_batches.extend(batch_fn(**kwargs))

    print(f"Good train prompts: {sum(len(b) for b in good_batches)}")
    print(f"Good eval prompts:  {sum(len(b) for b in good_eval_batches)}")
    print(f"Bad prompts:        {sum(len(b) for b in bad_batches)}")

    is_recon = (method == "reconstruction_error")
    _score_method = "dte" if method == "dte_glp" else method
    common_kwargs = dict(
        layers=layers, num_steps=num_steps,
        num_hutchinson_samples=num_hutchinson_samples,
        device=device, batch_size=batch_size,
        method=_score_method,
        reference_activations=reference_activations,
        K=dte_K,
        num_sigma_bins=dte_num_sigma_bins,
        normalize=normalize,
    )
    _recon_kwargs = dict(
        layers=layers, noise_level=noise_level, num_timesteps=rec_num_timesteps,
        device=device, batch_size=batch_size,
    )

    def _extract(texts):
        if is_recon:
            return extract_reconstruction_errors(
                texts, llm_model, llm_tokenizer, diffusion_model, **_recon_kwargs
            )
        return extract_log_probs(texts, llm_model, llm_tokenizer, diffusion_model, **common_kwargs)

    # benign_train — used only for reference/context; scores saved for completeness
    print(f"======= Computing scores for good train set (GPU {gpu_id}) =======")
    good_results = []
    for batch in tqdm(good_batches, desc="good_batches", mininterval=30, ncols=120):
        good_results.append(_extract(batch))
    assert good_results, "No good train batches processed"

    # benign_test — held-out good samples used for evaluation
    print(f"======= Computing scores for good eval set (GPU {gpu_id}) =======")
    good_eval_results = []
    for batch in tqdm(good_eval_batches, desc="good_eval_batches", mininterval=30, ncols=120):
        good_eval_results.append(_extract(batch))
    assert good_eval_results, "No good eval batches processed"

    # test samples labels (bad prompts)
    print(f"======= Computing scores for bad set (GPU {gpu_id}) =======")
    bad_results = []
    for batch in tqdm(bad_batches, desc="bad_batches", mininterval=30, ncols=120):
        bad_results.append(_extract(batch))
    assert bad_results, (
        f"No bad batches were processed for data_selection={data_selection!r} "
        f"with skip={gpu_id * num_samples}. Check dataset availability."
    )

    # calibration samples labels (to determine youden threshold)
    if hf_dataset:
        metric_bad_batches = _chunk(_hf["cal_bad_prompts"], batch_size)
        print(f"Metric bad prompts (HF calibration split): {sum(len(b) for b in metric_bad_batches)}")
        print(f"======= Computing scores for calibration set (GPU {gpu_id}) =======")
        metric_bad_results = []
        for batch in tqdm(metric_bad_batches, desc="cal_bad_batches", mininterval=30, ncols=120):
            metric_bad_results.append(_extract(batch))
    elif metric_data_selection is not None and metric_data_selection != data_selection:
        assert metric_data_selection in dataset_selections, (
            f"Invalid metric_data_selection, choose: {list(dataset_selections.keys())}"
        )
        metric_kwargs = {
            "skip": gpu_id * num_samples,
            "batch_size": batch_size,
            "num_samples": int(num_samples / len(dataset_selections[metric_data_selection])),
        }
        metric_bad_batches = []
        for batch_fn in dataset_selections[metric_data_selection]:
            metric_bad_batches.extend(batch_fn(**metric_kwargs))
        print(f"Metric bad prompts ({metric_data_selection}): {sum(len(b) for b in metric_bad_batches)}")
        print(f"======= Computing scores for metric bad set (GPU {gpu_id}) =======")
        metric_bad_results = []
        for batch in tqdm(metric_bad_batches, desc="metric_bad_batches", mininterval=30, ncols=120):
            metric_bad_results.append(_extract(batch))
    else:
        metric_bad_results = bad_results

    # store everything
    out_file = os.path.join(out_dir, f"logprob_results_{gpu_id}.th")
    save_dict = {
        "layers": layers,
        "method": method,
        "num_steps": num_steps,
        "num_hutchinson_samples": num_hutchinson_samples,
        "dte_K": dte_K,
        "dte_num_sigma_bins": dte_num_sigma_bins,
        "reference_num_samples": reference_num_samples if method in ("dte", "dte_glp") else 0,
        "glp_sample_steps": glp_sample_steps if method == "dte_glp" else 0,
        "noise_level": noise_level if is_recon else None,
        "rec_num_timesteps": rec_num_timesteps if is_recon else None,
        "data_selection": data_selection,
        "metric_data_selection": metric_data_selection,
        "hf_dataset": hf_dataset,
    }

    if is_recon:
        save_dict.update(
            good_recon_errors       = torch.cat(good_results,       dim=0),
            good_eval_recon_errors  = torch.cat(good_eval_results,  dim=0),
            bad_recon_errors        = torch.cat(bad_results,        dim=0),
            metric_bad_recon_errors = torch.cat(metric_bad_results, dim=0),
        )
    else:
        good_log_probs       = torch.cat([r["log_probs"] for r in good_results],       dim=0)
        good_eval_log_probs  = torch.cat([r["log_probs"] for r in good_eval_results],  dim=0)
        bad_log_probs        = torch.cat([r["log_probs"] for r in bad_results],        dim=0)
        good_probs           = torch.cat([r["probs"]     for r in good_results],       dim=0)
        good_eval_probs      = torch.cat([r["probs"]     for r in good_eval_results],  dim=0)
        bad_probs            = torch.cat([r["probs"]     for r in bad_results],        dim=0)
        metric_bad_log_probs = torch.cat([r["log_probs"] for r in metric_bad_results], dim=0)
        metric_bad_probs     = torch.cat([r["probs"]     for r in metric_bad_results], dim=0)
        save_dict.update(
            good_log_probs=good_log_probs,
            good_eval_log_probs=good_eval_log_probs,
            bad_log_probs=bad_log_probs,
            good_probs=good_probs,
            good_eval_probs=good_eval_probs,
            bad_probs=bad_probs,
            metric_bad_log_probs=metric_bad_log_probs,
            metric_bad_probs=metric_bad_probs,
        )
        if method == "dte":
            save_dict.update(
                good_expected_sigma       = torch.cat([r["expected_sigma"] for r in good_results],       dim=0),
                good_eval_expected_sigma  = torch.cat([r["expected_sigma"] for r in good_eval_results],  dim=0),
                bad_expected_sigma        = torch.cat([r["expected_sigma"] for r in bad_results],        dim=0),
                metric_bad_expected_sigma = torch.cat([r["expected_sigma"] for r in metric_bad_results], dim=0),
            )

    torch.save(save_dict, out_file)
    print(f"Saved to {out_file}")


def aggregate(out_dir: str) -> dict:
    """ Load all per-GPU result files in out_dir, concatenate, compute metrics, save JSON.

    Expects files named logprob_results_*.th produced by main().
    Writes results.json to out_dir and returns the metrics dict.

    For DTE, scores both p_clean and expected_sigma (distinct signals).
    Reports metrics for several layer-aggregation strategies:
      mean  — average score across layers (default)
      min   — most anomalous layer (reduces FNR, higher FPR)
      best  — single best layer by per-layer AUROC
    """
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("logprob_results_*.th"))
    assert files, f"No logprob_results_*.th files found in {out_dir}"

    chunks = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    print(f"Loaded {len(chunks)} GPU result file(s) from {out_dir}")

    tensor_keys = (
        "good_log_probs", "good_eval_log_probs", "bad_log_probs",
        "good_probs", "good_eval_probs", "bad_probs",
        "good_expected_sigma", "good_eval_expected_sigma", "bad_expected_sigma",
        "metric_bad_log_probs", "metric_bad_probs", "metric_bad_expected_sigma",
        "good_recon_errors", "good_eval_recon_errors", "bad_recon_errors", "metric_bad_recon_errors",
    )
    cat = {k: torch.cat([c[k] for c in chunks], dim=0)
           for k in tensor_keys if chunks[0].get(k) is not None}

    good_log_probs      = cat.get("good_log_probs", None)
    good_eval_log_probs = cat.get("good_eval_log_probs", good_log_probs)  # fallback for old files
    bad_log_probs       = cat.get("bad_log_probs", None)
    good_probs          = cat.get("good_probs", None)
    good_eval_probs     = cat.get("good_eval_probs", good_probs)
    bad_probs           = cat.get("bad_probs", None)
    good_es             = cat.get("good_expected_sigma")
    good_eval_es        = cat.get("good_eval_expected_sigma", good_es)
    bad_es              = cat.get("bad_expected_sigma")
    metric_bad_log_probs = cat.get("metric_bad_log_probs")
    metric_bad_probs     = cat.get("metric_bad_probs")
    metric_bad_es        = cat.get("metric_bad_expected_sigma")

    cfg = {k: v for k, v in chunks[0].items() if k not in tensor_keys}
    cfg["layers"] = [int(l) for l in cfg["layers"]]
    layers = cfg["layers"]
    method = cfg.get("method", "hutchinson")
    has_dte = method in ("dte", "dte_glp")
    is_recon = (method == "reconstruction_error")
    metric_ds = cfg.get("metric_data_selection")

    good_recon_errors       = cat.get("good_recon_errors")
    good_eval_recon_errors  = cat.get("good_eval_recon_errors")
    bad_recon_errors        = cat.get("bad_recon_errors")
    metric_bad_recon_errors = cat.get("metric_bad_recon_errors")

    if is_recon:
        if metric_bad_recon_errors is None:
            metric_bad_recon_errors = bad_recon_errors
    else:
        if metric_bad_log_probs is None:
            metric_bad_log_probs = bad_log_probs
            metric_bad_probs     = bad_probs
            metric_bad_es        = bad_es

    n_good_eval  = len(good_eval_recon_errors if is_recon else good_eval_log_probs)
    n_bad        = len(bad_recon_errors        if is_recon else bad_log_probs)
    n_metric_bad = len(metric_bad_recon_errors if is_recon else metric_bad_log_probs)
    print(f"  good_eval: {n_good_eval}  |  bad: {n_bad}  |  metric_bad: {n_metric_bad}  |  method: {method}  |  layers: {layers}")

    has_separate_metric = (metric_ds is not None and metric_ds != cfg.get("data_selection"))
    main_labels = np.concatenate([np.ones(n_good_eval), np.zeros(n_bad)])
    cal_labels  = np.concatenate([np.ones(n_good_eval), np.zeros(n_metric_bad)])
    results: dict = {"config": cfg, "n_good_eval": n_good_eval, "n_bad": n_bad,
                     "n_metric_bad": n_metric_bad, "per_layer": {}, "aggregate": {}}

    # agrgegation helper — evaluates on the main (eval) set but picks Youden from calibration
    plot_series: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _agg_section(label_str, g_scores, b_scores, score_key,
                     cal_g_scores=None, cal_b_scores=None, add_to_plot=True):
        if cal_g_scores is None:
            cal_g_scores, cal_b_scores = g_scores, b_scores
        scores     = np.concatenate([g_scores, b_scores])
        cal_scores = np.concatenate([cal_g_scores, cal_b_scores])
        cal_lbl    = np.concatenate([np.ones(len(cal_g_scores)), np.zeros(len(cal_b_scores))])
        youden_thr = _find_youden_threshold(cal_lbl, cal_scores)
        eval_lbl   = np.concatenate([np.ones(len(g_scores)), np.zeros(len(b_scores))])
        print(f"\n  {label_str}  good={g_scores.mean():.4f}±{g_scores.std():.4f}  "
              f"bad={b_scores.mean():.4f}±{b_scores.std():.4f}")
        m = _classification_metrics(eval_lbl, scores, label_str, youden_threshold=youden_thr)
        m["good_mean"] = float(g_scores.mean())
        m["bad_mean"]  = float(b_scores.mean())
        if add_to_plot:
            plot_series.append((label_str, eval_lbl, scores))
        return m

    # per layer metrics

    print("\n=================== Per-layer metrics ===================")
    layer_aurocs = {}  # score_name -> list of (layer, auroc)

    if is_recon:
        for li, layer in enumerate(layers):
            key = f"layer_{layer}"
            results["per_layer"][key] = {}
            g_err  = good_eval_recon_errors[:, li].numpy()
            b_err  = bad_recon_errors[:, li].numpy()
            mb_err = metric_bad_recon_errors[:, li].numpy()
            print(f"\n  Layer {layer}  "
                  f"recon_error good={g_err.mean():.3f}±{g_err.std():.3f}  bad={b_err.mean():.3f}±{b_err.std():.3f}")
            # negate: lower error = better (more benign); score = -error so good has higher score
            youden_thr = _find_youden_threshold(cal_labels, np.concatenate([-g_err, -mb_err]))
            m = _classification_metrics(main_labels, np.concatenate([-g_err, -b_err]),
                                         f"layer {layer} recon_error", youden_threshold=youden_thr)
            m["good_mean"] = float(g_err.mean())
            m["bad_mean"]  = float(b_err.mean())
            results["per_layer"][key]["recon_error"] = m
            layer_aurocs.setdefault("recon_error", []).append((layer, m["auroc"]))
    else:
        for li, layer in enumerate(layers):
            key = f"layer_{layer}"
            results["per_layer"][key] = {}
            g_lp  = good_eval_log_probs[:, li].numpy()
            b_lp  = bad_log_probs[:, li].numpy()
            mb_lp = metric_bad_log_probs[:, li].numpy()
            print(f"\n  Layer {layer}  "
                  f"log_prob good={g_lp.mean():.3f}±{g_lp.std():.3f}  bad={b_lp.mean():.3f}±{b_lp.std():.3f}")

            youden_thr = _find_youden_threshold(cal_labels, np.concatenate([g_lp, mb_lp]))
            m = _classification_metrics(main_labels, np.concatenate([g_lp, b_lp]),
                                         f"layer {layer} log-prob", youden_threshold=youden_thr)
            m["good_mean"] = float(g_lp.mean())
            m["bad_mean"]  = float(b_lp.mean())
            results["per_layer"][key]["log_prob"] = m
            layer_aurocs.setdefault("log_prob", []).append((layer, m["auroc"]))

            if has_dte:
                g_p  = good_eval_probs[:, li].numpy()
                b_p  = bad_probs[:, li].numpy()
                print(f"          p_clean     good={g_p.mean():.4f}±{g_p.std():.4f}  bad={b_p.mean():.4f}±{b_p.std():.4f}")
                results["per_layer"][key]["p_clean"] = {"good_mean": float(g_p.mean()), "bad_mean": float(b_p.mean())}

                if good_eval_es is not None:
                    g_es  = good_eval_es[:, li].numpy()
                    b_es  = bad_es[:, li].numpy()
                    mb_es = metric_bad_es[:, li].numpy()
                    print(f"          exp_sigma   good={g_es.mean():.4f}±{g_es.std():.4f}  bad={b_es.mean():.4f}±{b_es.std():.4f}")
                    youden_thr_es = _find_youden_threshold(cal_labels, np.concatenate([-g_es, -mb_es]))
                    m_es = _classification_metrics(main_labels, np.concatenate([-g_es, -b_es]),
                                                   f"layer {layer} exp_sigma", youden_threshold=youden_thr_es)
                    m_es["good_mean"] = float(g_es.mean())
                    m_es["bad_mean"]  = float(b_es.mean())
                    results["per_layer"][key]["expected_sigma"] = m_es
                    layer_aurocs.setdefault("expected_sigma", []).append((layer, m_es["auroc"]))

    print("\n=================== Aggregate metrics ===================")

    if is_recon:
        # mean across layers (lower mean error = better)
        print("\n--- mean recon_error across layers ---")
        results["aggregate"]["mean"] = {}
        results["aggregate"]["mean"]["recon_error"] = _agg_section(
            "mean recon_error",
            -good_eval_recon_errors.mean(1).numpy(), -bad_recon_errors.mean(1).numpy(), "recon_error",
            cal_g_scores=-good_eval_recon_errors.mean(1).numpy(),
            cal_b_scores=-metric_bad_recon_errors.mean(1).numpy())

        # max across layers (most anomalous layer wins — highest error = bad)
        print("\n--- max recon_error across layers (most anomalous layer) ---")
        results["aggregate"]["min"] = {}
        results["aggregate"]["min"]["recon_error"] = _agg_section(
            "max recon_error",
            -good_eval_recon_errors.max(1).values.numpy(), -bad_recon_errors.max(1).values.numpy(), "recon_error",
            cal_g_scores=-good_eval_recon_errors.max(1).values.numpy(),
            cal_b_scores=-metric_bad_recon_errors.max(1).values.numpy())

        # best single layer by AUROC
        print("\n--- best single layer (by AUROC) ---")
        results["aggregate"]["best_layer"] = {}
        for score_key, auroc_list in layer_aurocs.items():
            best_layer, best_auroc = max(auroc_list, key=lambda x: x[1])
            best_li = layers.index(best_layer)
            print(f"  best layer for {score_key}: layer {best_layer} (AUROC={best_auroc:.4f})")
            g  = -good_eval_recon_errors[:, best_li].numpy()
            b  = -bad_recon_errors[:, best_li].numpy()
            mb = -metric_bad_recon_errors[:, best_li].numpy()
            m = _agg_section(f"best-layer {score_key} (L{best_layer})", g, b, score_key,
                             cal_g_scores=g, cal_b_scores=mb)
            m["best_layer"] = best_layer
            results["aggregate"]["best_layer"][score_key] = m

    else:
        # mean across layers
        print("\n--- mean across layers ---")
        results["aggregate"]["mean"] = {}
        results["aggregate"]["mean"]["log_prob"] = _agg_section(
            "mean log_prob",
            good_eval_log_probs.mean(1).numpy(), bad_log_probs.mean(1).numpy(), "log_prob",
            cal_g_scores=good_eval_log_probs.mean(1).numpy(), cal_b_scores=metric_bad_log_probs.mean(1).numpy())
        if has_dte and good_eval_es is not None:
            results["aggregate"]["mean"]["expected_sigma"] = _agg_section(
                "mean exp_sigma",
                -good_eval_es.mean(1).numpy(), -bad_es.mean(1).numpy(), "expected_sigma",
                cal_g_scores=-good_eval_es.mean(1).numpy(), cal_b_scores=-metric_bad_es.mean(1).numpy())

        # min across layers (most anomalous layer wins — lower log_prob / higher sigma = bad)
        print("\n--- min-log_prob across layers (most anomalous layer) ---")
        results["aggregate"]["min"] = {}
        results["aggregate"]["min"]["log_prob"] = _agg_section(
            "min log_prob",
            good_eval_log_probs.min(1).values.numpy(), bad_log_probs.min(1).values.numpy(), "log_prob",
            cal_g_scores=good_eval_log_probs.min(1).values.numpy(), cal_b_scores=metric_bad_log_probs.min(1).values.numpy())
        if has_dte and good_eval_es is not None:
            results["aggregate"]["min"]["expected_sigma"] = _agg_section(
                "max exp_sigma",
                -good_eval_es.max(1).values.numpy(), -bad_es.max(1).values.numpy(), "expected_sigma",
                cal_g_scores=-good_eval_es.max(1).values.numpy(), cal_b_scores=-metric_bad_es.max(1).values.numpy())

        # best single layer by AUROC
        print("\n--- best single layer (by AUROC) ---")
        results["aggregate"]["best_layer"] = {}
        for score_key, auroc_list in layer_aurocs.items():
            best_layer, best_auroc = max(auroc_list, key=lambda x: x[1])
            best_li = layers.index(best_layer)
            print(f"  best layer for {score_key}: layer {best_layer} (AUROC={best_auroc:.4f})")
            if score_key == "log_prob":
                g  = good_eval_log_probs[:, best_li].numpy()
                b  = bad_log_probs[:, best_li].numpy()
                mb = metric_bad_log_probs[:, best_li].numpy()
            else:
                g  = -good_eval_es[:, best_li].numpy()
                b  = -bad_es[:, best_li].numpy()
                mb = -metric_bad_es[:, best_li].numpy()
            m = _agg_section(f"best-layer {score_key} (L{best_layer})", g, b, score_key,
                             cal_g_scores=g, cal_b_scores=mb)
            m["best_layer"] = best_layer
            results["aggregate"]["best_layer"][score_key] = m

    # generating plots
    print("\nGenerating plots...")
    try:
        _make_plots(out_dir, plot_series)
    except Exception as e:
        print(f"  Warning: could not generate plots: {e}")

    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_json}")
    return results


if __name__ == "__main__":
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
            layers=cfg["layers"],
            out_dir=cfg["out_dir"],
            model=cfg["model"],
            data_selection=cfg.get("data", "all_semantic"),
            metric_data_selection=cfg.get("metric_data"),
            num_samples=cfg["num_samples"] if "num_samples" in cfg else None,
            num_steps=cfg.get("num_steps", 100),
            num_hutchinson_samples=cfg.get("num_hutchinson_samples", 1),
            method=cfg.get("method", "hutchinson"),
            reference_num_samples=cfg.get("reference_num_samples", 512),
            glp_sample_steps=cfg.get("glp_sample_steps", 100),
            dte_K=cfg.get("dte_K", 5),
            dte_num_sigma_bins=cfg.get("dte_num_sigma_bins", 100),
            normalize=cfg.get("normalize", False),
            noise_level=cfg.get("noise_level", 0.5),
            rec_num_timesteps=cfg.get("rec_num_timesteps", 100),
            hf_dataset=cfg.get("hf_dataset"),
            num_gpus=cfg.get("num_gpus", 4),
            batch_size=cfg.get("batch_size"),
        )

    fire.Fire({"run": run, "aggregate": aggregate})
