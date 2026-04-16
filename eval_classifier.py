import json
import os
import random
import numpy as np
import torch
from pathlib import Path

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


def _classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    score_name: str,
    verbose: bool = True,
    target_tprs: tuple = tuple(np.arange(0.60, 1.00, 0.05).round(2)),
) -> dict:
    """Compute AUROC + metrics at Youden's-J threshold and fixed TPR targets.

    Returns a dict with auroc and per-threshold breakdowns.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    auroc = float(roc_auc_score(labels, scores))
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)

    # Youden's J
    youden = _threshold_metrics(labels, scores, float(thresholds[(tpr_arr - fpr_arr).argmax()]))

    # Fixed TPR targets (minimise FNR at acceptable recall floor)
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
    """Generate and save ROC curve and TPR/FNR-vs-threshold plots.

    Args:
        out_dir: directory where PNG files are written.
        named_scores: list of (label, labels_array, scores_array) tuples,
                      one per curve.
    """
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


def extract_activations(
    texts: list[str],
    llm_model,
    llm_tokenizer,
    diffusion_model,
    device: str = "cuda:0",
    batch_size: int = 8,
) -> torch.Tensor:
    """Run the LLM and return cached (N, L, D) activations on `device` in bfloat16."""
    activations = save_acts(
        hf_model=llm_model,
        hf_tokenizer=llm_tokenizer,
        text=texts,
        tracedict_config=diffusion_model.tracedict_config,
        token_idx="last",
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
    """Extract LLM activations and compute per-layer GLP scores.

    method="hutchinson" (default): CNF change-of-variables log p(x).
    method="dte": KNN-based DTE-InverseGamma posterior; requires
        `reference_activations` of shape (N_ref, L, D) on the same device.
    """
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


def main(
    gpu_id: int,
    layers: list[int],
    out_dir: str,
    model: str = "1b",
    data_selection: str = "all_semantic",
    num_samples: int = 128,
    num_steps: int = 100,
    num_hutchinson_samples: int = 1,
    method: str = "hutchinson",
    reference_num_samples: int = 512,
    dte_K: int = 5,
    dte_num_sigma_bins: int = 100,
    normalize: bool = False,
):
    torch.manual_seed(42)
    random.seed(42)

    device = f"cuda:{gpu_id}"

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
    print(f"[+] method:         {method}")
    if method == "dte":
        print(f"[+] dte_K:          {dte_K}")
        print(f"[+] dte_sigma_bins: {dte_num_sigma_bins}")
        print(f"[+] ref_samples:    {reference_num_samples}")
    print(f"[+] out_dir:        {out_dir}")
    print("================================================")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # --- good prompts (in-distribution) ---
    good_batches = list(
        fineweb_batches(skip=gpu_id * num_samples, batch_size=batch_size, num_samples=num_samples)
    )

    # --- DTE reference set (disjoint from eval set via skip offset) ---
    reference_activations = None
    if method == "dte":
        reference_skip = 10_000 + gpu_id * reference_num_samples
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

    # --- bad prompts ---
    assert data_selection in dataset_selections, (
        f"Invalid dataset selection, choose: {list(dataset_selections.keys())}"
    )
    # balance samples when picked from multiple datasets
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
    kwargs = {
        "skip": gpu_id * num_samples,
        "batch_size": batch_size,
        "num_samples": int(num_samples / len(dataset_selections[data_selection])),
    }
    bad_batches = []
    for batch_fn in dataset_selections[data_selection]:
        bad_batches.extend(batch_fn(**kwargs))

    print(f"Good prompts: {sum(len(b) for b in good_batches)}")
    print(f"Bad prompts:  {sum(len(b) for b in bad_batches)}")

    common_kwargs = dict(
        layers=layers, num_steps=num_steps,
        num_hutchinson_samples=num_hutchinson_samples,
        device=device, batch_size=batch_size,
        method=method,
        reference_activations=reference_activations,
        K=dte_K,
        num_sigma_bins=dte_num_sigma_bins,
        normalize=normalize,
    )

    # --- compute scores for good set ---
    print(f"======= Computing scores for good set (GPU {gpu_id}) =======")
    good_results = []
    for batch in good_batches:
        good_results.append(
            extract_log_probs(
                batch, llm_model, llm_tokenizer, diffusion_model,
                **common_kwargs,
            )
        )
    assert good_results, "No good batches were processed — check fineweb dataset / skip offset"
    good_log_probs = torch.cat([r["log_probs"] for r in good_results], dim=0)
    good_probs = torch.cat([r["probs"] for r in good_results], dim=0)

    # --- compute scores for bad set ---
    print(f"======= Computing scores for bad set (GPU {gpu_id}) =======")
    bad_results = []
    for batch in bad_batches:
        bad_results.append(
            extract_log_probs(
                batch, llm_model, llm_tokenizer, diffusion_model,
                **common_kwargs,
            )
        )
    assert bad_results, (
        f"No bad batches were processed for data_selection={data_selection!r} "
        f"with skip={gpu_id * num_samples}. Check dataset availability."
    )
    bad_log_probs = torch.cat([r["log_probs"] for r in bad_results], dim=0)
    bad_probs = torch.cat([r["probs"] for r in bad_results], dim=0)
    good_expected_sigma = torch.cat([r["expected_sigma"] for r in good_results], dim=0) if method == "dte" else None
    bad_expected_sigma  = torch.cat([r["expected_sigma"] for r in bad_results],  dim=0) if method == "dte" else None

    # --- summary statistics ---
    print("\n=================== Results ===================")
    for li, layer in enumerate(layers):
        g = good_log_probs[:, li]
        b = bad_log_probs[:, li]
        print(
            f"Layer {layer:2d}  |  "
            f"good: {g.mean():.2f} +/- {g.std():.2f}  |  "
            f"bad: {b.mean():.2f} +/- {b.std():.2f}  |  "
            f"delta: {(g.mean() - b.mean()):.2f}"
        )

    # aggregate across layers (mean log-prob)
    g_agg = good_log_probs.mean(dim=1)
    b_agg = bad_log_probs.mean(dim=1)
    print(
        f"{'Avg':>8s}  |  "
        f"good: {g_agg.mean():.2f} +/- {g_agg.std():.2f}  |  "
        f"bad: {b_agg.mean():.2f} +/- {b_agg.std():.2f}  |  "
        f"delta: {(g_agg.mean() - b_agg.mean()):.2f}"
    )

    if method == "dte":
        print("\n----------- DTE p_clean (prob, higher = in-dist) -----------")
        for li, layer in enumerate(layers):
            g = good_probs[:, li]
            b = bad_probs[:, li]
            print(
                f"Layer {layer:2d}  |  "
                f"good: {g.mean():.4f} +/- {g.std():.4f}  |  "
                f"bad: {b.mean():.4f} +/- {b.std():.4f}  |  "
                f"delta: {(g.mean() - b.mean()):.4f}"
            )
        g_prob_agg = good_probs.mean(dim=1)
        b_prob_agg = bad_probs.mean(dim=1)
        print(
            f"{'Avg':>8s}  |  "
            f"good: {g_prob_agg.mean():.4f} +/- {g_prob_agg.std():.4f}  |  "
            f"bad: {b_prob_agg.mean():.4f} +/- {b_prob_agg.std():.4f}  |  "
            f"delta: {(g_prob_agg.mean() - b_prob_agg.mean()):.4f}"
        )

    # --- AUROC + threshold-dependent metrics ---
    try:
        labels = torch.cat([torch.ones(len(g_agg)), torch.zeros(len(b_agg))]).numpy()
        print("\n----------- Classification metrics (avg across layers) -----------")
        _classification_metrics(labels, torch.cat([g_agg, b_agg]).numpy(), "log-prob")
        if method == "dte":
            _classification_metrics(labels, torch.cat([g_prob_agg, b_prob_agg]).numpy(), "p_clean")
    except Exception as e:
        print(f"\nCould not compute metrics: {e}")
    print("================================================")

    # --- save ---
    out_file = os.path.join(out_dir, f"logprob_results_{gpu_id}.th")
    torch.save({
        "good_log_probs": good_log_probs,
        "bad_log_probs": bad_log_probs,
        "good_probs": good_probs,
        "bad_probs": bad_probs,
        "layers": layers,
        "method": method,
        "num_steps": num_steps,
        "num_hutchinson_samples": num_hutchinson_samples,
        "dte_K": dte_K,
        "dte_num_sigma_bins": dte_num_sigma_bins,
        "reference_num_samples": reference_num_samples if method == "dte" else 0,
        "data_selection": data_selection,
        **({"good_expected_sigma": good_expected_sigma,
            "bad_expected_sigma":  bad_expected_sigma} if method == "dte" else {}),
    }, out_file)
    print(f"Saved to {out_file}")


def aggregate(out_dir: str) -> dict:
    """Load all per-GPU result files in out_dir, concatenate, compute metrics, save JSON.

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

    tensor_keys = ("good_log_probs", "bad_log_probs", "good_probs", "bad_probs",
                   "good_expected_sigma", "bad_expected_sigma")
    cat = {k: torch.cat([c[k] for c in chunks], dim=0)
           for k in tensor_keys if k in chunks[0]}

    good_log_probs = cat["good_log_probs"]
    bad_log_probs  = cat["bad_log_probs"]
    good_probs     = cat["good_probs"]
    bad_probs      = cat["bad_probs"]
    good_es = cat.get("good_expected_sigma")  # (N, L) or None
    bad_es  = cat.get("bad_expected_sigma")

    cfg = {k: v for k, v in chunks[0].items() if k not in tensor_keys}
    cfg["layers"] = [int(l) for l in cfg["layers"]]
    layers = cfg["layers"]
    method = cfg.get("method", "hutchinson")
    has_dte = method == "dte"

    n_good, n_bad = len(good_log_probs), len(bad_log_probs)
    print(f"  good: {n_good}  |  bad: {n_bad}  |  method: {method}  |  layers: {layers}")

    labels = np.concatenate([np.ones(n_good), np.zeros(n_bad)])
    results: dict = {"config": cfg, "n_good": n_good, "n_bad": n_bad,
                     "per_layer": {}, "aggregate": {}}

    # ── per-layer ──────────────────────────────────────────────────────────
    print("\n=================== Per-layer metrics ===================")
    layer_aurocs = {}  # score_name -> list of (layer, auroc)
    for li, layer in enumerate(layers):
        key = f"layer_{layer}"
        results["per_layer"][key] = {}
        g_lp = good_log_probs[:, li].numpy()
        b_lp = bad_log_probs[:, li].numpy()
        print(f"\n  Layer {layer}  "
              f"log_prob good={g_lp.mean():.3f}±{g_lp.std():.3f}  bad={b_lp.mean():.3f}±{b_lp.std():.3f}")
        m = _classification_metrics(labels, np.concatenate([g_lp, b_lp]), f"layer {layer} log-prob")
        m["good_mean"] = float(g_lp.mean())
        m["bad_mean"]  = float(b_lp.mean())
        results["per_layer"][key]["log_prob"] = m
        layer_aurocs.setdefault("log_prob", []).append((layer, m["auroc"]))

        if has_dte:
            g_p = good_probs[:, li].numpy()
            b_p = bad_probs[:, li].numpy()
            print(f"          p_clean     good={g_p.mean():.4f}±{g_p.std():.4f}  bad={b_p.mean():.4f}±{b_p.std():.4f}")
            # p_clean and log_prob are monotone transforms — skip redundant metrics
            results["per_layer"][key]["p_clean"] = {"good_mean": float(g_p.mean()), "bad_mean": float(b_p.mean())}

            if good_es is not None:
                g_es = good_es[:, li].numpy()
                b_es = bad_es[:, li].numpy()
                # expected_sigma: higher = more anomalous, so negate for scoring
                print(f"          exp_sigma   good={g_es.mean():.4f}±{g_es.std():.4f}  bad={b_es.mean():.4f}±{b_es.std():.4f}")
                m_es = _classification_metrics(labels, np.concatenate([-g_es, -b_es]), f"layer {layer} exp_sigma")
                m_es["good_mean"] = float(g_es.mean())
                m_es["bad_mean"]  = float(b_es.mean())
                results["per_layer"][key]["expected_sigma"] = m_es
                layer_aurocs.setdefault("expected_sigma", []).append((layer, m_es["auroc"]))

    # ── aggregate strategies ───────────────────────────────────────────────
    plot_series: list[tuple[str, np.ndarray, np.ndarray]] = []  # (name, labels, scores)

    def _agg_section(label_str, g_scores, b_scores, score_key, add_to_plot=True):
        scores = np.concatenate([g_scores, b_scores])
        print(f"\n  {label_str}  good={g_scores.mean():.4f}±{g_scores.std():.4f}  "
              f"bad={b_scores.mean():.4f}±{b_scores.std():.4f}")
        m = _classification_metrics(labels, scores, label_str)
        m["good_mean"] = float(g_scores.mean())
        m["bad_mean"]  = float(b_scores.mean())
        if add_to_plot:
            plot_series.append((label_str, labels, scores))
        return m

    print("\n=================== Aggregate metrics ===================")

    # mean across layers
    print("\n--- mean across layers ---")
    results["aggregate"]["mean"] = {}
    results["aggregate"]["mean"]["log_prob"] = _agg_section(
        "mean log_prob", good_log_probs.mean(1).numpy(), bad_log_probs.mean(1).numpy(), "log_prob")
    if has_dte and good_es is not None:
        results["aggregate"]["mean"]["expected_sigma"] = _agg_section(
            "mean exp_sigma", -good_es.mean(1).numpy(), -bad_es.mean(1).numpy(), "expected_sigma")

    # min across layers (most anomalous layer wins — lower log_prob / higher sigma = bad)
    print("\n--- min-log_prob across layers (most anomalous layer) ---")
    results["aggregate"]["min"] = {}
    results["aggregate"]["min"]["log_prob"] = _agg_section(
        "min log_prob", good_log_probs.min(1).values.numpy(), bad_log_probs.min(1).values.numpy(), "log_prob")
    if has_dte and good_es is not None:
        results["aggregate"]["min"]["expected_sigma"] = _agg_section(
            "max exp_sigma", -good_es.max(1).values.numpy(), -bad_es.max(1).values.numpy(), "expected_sigma")

    # best single layer by AUROC
    print("\n--- best single layer (by AUROC) ---")
    results["aggregate"]["best_layer"] = {}
    for score_key, auroc_list in layer_aurocs.items():
        best_layer, best_auroc = max(auroc_list, key=lambda x: x[1])
        best_li = layers.index(best_layer)
        print(f"  best layer for {score_key}: layer {best_layer} (AUROC={best_auroc:.4f})")
        if score_key == "log_prob":
            g = good_log_probs[:, best_li].numpy()
            b = bad_log_probs[:, best_li].numpy()
        else:
            g = -good_es[:, best_li].numpy()
            b = -bad_es[:, best_li].numpy()
        m = _agg_section(f"best-layer {score_key} (L{best_layer})", g, b, score_key)
        m["best_layer"] = best_layer
        results["aggregate"]["best_layer"][score_key] = m

    # ── plots ─────────────────────────────────────────────────────────────
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
            data_selection=cfg["data"],
            num_samples=cfg["num_samples"],
            num_steps=cfg.get("num_steps", 100),
            num_hutchinson_samples=cfg.get("num_hutchinson_samples", 1),
            method=cfg.get("method", "hutchinson"),
            reference_num_samples=cfg.get("reference_num_samples", 512),
            dte_K=cfg.get("dte_K", 5),
            dte_num_sigma_bins=cfg.get("dte_num_sigma_bins", 100),
            normalize=cfg.get("normalize", False),
        )

    fire.Fire({"run": run, "aggregate": aggregate})
