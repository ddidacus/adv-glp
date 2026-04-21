"""
DiffMean steering-vector classifier on LLM residual-stream activations.

Computes a per-layer steering vector via difference-of-means using:
  - HF train split as negative examples (benign)
  - HF calibration split as positive examples (adversarial)

Steering vector per layer: sv[l] = normalize(mean(adv[l]) - mean(benign[l]))

Classification score = dot(query_acts, sv) / norm(query_acts)
Higher score → more adversarial.

Youden's J threshold is derived from the calibration set and transferred to the
test set, matching the evaluation protocol of eval_classifier.py exactly.

Two-pass workflow (mirrors eval_linear_probe.py):
  Pass 1 — activation extraction (one job per GPU):
      python eval_diffmean.py run --config=cfg.yaml --gpu_id=0

  Pass 2 — steering vector computation + evaluation (single job, CPU-capable):
      python eval_diffmean.py aggregate --out_dir=results/eval-diffmean
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_classifier import (
    _threshold_metrics,
    _find_youden_threshold,
    _classification_metrics,
    _make_plots,
    extract_activations,
    _chunk,
    load_hf_splits,
)
from eval_by_component import (
    fineweb_batches,
    harmeval_batches,
    harmeval_gcg_batches,
    jailbreakbench_batches,
    sg_bench_batches,
    vanilla_wildjailbreak_batches,
    adversarial_wildjailbreak_batches,
)
from glp.denoiser import load_glp


# ── Dataset registry (mirrors eval_linear_probe.py) ──────────────────────────

DATASET_SELECTIONS = {
    "all_semantic": [
        harmeval_batches,
        jailbreakbench_batches,
        sg_bench_batches,
        vanilla_wildjailbreak_batches,
    ],
    "harmeval":              [harmeval_batches],
    "harmeval_gcg":          [harmeval_gcg_batches],
    "jailbreakbench":        [jailbreakbench_batches],
    "sgbench":               [sg_bench_batches],
    "vanilla_wildjailbreak": [vanilla_wildjailbreak_batches],
}


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_diffmean(acts: torch.Tensor, sv: np.ndarray) -> np.ndarray:
    """dot(acts, sv) / norm(acts) — higher means more adversarial."""
    sv_t = torch.from_numpy(sv).float()
    a = acts.float()  # (N, D)
    dots = a @ sv_t  # (N,)
    norms = a.norm(dim=1)  # (N,)
    return (dots / (norms + 1e-8)).numpy()


# ── Activation extraction helper ─────────────────────────────────────────────

def _extract_bad_acts(
    dataset_key: str,
    skip: int,
    batch_size: int,
    num_samples: int,
    llm_model,
    llm_tokenizer,
    diffusion_model,
    device: str,
    token_pooling: str = "mean",
) -> torch.Tensor:
    from tqdm import tqdm
    fns = DATASET_SELECTIONS[dataset_key]
    per_fn = int(num_samples)
    kwargs = dict(skip=skip, batch_size=batch_size, num_samples=per_fn)
    chunks = []
    for fn in fns:
        for batch in tqdm(list(fn(**kwargs)), desc=fn.__name__, mininterval=30, ncols=120):
            chunks.append(
                extract_activations(batch, llm_model, llm_tokenizer, diffusion_model,
                                    device=device, batch_size=batch_size,
                                    token_pooling=token_pooling).cpu()
            )
    return torch.cat(chunks, dim=0)  # (N, L, D)


# ── Pass 1: activation extraction (per GPU) ──────────────────────────────────

def main(
    gpu_id: int,
    layers: list[int],
    out_dir: str,
    model: str = "1b",
    data_selection: str = "all_semantic",
    metric_data_selection: str | None = None,
    num_samples: int | None = None,
    hf_dataset: str | None = None,
    num_gpus: int = 4,
    token_pooling: str = "mean",
    good_data: str = "fineweb",
):
    """Extract and save activations for good, metric-bad, and test-bad sets."""
    torch.manual_seed(42)
    random.seed(42)

    device = f"cuda:{gpu_id}"

    if model == "1b":
        batch_size   = 64
        llm_model_id = "unsloth/Llama-3.2-1B"
        glp_model_id = "generative-latent-prior/glp-llama1b-d12-multi"
    elif model == "8b":
        batch_size   = 32
        llm_model_id = "meta-llama/Llama-3.1-8B"
        glp_model_id = "generative-latent-prior/glp-llama8b-d6"
    else:
        raise NotImplementedError(f"Unknown model: {model}")

    if not hf_dataset:
        assert data_selection in DATASET_SELECTIONS, (
            f"Invalid data_selection, choose: {list(DATASET_SELECTIONS.keys())}"
        )
        if metric_data_selection is None:
            metric_data_selection = data_selection
        assert metric_data_selection in DATASET_SELECTIONS, (
            f"Invalid metric_data_selection, choose: {list(DATASET_SELECTIONS.keys())}"
        )
    else:
        data_selection        = "hf_test"
        metric_data_selection = "hf_calibration"

    print("================================================")
    print(f"[+] LLM:              {llm_model_id}")
    print(f"[+] batch_size:       {batch_size}")
    print(f"[+] num_samples:      {num_samples}")
    print(f"[+] layers:           {layers}")
    print(f"[+] data_selection:   {data_selection}")
    print(f"[+] metric_data:      {metric_data_selection}")
    print(f"[+] token_pooling:    {token_pooling}")
    print(f"[+] good_data:        {good_data}")
    print(f"[+] out_dir:          {out_dir}")
    print("================================================")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    llm_model     = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    diffusion_model = load_glp(glp_model_id, device=device, checkpoint="final")
    diffusion_model.tracedict_config.layers = layers

    common = dict(llm_model=llm_model, llm_tokenizer=llm_tokenizer,
                  diffusion_model=diffusion_model, device=device,
                  token_pooling=token_pooling)
    skip = gpu_id * num_samples if num_samples else 0

    def _acts_from_batches(batches: list[list[str]]) -> torch.Tensor:
        from tqdm import tqdm
        chunks = [extract_activations(b, **common, batch_size=batch_size).cpu()
                  for b in tqdm(batches, desc="batches", mininterval=30, ncols=120)]
        return torch.cat(chunks, dim=0)

    if hf_dataset:
        _hf = load_hf_splits(hf_dataset, gpu_id, num_gpus, num_samples, batch_size)
        print("Extracting good train activations (benign_train)...")
        good_acts = _acts_from_batches(_chunk(_hf["good_prompts"], batch_size))
        print(f"  good_acts: {tuple(good_acts.shape)}")

        print("Extracting good eval activations (benign_test, held-out)...")
        good_eval_acts = _acts_from_batches(_chunk(_hf["good_eval_prompts"], batch_size))
        print(f"  good_eval_acts: {tuple(good_eval_acts.shape)}")

        print("Extracting metric bad activations (adversarial_calibration)...")
        metric_bad_acts = _acts_from_batches(_chunk(_hf["cal_bad_prompts"], batch_size))
        print(f"  metric_bad_acts: {tuple(metric_bad_acts.shape)}")

        print("Extracting test bad activations (adversarial_test)...")
        bad_acts = _acts_from_batches(_chunk(_hf["test_bad_prompts"], batch_size))
        print(f"  bad_acts: {tuple(bad_acts.shape)}")
        has_separate_test = True
    else:
        assert num_samples is not None, "num_samples is required when not using hf_dataset"
        from tqdm import tqdm
        assert good_data in ("fineweb", "wildjailbreak_benign"), \
            f"good_data must be 'fineweb' or 'wildjailbreak_benign', got {good_data!r}"
        print(f"Extracting good activations [{good_data}]...")
        if good_data == "fineweb":
            good_batches = list(fineweb_batches(skip=skip, batch_size=batch_size, num_samples=num_samples))
        else:
            good_batches = list(adversarial_wildjailbreak_batches(
                skip=skip, batch_size=batch_size, num_samples=num_samples, benign=True,
            ))
        good_chunks = []
        for batch in tqdm(good_batches, desc=f"good_{good_data}", mininterval=30, ncols=120):
            good_chunks.append(
                extract_activations(batch, **common, batch_size=batch_size).cpu()
            )
        good_acts = torch.cat(good_chunks, dim=0)
        print(f"  good_acts: {tuple(good_acts.shape)}")

        print(f"Extracting metric bad activations [{metric_data_selection}]...")
        metric_bad_acts = _extract_bad_acts(
            metric_data_selection, skip=skip, batch_size=batch_size,
            num_samples=num_samples, **common,
        )
        print(f"  metric_bad_acts: {tuple(metric_bad_acts.shape)}")

        has_separate_test = data_selection != metric_data_selection
        if has_separate_test:
            print(f"Extracting test bad activations [{data_selection}]...")
            bad_acts = _extract_bad_acts(
                data_selection, skip=skip, batch_size=batch_size,
                num_samples=num_samples, **common,
            )
            print(f"  bad_acts: {tuple(bad_acts.shape)}")
        else:
            bad_acts = metric_bad_acts

    out_file = os.path.join(out_dir, f"acts_{gpu_id}.th")
    torch.save({
        "good_acts":             good_acts,
        "good_eval_acts":        good_eval_acts if hf_dataset else good_acts,
        "metric_bad_acts":       metric_bad_acts,
        "bad_acts":              bad_acts,
        "has_separate_test":     has_separate_test,
        "layers":                layers,
        "data_selection":        data_selection,
        "metric_data_selection": metric_data_selection,
        "model":                 model,
        "num_samples":           num_samples,
        "hf_dataset":            hf_dataset,
        "token_pooling":         token_pooling,
        "good_data":             good_data,
    }, out_file)
    print(f"[GPU {gpu_id}] Saved activations to {out_file}")


# ── Pass 2: steering vector computation + evaluation ─────────────────────────

def aggregate(out_dir: str) -> dict:
    """Compute per-layer DiffMean steering vectors and evaluate on the test set.

    Youden's J threshold is derived from calibration (metric) predictions and
    transferred to test evaluation, matching eval_classifier.py exactly.
    """
    out_dir = Path(out_dir)
    files   = sorted(out_dir.glob("acts_*.th"))
    assert files, f"No acts_*.th files found in {out_dir}"

    chunks = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    print(f"Loaded {len(chunks)} GPU activation file(s) from {out_dir}")

    good_acts       = torch.cat([c["good_acts"]       for c in chunks], dim=0)  # (N_train, L, D) — DiffMean training
    good_eval_acts  = torch.cat([c["good_eval_acts"]  for c in chunks], dim=0)  # (N_eval,  L, D) — held-out good
    metric_bad_acts = torch.cat([c["metric_bad_acts"] for c in chunks], dim=0)  # (N_cal,   L, D)
    bad_acts        = torch.cat([c["bad_acts"]        for c in chunks], dim=0)  # (N_bad,   L, D)

    cfg = {k: v for k, v in chunks[0].items()
           if k not in ("good_acts", "good_eval_acts", "metric_bad_acts", "bad_acts")}
    cfg["layers"] = [int(l) for l in cfg["layers"]]
    layers            = cfg["layers"]
    data_sel          = cfg["data_selection"]
    metric_sel        = cfg["metric_data_selection"]
    has_separate_test = cfg["has_separate_test"]
    model             = cfg.get("model", "unknown")

    n_good       = len(good_acts)       # DiffMean training (mean computation)
    n_good_eval  = len(good_eval_acts)  # held-out good evaluation
    n_metric_bad = len(metric_bad_acts)
    n_bad        = len(bad_acts)

    print(f"  good_train: {n_good}  |  good_eval: {n_good_eval}  |  "
          f"metric_bad: {n_metric_bad}  |  bad: {n_bad}")
    print(f"  layers: {layers}")

    # ── Compute DiffMean steering vectors from benign_train + adversarial_calibration
    print("\n=================== Computing DiffMean steering vectors ===================")
    steering_directions: dict[int, np.ndarray] = {}
    for li, layer in enumerate(layers):
        pos_mean = metric_bad_acts[:, li, :].float().mean(0).numpy()
        neg_mean = good_acts[:, li, :].float().mean(0).numpy()
        d = pos_mean - neg_mean
        norm = np.linalg.norm(d)
        d = d / (norm + 1e-12)
        steering_directions[layer] = d.astype(np.float32)
        print(f"  Layer {layer:2d}  sv norm (before norm): {norm:.4f}")

    sv_pt = {layer: torch.from_numpy(sv) for layer, sv in steering_directions.items()}
    torch.save(sv_pt, out_dir / "steering_vector.pt")
    print(f"Steering vector saved to {out_dir / 'steering_vector.pt'}")

    # ── Score held-out good eval + adversarial sets ───────────────────────
    good_eval_scores  = np.zeros((n_good_eval,  len(layers)), dtype=np.float32)
    bad_scores        = np.zeros((n_bad,         len(layers)), dtype=np.float32)
    metric_bad_scores = np.zeros((n_metric_bad,  len(layers)), dtype=np.float32)

    print("\n=================== Scoring ===================")
    for li, layer in enumerate(layers):
        sv = steering_directions[layer]
        good_eval_scores[:, li]  = _score_diffmean(good_eval_acts[:, li, :], sv)
        metric_bad_scores[:, li] = _score_diffmean(metric_bad_acts[:, li, :], sv)
        bad_scores[:, li] = (
            _score_diffmean(bad_acts[:, li, :], sv)
            if has_separate_test else metric_bad_scores[:, li]
        )
        g, mb, b = good_eval_scores[:, li], metric_bad_scores[:, li], bad_scores[:, li]
        print(f"  Layer {layer:2d}  good_eval={g.mean():.3f}±{g.std():.3f}  "
              f"metric_bad={mb.mean():.3f}±{mb.std():.3f}  bad={b.mean():.3f}±{b.std():.3f}")

    # labels: 0 = benign (good_eval), 1 = adversarial
    cal_labels  = np.concatenate([np.zeros(n_good_eval), np.ones(n_metric_bad)])
    main_labels = np.concatenate([np.zeros(n_good_eval), np.ones(n_bad)])

    results: dict = {
        "config": cfg,
        "n_good_train": n_good, "n_good_eval": n_good_eval,
        "n_bad": n_bad, "n_metric_bad": n_metric_bad,
        "per_layer": {}, "aggregate": {},
    }

    print("\n=================== Per-layer metrics ===================")
    layer_aurocs = []
    for li, layer in enumerate(layers):
        key = f"layer_{layer}"
        g   = good_eval_scores[:, li]
        b   = bad_scores[:, li]
        mb  = metric_bad_scores[:, li]

        cal_sc  = np.concatenate([g, mb])
        youden_thr = _find_youden_threshold(cal_labels, cal_sc)
        m = _classification_metrics(
            main_labels, np.concatenate([g, b]),
            f"layer {layer} diffmean", youden_threshold=youden_thr,
        )
        m["good_mean"] = float(g.mean())
        m["bad_mean"]  = float(b.mean())
        results["per_layer"][key] = m
        layer_aurocs.append((layer, m["auroc"]))

    # ── Aggregate strategies ──────────────────────────────────────────────
    plot_series: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _agg_section(label_str, g_sc, b_sc, cal_g_sc=None, cal_b_sc=None):
        if cal_g_sc is None:
            cal_g_sc, cal_b_sc = g_sc, b_sc
        cal_lbl  = np.concatenate([np.zeros(len(cal_g_sc)), np.ones(len(cal_b_sc))])
        eval_lbl = np.concatenate([np.zeros(len(g_sc)),     np.ones(len(b_sc))])
        cal_sc   = np.concatenate([cal_g_sc, cal_b_sc])
        eval_sc  = np.concatenate([g_sc, b_sc])
        ythr     = _find_youden_threshold(cal_lbl, cal_sc)
        print(f"\n  {label_str}  good={g_sc.mean():.4f}±{g_sc.std():.4f}  "
              f"bad={b_sc.mean():.4f}±{b_sc.std():.4f}")
        m = _classification_metrics(eval_lbl, eval_sc, label_str, youden_threshold=ythr)
        m["good_mean"] = float(g_sc.mean())
        m["bad_mean"]  = float(b_sc.mean())
        plot_series.append((label_str, eval_lbl, eval_sc))
        return m

    print("\n=================== Aggregate metrics ===================")

    print("\n--- mean across layers ---")
    results["aggregate"]["mean"] = _agg_section(
        "mean diffmean-score",
        good_eval_scores.mean(1), bad_scores.mean(1),
        cal_g_sc=good_eval_scores.mean(1), cal_b_sc=metric_bad_scores.mean(1),
    )

    print("\n--- max across layers (most adversarial layer) ---")
    results["aggregate"]["max"] = _agg_section(
        "max diffmean-score",
        good_eval_scores.max(1), bad_scores.max(1),
        cal_g_sc=good_eval_scores.max(1), cal_b_sc=metric_bad_scores.max(1),
    )

    print("\n--- best single layer (by AUROC) ---")
    best_layer, best_auroc = max(layer_aurocs, key=lambda x: x[1])
    best_li = layers.index(best_layer)
    print(f"  best layer: {best_layer}  (AUROC={best_auroc:.4f})")
    m_best = _agg_section(
        f"best-layer diffmean-score (L{best_layer})",
        good_eval_scores[:, best_li], bad_scores[:, best_li],
        cal_g_sc=good_eval_scores[:, best_li], cal_b_sc=metric_bad_scores[:, best_li],
    )
    m_best["best_layer"] = best_layer
    results["aggregate"]["best_layer"] = m_best

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    try:
        _make_plots(out_dir, plot_series)
    except Exception as e:
        print(f"  Warning: could not generate plots: {e}")

    class _Encoder(json.JSONEncoder):
        def default(self, o):
            import torch, numpy as np
            if isinstance(o, torch.Tensor):
                return o.tolist()
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            return super().default(o)

    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, cls=_Encoder)
    print(f"Saved to {out_json}")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

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
            data_selection=cfg.get("data_selection", cfg.get("data", "all_semantic")),
            metric_data_selection=cfg.get("metric_data"),
            num_samples=cfg["num_samples"],
            hf_dataset=cfg.get("hf_dataset"),
            num_gpus=cfg.get("num_gpus", 4),
            token_pooling=cfg.get("token_pooling", "mean"),
            good_data=cfg.get("good_data", "fineweb"),
        )

    fire.Fire({"run": run, "aggregate": aggregate})
