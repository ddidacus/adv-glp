"""
Linear-probe classifier on LLM residual-stream activations.

Trains one logistic-regression probe per layer on the metric_data_selection
(calibration) set, then evaluates on the data_selection (test) set with the
Youden threshold transferred from calibration predictions — matching the
evaluation protocol of eval_classifier.py exactly.

Two-pass workflow (mirrors eval_classifier.py):
  Pass 1 — activation extraction (one job per GPU):
      python eval_linear_probe.py run --config=cfg.yaml --gpu_id=0

  Pass 2 — probe training + evaluation (single job, CPU-capable):
      python eval_linear_probe.py aggregate --out_dir=results/lp_run \\
          [--probe_lr=1e-3] [--probe_epochs=100] [--probe_wd=1e-4] \\
          [--probe_batch_size=64] [--device=cpu]
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
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


DATASET_SELECTIONS = {
    "all_semantic": [
        harmeval_batches,
        jailbreakbench_batches,
        sg_bench_batches,
        adversarial_wildjailbreak_batches,
    ],
    "harmeval":              [harmeval_batches],
    "harmeval_gcg":          [harmeval_gcg_batches],
    "jailbreakbench":        [jailbreakbench_batches],
    "sgbench":               [sg_bench_batches],
    "vanilla_wildjailbreak": [vanilla_wildjailbreak_batches],
    "adversarial_wildjailbreak": [adversarial_wildjailbreak_batches],
}


# classifier model (linear probe)

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)  # (N,) raw logits

# probe training routine

def _train_probe(
    acts: torch.Tensor,     # (N, D) float32
    labels: torch.Tensor,   # (N,) float32  1=good 0=bad
    lr: float,
    num_epochs: int,
    weight_decay: float,
    batch_size: int,
    device: str,
) -> LinearProbe:
    """ standard supervised learning """
    probe = LinearProbe(acts.shape[1]).to(device)
    opt   = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    ds     = torch.utils.data.TensorDataset(
        acts.to(device=device, dtype=torch.float32),
        labels.to(device=device, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    probe.train()
    for _ in range(num_epochs):
        for x, y in loader:
            opt.zero_grad()
            loss_fn(probe(x), y).backward()
            opt.step()

    return probe


@torch.no_grad()
def _score_probe(probe: LinearProbe, acts: torch.Tensor, device: str, batch_size: int) -> np.ndarray:
    """Sigmoid probability that a sample is good (in-distribution); higher = more in-dist."""
    probe.eval()
    acts = acts.to(device=device, dtype=torch.float32)
    probs = []
    for i in range(0, len(acts), batch_size):
        probs.append(torch.sigmoid(probe(acts[i:i + batch_size])).cpu())
    return torch.cat(probs).numpy()


# extract activations on test prompts

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
    fns     = DATASET_SELECTIONS[dataset_key]
    per_fn  = int(num_samples)
    kwargs  = dict(skip=skip, batch_size=batch_size, num_samples=per_fn)
    chunks  = []
    for fn in fns:
        for batch in tqdm(list(fn(**kwargs)), desc=fn.__name__, mininterval=30, ncols=120):
            chunks.append(
                extract_activations(batch, llm_model, llm_tokenizer, diffusion_model,
                                    device=device, batch_size=batch_size,
                                    token_pooling=token_pooling).cpu()
            )
    return torch.cat(chunks, dim=0)  # (N, L, D)


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
    torch.manual_seed(42)
    random.seed(42)

    device = f"cuda:{gpu_id}"

    # loading models

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
    skip = gpu_id * num_samples if num_samples is not None else 0

    print(f"[+] token_pooling:    {token_pooling}")
    print(f"[+] good_data:        {good_data}")

    def _acts_from_batches(batches: list[list[str]]) -> torch.Tensor:
        chunks = [extract_activations(b, **common, batch_size=batch_size).cpu() for b in batches]
        return torch.cat(chunks, dim=0)

    def _token_lengths(prompts: list[str]) -> list[int]:
        enc = llm_tokenizer(prompts, truncation=True, max_length=2048)
        return [len(ids) for ids in enc["input_ids"]]

    # load hf unified dataset

    if hf_dataset:
        _hf = load_hf_splits(hf_dataset, gpu_id, num_gpus, num_samples, batch_size)
        _overlap = set(_hf["good_prompts"]) & set(_hf["good_eval_prompts"])
        assert not _overlap, (
            f"good_prompts and good_eval_prompts overlap: {len(_overlap)} shared prompts"
        )
        print(f"  [check] good_prompts ∩ good_eval_prompts = 0 (OK)")
        print("Extracting good train activations (HF train first half)...")
        good_acts = _acts_from_batches(_chunk(_hf["good_prompts"], batch_size))
        good_token_lengths = _token_lengths(_hf["good_prompts"])
        print(f"  good_acts: {tuple(good_acts.shape)}")

        print("Extracting good eval activations (HF train second half, held-out)...")
        good_eval_acts = _acts_from_batches(_chunk(_hf["good_eval_prompts"], batch_size))
        good_eval_token_lengths = _token_lengths(_hf["good_eval_prompts"])
        print(f"  good_eval_acts: {tuple(good_eval_acts.shape)}")

        print("Extracting metric bad activations (HF calibration split)...")
        metric_bad_acts = _acts_from_batches(_chunk(_hf["cal_bad_prompts"], batch_size))
        metric_bad_token_lengths = _token_lengths(_hf["cal_bad_prompts"])
        print(f"  metric_bad_acts: {tuple(metric_bad_acts.shape)}")

        print("Extracting test bad activations (HF test split)...")
        bad_acts = _acts_from_batches(_chunk(_hf["test_bad_prompts"], batch_size))
        bad_token_lengths = _token_lengths(_hf["test_bad_prompts"])
        print(f"  bad_acts: {tuple(bad_acts.shape)}")
        has_separate_test = True

    # load batches from raw sources (outdated)
    else:
        assert num_samples is not None, "num_samples is required when not using hf_dataset"
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

    # store extracted activations
    out_file = os.path.join(out_dir, f"acts_{gpu_id}.th")
    save_dict = {
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
        "good_data":             good_data if not hf_dataset else "hf",
    }
    if hf_dataset:
        save_dict.update(
            good_token_lengths=good_token_lengths,
            good_eval_token_lengths=good_eval_token_lengths,
            metric_bad_token_lengths=metric_bad_token_lengths,
            bad_token_lengths=bad_token_lengths,
        )
    torch.save(save_dict, out_file)
    print(f"[GPU {gpu_id}] Saved activations to {out_file}")


def _run_sanity_checks(
    good_acts: torch.Tensor,        # (N_train, L, D)
    good_eval_acts: torch.Tensor,   # (N_eval,  L, D)
    metric_bad_acts: torch.Tensor,  # (N_cal,   L, D)
    bad_acts: torch.Tensor,         # (N_bad,   L, D)
    layers: list[int],
    train_labels: torch.Tensor,
    probe_lr: float,
    probe_epochs: int,
    probe_wd: float,
    probe_batch_size: int,
    device: str,
    good_eval_lengths: np.ndarray | None,
    bad_lengths: np.ndarray | None,
    metric_bad_lengths: np.ndarray | None,
    good_eval_scores: np.ndarray | None = None,  # (N_eval, L) — from real probes
    bad_scores: np.ndarray | None = None,        # (N_bad,  L)
) -> dict:
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    n_good_eval  = len(good_eval_acts)
    n_bad        = len(bad_acts)
    n_metric_bad = len(metric_bad_acts)
    main_labels  = np.concatenate([np.ones(n_good_eval), np.zeros(n_bad)])

    print("\n\n=================== SANITY CHECKS ===================")
    sanity: dict = {}

    # ── 2a: norm-only AUROC per layer ─────────────────────────────────────
    print("\n[norm-AUROC]  L2 norm of last-token activation as the only feature")
    norm_aurocs = {}
    for li, layer in enumerate(layers):
        g_norm  = good_eval_acts[:, li, :].float().norm(dim=1).numpy()
        b_norm  = bad_acts[:, li, :].float().norm(dim=1).numpy()
        scores  = np.concatenate([g_norm, b_norm])
        auroc   = float(roc_auc_score(main_labels, scores))
        # check both polarities (good may have higher or lower norm)
        auroc   = max(auroc, 1.0 - auroc)
        print(f"  layer {layer:2d}:  AUROC={auroc:.4f}  "
              f"mean_norm  good={g_norm.mean():.1f}±{g_norm.std():.1f}  "
              f"bad={b_norm.mean():.1f}±{b_norm.std():.1f}")
        norm_aurocs[f"layer_{layer}"] = {"auroc": auroc,
                                          "good_norm_mean": float(g_norm.mean()),
                                          "bad_norm_mean":  float(b_norm.mean())}
    sanity["norm_auroc"] = norm_aurocs

    # ── 2b: shuffled-label probe (null hypothesis) ────────────────────────
    print("\n[shuffled-label probe]  Probe trained on randomly permuted labels (expected AUROC≈0.5)")
    N_SHUFFLES = 3
    shuffled_aurocs = []
    rng = np.random.default_rng(0)
    for run_i in range(N_SHUFFLES):
        shuffled = train_labels[rng.permutation(len(train_labels))]
        all_layer_aurocs = []
        for li, layer in enumerate(layers):
            train_acts_li = torch.cat([good_acts[:, li, :], metric_bad_acts[:, li, :]], dim=0)
            probe = _train_probe(train_acts_li, shuffled,
                                 lr=probe_lr, num_epochs=probe_epochs,
                                 weight_decay=probe_wd, batch_size=probe_batch_size,
                                 device=device)
            g_sc = _score_probe(probe, good_eval_acts[:, li, :], device, probe_batch_size)
            b_sc = _score_probe(probe, bad_acts[:, li, :], device, probe_batch_size)
            all_layer_aurocs.append(roc_auc_score(main_labels, np.concatenate([g_sc, b_sc])))
        mean_auroc = float(np.mean(all_layer_aurocs))
        shuffled_aurocs.append(mean_auroc)
        print(f"  run {run_i+1}: mean AUROC across layers = {mean_auroc:.4f}")
    sanity["shuffled_probe"] = {
        "mean_auroc": float(np.mean(shuffled_aurocs)),
        "std_auroc":  float(np.std(shuffled_aurocs)),
        "runs": shuffled_aurocs,
    }
    print(f"  → mean={np.mean(shuffled_aurocs):.4f}  std={np.std(shuffled_aurocs):.4f}")

    # ── 2c: length-based diagnostics (only if lengths were saved) ─────────
    if good_eval_lengths is not None and bad_lengths is not None:
        print("\n[length stats]  Token length distributions")
        print(f"  good_eval : mean={good_eval_lengths.mean():.1f}  "
              f"std={good_eval_lengths.std():.1f}  "
              f"min={good_eval_lengths.min()}  max={good_eval_lengths.max()}")
        print(f"  bad       : mean={bad_lengths.mean():.1f}  "
              f"std={bad_lengths.std():.1f}  "
              f"min={bad_lengths.min()}  max={bad_lengths.max()}")

        # length-only logistic regression
        all_lengths = np.concatenate([good_eval_lengths, bad_lengths]).reshape(-1, 1).astype(float)
        lr_clf = LogisticRegression(max_iter=1000)
        lr_clf.fit(all_lengths, main_labels)
        length_scores = lr_clf.predict_proba(all_lengths)[:, 1]
        length_auroc  = float(roc_auc_score(main_labels, length_scores))
        print(f"\n[length-only AUROC]  LogisticRegression on scalar token length: {length_auroc:.4f}")

        # length-stratified probe AUROC (mean across layers)
        combined_lengths = np.concatenate([good_eval_lengths, bad_lengths])
        q33, q66 = np.percentile(combined_lengths, [33, 66])
        buckets = {
            f"short (≤{int(q33)} tok)":  combined_lengths <= q33,
            f"mid ({int(q33)+1}–{int(q66)} tok)": (combined_lengths > q33) & (combined_lengths <= q66),
            f"long (>{int(q66)} tok)":   combined_lengths > q66,
        }
        print("\n[length-stratified AUROC]  Real probe (mean-layer score), within length bucket")
        strat: dict = {}
        if good_eval_scores is not None and bad_scores is not None:
            combined_scores = np.concatenate([good_eval_scores.mean(1), bad_scores.mean(1)])
        else:
            combined_scores = None
        for name, mask in buckets.items():
            n_in = mask.sum()
            lbl_in = main_labels[mask]
            n_good_in = int(lbl_in.sum())
            n_bad_in  = int((lbl_in == 0).sum())
            if n_good_in < 2 or n_bad_in < 2:
                print(f"  {name}: skipped (too few samples: good={n_good_in} bad={n_bad_in})")
                continue
            entry = {"n": int(n_in), "n_good": n_good_in, "n_bad": n_bad_in}
            if combined_scores is not None:
                sc_in  = combined_scores[mask]
                auroc_in = float(roc_auc_score(lbl_in, sc_in))
                entry["auroc"] = auroc_in
                print(f"  {name}: n={n_in}  (good={n_good_in}  bad={n_bad_in})  AUROC={auroc_in:.4f}")
            else:
                print(f"  {name}: n={n_in}  (good={n_good_in}  bad={n_bad_in})  AUROC=n/a")
            strat[name] = entry

        sanity["length"] = {
            "good_eval_mean": float(good_eval_lengths.mean()),
            "bad_mean":       float(bad_lengths.mean()),
            "length_only_auroc": length_auroc,
            "strat_buckets":  strat,
        }
    else:
        print("\n[length checks]  Skipped — re-run extraction to save token lengths.")
        sanity["length"] = None

    print("=====================================================\n")
    return sanity


def aggregate(
    out_dir: str,
    probe_lr: float = 1e-3,
    probe_epochs: int = 100,
    probe_wd: float = 1e-4,
    probe_batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    """ Combine gpu shards of activations, train and eval linear probe """
    out_dir = Path(out_dir)
    files   = sorted(out_dir.glob("acts_*.th"))
    assert files, f"No acts_*.th files found in {out_dir}"

    chunks = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    print(f"Loaded {len(chunks)} GPU activation file(s) from {out_dir}")

    good_acts        = torch.cat([c["good_acts"]        for c in chunks], dim=0)  # (N_train, L, D) — probe training only
    good_eval_acts   = torch.cat([c["good_eval_acts"]   for c in chunks], dim=0)  # (N_eval,  L, D) — held-out good
    metric_bad_acts  = torch.cat([c["metric_bad_acts"]  for c in chunks], dim=0)  # (N_cal,   L, D)
    bad_acts         = torch.cat([c["bad_acts"]         for c in chunks], dim=0)  # (N_bad,   L, D)

    _length_keys = ("good_token_lengths", "good_eval_token_lengths",
                    "metric_bad_token_lengths", "bad_token_lengths")
    _tensor_keys = ("good_acts", "good_eval_acts", "metric_bad_acts", "bad_acts")

    # concatenate lengths across GPU shards if present
    def _cat_lengths(key):
        if chunks[0].get(key) is None:
            return None
        return np.array([l for c in chunks for l in c[key]], dtype=np.int32)

    good_token_lengths      = _cat_lengths("good_token_lengths")
    good_eval_token_lengths = _cat_lengths("good_eval_token_lengths")
    metric_bad_token_lengths= _cat_lengths("metric_bad_token_lengths")
    bad_token_lengths       = _cat_lengths("bad_token_lengths")

    cfg = {k: v for k, v in chunks[0].items()
           if k not in _tensor_keys and k not in _length_keys}
    cfg["layers"] = [int(l) for l in cfg["layers"]]
    layers     = cfg["layers"]
    data_sel   = cfg["data_selection"]
    metric_sel = cfg["metric_data_selection"]
    has_separate_test = cfg["has_separate_test"]

    n_good       = len(good_acts)       # probe training
    n_good_eval  = len(good_eval_acts)  # held-out good evaluation
    n_metric_bad = len(metric_bad_acts)
    n_bad        = len(bad_acts)

    print(f"  good_train: {n_good}  |  good_eval: {n_good_eval}  |  "
          f"metric_bad: {n_metric_bad}  |  bad: {n_bad}")
    print(f"  layers: {layers}")
    print(f"  probe_lr={probe_lr}  probe_epochs={probe_epochs}  "
          f"probe_wd={probe_wd}  probe_batch_size={probe_batch_size}  device={device}")

    # probe trained on good_train + cal_bad; Youden calibrated on good_eval + cal_bad
    train_labels = torch.cat([torch.ones(n_good), torch.zeros(n_metric_bad)])

    cal_labels  = np.concatenate([np.ones(n_good_eval), np.zeros(n_metric_bad)])
    main_labels = np.concatenate([np.ones(n_good_eval), np.zeros(n_bad)])

    results: dict = {
        "config": {**cfg, "probe_lr": probe_lr, "probe_epochs": probe_epochs,
                   "probe_wd": probe_wd, "probe_batch_size": probe_batch_size},
        "n_good_train": n_good, "n_good_eval": n_good_eval,
        "n_bad": n_bad, "n_metric_bad": n_metric_bad,
        "per_layer": {}, "aggregate": {},
    }

    good_eval_scores  = np.zeros((n_good_eval,  len(layers)), dtype=np.float32)
    bad_scores        = np.zeros((n_bad,         len(layers)), dtype=np.float32)
    metric_bad_scores = np.zeros((n_metric_bad,  len(layers)), dtype=np.float32)

    print("\n=================== Training probes ===================")
    layer_aurocs = []

    for li, layer in enumerate(layers):
        train_acts_li = torch.cat([good_acts[:, li, :], metric_bad_acts[:, li, :]], dim=0)
        probe = _train_probe(
            train_acts_li, train_labels,
            lr=probe_lr, num_epochs=probe_epochs,
            weight_decay=probe_wd, batch_size=probe_batch_size,
            device=device,
        )

        good_eval_scores[:, li]  = _score_probe(probe, good_eval_acts[:, li, :], device, probe_batch_size)
        metric_bad_scores[:, li] = _score_probe(probe, metric_bad_acts[:, li, :], device, probe_batch_size)
        bad_scores[:, li]        = (
            _score_probe(probe, bad_acts[:, li, :], device, probe_batch_size)
            if has_separate_test else metric_bad_scores[:, li]
        )

        g, mb, b = good_eval_scores[:, li], metric_bad_scores[:, li], bad_scores[:, li]
        print(f"  Layer {layer:2d}  good_eval={g.mean():.3f}±{g.std():.3f}  "
              f"metric_bad={mb.mean():.3f}±{mb.std():.3f}  bad={b.mean():.3f}±{b.std():.3f}")

    print("\n=================== Per-layer metrics ===================")
    for li, layer in enumerate(layers):
        key = f"layer_{layer}"
        g   = good_eval_scores[:, li]
        b   = bad_scores[:, li]
        mb  = metric_bad_scores[:, li]

        youden_thr = _find_youden_threshold(cal_labels, np.concatenate([g, mb]))
        m = _classification_metrics(
            main_labels, np.concatenate([g, b]),
            f"layer {layer} probe", youden_threshold=youden_thr,
        )
        m["good_mean"] = float(g.mean())
        m["bad_mean"]  = float(b.mean())
        results["per_layer"][key] = m
        layer_aurocs.append((layer, m["auroc"]))

    # aggregate layer metrics

    plot_series: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _agg_section(label_str, g_sc, b_sc, cal_g_sc=None, cal_b_sc=None):
        if cal_g_sc is None:
            cal_g_sc, cal_b_sc = g_sc, b_sc
        cal_lbl  = np.concatenate([np.ones(len(cal_g_sc)), np.zeros(len(cal_b_sc))])
        eval_lbl = np.concatenate([np.ones(len(g_sc)),     np.zeros(len(b_sc))])
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
        "mean probe-score",
        good_eval_scores.mean(1), bad_scores.mean(1),
        cal_g_sc=good_eval_scores.mean(1), cal_b_sc=metric_bad_scores.mean(1),
    )

    print("\n--- min across layers (most anomalous layer) ---")
    results["aggregate"]["min"] = _agg_section(
        "min probe-score",
        good_eval_scores.min(1), bad_scores.min(1),
        cal_g_sc=good_eval_scores.min(1), cal_b_sc=metric_bad_scores.min(1),
    )

    print("\n--- best single layer (by AUROC) ---")
    best_layer, best_auroc = max(layer_aurocs, key=lambda x: x[1])
    best_li = layers.index(best_layer)
    print(f"  best layer: {best_layer}  (AUROC={best_auroc:.4f})")
    m_best = _agg_section(
        f"best-layer probe-score (L{best_layer})",
        good_eval_scores[:, best_li], bad_scores[:, best_li],
        cal_g_sc=good_eval_scores[:, best_li], cal_b_sc=metric_bad_scores[:, best_li],
    )
    m_best["best_layer"] = best_layer
    results["aggregate"]["best_layer"] = m_best

    # sanity checks
    results["sanity"] = _run_sanity_checks(
        good_acts=good_acts,
        good_eval_acts=good_eval_acts,
        metric_bad_acts=metric_bad_acts,
        bad_acts=bad_acts,
        layers=layers,
        train_labels=train_labels,
        probe_lr=probe_lr,
        probe_epochs=probe_epochs,
        probe_wd=probe_wd,
        probe_batch_size=probe_batch_size,
        device=device,
        good_eval_lengths=good_eval_token_lengths,
        bad_lengths=bad_token_lengths,
        metric_bad_lengths=metric_bad_token_lengths,
        good_eval_scores=good_eval_scores,
        bad_scores=bad_scores,
    )

    # plotting
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
            data_selection=cfg.get("data_selection", cfg.get("data", "all_semantic")),
            metric_data_selection=cfg.get("metric_data"),
            num_samples=cfg["num_samples"],
            hf_dataset=cfg.get("hf_dataset"),
            num_gpus=cfg.get("num_gpus", 4),
            token_pooling=cfg.get("token_pooling", "mean"),
            good_data=cfg.get("good_data", "fineweb"),
        )

    fire.Fire({"run": run, "aggregate": aggregate})
