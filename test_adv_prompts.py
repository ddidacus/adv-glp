"""
test_adv_prompts.py

Feed adversarial prompts from centrepourlasecuriteia/jailbreak-dataset through
Llama-3.2-1B-Instruct, then evaluate each (question, answer) pair with
Llama Guard 3-8B.  Output is a JSON list of {question, answer, safety_label}.

Usage:
  python test_adv_prompts.py [--n_samples N] [--batch_size B]
                             [--gen_device cuda:0] [--guard_device cuda:1]
                             [--out results/adv_prompts_eval.json]
"""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from merge_datasets import (
    generate_responses_batch,
    load_judge,
)

LLM_MODEL_ID  = "meta-llama/Llama-3.2-1B-Instruct"
JUDGE_MODEL_ID = "meta-llama/Llama-Guard-3-8B"


def parse_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--dataset", type=str, default="centreia",
    #                choices=["centreia", "wildjailbreak"],
    #                help="Dataset to evaluate: 'centreia' (centrepourlasecuriteia/jailbreak-dataset) "
    #                     "or 'wildjailbreak' (allenai/wildjailbreak eval split, "
    #                     "adversarial_benign + adversarial_harmful)")
    # p.add_argument("--n_samples", type=int, default=None,
    #                help="Max prompts per split to evaluate (default: all)")
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle seed for dataset splits")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Generation batch size (default: 256, tuned for 80GB GPU + 1B model)")
    p.add_argument("--guard_batch_size", type=int, default=32,
                   help="Judge batch size (default: 64, tuned for 80GB GPU + 8B guard)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--gen_device",   type=str, default="cuda:0")
    p.add_argument("--guard_device", type=str, default="cuda:0")
    p.add_argument("--out", type=str, default="results/adv_prompts_eval.json")
    p.add_argument("--shard_id",    type=int, default=0,
                   help="Index of this shard (0-based)")
    p.add_argument("--num_shards",  type=int, default=1,
                   help="Total number of shards")
    return p.parse_args()


def _load_centreia(n_samples, seed, shard_id, num_shards):
    print("Loading dataset: centrepourlasecuriteia/jailbreak-dataset")
    ds = load_dataset("centrepourlasecuriteia/jailbreak-dataset")
    split = ds["train"] if hasattr(ds, "keys") else ds
    prompts = [row["prompt"] for row in split if row["category"] != "Benign"]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    if n_samples is not None:
        prompts = prompts[:n_samples]
    if num_shards > 1:
        shard_size = (len(prompts) + num_shards - 1) // num_shards
        prompts = prompts[shard_id * shard_size : (shard_id + 1) * shard_size]
    print(f"  {len(prompts)} adversarial prompts (shard {shard_id}/{num_shards})")
    # returns (harmful_prompts, benign_prompts); centreia has no benign split
    return prompts, []


def _load_wildjailbreak(n_samples, seed, shard_id, num_shards):
    print("Loading dataset: allenai/wildjailbreak (eval split)")
    ds = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)
    split = ds["train"]  # the eval config has a single "train" split

    harmful = [row["adversarial"] for row in split
               if row["data_type"] == "adversarial_harmful" and row["adversarial"]]
    benign  = [row["adversarial"] for row in split
               if row["data_type"] == "adversarial_benign"  and row["adversarial"]]

    rng = random.Random(seed)
    rng.shuffle(harmful)
    rng.shuffle(benign)

    if n_samples is not None:
        harmful = harmful[:n_samples]
        benign  = benign[:n_samples]

    if num_shards > 1:
        shard_size = (len(harmful) + num_shards - 1) // num_shards
        harmful = harmful[shard_id * shard_size : (shard_id + 1) * shard_size]
        shard_size = (len(benign) + num_shards - 1) // num_shards
        benign  = benign [shard_id * shard_size : (shard_id + 1) * shard_size]

    print(f"  {len(harmful)} adversarial_harmful, {len(benign)} adversarial_benign "
          f"(shard {shard_id}/{num_shards})")
    return harmful, benign


@torch.no_grad()
def judge_batch(guard_model, guard_tokenizer, prompts, responses, device, max_input_tokens=1024):
    """Run Llama Guard on (prompt, response) pairs; returns list of 'safe'/'unsafe' strings."""
    pad_id = guard_tokenizer.pad_token_id or 0
    encoded = [
        guard_tokenizer.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
            tokenize=True,
        )
        for p, r in zip(prompts, responses)
    ]
    encoded = [e[-max_input_tokens:] if len(e) > max_input_tokens else e for e in encoded]
    max_len = max(len(e) for e in encoded)
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(e)) + e for e in encoded], device=device
    )
    attention_mask = torch.tensor(
        [[0] * (max_len - len(e)) + [1] * len(e) for e in encoded], device=device
    )
    outputs = guard_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=20,
        pad_token_id=pad_id,
    )
    labels = []
    for out in outputs:
        text = guard_tokenizer.decode(out[max_len:], skip_special_tokens=True).strip()
        verdict = text.splitlines()[0].strip().lower()
        labels.append("unsafe" if verdict == "unsafe" else "safe")
    return labels


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    # if args.dataset == "centreia":
    #     adversarial, benign = _load_centreia(args.n_samples, args.seed, args.shard_id, args.num_shards)
    # else:
    #     adversarial, benign = _load_wildjailbreak(args.n_samples, args.seed, args.shard_id, args.num_shards)

    dataset = load_dataset("ddidacus/guard-glp-data", split="train")
    adversarial = [sample["prompt"] for sample in dataset if sample["adversarial"]]
    benign = [sample["prompt"] for sample in dataset if not sample["adversarial"]]

    all_prompts = adversarial + benign
    prompt_labels = ["harmful"] * len(adversarial) + ["benign"] * len(benign)

    # ── Load LLM ─────────────────────────────────────────────────────────────
    print(f"Loading LLM: {LLM_MODEL_ID}")
    llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    llm_tok.padding_side = "left"
    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID, torch_dtype=torch.bfloat16
    ).to(args.gen_device)
    llm.eval()

    # ── Generate responses ────────────────────────────────────────────────────
    responses = []
    for i in tqdm(range(0, len(all_prompts), args.batch_size), desc="generating"):
        batch = all_prompts[i : i + args.batch_size]
        out = generate_responses_batch(
            llm, llm_tok, batch, args.gen_device,
            max_new_tokens=args.max_new_tokens,
        )
        responses.extend(out)

    # ── Free LLM memory, load Llama Guard ────────────────────────────────────
    del llm
    torch.cuda.empty_cache()

    print(f"Loading Llama Guard: {JUDGE_MODEL_ID}")
    guard_model, guard_tok = load_judge(args.guard_device)

    # ── Evaluate with Llama Guard ────────────────────────────────────────────
    labels = []
    guard_bs = args.guard_batch_size
    i = 0
    pbar = tqdm(total=len(all_prompts), desc="judging")
    while i < len(all_prompts):
        batch_q = all_prompts[i : i + guard_bs]
        batch_r = responses[i : i + guard_bs]
        while True:
            try:
                batch_labels = judge_batch(
                    guard_model, guard_tok, batch_q, batch_r, args.guard_device
                )
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if guard_bs == 1:
                    print(f"OOM in guard with batch_size=1. Skipping {len(batch_q)} sample(s).")
                    batch_labels = ["unknown"] * len(batch_q)
                    break
                guard_bs = max(1, guard_bs // 2)
                print(f"OOM in guard, retrying with batch_size={guard_bs}")
                batch_q = all_prompts[i : i + guard_bs]
                batch_r = responses[i : i + guard_bs]
        labels.extend(batch_labels)
        pbar.update(len(batch_q))
        i += len(batch_q)
    pbar.close()

    # ── Save results ──────────────────────────────────────────────────────────
    records = [
        {"question": q, "answer": a, "safety_label": l, "prompt_label": pl}
        for q, a, l, pl in zip(all_prompts, responses, labels, prompt_labels)
    ]
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {out_path}")

    harmful_recs = [r for r in records if r["prompt_label"] == "harmful"]
    benign_recs  = [r for r in records if r["prompt_label"] == "benign"]

    if harmful_recs:
        n_unsafe = sum(r["safety_label"] == "unsafe" for r in harmful_recs)
        print(f"  harmful  — unsafe (ASR): {n_unsafe}/{len(harmful_recs)} "
              f"({100*n_unsafe/len(harmful_recs):.1f}%)")
    if benign_recs:
        n_refused = sum(r["safety_label"] == "safe" for r in benign_recs)
        n_unsafe_b = sum(r["safety_label"] == "unsafe" for r in benign_recs)
        print(f"  benign   — safe (correct): {n_refused}/{len(benign_recs)} "
              f"({100*n_refused/len(benign_recs):.1f}%)  |  "
              f"refused (false-positive): {n_unsafe_b}/{len(benign_recs)} "
              f"({100*n_unsafe_b/len(benign_recs):.1f}%)")


if __name__ == "__main__":
    main()
