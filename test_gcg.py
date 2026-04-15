"""
Load the harmeval_gcg dataset, show the LLM's answer for every adversarial prompt,
and judge each response with Llama Guard 3 8B.

Usage:
    python test_gcg.py
    python test_gcg.py --model=1b
    python test_gcg.py --model=8b --dataset_path=data/harmeval_gcg/harmeval_gcg
    python test_gcg.py --model=8b --max_new_tokens=300 --num_samples=10
    python test_gcg.py --judge_device=cuda:1
"""

import torch
import fire
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_IDS = {
    "1b": "unsloth/Llama-3.2-1B",
    "8b": "meta-llama/Llama-3.1-8B",
}

JUDGE_MODEL_ID = "meta-llama/Llama-Guard-3-8B"


def load_judge(device: str):
    print(f"Loading judge: {JUDGE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def judge_response(judge_model, judge_tokenizer, question: str, response: str, device: str) -> tuple[str, str]:
    """
    Returns (verdict, categories) where verdict is 'safe' or 'unsafe'
    and categories is the violated category string (empty if safe).
    """
    chat = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    input_ids = judge_tokenizer.apply_chat_template(
        chat, return_tensors="pt"
    ).to(device)
    output = judge_model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        pad_token_id=0,
    )
    result = judge_tokenizer.decode(
        output[0][input_ids.shape[-1]:], skip_special_tokens=True
    ).strip()

    # result is either "safe" or "unsafe\nS1,S2,..."
    lines = result.splitlines()
    verdict = lines[0].strip().lower()
    categories = lines[1].strip() if len(lines) > 1 else ""
    return verdict, categories


def main(
    model: str = "1b",
    dataset_path: str = "data/harmeval_gcg/harmeval_gcg",
    max_new_tokens: int = 200,
    num_samples: int = None,
    device: str = "cuda:0",
    judge_device: str = None,
):
    assert model in MODEL_IDS, f"--model must be one of {list(MODEL_IDS)}, got '{model}'"
    model_id = MODEL_IDS[model]
    if judge_device is None:
        judge_device = device

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    llm.eval()

    judge_model, judge_tokenizer = load_judge(judge_device)

    print(f"Loading dataset from: {dataset_path}")
    ds = load_from_disk(dataset_path)
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    print(f"Running inference on {len(ds)} prompts...\n")

    n_unsafe = 0
    print("=" * 80)

    for i, row in enumerate(ds):
        adv_prompt = row["adv_prompt"]
        question = row["question"]
        attack = row["attack"]

        # Generate response
        inputs = tokenizer(adv_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Judge the (question, response) pair
        verdict, categories = judge_response(
            judge_model, judge_tokenizer, question, generated_text, judge_device
        )
        is_unsafe = verdict == "unsafe"
        if is_unsafe:
            n_unsafe += 1

        label = f"UNSAFE [{categories}]" if is_unsafe else "safe"
        print(f"[{i+1}/{len(ds)}] Question: {question}")
        print(f"Attack suffix: {attack!r}")
        print(f"Answer:\n{generated_text}")
        print(f"Judge: {label}")
        print("=" * 80)

    print(f"\nSummary: {n_unsafe}/{len(ds)} responses judged UNSAFE ({100*n_unsafe/len(ds):.1f}%)")


if __name__ == "__main__":
    fire.Fire(main)
