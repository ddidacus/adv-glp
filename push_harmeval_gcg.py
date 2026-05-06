"""Push data/harmeval_gcg_512/harmeval_gcg to the HuggingFace Hub."""
from datasets import load_from_disk
from huggingface_hub import HfApi

DISK_PATH = "data/harmeval_gcg_512/harmeval_gcg"
REPO_ID   = "ddidacus/harmeval-gcg-llama3-1b"

DATASET_CARD = """\
---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- safety
- jailbreak
- gcg
- adversarial
- llm
---

# harmeval-gcg-llama3-1b

HarmEval ([SoftMINER-Group/HarmEval](https://huggingface.co/datasets/SoftMINER-Group/HarmEval)) augmented with adversarial suffixes generated via the **Greedy Coordinate Gradient (GCG)** attack method, optimized specifically against **Llama-3.2-1B-Instruct**.

Each harmful prompt is paired with a GCG-optimized adversarial suffix that, when appended to the original question, maximizes the probability of the model producing a target harmful response.

## Columns

| Column | Description |
|---|---|
| `question` | Original harmful prompt from HarmEval |
| `target` | Target response prefix used during GCG optimization (e.g. `"Sure, here is the answer:"`) |
| `attack` | Raw GCG adversarial suffix (gibberish token sequence) |
| `adv_prompt` | Full adversarial prompt: `question + attack` (ready to use as model input) |
| `best_loss` | Cross-entropy loss of the target at the end of optimization (lower = stronger attack) |

## Statistics

- **550 rows** (one per HarmEval question)
- Suffix search run for a fixed budget of GCG steps targeting Llama-3.2-1B-Instruct

## Intended use

Evaluating the robustness of safety classifiers (e.g. linear probes, GLP diffusion models) against GCG-style adversarial inputs at the activation level.
"""


def main():
    print(f"Loading dataset from {DISK_PATH}...")
    ds = load_from_disk(DISK_PATH)
    print(ds)

    print(f"Pushing to hub: {REPO_ID}...")
    ds.push_to_hub(REPO_ID)

    print("Pushing dataset card...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print("Done.")


if __name__ == "__main__":
    main()
