"""Count samples per data_type split in allenai/wildjailbreak."""
import json
from datasets import load_dataset

dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)

splits = ["adversarial_harmful", "adversarial_benign", "vanilla_harmful", "vanilla_benign"]
counts = {}
for split in splits:
    filtered = dataset.filter(lambda x: x["data_type"] == split)
    counts[split] = len(filtered["train"])

with open("split_counts.json", "w") as f:
    json.dump(counts, f, indent=2)

print(json.dumps(counts, indent=2))
