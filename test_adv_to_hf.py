import json
import datasets

# Store successful adversarial examples
with open("results/adv_prompts_eval.json") as fp:
    data = json.load(fp)

unsafe_samples = [sample for sample in data if sample["safety_label"] == "unsafe"]
keys = list(unsafe_samples[0].keys())
data_dict = {k:[] for k in keys}

for s in unsafe_samples:
    for k in keys:
        data_dict[k].append(s[k])

prompts_adversarial = data_dict["question"]
labels_adversarial = ["adversarial_successful"] * len(prompts_adversarial)
print(f"# adv prompts: {len(prompts_adversarial)}")

# Store benign examples
b_ds = datasets.load_dataset("centrepourlasecuriteia/jailbreak-dataset")
b_ds = b_ds["train"].filter(lambda x: x["category"] == "Benign")["prompt"]
prompts_benign = list(b_ds)
labels_benign = ["benign"] * len(prompts_benign)
print(f"# benign prompts: {len(prompts_benign)}")

ds = datasets.Dataset.from_dict({
    "prompt": prompts_adversarial + prompts_benign,
    "label": labels_adversarial + labels_benign
})
print(ds)
ds.save_to_disk("data/centreia_llama1b_prompts")