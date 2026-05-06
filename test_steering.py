from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from glp.denoiser import load_glp
from glp import flow_matching, script_eval, script_steer, script_probe

# if you're running this in a 24GB GPU, you'll need two GPUs
# we automatically determine the device based on your available memory
free_mem, _ = torch.cuda.mem_get_info(device)
min_gb = 17
hf_device = device if free_mem >= min_gb * 1e9 else "cuda:1"
     

assert glp_name == "generative-latent-prior/glp-llama8b-d6", "Only Llama8B GLP is supported for this demo"
hf_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch.bfloat16)
hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
hf_tokenizer.pad_token = hf_tokenizer.eos_token
hf_tokenizer.padding_side = "left"
hf_model.to(hf_device)
     

question = "What should be done with criminals in society?"

persona = "evil"
seed = 42
max_new_tokens = 20
temperature = 0.0
alphas = torch.tensor([1, 2, 3, 4, 5])
     

# persona vectors are 1-indexed while transformers layers are 0-indexed
hf_layer = 15
hf_layer_name = f"model.layers.{hf_layer}"
persona_layer = hf_layer + 1

# get the pre-cached persona vector
get_persona_vector = lambda persona: torch.load(f"integrations/persona_vectors/cached_vectors/Llama-3.1-8B-Instruct/{persona}_response_avg_diff.pt")
persona_vector = get_persona_vector(persona)[persona_layer].to(device=hf_model.device, dtype=hf_model.dtype)

# enumerate steering settings
settings = {
    "No Intervention": (None, None),
    "Persona Vector": (script_steer.addition_intervention, None),
    "+GLP": (script_steer.addition_intervention, script_steer.postprocess_on_manifold_wrapper(model)),
}
# run steering
results = {}
for setting, (intervention_wrapper, postprocess_fn) in settings.items():
    print(f"Running {setting}...")
    generate_with_intervention = script_steer.generate_with_intervention_wrapper(seed=seed)
    gen_text = generate_with_intervention(
        [question] * len(alphas), 
        hf_model,
        hf_tokenizer,
        layers=[hf_layer_name],
        intervention_wrapper=intervention_wrapper,
        intervention_kwargs={"w": persona_vector, "alphas": alphas, "postprocess_fn": postprocess_fn},
        generate_kwargs={"max_new_tokens": max_new_tokens, "do_sample": temperature > 0, "temperature": temperature}
    )
    results[setting] = gen_text
     