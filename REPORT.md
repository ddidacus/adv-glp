# Anomaly Detection via Generative Latent Prior: Reconstruction Error Analysis

## Setup

- **LLM**: Llama-3.2-1B (`unsloth/Llama-3.2-1B`, bfloat16)
- **GLP model**: `generative-latent-prior/glp-llama1b-d6` (final checkpoint)
- **Noise level**: 0.5
- **Denoising timesteps**: 100
- **Samples per dataset**: 1024
- **Batch size**: 16
- **Hardware**: NVIDIA RTX 8000 (48 GB)

## Method

For each input string:

1. Extract last-token activations from the LLM across all traced layers.
2. Normalize activations using the GLP's pre-computed statistics.
3. Add noise at level `u = 0.5` via flow matching interpolation.
4. Denoise back with `sample_on_manifold` (100 Euler steps).
5. Compute L2 reconstruction error between original and denoised activations.

The hypothesis is that in-distribution text (benign web data) should be reconstructed more faithfully than out-of-distribution text (adversarial jailbreak prompts), yielding lower reconstruction error.

## Datasets

| Dataset | Description | Source |
|---|---|---|
| **FineWeb** | Benign web text | `HuggingFaceFW/fineweb` (subset `sample-10BT`, split `train`) |
| **WildJailbreak** | Adversarial harmful prompts | `allenai/wildjailbreak` (filtered to `adversarial_harmful`) |

## Results

| Dataset | Min | Max | Median | Mean |
|---|---|---|---|---|
| FineWeb (benign) | 1.367 | 3.016 | 1.793 | 1.844 |
| WildJailbreak (adversarial) | 1.539 | 2.781 | 1.961 | 1.997 |

## Analysis

- **Mean reconstruction error** is 8.3% higher for adversarial prompts (1.997 vs 1.844).
- **Median reconstruction error** is 9.4% higher for adversarial prompts (1.961 vs 1.793).
- The benign distribution has a wider range (1.367--3.016) compared to the adversarial one (1.539--2.781), suggesting that some long or unusual web documents can also produce high error, while jailbreak prompts are more consistently out-of-distribution.
- The distributions overlap, so a simple threshold would not cleanly separate the two classes. Further work could explore per-layer error decomposition, different noise levels, or combining reconstruction error with other features.

## Throughput

- FineWeb batches: ~5--15 s per batch of 16 (variable due to sequence length; `max_length=2048`, `padding="longest"`).
- WildJailbreak batches: ~1--3 s per batch of 16 (prompts are much shorter).
- Total wall time: ~15 min for 2048 samples (1024 per dataset).
