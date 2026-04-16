# Density Estimation with GLP

This document describes how to compute `log p(x)` — the log-probability of a sample under the distribution learned by the GLP flow matching model.

## Background

GLP is a **flow matching** model (a type of continuous normalizing flow). It learns a velocity field `v(x, t)` that defines an ODE transporting between the data distribution and a standard Gaussian `N(0, I)`.

For continuous normalizing flows, exact log-likelihood is available via the **instantaneous change of variables** formula:

```
log p(x_0) = log p_1(x_1) - integral_0^1  tr(dv/dx(x_t, t)) dt
```

where:
- `x_0` is the data point (at `t=0`)
- `x_1` is the endpoint after solving the forward ODE `dx/dt = v(x, t)` from `t=0` to `t=1` (should land in `N(0, I)`)
- `tr(dv/dx)` is the divergence of the velocity field (trace of its Jacobian)

## Usage

```python
from glp.denoiser import load_glp

# Load model
model = load_glp("path/to/weights", device="cuda:0")

# latents: (batch, seq, dim) — raw (unnormalized) activations
result = model.log_prob(
    latents,
    num_steps=100,              # Euler ODE steps (higher = more accurate)
    num_hutchinson_samples=1,   # random vectors for trace estimation
    layer_idx=None,             # for multi-layer models
    normalize=True,             # set False if latents are already normalized
)

log_p = result.log_prob         # (batch,) log-probability in data space
z = result.z                    # (batch, seq, dim) endpoint in base distribution
```

You can also call the function directly:

```python
from glp.flow_matching import log_prob

result = log_prob(model, latents, num_steps=100)
```

### Return values

| Field | Shape | Description |
|-------|-------|-------------|
| `log_prob` | `(batch,)` | Log-probability in data space |
| `z` | `(batch, seq, dim)` | Endpoint in base distribution space |
| `log_p_base` | `(batch,)` | `log N(0,I)` evaluated at `z` |
| `log_det_flow` | `(batch,)` | Accumulated log-det from the ODE |
| `log_det_normalize` | scalar | Log-det of the normalization transform |

The total is: `log_prob = log_p_base + log_det_flow + log_det_normalize`.

## Parameters

### `num_steps`

Number of Euler discretization steps for the ODE. More steps = better accuracy but slower.

| Steps | Use case |
|-------|----------|
| 20 | Quick sanity check |
| 100 | Reasonable accuracy for most uses |
| 500+ | High accuracy (e.g., for reported numbers) |

Run with increasing step counts to check convergence for your use case.

### `num_hutchinson_samples`

The divergence `tr(dv/dx)` is estimated stochastically using the Hutchinson trace estimator:

```
tr(J) ~ eps^T J eps,    eps ~ N(0, I)
```

Each sample requires one backward pass through the denoiser. More samples reduce variance but increase cost linearly. In practice, `1` is often sufficient; use `5-10` if you need lower-variance estimates (e.g., for comparing close samples).

### `normalize`

- `True` (default): Pass raw activations; the function normalizes them internally and includes the log-determinant correction from the normalization transform `z = (x - mu) / sqrt(var)`.
- `False`: Pass pre-normalized activations; no correction is applied.

## How it works

1. **Normalize** the input activations and compute the normalization log-det: `log |det(dz/dx)| = -0.5 * sum(log(var))`
2. **Solve the forward ODE** from `sigma=0` (data) to `sigma=1` (noise) using Euler integration:
   - At each step, compute the velocity `v = denoiser(x, t)`
   - Estimate the divergence via Hutchinson: draw `eps ~ N(0,I)`, compute `eps^T (dv/dx) eps` using `torch.autograd.grad`
   - Update: `x <- x + dt * v`, `log_det -= dt * div(v)`
3. **Evaluate the base distribution** `log N(0, I)` at the endpoint `z`
4. **Combine**: `log p(x) = log N(z) + log_det_flow + log_det_normalize`

## Bits per dimension

To convert to bits per dimension (a standard metric):

```python
bpd = -result.log_prob / (d * s * math.log(2))
```

where `d` is the activation dimension and `s` is the sequence length.

## Verification checklist

- **Roundtrip**: Sample `z ~ N(0,I)`, reverse-ODE to get `x`, then `log_prob(x)` — the reconstructed `z` should match the original.
- **Ordering**: Model-generated samples should score higher than random noise.
- **Convergence**: `log_prob` values should stabilize as `num_steps` increases.
- **Hutchinson variance**: Multiple runs with `num_hutchinson_samples=1` should have low variance; increasing samples should reduce it further.
