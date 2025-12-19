# Shape Diffusion: Score-Based Generative Modeling for 2D Shapes

This project implements score-based diffusion models for generating and manipulating 2D shapes. We use **Elliptic Fourier Descriptors (EFD)** to represent shapes in the frequency domain and train diffusion models to learn the score function (gradient of the log-probability) over Fourier coefficients.

We use the contours of digits from the MNIST dataset for our example.

## Overview

## Mathematical Framework

### Elliptic Fourier Descriptors

A closed 2D shape with $N$ landmarks is represented as $x(t) \in \R^{N \times 2}$. The EFD representation computes the Discrete Fourier Transform (DFT) of the landmark sequence:
```
X_k = ℱ[x] = Σ_{n=0}^{N-1} x_n e^{-2πikn/N}
```

where:
- `k = 0, 1, ..., ⌊N/2⌋` (frequency indices)
- Each coefficient is complex: `X_k = Re(X_k) + i·Im(X_k)`

We store coefficients as: **[Re, Im]** pairs for each frequency mode.

### Frequency-Dependent Noise Scaling

We apply a **frequency-dependent noise scaling**:

```
z_k ~ N(0, 1/(k+1)^2)  for frequency k
```


## Files Description

| File | Purpose |
|------|---------|
| `simple_network.py` | ScoreNet architecture with time conditioning |
| `sde.py` | OU-SDE implementation (forward/reverse processes) |
| `utils.py` | Fourier transforms, noise scaling, helper functions |
| `dataset.py` | MNIST shape dataset (landmarks extraction) |
| `train_base_model.py` | Unconditional diffusion model training |
| `train_conditional.py` | Conditional diffusion model training |
| `uncond_sample.py` | Unconditional sampling at different landmark resolutions |
| `cond_sample_finitedim.py` | Conditional sampling with guidance |
| `efd_shape.py` | EFD visualization and reconstruction analysis |

## Usage

### 1. Training an Unconditional Model

```bash
python train_base_model.py
```

Trains a score-based model to generate random MNIST digit 3 shapes. Saves model checkpoints to `base_model/model.pt` and sample visualizations to `base_model/imgs/`.

**Configuration** (`config/base_model.yaml`):
- Model hidden dimension: 1024
- Number of Fourier modes: 16
- Number of landmarks: 64
- Number of epochs: 5000
- Learning rate: 1e-4

### 2. Unconditional Sampling

```bash
python uncond_sample.py
```

Generates new random shapes at multiple landmark resolutions (32, 64, 128 points) and saves to `uncond_shape_samples.png`.

### 3. Training a Conditional Model

```bash
python train_conditional.py
```

Trains a model conditioned on the first 5 Fourier coefficients to perform shape completion. Results saved to `conditional_training_results/`.

### 4. Conditional Sampling with DPS

```bash
python dps_sampling.py \
    --mask_ratio 0.4 \
    --lambd 50.0 \
    --num_samples 5 \
    --num_dataset_elements 10
```

**Arguments:**
- `--mask_ratio`: Fraction of landmarks to mask (0.4 = 40%)
- `--lambd`: Guidance strength (higher = more constrained to observations, but also unstable)
- `--num_samples`: Number of samples per measurement
- `--num_dataset_elements`: Number of ground truth shapes to process
- `--num_timesteps`: Number of reverse SDE steps (default 1000)

### 5. Visualize EFD Reconstruction

```bash
python efd_shape.py
```

Shows reconstruction quality at different numbers of Fourier bases.
