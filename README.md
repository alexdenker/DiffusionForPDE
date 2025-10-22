# Learning Guidance for Diffusions in Function Spaces 

For finite-dimensional diffusion processes we are able to estimate the (generalised) h-transform for conditional sampling from diffusion models, see [DEFT](https://arxiv.org/abs/2406.01781). We want to extend this to infinite-dimensional diffusion models. This is based on the extensions of the [h-transform to infinite-dimensional diffusion processes](https://arxiv.org/abs/2402.01434). 

We want to apply this framework to several PDE-based problems. 

Structure:
- `NonUniformMesh/`: This folder contains the code for training infinite-dimensional diffusion models on a non-uniform mesh and a first implementation of conditional sampling for Poisson. 

All other folders just have some code blocks.

## Installation 

You will need to install FenicsX (as the backend for solving PDEs):

```python
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

FenicsX is compatible with pytorch. The code is tested with pytorch version 2.3.1 and CUDA 12.1. This can be installed via:

```python
conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```