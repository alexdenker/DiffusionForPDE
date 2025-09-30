# Poisson Inverse Problem using Infinite Dimensional Diffusion

This implementation used the non-uniform FFT from *https://github.com/johertrich/simple_torch_NFFT*

It requires only PyTorch (>= 2.5 recommended) and NumPy and can be installed with
```
pip install git+https://github.com/johertrich/simple_torch_NFFT
```

We want to solve the inverse problem governed the PDE

$$ - \Delta u = a(x) \quad \text{in } \Omega $$
$$ u = 0 \quad \text{on } \delta \Omega $$

We assume that we measure $u$ at some points inside the domain $\Omega$ and want to recover the parameter $a: \Omega \to \R$.


## Files 

- `create_mesh.py`:  This script creates the mesh and saves it in `data/` 
- `diffusion.py`: This is the forward diffusion process (based on DDPM). For $t \in \{0,...,100\}$ we have $x_t = \sqrt{\alpha_t} x_0 + \sqrt{(1-\alpha_t)} z$.
- `poisson_inverse.py`: Implementation of FunDPS for the Poisson inverse problem 
- `train_fno_pixelwisenoise.py`: Training of the (time-dependent) neural operator using the diffusion process from `diffusion.py` using i.i.d. Gaussian noise for every mesh position.  
