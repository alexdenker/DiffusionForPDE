# Infinite-dimensional Diffusion for regular 2D-grids 

Here, we give an implementation for data on regular 2D-grids (aka 'images'). We assume that the data we want to model has 
resolution $N_i \times N_i$ for some $N_i \in \{ 32, 64, 128, \dots \}$. 

This setting makes it possible to exploit the FFT in the architecture and the diffusion process.

This is the setting in most of the current works, e.g.
- [Guided Diffusion Sampling on Function Spaces with Applications to PDEs](https://www.arxiv.org/abs/2505.17004)
- [DiffusionPDE: Generative PDE-Solving Under Partial Observation](https://arxiv.org/abs/2406.17763)