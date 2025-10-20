# Infinite Dimensional Diffusion: Toy Example 

Learn a model for functions on a circular mesh. The functions have a zero background and contain a number of ellipses with varying intensity.  

We use the following forward SDE 

$$ d X_t = -1/2 \beta(t) X_t dt+ \sqrt{\beta(t) C} dW_t $$

for $t \in [0,1]$ and $\beta(t) = \beta_{min} + t (\beta_{max} - \beta_{min})$. 

This SDE has the following reverse SDE

$$ d Y_s = 1/2 \beta(1-s) Y_s dt + \beta(1-s) C \nabla \log p_{1-s}(Y_s) dt + \sqrt{\beta(1-s)C} dW_t $$

We can simulate this with euler maruyama 

$$ Y_{t+\Delta t} = Y_t +  1/2 \beta(1-t) Y_t \Delta t + \beta(1-t) C \nabla \log p_{1-t}(Y_t) \Delta t + \sqrt{\beta(1-t) \Delta t} z $$

with $z \sim N(0,C)$.

### Usage

First use the script `create_dataset.py` to create a dataset for training. Then change the `WANDB_PROJECT` and `WANDB_ENTITY` in `configs/wandb_configs.py`. Train the model using `train.py`. Finally, sample using `uncond_sampling.py` (add the path of the saved model to the script).

### Dataset 

We create a dataset of random Gaussians (blobs \ ellipses) on a circular domain.

### Installation

This implementation used the non-uniform FFT from *https://github.com/johertrich/simple_torch_NFFT*

It requires only PyTorch (>= 2.5 recommended) and NumPy and can be installed with
```
pip install git+https://github.com/johertrich/simple_torch_NFFT
```


### TODO: 

- Neural Operator: Add U-Net logic, currently the network works only on the same resolution. However, downsampling data on a non-uniform / unstructured mesh is a bit harder. 
- Noise type: The noise supported is given by radial basis functions $C_{i,j} = \exp(- || x_i - x_j ||_2^2 / (2s^2))$ for different $s$. Different choices? 
- Discretisation: Currently the function are discretised on the mesh. We could also work in a different discretisation? 
- PDE: Currently only the PoissonPDE is implemented, different choices?