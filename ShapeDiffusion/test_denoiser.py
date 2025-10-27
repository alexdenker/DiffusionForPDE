

import torch
import numpy as np 

import matplotlib.pyplot as plt
from tqdm import tqdm 

from dataset import MNISTShapesDataset
from noise_sampler import ShapeNoise
from simple_neural_operator import fourier_to_shape, shape_to_fourier, complex_to_real, real_to_complex, ScoreNet
from sde import OU

device = "cuda"
batch_size = 8

num_landmarks = 64

dataset = MNISTShapesDataset(class_label=4, num_landmarks=num_landmarks)

noise_sampler = ShapeNoise(num_landmarks=num_landmarks, alpha=3.0, device="cpu")


num_modes = 32
model = ScoreNet(input_dim=num_modes*4, hidden_dim=512, time_embed_dim=32, num_blocks=8, max_period=2.0)
model.load_state_dict(torch.load("model.pt"))
model.to(device)
model.train()

sde = OU(beta_min=0.001, beta_max=15.0)

num_epochs = 10000

data_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=6)

def pts_to_fourier(pts, num_modes):
    """
    pts: [batch_size, num_landmarks, 2]

    return [batch_size, num_modes*4] (pos + neg, real + imag)
    """
    x_trunc = shape_to_fourier(pts, num_modes)
    x_real = complex_to_real(x_trunc)
    x0 = x_real.view(pts.shape[0], -1)
    
    return x0

def fourier_to_pts(z, num_landmarks, num_modes):
    z = real_to_complex(z, num_modes)
    pts = fourier_to_shape(z, num_landmarks, num_modes)

    return pts 

def model_fun(xt, t):
    pred = model(xt, t) 
    cov_t = sde.cov_t_scaling(t, xt)

    return pred / cov_t



model.eval()

x = next(iter(data_loader)).to(device)

    
        
random_t = torch.rand((x.shape[0],), device=x.device) * (1-0.001) + 0.001
z = noise_sampler.sample(x.shape[0]).to(device)

mean_t = sde.mean_t(random_t, x)
mean_t_scale = sde.mean_t_scaling(random_t, x)

cov_t = sde.cov_t_scaling(random_t, x)

xt = mean_t + cov_t * z 

with torch.no_grad():
    xt_inp = pts_to_fourier(xt, num_modes=num_modes)
    pred = model_fun(xt_inp, random_t) 
    pred = fourier_to_pts(pred, num_landmarks=num_landmarks, num_modes=num_modes)
    res = pred + z / cov_t 

    denoised = (xt + cov_t**2 * pred)/mean_t_scale


fig, axes = plt.subplots(2, x.shape[0]//2, figsize=(16,8))

for idx, ax in enumerate(axes.ravel()):
    ax.plot(x[idx,:,0].cpu().numpy(), x[idx,:,1].cpu().numpy(), '-o', label="clean")
    ax.plot(xt[idx,:,0].cpu().numpy(), xt[idx,:,1].cpu().numpy(), '-o', label="noisy")
    ax.plot(denoised[idx,:,0].cpu().numpy(), denoised[idx,:,1].cpu().numpy(), '-o', label="denoised")
    ax.set_title(f"t={random_t[idx].item():.4f}")

ax.legend()

plt.show()