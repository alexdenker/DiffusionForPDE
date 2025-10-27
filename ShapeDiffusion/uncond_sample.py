

import torch
import numpy as np 

import matplotlib.pyplot as plt
from tqdm import tqdm 

from dataset import MNISTShapesDataset
from noise_sampler import ShapeNoise
from simple_neural_operator import fourier_to_shape, shape_to_fourier, complex_to_real, real_to_complex, ScoreNet
from sde import OU

device = "cuda"

num_landmarks = 64
dataset = MNISTShapesDataset(class_label=4, num_landmarks=num_landmarks)

noise_sampler = ShapeNoise(num_landmarks=num_landmarks, alpha=3.0, device="cpu")


num_modes = 32
model = ScoreNet(input_dim=num_modes*4, hidden_dim=512, time_embed_dim=32, num_blocks=8, max_period=2.0)
model.load_state_dict(torch.load("model.pt"))
model.to(device)
model.train()


sde = OU(beta_min=0.001, beta_max=15.0)

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

# sample 
batch_size_smpl = 8
num_timesteps = 1000
ts = torch.linspace(1e-3, 1, num_timesteps).to(device)

delta_t = ts[1] - ts[0]
x_init = noise_sampler.sample(batch_size_smpl).to(device) # N(0,C)

xt = x_init.clone()

for ti in tqdm(reversed(ts), total=len(ts)):
    print(ti)
    t = torch.ones((batch_size_smpl,)).to(xt.device)* ti

    with torch.no_grad():
        xt_inp = pts_to_fourier(xt, num_modes=num_modes)
        score = model(xt_inp, t)
        score = fourier_to_pts(score, num_landmarks=num_landmarks, num_modes=num_modes)

    beta_t = sde.beta_t(t).view(-1, 1, 1)
    noise = noise_sampler.sample(batch_size_smpl).to(device) # N(0,C)
    
    xt = xt + beta_t/2.0 * delta_t*xt + beta_t * delta_t * score + beta_t.sqrt()*delta_t.sqrt() * noise 

print("samples: ", xt.shape)
fig, axes = plt.subplots(2,4, figsize=(12,5))

for idx, ax in enumerate(axes.ravel()):
    ax.plot(xt[idx,:,0].cpu().numpy(), xt[idx,:,1].cpu().numpy(), '-o', label="sample")
    ax.plot(x_init[idx,:,0].cpu().numpy(), x_init[idx,:,1].cpu().numpy(), '-o', label="init noise")

ax.legend()
plt.show()