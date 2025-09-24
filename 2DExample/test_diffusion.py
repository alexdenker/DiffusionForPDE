

import matplotlib.pyplot as plt 

import torch 
import numpy as np 

from utils import gen_conductivity
from diffusion import Diffusion

device = "cuda"

# create image grid 
B, C, H, W = 4, 1, 256, 256

x = np.linspace(-1, 1, 256)
y = np.linspace(-1, 1, 256)
X, Y = np.meshgrid(x, y)
xy = np.vstack((X.flatten(), Y.flatten())).T

def create_sample(mesh_pos):
    sigma_mesh = gen_conductivity(
        mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=1.0
    )
    return torch.from_numpy(sigma_mesh).float().reshape(1, 1, H, W)

def draw_batch(batch_size, mesh_pos):
    x = [] 
    for _ in range(batch_size):
        x.append(create_sample(mesh_pos))

    return torch.cat(x, dim=0)


pos = torch.from_numpy(xy).float().to("cuda")
pos = torch.repeat_interleave(pos, repeats=B, dim=0)

diffusion = Diffusion(beta_start=1e-5, beta_end=1e-2)
spectral_filter = diffusion.make_spectral_filter(H, W, power=2.0, cutoff=120, device=device)

print("spectral filter: ", spectral_filter.shape)

print(spectral_filter)
spectral_filter_plot = torch.fft.fftshift(spectral_filter, dim=(-2, -1))  # shift zero freq to center
plt.figure()
plt.imshow(spectral_filter_plot[0,0].cpu().numpy(), cmap='jet', interpolation="nearest")
plt.colorbar()
plt.title("Spectral Filter")
plt.axis('image')
plt.show()

np.random.seed(123)
sigma = draw_batch(B,xy)
sigma = sigma.to("cuda")


z = diffusion.start_sample(sigma.shape, spectral_filter)
fig, axes = plt.subplots(2,2, figsize=(9,9))
for idx, ax in enumerate(axes.flatten()):
    im = ax.imshow(z[idx,0].cpu().numpy(), cmap='jet', interpolation="nearest")
    ax.axis('image')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Sample {idx} from N(0,C)")
    ax.axis("off")
    fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)

plt.show()



print("sigma: ", sigma.shape)
print("pos: ", pos.shape)


for t in [0, 10, 50, 100, 200, 500, 800, 999]:
    t_idx = torch.ones(sigma.shape[0], device=sigma.device).long() * t
    
    print(t_idx)
    x_t, noise = diffusion.q_sample(sigma, t_idx, spectral_filter=spectral_filter)


    fig, axes = plt.subplots(2,2, figsize=(9,9))
    for idx, ax in enumerate(axes.flatten()):
        im = ax.imshow(x_t[idx,0].cpu().numpy(), cmap='jet', interpolation="nearest")
        ax.axis('image')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Sample {idx} at t={t}")
        ax.axis("off")
        fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)

    plt.show()



fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14,7))

