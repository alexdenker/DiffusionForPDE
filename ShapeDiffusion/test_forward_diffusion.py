import torch 
import numpy as np
import matplotlib.pyplot as plt


from dataset import MNISTShapesDataset
from noise_sampler import ShapeNoise
from sde import OU

device = 'cpu'


dataset = MNISTShapesDataset() 
pts = dataset[0].unsqueeze(0)

print(pts.shape)

sde = OU(beta_min=0.001, beta_max=15.0)
alpha = 1.5
noise_sampler = ShapeNoise(num_landmarks=pts.shape[1], alpha=alpha, device='cpu')

T = 1.0
num_time_points = 10
ts = torch.linspace(0, T, num_time_points)

fig, axes = plt.subplots(2,num_time_points // 2, figsize=(14,8))
ax = axes.ravel()

for idx, t in enumerate(ts):
    noise = noise_sampler.sample(1)
    t = t.to(device)

    mean_t = sde.mean_t(t, pts)
    cov_t = sde.cov_t_scaling(t, pts)

    xt = mean_t + cov_t * noise 

    im = ax[idx].plot(np.append(xt[0,:,0].cpu().numpy(), xt[0,0,0].cpu().numpy()),
                        np.append(xt[0,:,1].cpu().numpy(), xt[0,0,1].cpu().numpy()), '-o', markersize=3)
    ax[idx].set_title(f"t = {t.item():.4f}")
    

fig.suptitle(f"Forward Diffusion with shape noise (alpha={alpha})", fontsize=16)
plt.show()