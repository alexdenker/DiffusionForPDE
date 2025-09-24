
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation


import yaml 
import os 

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 

import numpy as np 

from neural_operator.fourier_neural_operator import FNO
from utils import gen_conductivity
from diffusion import Diffusion

configs = {
    "num_epochs": 1200,
    "lr": 1e-4, 
    "save_dir": "exp/fno",
    "model": {
    "modes": 22, 
    "width": 32 }
}


device = "cuda"


log_dir = os.path.join(configs["save_dir"])
print("save model to ", log_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

model = FNO(modes=configs["model"]["modes"], width=configs["model"]["width"])
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

model.to(device)
model.train()

B, C, H, W = 4, 1, 128, 128

x = np.linspace(-1, 1, H)
y = np.linspace(-1, 1, W)
X, Y = np.meshgrid(x, y)
xy = np.vstack((X.flatten(), Y.flatten())).T


with open(os.path.join(log_dir, "config.yaml"), "w") as file:
    yaml.dump(configs, file)

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
min_loss = 1e6

num_epochs = configs["num_epochs"]
optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])

num_iters = 20000
save_every = 100
print_every = 10
plot_every = 50
batch_size = B

running_loss = 0

model.train()

diffusion = Diffusion(beta_start=1e-5, beta_end=1e-2)
spectral_filter = diffusion.make_spectral_filter(H, W, power=2.0, cutoff=120, device=device)

pos = np.stack([X, Y], axis=0)

pos = torch.from_numpy(pos).float().to("cuda").unsqueeze(0)
print("pos: ", pos.shape)

pos = torch.repeat_interleave(pos, repeats=batch_size, dim=0)
print("pos: ", pos.shape)
for step in range(num_iters):
    optimizer.zero_grad() 
    
    mean_loss = []
    
    sigma = draw_batch(batch_size,xy)
    sigma = sigma.to("cuda")
    
    random_t = torch.randint(1, diffusion.num_diffusion_timesteps, (sigma.shape[0],), device=sigma.device)


    x_t, noise = diffusion.q_sample(sigma, random_t, spectral_filter=spectral_filter)

    inp = torch.cat([pos, x_t], dim=1)

    pred = model(inp, random_t)
    loss = torch.sum((pred - noise)**2)/pred.shape[0] # loss function w.r.t basis elements # 
    loss.backward() 

    #print("Loss: ", loss.item())
    running_loss = (1-0.99)*loss.item() + 0.99 * running_loss
    optimizer.step() 
    
    if step % plot_every == 0:
        pred_fft = torch.fft.fft2(pred, dim=(-2,-1))
        filtered_pred_fft = pred_fft * spectral_filter  # shape (B,1,H,W)
        filtered_pred = torch.fft.ifft2(filtered_pred_fft, dim=(-2,-1)).real

        alpha_t = diffusion.alpha(random_t).view(-1, 1,1, 1)
        x0_pred = (x_t - filtered_pred * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14,7))

        im = ax1.imshow(sigma[0,0].detach().cpu().numpy(), cmap='jet', interpolation="nearest")
        ax1.axis('image')
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title("Phantom")
        ax1.axis("off")
        fig.colorbar(im, ax=ax1,fraction=0.046, pad=0.04)

        im = ax2.imshow(x_t[0,0].detach().cpu().numpy(), cmap='jet', interpolation="nearest")
        ax2.axis('image')
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_title(f"Noisy phantom at t={random_t[0].item()}")
        ax2.axis("off")
        fig.colorbar(im, ax=ax2,fraction=0.046, pad=0.04)

        im = ax3.imshow(x0_pred[0,0].detach().cpu().numpy(), cmap='jet', interpolation="nearest")
        ax3.axis('image')
        ax3.set_aspect('equal', adjustable='box')
        ax3.set_title("Denoised pred")
        ax3.axis("off")
        fig.colorbar(im, ax=ax3,fraction=0.046, pad=0.04)

        plt.savefig(f"train_imgs/{step}.png")
        plt.close()

    if step % save_every == 0:
        torch.save(model.state_dict(), os.path.join(log_dir,"fno_model.pt"))
    
    if step % print_every == 0:
        print(f"Running loss {running_loss:.4f} at step {step}. Current loss {loss.item():.4f}")