
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

from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx.fem import (Constant, Function, functionspace, assemble_scalar, form)

import wandb 

from neural_operator.fourier_neural_operator_dse import FNO_dse, VFT
from utils import gen_conductivity
from diffusion import Diffusion

configs = {
    "model": "fno_dse",
    "mesh_name": "disk_dense",
    "resume_from": None, 
    "num_epochs": 1200,
    "lr": 2e-3, 
    "save_dir": "exp/fno_dse",
    "model": {
    "modes": 14, 
    "width": 48 }
}


mesh_name = configs["mesh_name"]

omega, _, _ = gmshio.read_from_msh(f"{mesh_name}.msh", MPI.COMM_WORLD, gdim=2)
omega.topology.create_connectivity(1, 2)

xy = omega.geometry.x
cells = omega.geometry.dofmap.reshape((-1, omega.topology.dim + 1))

tri = Triangulation(xy[:, 0], xy[:, 1], cells)


V = functionspace(omega, ("DG", 0)) # piecewise constant
mesh_pos = np.array(V.tabulate_dof_coordinates()[:,:2])

log_dir = os.path.join(configs["save_dir"], "circle")
print("save model to ", log_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

model = FNO_dse(modes=configs["model"]["modes"], width=configs["model"]["width"])
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

model.to("cuda")
model.train()

with open(os.path.join(log_dir, "config.yaml"), "w") as file:
    yaml.dump(configs, file)

def create_sample(mesh_pos):
    sigma_mesh = gen_conductivity(
        mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=1.0
    )
    return torch.from_numpy(sigma_mesh).float().unsqueeze(0)

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
batch_size = 32

running_loss = 0

pos = torch.from_numpy(mesh_pos).float().to("cuda").unsqueeze(0)
pos = torch.repeat_interleave(pos, repeats=batch_size, dim=0)

print("pos: ", pos.shape)

model.train()

diffusion = Diffusion()

x1 = pos[:,:,0] - torch.min(pos[:,:,0])
x2 = pos[:,:,1] - torch.min(pos[:,:,1])

nu_fft = VFT(x2, x2, modes=configs["model"]["modes"])

def forward(x):
    x = x + 0j
    x_ft = nu_fft.forward(x) #[4, 20, 32, 16] 
    x_ft = x_ft.permute(0, 2, 1)
    x_ft = torch.reshape(x_ft, (batch_size, 1, 2*configs["model"]["modes"], 2*configs["model"]["modes"]-1))
    return torch.view_as_real(x_ft)

def inverse(x):
    x = torch.view_as_complex(x)[:,0,:,:]
    print("x comples: ", x.shape)
    x_ft = x.reshape(x.shape[0], -1, 1) # x.permute(0, 2, 1)
    print("x ft shape: ", x_ft.shape)
    x = nu_fft.inverse(x_ft).real # x [4, 20, 512, 512]
    print("x after nu_fft inverse ", x.shape)
    #x = x.permute(0, 2, 1)
    return x 


#import torchkbnufft as tkbn

#nufft_adj = tkbn.KbNufftAdjoint(im_size=(14,14)).to("cuda")

for step in range(num_iters):
    optimizer.zero_grad() 
    
    mean_loss = []
    
    sigma = draw_batch(batch_size,mesh_pos).unsqueeze(-1)
    sigma = sigma.to("cuda")
    print("sigma: ", sigma.shape)
    

    # Now take a regular FFT (uniform frequency grid)
    #sigma_fft = torch.fft.fftshift(torch.fft.fft2(img_grid), dim=(-2, -1))
        
    sigma_fft = forward(sigma)
    print("sigma: ", sigma.shape)
    print("sigma_fft: ", sigma_fft.shape)

    random_t = torch.randint(1, diffusion.num_diffusion_timesteps, (sigma.shape[0],), device=sigma.device)
    z = torch.randn_like(sigma_fft)
    #print("random_t.shape: ", random_t.shape)
    alpha_t = diffusion.alpha(random_t).view(-1, 1,1, 1, 1)
    print(alpha_t.shape, sigma_fft.shape)
    # add noise to basis elements (choose basis)
    perturbed_sigma_fft = alpha_t.sqrt() * sigma_fft + (1 - alpha_t).sqrt() * z 

    perturbed_sigma = inverse(perturbed_sigma_fft)
    print("Perturbed sigma: ", perturbed_sigma.shape)
    inp = torch.cat([pos, perturbed_sigma], dim=-1)

    pred = model(inp, random_t)
    pred_fft = forward(pred)
    #print("pred: ", pred.shape, " sigma: ", sigma.shape)
    loss = torch.sum((pred_fft - z)**2)/pred.shape[0] # loss function w.r.t basis elements # 
    loss.backward() 

    #print("Loss: ", loss.item())
    running_loss = (1-0.99)*loss.item() + 0.99 * running_loss
    optimizer.step() 
    
    if step % plot_every == 0:
        alpha_t = diffusion.alpha(random_t).view(-1, 1,1, 1, 1)
        x0_pred_fft = (perturbed_sigma_fft - pred_fft * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
        x0_pred = inverse(x0_pred_fft)
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14,7))

        im = ax1.tripcolor(tri, sigma[0].cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
        ax1.axis('image')
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title("Phantom")
        ax1.axis("off")
        fig.colorbar(im, ax=ax1,fraction=0.046, pad=0.04)

        im = ax2.tripcolor(tri, perturbed_sigma[0].cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
        ax2.axis('image')
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_title(f"Noisy phantom at t={random_t[0].item()}")
        ax2.axis("off")
        fig.colorbar(im, ax=ax2,fraction=0.046, pad=0.04)

        im = ax3.tripcolor(tri, x0_pred[0].detach().cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
        ax3.axis('image')
        ax3.set_aspect('equal', adjustable='box')
        ax3.set_title("Denoised pred")
        ax3.axis("off")
        fig.colorbar(im, ax=ax3,fraction=0.046, pad=0.04)

        plt.savefig(f"train_imgs/{step}.png")
        plt.close()

    if step % save_every == 0:
        torch.save(model.state_dict(), os.path.join(log_dir,"lno_model.pt"))
    
    if step % print_every == 0:
        print(f"Running loss {running_loss:.4f} at step {step}. Current loss {loss.item():.4f}")