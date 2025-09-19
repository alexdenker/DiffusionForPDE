


import yaml 
import os 

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from matplotlib.tri import Triangulation

import numpy as np 

from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx.fem import (Constant, Function, functionspace, assemble_scalar, form)

import matplotlib.pyplot as plt 

from tqdm import tqdm 
from neural_operator.fourier_neural_operator_dse import FNO_dse
from utils import gen_conductivity
from diffusion import Diffusion

configs = {
    "model": "fno_dse",
    "mesh_name": "disk_dense",
    "lr": 1e-3, 
    "save_dir": "exp/fno_dse",
    "model": {
    "modes": 14, 
    "width": 48 }
}


mesh_name = configs["mesh_name"]

omega, _, _ = gmshio.read_from_msh(f"{mesh_name}.msh", MPI.COMM_WORLD, gdim=2)
omega.topology.create_connectivity(1, 2)

V = functionspace(omega, ("DG", 0))
mesh_pos = np.array(V.tabulate_dof_coordinates()[:,:2])

model = FNO_dse(modes=configs["model"]["modes"], width=configs["model"]["width"])
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
model.load_state_dict(torch.load("exp/fno_dse/circle/lno_model.pt"))
model.to("cuda")
model.eval()

xy = omega.geometry.x
cells = omega.geometry.dofmap.reshape((-1, omega.topology.dim + 1))

tri = Triangulation(xy[:, 0], xy[:, 1], cells)

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

fig, axes = plt.subplots(2,3, figsize=(14,7))

for idx, ax in enumerate(axes.ravel()):

    im = ax.tripcolor(tri, create_sample(mesh_pos)[0].cpu().numpy().flatten(), cmap='jet', shading='flat', vmin=0.01, vmax=4.0,edgecolors='k')
    ax.axis('image')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Phantom")
    ax.axis("off")
    fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)



plt.savefig("true_samples.png")

min_loss = 1e6


batch_size = 6

pos = torch.from_numpy(mesh_pos).float().to("cuda").unsqueeze(0)
pos = torch.repeat_interleave(pos, repeats=batch_size, dim=0)

print("pos: ", pos.shape)

model.eval()

diffusion = Diffusion()

ts = torch.arange(0, diffusion.num_diffusion_timesteps).to("cuda")[::10]
x = torch.randn((batch_size, pos.shape[1], 1)).to("cuda")

print("x: ", x.shape)

n = x.size(0)
ss = [-1] + list(ts[:-1])
xt_s = [x.cpu()]
x0_s = []

xt = x
eta = 1.0
for ti, si in tqdm(zip(reversed(ts), reversed(ss)), total=len(ts)):
    t = torch.ones(n).to(x.device).long() * ti
    s = torch.ones(n).to(x.device).long() * si

    alpha_t = diffusion.alpha(t).view(-1, 1, 1)
    alpha_s = diffusion.alpha(s).view(-1, 1, 1)
    c1 = (
        (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)
    ).sqrt() * eta
    c2 = ((1 - alpha_s) - c1**2).sqrt()
    with torch.no_grad():
        inp = torch.cat([pos, xt], dim=-1)
        
        et = model(inp, t)

    x0_pred = (xt - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
    x0_pred = torch.clamp(x0_pred, 0, 5)
    xs = alpha_s.sqrt() * x0_pred + c1 * torch.randn_like(xt) + c2 * et
    # xt_s.append(xs.cpu())
    # x0_s.append(x0_pred.cpu())
    xt = xs

    """  
    fig, axes = plt.subplots(2,3, figsize=(14,7))

    for idx, ax in enumerate(axes.ravel()):

        im = ax.tripcolor(tri, xt[idx].cpu().numpy().flatten(), cmap='jet', shading='flat', edgecolors='k')
        ax.axis('image')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Phantom")
        ax.axis("off")
        fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)



    plt.savefig(f"imgs/{ti}.png")
    plt.close()
    """


print(xt.shape)


fig, axes = plt.subplots(2,3, figsize=(14,7))

for idx, ax in enumerate(axes.ravel()):
    if idx < xt.shape[0]:
        im = ax.tripcolor(tri, xt[idx].cpu().numpy().flatten(), cmap='jet', shading='flat', vmin=0.01, vmax=4.0, edgecolors='k')
        ax.axis('image')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Sample")
        ax.axis("off")
        fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
    else:
        ax.axis("off")


plt.savefig("generated_samples.png")




