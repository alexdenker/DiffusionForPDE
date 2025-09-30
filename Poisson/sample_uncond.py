


import yaml 
import os 

import torch 
from matplotlib.tri import Triangulation

import numpy as np 

use_dolfin = True 
try: 
    from dolfinx.io import gmshio
    from mpi4py import MPI
    from dolfinx.fem import functionspace
except ImportError as e:
    print("dolfin not installed, fall back to saved numpy files")
    use_dolfin = False 


import matplotlib.pyplot as plt 

from tqdm import tqdm 
from neural_operator.nfft_neural_operator import NUFNO
from utils import gen_conductivity
from diffusion import Diffusion

configs = {
    "mesh_name": "disk_dense",
    "lr": 1e-3, 
    "save_dir": "exp/fno_dse",
    "model": {
    "modes": 14, 
    "width": 32 }
}


mesh_name = configs["mesh_name"]
if use_dolfin:
    omega, _, _ = gmshio.read_from_msh(f"{mesh_name}.msh", MPI.COMM_WORLD, gdim=2)
    omega.topology.create_connectivity(1, 2)

    V = functionspace(omega, ("DG", 0))
    mesh_pos = np.array(V.tabulate_dof_coordinates()[:,:2])

    xy = omega.geometry.x
    cells = omega.geometry.dofmap.reshape((-1, omega.topology.dim + 1))
else:
    xy = np.load(f"data/disk/{mesh_name}_xy.npy")
    cells = np.load(f"data/disk/{mesh_name}_cells.npy")
    mesh_pos = np.load(f"data/disk/{mesh_name}_mesh_pos.npy")

model = NUFNO(n_layers=4, 
              modes=configs["model"]["modes"], 
              width=configs["model"]["width"],
              in_channels=3,
              timestep_embedding_dim=33)
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
model.load_state_dict(torch.load("exp/fno_dse/circle/lno_model.pt"))
model.to("cuda")
model.eval()

tri = Triangulation(xy[:, 0], xy[:, 1], cells)


batch_size = 6

pos = torch.from_numpy(mesh_pos).float().to("cuda").unsqueeze(0)
pos = torch.repeat_interleave(pos, repeats=batch_size, dim=0)

pos[:,:,0] = (pos[:,:,0] - torch.min(pos[:,:,0]))/(torch.max(pos[:,:,0]) - torch.min(pos[:,:,0])) - 0.5
pos[:,:,1] = (pos[:,:,1] - torch.min(pos[:,:,1]))/(torch.max(pos[:,:,1]) - torch.min(pos[:,:,1])) - 0.5


print("pos: ", pos.shape)

model.eval()

diffusion = Diffusion(beta_start=1e-4, beta_end=6e-3)

ts = torch.arange(0, diffusion.num_diffusion_timesteps).to("cuda")[::5]
x = torch.randn((batch_size, 1, pos.shape[1])).to("cuda")

print("x: ", x.shape)

def model_fn(x, t, pos):
    inp = torch.cat([pos.permute(0, 2, 1), x], dim=1)

    # output of models as (xt - sqrt(alpha_t) * model) / sqrt(1-alpha_t)
    sqrt_ab = diffusion.alpha(t).sqrt().view(-1, 1, 1)
    sqrt_omb = (1 - diffusion.alpha(t)).sqrt().view(-1, 1, 1)

    pred = model(inp, t, pos.unsqueeze(1))
    return (x - sqrt_ab * pred) / sqrt_omb

n = x.size(0)
ss = [-1] + list(ts[:-1])

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
        
        et = model_fn(xt, t, pos)

    x0_pred = (xt - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
    xt = alpha_s.sqrt() * x0_pred + c1 * torch.randn_like(xt) + c2 * et

print(xt.shape)


fig, axes = plt.subplots(2,3, figsize=(14,7))

for idx, ax in enumerate(axes.ravel()):
    if idx < xt.shape[0]:
        im = ax.tripcolor(tri, xt[idx].cpu().numpy().flatten(), cmap='jet', shading='flat', edgecolors='k')
        ax.axis('image')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Sample")
        ax.axis("off")
        fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
    else:
        ax.axis("off")


plt.savefig("generated_samples.png")
plt.show()



