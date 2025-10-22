"""
We are looking to solve a standard Poisson inverse problem
    - nabla( nabla u) = a(x)  in Omega
                   u  = 1  on Boundary

Calerdon 
- nabla(a(x) nabla u) = 0  in Omega
                   u  = 1  on Boundary


We assume that we have some internal measurements of the solution u. 

FunDPS for Poisson 

Discretise the PDE 
    M u = b(a)
    u = M^{-1} b(a)

d/da || y - M^{-1} b(a) ||

d/da M^{-1} f(a) = M^{-1} d/da f(a)

"""


import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation

import os 
from omegaconf import OmegaConf

from physics.utils import gen_conductivity
from neural_operator.nfft_neural_operator import NUFNO
from neural_operator.sde import OU
from neural_operator.noise_sampler import RBFKernel
from physics.poisson import PoissonOperator
from neural_operator.score_model import ScoreModel
from samplers.dps import DPS

import torch 
from tqdm import tqdm 

device = "cuda"
###
load_path = "exp/NonUniformFNO/20251020_101410"
###

cfg = OmegaConf.load(os.path.join(load_path, "config.yaml"))


mesh_name = cfg.model.mesh_name

device = "cuda"
mesh_name = "disk_dense" #"disk_256" #"disk_dense"

poisson = PoissonOperator(mesh_path=f"data/disk/{mesh_name}.msh",
                          device=device)

xy = poisson.omega.geometry.x
cells = poisson.omega.geometry.dofmap.reshape((-1, poisson.omega.topology.dim + 1))
tri = Triangulation(xy[:, 0], xy[:, 1], cells)
mesh_pos = np.array(poisson.W.tabulate_dof_coordinates()[:,:2])

mask = np.random.choice(poisson.dofs_pl, int(0.8*poisson.dofs_pl), replace=False)

B = torch.eye(poisson.dofs_pl,device=device)[mask]
poisson.set_observation_operator(B)

# Create ground truth parameter (conductivity)
np.random.seed(123)
a = sigma_mesh = gen_conductivity(
        mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=0.0
    )
a = torch.from_numpy(a).float().to(device).unsqueeze(-1)

sol = poisson.solve_linear_system(a)
y = torch.matmul(B, sol).T
y = y + 0.00 * torch.randn_like(y)

model = NUFNO(n_layers=cfg.model.n_layers,  
            modes=cfg.model.modes,
            width=cfg.model.width,
            in_channels=3,
            timestep_embedding_dim=cfg.model.timestep_embedding_dim,
            max_period=cfg.model.max_period)


noise_sampler = RBFKernel(torch.from_numpy(mesh_pos).float().to(device), scale=cfg.noise.scale, eps=cfg.noise.eps, device=device)

sde = OU(beta_min=cfg.sde.beta_min, beta_max=cfg.sde.beta_max)
score_model = ScoreModel(model, sde, noise_sampler, cfg)    

print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
score_model.model.load_state_dict(torch.load(os.path.join(load_path, "fno_model.pt")))
score_model.model.to(device)
score_model.model.eval()

batch_size = 1

pos = torch.from_numpy(mesh_pos).float().to(device).unsqueeze(0)
pos = torch.repeat_interleave(pos, repeats=batch_size, dim=0)
pos[:,:,0] = (pos[:,:,0] - torch.min(pos[:,:,0]))/(torch.max(pos[:,:,0]) - torch.min(pos[:,:,0])) - 0.5
pos[:,:,1] = (pos[:,:,1] - torch.min(pos[:,:,1]))/(torch.max(pos[:,:,1]) - torch.min(pos[:,:,1])) - 0.5

pos_inp = torch.repeat_interleave(pos, repeats=batch_size, dim=0)

num_timesteps = 500
ts = torch.linspace(1e-3, 1, num_timesteps).to("cuda")

cfg = OmegaConf.create({"gamma": 3.0})
dps_sampler = DPS(score_model, poisson, cfg)

print(y.shape, pos_inp.shape, ts.shape)

xt = dps_sampler.sample(y, pos_inp, ts)


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))


im = ax1.tripcolor(tri, xt[0].cpu().numpy().flatten(), shading='flat', edgecolors='k')
ax1.axis('image')
ax1.set_aspect('equal', adjustable='box')
ax1.set_title("Reconstruction")
ax1.axis("off")
fig.colorbar(im, ax=ax1)
im = ax2.tripcolor(tri, a.cpu().numpy().flatten(), shading='flat',  edgecolors='k')
ax2.axis('image')
ax2.set_aspect('equal', adjustable='box')
ax2.set_title("Ground truth")
ax2.axis("off")

fig.colorbar(im, ax=ax2)
plt.savefig("reconstruction.png")
plt.show()



