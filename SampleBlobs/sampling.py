"""
Train a diffusion models based on the NonUniformFFT Neural Operator with i.i.d. white Gaussian noise in pixel space.
The forward process is 
    x_t = alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * z 

Here, x_0 is the parameter on the given mesh (shape [batch_size, dim, num_mesh_points]). 
This means that we are adding i.i.d. Gaussian noise to every mesh position.

"""

import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation

import yaml 
import os 
from tqdm import tqdm 

import torch 
import numpy as np 

from nfft_neural_operator import NUFNO
from sampler.noise_sampler import RBFKernel
from sde import OU

use_dolfin = True 
try: 
    from dolfinx.io import gmshio
    from mpi4py import MPI
    from dolfinx.fem import functionspace
except ImportError as e:
    print("dolfin not installed, fall back to saved numpy files")
    use_dolfin = False 




log_img_dir = "samples/"
if not os.path.exists(log_img_dir):
    os.makedirs(log_img_dir)

###
load_path = "exp/NonUniformFNO/20251002_074424"
###

with open(os.path.join(load_path,"config.yaml"), "r") as f:
    configs = yaml.safe_load(f)

mesh_name = configs["mesh_name"]

if use_dolfin:
    omega, _, _ = gmshio.read_from_msh(f"{mesh_name}.msh", MPI.COMM_WORLD, gdim=2)
    omega.topology.create_connectivity(1, 2)

    xy = omega.geometry.x

    cells = omega.geometry.dofmap.reshape((-1, omega.topology.dim + 1))

    V = functionspace(omega, ("DG", 0)) # piecewise constant
    mesh_pos = np.array(V.tabulate_dof_coordinates()[:,:2])
else:
    xy = np.load(f"data/disk/{mesh_name}_xy.npy")
    cells = np.load(f"data/disk/{mesh_name}_cells.npy")
    mesh_pos = np.load(f"data/disk/{mesh_name}_mesh_pos.npy")


tri = Triangulation(xy[:, 0], xy[:, 1], cells)



model = NUFNO(n_layers=configs["model"]["n_layers"], 
            modes=configs["model"]["modes"], 
            width=configs["model"]["width"],
            in_channels=3,
            timestep_embedding_dim=33,
            max_period=configs["model"]["max_period"])
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
model.load_state_dict(torch.load(os.path.join(load_path, "fno_model.pt")))
model.to("cuda")

  
device = "cuda"

# the positions need to the scaled to [-0.5, 0.5] for the non-uniform fast fourier transform
pos = torch.from_numpy(mesh_pos).float().to("cuda").unsqueeze(0)
pos[:,:,0] = (pos[:,:,0] - torch.min(pos[:,:,0]))/(torch.max(pos[:,:,0]) - torch.min(pos[:,:,0])) - 0.5
pos[:,:,1] = (pos[:,:,1] - torch.min(pos[:,:,1]))/(torch.max(pos[:,:,1]) - torch.min(pos[:,:,1])) - 0.5

scale = 0.6
eps = 0.01 
noise_sampler = RBFKernel(torch.from_numpy(mesh_pos).float().to(device), scale=scale, eps=eps, device=device)

sde = OU(beta_min=configs["beta_min"], beta_max=configs["beta_max"])

model.eval()

def model_fn(x, t, pos):
    inp = torch.cat([pos.permute(0, 2, 1), x], dim=1)

    var_factor = sde.cov_t_scaling(t, x)

    pred = model(inp, t, pos.unsqueeze(1))
    return pred / var_factor

batch_size = 6

num_timesteps = 200
ts = torch.linspace(1e-3, 1, num_timesteps).to("cuda")

delta_t = ts[1] - ts[0]
print("delta_t: ", delta_t)

xt = noise_sampler.sample(batch_size).unsqueeze(1) # N(0,C)

pos_inp = torch.repeat_interleave(pos, repeats=batch_size, dim=0)
for ti in tqdm(reversed(ts), total=len(ts)):
    print(ti)
    t = torch.ones(batch_size).to(xt.device)* ti

    with torch.no_grad():
        score = model_fn(xt, t, pos_inp)

    beta_t = sde.beta_t(t).view(-1, 1, 1)
    noise = noise_sampler.sample(batch_size).unsqueeze(1) # N(0,C)
    print(torch.linalg.norm(score), torch.linalg.norm(xt))
    xt = xt + beta_t/2.0 * delta_t*xt  + beta_t* delta_t * score + beta_t.sqrt()*delta_t.sqrt() * noise 
    
    print(torch.max(xt.abs()))
    #x0_pred = torch.clamp(x0_pred, 0, 10)
    #xt = scaling* x0_pred + (1 - scaling).sqrt().view(-1, 1, 1) * noise 

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


    plt.savefig(os.path.join(log_img_dir, f"t={ti:.4f}.png"))
    plt.close()

    
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




