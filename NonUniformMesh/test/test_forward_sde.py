"""
Testing out the forward diffusion process from https://arxiv.org/pdf/2302.10130

x_t = exp(-t) x_0 + sqrt(1 - exp(-t)) noise, noise ~ N(0,C)


"""
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation

import torch
import numpy as np 

import sys, os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neural_operator.noise_sampler import RBFKernel
from physics.utils import gen_conductivity
from neural_operator.sde import OU

use_dolfin = True 
try: 
    from dolfinx.io import gmshio
    from mpi4py import MPI
    from dolfinx.fem import functionspace
except ImportError as e:
    print("dolfin not installed, fall back to saved numpy files")
    use_dolfin = False 

def create_sample(mesh_pos):
    sigma_mesh = gen_conductivity(
        mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=0.0
    )
    return torch.from_numpy(sigma_mesh).float().unsqueeze(0)

device = "cuda"

if use_dolfin:
    omega, _, _ = gmshio.read_from_msh("data/disk/disk_dense.msh", MPI.COMM_WORLD, gdim=2)
    omega.topology.create_connectivity(1, 2)

    xy = omega.geometry.x
    cells = omega.geometry.dofmap.reshape((-1, omega.topology.dim + 1))

    V = functionspace(omega, ("DG", 0)) # piecewise constant
    mesh_pos = np.array(V.tabulate_dof_coordinates()[:,:2])
else:
    xy = np.load("data/disk/disk_dense_xy.npy")
    cells = np.load("data/disk/disk_dense_cells.npy")
    mesh_pos = np.load("data/disk/disk_dense_mesh_pos.npy")


tri = Triangulation(xy[:, 0], xy[:, 1], cells)



scale = 0.6
eps = 0.01 
noise_sampler = RBFKernel(torch.from_numpy(mesh_pos).float().to(device), scale=scale, eps=eps, device=device)

z = noise_sampler.sample(4)

sigma = create_sample(mesh_pos).to(device)

sde = OU(beta_min=0.001,beta_max=10)

T = 1.0
num_time_points = 6
ts = torch.linspace(0, T, num_time_points)

fig, axes = plt.subplots(1,num_time_points, figsize=(12,4))

for idx, t in enumerate(ts):
    noise = noise_sampler.sample(1)
    t = t.to(device)

    mean_t = sde.mean_t(t, sigma)
    cov_t = sde.cov_t_scaling(t, sigma)

    xt = mean_t + cov_t * noise 

    im = axes[idx].tripcolor(tri, xt.cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
    axes[idx].axis('image')
    axes[idx].set_aspect('equal', adjustable='box')
    axes[idx].set_title(f"t = {t.item():.4f}")
    axes[idx].axis("off")
    fig.colorbar(im, ax=axes[idx],fraction=0.046, pad=0.04)

fig.suptitle(f"Forward Diffusion with RBF Kernel Noise with scale={scale}, eps={eps}", fontsize=16)
plt.savefig("forward_sde.png")
plt.show()

# simulate with Euler-Maruyama 

T = 1.0
num_time_points = 100
ts = torch.linspace(0, T, num_time_points)
delta_t = ts[1] - ts[0]
print("delta_t: ", delta_t)

xt = sigma 

fig, axes = plt.subplots(1,6, figsize=(12,4))

plot_idx = 0
for idx, t in enumerate(ts):
    drift = sde.drift(t, xt)
    diffuson = sde.diffusion(t, xt)
    noise = noise_sampler.sample(1)
    xt = xt + drift * delta_t + diffuson * delta_t.sqrt() * noise 

    if idx in [0, 10, 30, 40, 60, 90]:
        im = axes[plot_idx].tripcolor(tri, xt.cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
        axes[plot_idx].axis('image')
        axes[plot_idx].set_aspect('equal', adjustable='box')
        axes[plot_idx].set_title(f"t = {float(np.round(t.item(),4))}")
        axes[plot_idx].axis("off")
        fig.colorbar(im, ax=axes[plot_idx],fraction=0.046, pad=0.04)
        plot_idx += 1
fig.suptitle(f"Forward Diffusion with RBF Kernel Noise with scale={scale}, eps={eps}", fontsize=16)
plt.savefig("euler_maruyama.png")
plt.show()