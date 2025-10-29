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

for scale in [0.1, 0.3, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]:

    eps = 0.01 
    noise_sampler = RBFKernel(torch.from_numpy(mesh_pos).float().to(device), scale=scale, eps=eps, device=device)

    z = noise_sampler.sample(6)

    fig, axes = plt.subplots(2, 3, figsize=(15,10))

    for idx, ax in enumerate(axes.flatten()):

        im = ax.tripcolor(tri, z[idx].cpu().numpy().flatten(), cmap='jet', shading='flat') #,edgecolors='k')
        ax.axis('image')
        ax.set_aspect('equal', adjustable='box')
        ax.axis("off")
        fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)

    fig.suptitle(f"RBF Kernel Noise with scale={scale}, eps={eps}", fontsize=16)
    plt.savefig(f"noise_with_scale={scale}.png")
    plt.show()