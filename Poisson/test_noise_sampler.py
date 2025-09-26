


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation

import torch
import numpy as np 

from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx.fem import functionspace

from sampler.noise_sampler import RBFKernel

device = "cuda"

omega, _, _ = gmshio.read_from_msh("disk_dense.msh", MPI.COMM_WORLD, gdim=2)
omega.topology.create_connectivity(1, 2)

xy = omega.geometry.x
cells = omega.geometry.dofmap.reshape((-1, omega.topology.dim + 1))

tri = Triangulation(xy[:, 0], xy[:, 1], cells)

V = functionspace(omega, ("DG", 0)) # piecewise constant
mesh_pos = np.array(V.tabulate_dof_coordinates()[:,:2])

scale = 0.2
eps = 0.01 
noise_sampler = RBFKernel(torch.from_numpy(mesh_pos).float().to(device), scale=scale, eps=eps, device=device)

z = noise_sampler.sample(4)


fig, axes = plt.subplots(2,2, figsize=(9,9))

for idx, ax in enumerate(axes.flatten()):
    im = ax.tripcolor(tri, z[idx].cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
    ax.axis('image')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Sample {idx+1}")
    ax.axis("off")
    fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)

fig.suptitle(f"RBF Kernel Noise Samples on Irregular Mesh with scale={scale}, eps={eps}", fontsize=16)
plt.savefig("RBF_noise.png")
plt.close()

