import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation

from tqdm import tqdm 
import os 

import torch  
import numpy as np 

use_dolfin = True 
try: 
    from dolfinx.io import gmshio
    from mpi4py import MPI
    from dolfinx.fem import functionspace
except ImportError as e:
    print("dolfin not installed, fall back to saved numpy files")
    use_dolfin = False 

from physics.utils import gen_conductivity

show_images = False 

def create_sample(mesh_pos):
    sigma_mesh = gen_conductivity(
        mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=0.0
    )
    return torch.from_numpy(sigma_mesh).float().unsqueeze(0)

num_examples = 4000

dataset_dir = os.path.join("dataset", "mesh_dg0")

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

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
print(mesh_pos.shape)

if show_images:

    tri = Triangulation(xy[:, 0], xy[:, 1], cells)


for i in tqdm(range(num_examples)):
    sigma = create_sample(mesh_pos)

    if show_images:

        fig, ax = plt.subplots(1,1, figsize=(7,7))


        im = ax.tripcolor(tri, sigma.flatten(), cmap='Blues', shading='flat', edgecolors='k')
        ax.axis('image')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Reconstruction")
        ax.axis("off")
        fig.colorbar(im, ax=ax)

        plt.show()

    #print(sigma.shape)
    np.save(os.path.join(dataset_dir, f"example_{i}.npy"), sigma.cpu().numpy())
        