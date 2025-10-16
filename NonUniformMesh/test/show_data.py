import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation


import torch 
import numpy as np 
import sys, os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from physics.dataset import EllipsesDataset

use_dolfin = True 
try: 
    from dolfinx.io import gmshio
    from mpi4py import MPI
    from dolfinx.fem import functionspace
except ImportError as e:
    print("dolfin not installed, fall back to saved numpy files")
    use_dolfin = False 


mesh_name = "disk_dense"

if use_dolfin:
    omega, _, _ = gmshio.read_from_msh(f"data/disk/{mesh_name}.msh", MPI.COMM_WORLD, gdim=2)
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

dataset = EllipsesDataset(base_path="dataset/mesh_dg0")

plot_batch = [dataset[i] for i in range(6)]
plot_batch = torch.cat(plot_batch, dim=0).to("cpu").unsqueeze(1)

fig, axes = plt.subplots(1,6, figsize=(12,5))

for idx in range(6):

    im = axes[idx].tripcolor(tri, plot_batch[idx,0].cpu().numpy().flatten(), shading='flat')#,edgecolors='k')
    axes[idx].axis('image')
    axes[idx].set_aspect('equal', adjustable='box')
    axes[idx].axis("off")
    fig.colorbar(im, ax=axes[idx],fraction=0.046, pad=0.04)

fig.suptitle("Training data")
fig.tight_layout()

plt.show()