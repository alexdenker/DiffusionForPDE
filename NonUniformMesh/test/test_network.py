import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation

import torch 
import numpy as np 

import sys, os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_operator.nfft_neural_operator_version2 import NUFNO, timestep_embedding

use_dolfin = True 
try: 
    from dolfinx.io import gmshio
    from mpi4py import MPI
    from dolfinx.fem import functionspace
except ImportError as e:
    print("dolfin not installed, fall back to saved numpy files")
    use_dolfin = False 


configs = {
    "mesh_name": "disk_dense",
    "num_epochs": 1000,
    "lr": 1e-3, 
    "save_dir": "exp/NonUniformFNO",
    "beta_min": 0.001,
    "beta_max": 15., 
    "model": {
    "n_layers": 6,
    "modes": 16, 
    "width": 32,
    "timestep_embedding_dim": 33,
    "max_period": 10, }
}

device = "cuda"

mesh_name = configs["mesh_name"]

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

"""
model = NUFNO(n_layers=configs["model"]["n_layers"], 
            modes=configs["model"]["modes"], 
            width=configs["model"]["width"],
            in_channels=3,
            timestep_embedding_dim=configs["model"]["timestep_embedding_dim"],
            max_period=configs["model"]["max_period"])
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

model.to(device)
model.train()
"""
ts = torch.linspace(1e-3, 1.0, 100)


emb = timestep_embedding(ts, configs["model"]["timestep_embedding_dim"], max_period=configs["model"]["max_period"])

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.imshow(emb)
ax1.set_aspect("auto")
ax1.set_title("Time embedding")
ax1.set_xlabel("embedding dim.")
ax1.set_ylabel("time dim.")

ax2.plot(emb[0,:], label=f"t={ts[0].item():.4f}")
ax2.plot(emb[20,:], label=f"t={ts[20].item():.4f}")
ax2.plot(emb[40,:], label=f"t={ts[40].item():.4f}")
ax2.plot(emb[60,:], label=f"t={ts[60].item():.4f}")
ax2.plot(emb[80,:], label=f"t={ts[80].item():.4f}")
ax2.legend()

plt.show()