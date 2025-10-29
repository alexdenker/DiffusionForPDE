

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation
import torch 
from tqdm import tqdm 

from physics.conjugate_gradient import cg 
from physics.poisson import PoissonOperator
from physics.dataset import EllipsesDataset
from physics.utils import gen_conductivity

device = "cuda"
mesh_name = "disk_dense" #"disk_256" #"disk_dense"

poisson = PoissonOperator(mesh_path=f"data/disk/{mesh_name}.msh",
                          device=device)

xy = poisson.omega.geometry.x
cells = poisson.omega.geometry.dofmap.reshape((-1, poisson.omega.topology.dim + 1))
tri = Triangulation(xy[:, 0], xy[:, 1], cells)
mesh_pos = np.array(poisson.W.tabulate_dof_coordinates()[:,:2])

mask = np.random.choice(poisson.dofs_pl, int(0.6*poisson.dofs_pl), replace=False)

B = torch.eye(poisson.dofs_pl,device=device)[mask]
poisson.set_observation_operator(B)

dataset = EllipsesDataset(base_path="dataset/gaussian_bumps")
a = dataset[2].to(device).T



print("a: ", a.shape)

sol = poisson.solve_linear_system(a)
y = torch.matmul(B, sol)
y = y + 0.00 * torch.randn_like(y)


a_pred = torch.zeros((poisson.dofs_pc,1), device=device)
a_pred.requires_grad_()

optimiser = torch.optim.SGD([a_pred], lr=2.0)
num_iters = 1000

for i in tqdm(range(num_iters)):
    optimiser.zero_grad()
    
    Hx = poisson.forward(a_pred)
    loss = 1/2*((Hx - y) ** 2).sum()
    loss.backward()

    print("Loss: ", loss.item(), " iter: ", i+1)
    """
    with torch.no_grad():
        gradient = poisson.adjoint(Hx - y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    pred = a_pred.grad.cpu().numpy().flatten()
    im = ax1.tripcolor(tri, pred, cmap="jet", shading="flat")
    ax1.axis("image")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("gradient torch")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.axis("off")

    pred = gradient.cpu().numpy().flatten()
    im = ax2.tripcolor(tri, pred, cmap="jet", shading="flat")
    ax2.axis("image")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("gradient analytical")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis("off")
    
    plt.show()
    """
    optimiser.step()

a_pred = a_pred.detach()

u_pred = poisson.solve_linear_system(a_pred)

print("a_pred: ", a_pred.shape)

print("a: ", a.shape)

print("MSE: ", torch.mean((a_pred - a)**2))

xy_2d = xy[:, 0:2]
mask_xy = xy[mask,:]

fig, axes = plt.subplots(2, 2, figsize=(13, 6))

pred = a.cpu().numpy().flatten()
im = axes[0,0].tripcolor(tri, pred, cmap="jet", shading="flat")
axes[0,0].axis("image")
axes[0,0].set_aspect("equal", adjustable="box")
axes[0,0].set_title("Parameter a")
fig.colorbar(im, ax=axes[0,0], fraction=0.046, pad=0.04)
axes[0,0].axis("off")

pred = sol.cpu().numpy().flatten()
im = axes[0,1].tripcolor(tri, pred, cmap="Reds", shading="flat")
axes[0,1].axis("image")
axes[0,1].set_aspect("equal", adjustable="box")
axes[0,1].set_title("Solution u")
fig.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)
axes[0,1].axis("off")
axes[0,1].scatter(mask_xy[:,0], mask_xy[:,1],
               c="black", s=10, marker="o", label="Sensors")



pred = a_pred.cpu().numpy().flatten()
im = axes[1,0].tripcolor(tri, pred, cmap="jet", shading="flat")
axes[1,0].axis("image")
axes[1,0].set_aspect("equal", adjustable="box")
axes[1,0].set_title("Parameter a_pred")
fig.colorbar(im, ax=axes[1,0], fraction=0.046, pad=0.04)
axes[1,0].axis("off")

pred = u_pred.cpu().numpy().flatten()
im = axes[1,1].tripcolor(tri, pred, cmap="Reds", shading="flat")
axes[1,1].axis("image")
axes[1,1].set_aspect("equal", adjustable="box")
axes[1,1].set_title("Solution u_pred")
fig.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)
axes[1,1].axis("off")
axes[1,1].scatter(mask_xy[:,0], mask_xy[:,1],
               c="black", s=10, marker="o", label="Sensors")
axes[0,1].legend()

plt.show()