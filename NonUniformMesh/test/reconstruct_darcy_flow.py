
import torch 
from tqdm import tqdm 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation
import sys, os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from physics.darcy_flow import DarcyFlowOperator
from neural_operator.noise_sampler import RBFKernel

device = "cuda"
darcy = DarcyFlowOperator(mesh_path= "data/disk/disk_256.msh",
                          device=device)

xy = darcy.omega.geometry.x
cells = darcy.omega.geometry.dofmap.reshape((-1, darcy.omega.topology.dim + 1))
tri = Triangulation(xy[:, 0], xy[:, 1], cells)

mesh_pos = np.array(darcy.W.tabulate_dof_coordinates()[:,:2])

# create a random parameter field
freq = 0.5 * torch.pi  
a = 5.0 * (np.sin(freq * mesh_pos[:, 0]) * np.cos(freq * mesh_pos[:, 1])) + 6.0

noise_sampler = RBFKernel(mesh_points=torch.from_numpy(mesh_pos).float().to(device), scale=0.8, eps=1e-3, device=device)

torch.manual_seed(12)
a = noise_sampler.sample(N=1).T
a[a > 0] = 12 
a[a <= 0] = 3

fig, ax = plt.subplots(1,1, figsize=(6,6))
im = ax.tripcolor(tri, a.cpu().numpy().ravel(), cmap="jet", shading="flat")
ax.axis("image")
ax.set_aspect("equal", adjustable="box")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.axis("off")
plt.show()
print("a: ",a.shape)
#a = torch.from_numpy(a).float().unsqueeze(1).to(device)

# forward solve
with torch.no_grad():
    u = darcy.forward(a)

print("Solution u shape: ", u.shape)


mean = 7.5
a_torch = torch.ones_like(a)*mean
a_torch.requires_grad = True

print("a_torch shape: ", a_torch.shape)
Linv_a = noise_sampler.apply_L_inv(a_torch.T - mean)

print("Linv_a shape: ", Linv_a.shape)
print(Linv_a)

optim = torch.optim.Adam([a_torch],lr=0.1)

alpha = 1e-7

loss_list =[] 
for i in tqdm(range(400)):
    optim.zero_grad()

    u_pred = darcy.forward(a_torch)
    Linv_a = noise_sampler.apply_L_inv(a_torch.T - mean)
    loss_data = torch.sum((u_pred - u)**2)
    loss_reg = torch.sum(Linv_a**2 )
    loss = loss_data + alpha * loss_reg
    loss.backward() 
    optim.step() 
    with torch.no_grad():
        a_torch.data.clamp_(0)

    print(f"loss {loss.item():.8f}, loss data {loss_data.item():.8f}, loss reg {loss_reg.item():.8f} at iter {i}")
    loss_list.append(loss.item())



with torch.no_grad():
    u_pred = darcy.forward(a_torch)

plt.figure()
plt.semilogy(loss_list)

fig, axes = plt.subplots(2, 2, figsize=(13, 6))


im = axes[0,0].tripcolor(tri, a.detach().cpu().numpy().ravel(), cmap="jet", shading="flat")
axes[0,0].axis("image")
axes[0,0].set_aspect("equal", adjustable="box")
axes[0,0].set_title("Parameter a")
fig.colorbar(im, ax=axes[0,0], fraction=0.046, pad=0.04)
axes[0,0].axis("off")

im = axes[0,1].tripcolor(tri, u.cpu().numpy().flatten(), cmap="Reds", shading="gouraud")
axes[0,1].axis("image")
axes[0,1].set_aspect("equal", adjustable="box")
axes[0,1].set_title("Solution u")
fig.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)
axes[0,1].axis("off")

im = axes[1,0].tripcolor(tri, a_torch.detach().cpu().numpy().ravel(), cmap="jet", shading="flat")
axes[1,0].axis("image")
axes[1,0].set_aspect("equal", adjustable="box")
axes[1,0].set_title("Parameter a_pred")
fig.colorbar(im, ax=axes[1,0], fraction=0.046, pad=0.04)
axes[1,0].axis("off")

im = axes[1,1].tripcolor(tri, u_pred.cpu().numpy().flatten(), cmap="Reds", shading="gouraud")
axes[1,1].axis("image")
axes[1,1].set_aspect("equal", adjustable="box")
axes[1,1].set_title("Solution u_pred")
fig.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)
axes[1,1].axis("off")

plt.show()