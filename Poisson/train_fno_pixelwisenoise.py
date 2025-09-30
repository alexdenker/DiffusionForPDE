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

from neural_operator.nfft_neural_operator import NUFNO
from utils import gen_conductivity
from diffusion import Diffusion

configs = {
    "mesh_name": "disk_dense",
    "resume_from": None, 
    "num_epochs": 1200,
    "lr": 8e-4, 
    "save_dir": "exp/fno_dse",
    "model": {
    "modes": 14, 
    "width": 32 }
}

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

log_dir = os.path.join(configs["save_dir"], "circle")
print("save model to ", log_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists("train_imgs"):
    os.makedirs("train_imgs")

model = NUFNO(n_layers=4, 
              modes=configs["model"]["modes"], 
              width=configs["model"]["width"],
              in_channels=3,
              timestep_embedding_dim=33)
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

model.to("cuda")
model.train()

with open(os.path.join(log_dir, "config.yaml"), "w") as file:
    yaml.dump(configs, file)

def create_sample(mesh_pos):
    sigma_mesh = gen_conductivity(
        mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=0.0
    )
    return torch.from_numpy(sigma_mesh).float().unsqueeze(0)

def draw_batch(batch_size, mesh_pos):
    x = [] 
    for _ in range(batch_size):
        x.append(create_sample(mesh_pos))

    return torch.cat(x, dim=0)

min_loss = 1e6

num_epochs = configs["num_epochs"]
optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])

num_iters = 20000
save_every = 100
print_every = 10
plot_every = 50
batch_size = 16

running_loss = 0

pos = torch.from_numpy(mesh_pos).float().to("cuda").unsqueeze(0)


pos = torch.repeat_interleave(pos, repeats=batch_size, dim=0)

pos[:,:,0] = (pos[:,:,0] - torch.min(pos[:,:,0]))/(torch.max(pos[:,:,0]) - torch.min(pos[:,:,0])) - 0.5
pos[:,:,1] = (pos[:,:,1] - torch.min(pos[:,:,1]))/(torch.max(pos[:,:,1]) - torch.min(pos[:,:,1])) - 0.5

print("pos: ", pos.shape)

model.train()

diffusion = Diffusion(beta_start=1e-4, beta_end=6e-3)

def model_fn(x, t, pos):
    inp = torch.cat([pos.permute(0, 2, 1), x], dim=1)

    # output of models as (xt - sqrt(alpha_t) * model) / sqrt(1-alpha_t)
    sqrt_ab = diffusion.alpha(t).sqrt().view(-1, 1, 1)
    sqrt_omb = (1 - diffusion.alpha(t)).sqrt().view(-1, 1, 1)

    pred = model(inp, t, pos.unsqueeze(1))
    return (x - sqrt_ab * pred) / sqrt_omb



for step in range(num_iters):
    optimizer.zero_grad() 
    
    mean_loss = []
    
    sigma = draw_batch(batch_size,mesh_pos).unsqueeze(1)
    sigma = sigma.to("cuda")
    
    random_t = torch.randint(1, diffusion.num_diffusion_timesteps, (sigma.shape[0],), device=sigma.device)
    z = torch.randn_like(sigma)
    #print("sigma: ", sigma.shape)
    alpha_t = diffusion.alpha(random_t).view(-1, 1, 1)
    perturbed_sigma = alpha_t.sqrt() * sigma + (1 - alpha_t).sqrt() * z 

    #print("perturbed_sigma: ", sigma.shape)
    #print("pos: ", pos.shape)
    #inp = torch.cat([pos.permute(0, 2, 1), perturbed_sigma], dim=1)

    pred = model_fn(perturbed_sigma, random_t, pos)
    

    loss = torch.sum((pred - z)**2)/pred.shape[0] # loss function w.r.t basis elements # 
    loss.backward() 

    #print("Loss: ", loss.item())
    running_loss = (1-0.99)*loss.item() + 0.99 * running_loss
    optimizer.step() 
    
    if step % plot_every == 0:

        x0_pred = (perturbed_sigma - pred * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14,7))

        im = ax1.tripcolor(tri, sigma[0].cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
        ax1.axis('image')
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title("Phantom")
        ax1.axis("off")
        fig.colorbar(im, ax=ax1,fraction=0.046, pad=0.04)

        im = ax2.tripcolor(tri, perturbed_sigma[0].cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
        ax2.axis('image')
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_title(f"Noisy phantom at t={random_t[0].item()}")
        ax2.axis("off")
        fig.colorbar(im, ax=ax2,fraction=0.046, pad=0.04)

        im = ax3.tripcolor(tri, x0_pred[0].detach().cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
        ax3.axis('image')
        ax3.set_aspect('equal', adjustable='box')
        ax3.set_title("Denoised pred")
        ax3.axis("off")
        fig.colorbar(im, ax=ax3,fraction=0.046, pad=0.04)

        plt.savefig(f"train_imgs/{step}.png")
        plt.close()

    if step % save_every == 0:
        torch.save(model.state_dict(), os.path.join(log_dir,"lno_model.pt"))
    
    if step % print_every == 0:
        print(f"Running loss {running_loss:.4f} at step {step}. Current loss {loss.item():.4f}")