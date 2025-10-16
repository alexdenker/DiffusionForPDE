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
import wandb 
from datetime import datetime

from configs.wandb_configs import WANDB_PROJECT, WANBD_ENTITY

from neural_operator.nfft_neural_operator_version2 import NUFNO
from physics.dataset import EllipsesDataset
from neural_operator.noise_sampler import RBFKernel
from neural_operator.sde import OU

use_dolfin = True 
try: 
    from dolfinx.io import gmshio
    from mpi4py import MPI
    from dolfinx.fem import functionspace
except ImportError as e:
    print("dolfin not installed, fall back to saved numpy files")
    use_dolfin = False 


log_wandb = False 


configs = {
    "mesh_name": "disk_dense",
    "num_epochs": 1000,
    "lr": 1e-4, 
    "save_dir": "exp/NonUniformFNO",
    "beta_min": 0.001,
    "beta_max": 15., 
    "loss_scaling": "positive", # scaling the loss by sigma(t)
    "model": {
    "n_layers": 6,
    "modes": 16, 
    "width": 32,
    "timestep_embedding_dim": 33,
    "max_period": 10. }
}


if not os.path.exists("wandb/"):
    os.makedirs("wandb/")

wandb_kwargs = {
    "project": WANDB_PROJECT,
    "entity": WANBD_ENTITY,
    "config": configs,
    "name": "NUFNO_Diffusion_rbfnoise",
    "mode": "online" if log_wandb else "disabled" , 
    "settings": wandb.Settings(code_dir="wandb/"),
    "dir": "wandb/",
}
device = "cuda"
with wandb.init(**wandb_kwargs) as run:

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

    name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(configs["save_dir"], name)
    print("save model to ", log_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, "config.yaml"), "w") as file:
        yaml.dump(configs, file)

    model = NUFNO(n_layers=configs["model"]["n_layers"], 
                modes=configs["model"]["modes"], 
                width=configs["model"]["width"],
                in_channels=3,
                timestep_embedding_dim=configs["model"]["timestep_embedding_dim"],
                max_period=configs["model"]["max_period"])
    print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

    model.to("cuda")
    model.train()

    sde = OU(beta_min=configs["beta_min"], beta_max=configs["beta_max"])

    num_epochs = configs["num_epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    dataset = EllipsesDataset(base_path="dataset/mesh_dg0")
    
    plot_batch = [dataset[i] for i in range(6)]
    plot_batch = torch.cat(plot_batch, dim=0).to(device).unsqueeze(1)
    print("plot batch: ", plot_batch.shape)
    batch_size = 16

    data_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=6)
    
    pos = torch.from_numpy(mesh_pos).float().to("cuda").unsqueeze(0)
    pos[:,:,0] = (pos[:,:,0] - torch.min(pos[:,:,0]))/(torch.max(pos[:,:,0]) - torch.min(pos[:,:,0])) - 0.5
    pos[:,:,1] = (pos[:,:,1] - torch.min(pos[:,:,1]))/(torch.max(pos[:,:,1]) - torch.min(pos[:,:,1])) - 0.5

    scale = 0.6
    eps = 0.01 
    noise_sampler = RBFKernel(torch.from_numpy(mesh_pos).float().to(device), scale=scale, eps=eps, device=device)

    def model_fn(x, t, pos):
        inp = torch.cat([pos.permute(0, 2, 1), x], dim=1)

        var_factor = sde.cov_t_scaling(t, x)

        pred = model(inp, t, pos.unsqueeze(1))
        return pred / var_factor

    for epoch in range(configs["num_epochs"]):
        model.train() 

        mean_loss = []
        for idx, x in tqdm(enumerate(data_loader), total=len(data_loader)):
            optimizer.zero_grad() 
            
            mean_loss = []
            pos_inp = torch.repeat_interleave(pos, repeats=batch_size, dim=0)

            x0 = x.to("cuda")

            random_t = torch.rand((x0.shape[0],), device=x0.device) * (1-0.001) + 0.001

            z = noise_sampler.sample(x0.shape[0]).unsqueeze(1) # N(0,C)

            mean_t = sde.mean_t(random_t, x0)
            cov_t = sde.cov_t_scaling(random_t, x0)
            xt = mean_t + cov_t * z 

            # pos_in: mesh positions (for the nFFT in the model)
            pred = model_fn(xt, random_t, pos_inp) # last layer / cov_t

            residual = pred + z/cov_t

            if configs["loss_scaling"] == "positive":
                residual = residual * cov_t

            residual_Linv = residual  #noise_sampler.apply_L_inv(residual.squeeze())
            loss = torch.sum(residual_Linv**2)/pred.shape[0] # loss function (mulitplied by (1-exp(-t)).sqrt()
            loss.backward() 
            mean_loss.append(loss.item())

            optimizer.step() 
            wandb.log({"train/loss": loss.item()})

        wandb.log({"train/mean_loss": float(np.mean(mean_loss))})
        lr_scheduler.step()
        model.eval() 
        wandb_log_dict = {}
        with torch.no_grad():
            times = torch.rand((plot_batch.shape[0],), device=plot_batch.device) * (1-0.001) + 0.001

            mean_t = sde.mean_t(times, plot_batch)
            mean_t_scaling = sde.mean_t_scaling(times, plot_batch)
            cov_t = sde.cov_t_scaling(times, plot_batch)

            z = noise_sampler.sample(plot_batch.shape[0]).unsqueeze(1)
            noisy_batch = mean_t + cov_t * z
            pos_inp = torch.repeat_interleave(pos, repeats=noisy_batch.shape[0], dim=0)
            with torch.no_grad():
                score_t = model_fn(noisy_batch, times, pos_inp) #model(noisy_batch, times, pos_inp) 

                x0_pred = (noisy_batch + score_t *cov_t**2) / mean_t_scaling
            
            fig, axes = plt.subplots(3, plot_batch.shape[0], figsize=(16,6))

            for idx in range(plot_batch.shape[0]):

                im = axes[0,idx].tripcolor(tri, plot_batch[idx,0].cpu().numpy().flatten(), cmap='Blues', shading='flat')#,edgecolors='k')
                axes[0,idx].axis('image')
                axes[0,idx].set_aspect('equal', adjustable='box')
                axes[0,idx].set_title("Phantom")
                axes[0,idx].axis("off")
                fig.colorbar(im, ax=axes[0,idx],fraction=0.046, pad=0.04)

                im = axes[1,idx].tripcolor(tri, noisy_batch[idx,0].cpu().numpy().flatten(), cmap='Blues', shading='flat')#,edgecolors='k')
                axes[1,idx].axis('image')
                axes[1,idx].set_aspect('equal', adjustable='box')
                axes[1,idx].set_title(f"Noisy phantom at t={times[idx].item():.4f}")
                axes[1,idx].axis("off")
                fig.colorbar(im, ax=axes[1,idx],fraction=0.046, pad=0.04)

                im = axes[2,idx].tripcolor(tri, x0_pred[idx,0].detach().cpu().numpy().flatten(), cmap='Blues', shading='flat')#,edgecolors='k')
                axes[2,idx].axis('image')
                axes[2,idx].set_aspect('equal', adjustable='box')
                axes[2,idx].set_title("Denoised pred")
                axes[2,idx].axis("off")
                fig.colorbar(im, ax=axes[2,idx],fraction=0.046, pad=0.04)

            wandb_log_dict["denoised_example"] = wandb.Image(plt)
            plt.close()

        wandb.log(wandb_log_dict)
        torch.save(model.state_dict(), os.path.join(log_dir,"fno_model.pt"))
        