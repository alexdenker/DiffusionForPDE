"""
Train a diffusion models based on the NonUniformFFT Neural Operator with i.i.d. white Gaussian noise in pixel space.
The forward process is 
    x_t = alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * z 

Here, x_0 is the parameter on the given mesh (shape [batch_size, dim, num_mesh_points]). 
This means that we are adding i.i.d. Gaussian noise to every mesh position.

"""

import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation

import os 
import argparse

import torch 
import numpy as np 
from omegaconf import OmegaConf

from configs.model_configs import Config

from neural_operator.nfft_neural_operator import NUFNO
from neural_operator.score_model import ScoreModel
from samplers.euler_maruyama import EulerMaruyama, EulerMaruyamaExponential
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

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, required=True, help="Path to experiment folder")
parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., cuda or cpu)")


def main(args):
    device = args.device
    load_path = args.load_path
    
    cfg = OmegaConf.structured(Config) 
    cfg = OmegaConf.merge(cfg, OmegaConf.load(os.path.join(load_path, "config.yaml")))
    print(cfg)

    mesh_name = cfg.model.mesh_name

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

    model = NUFNO(n_layers=cfg.model.n_layers,
                modes=cfg.model.modes,
                width=cfg.model.width,
                in_channels=3,
                timestep_embedding_dim=cfg.model.timestep_embedding_dim,
                max_period=cfg.model.max_period)
                

    print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
    noise_sampler = RBFKernel(torch.from_numpy(mesh_pos).float().to(device), scale=cfg.noise.scale, eps=cfg.noise.eps, device=device)

    sde = OU(beta_min=cfg.sde.beta_min, beta_max=cfg.sde.beta_max)

    score_model = ScoreModel(model, sde, noise_sampler, cfg)
    score_model.model.load_state_dict(torch.load(os.path.join(load_path, "fno_model_ema.pt")))
    score_model.model.to("cuda")
    score_model.model.eval()

    pos = torch.from_numpy(mesh_pos).float().to("cuda").unsqueeze(0)
    batch_size = 6
    pos_inp = torch.repeat_interleave(pos, repeats=batch_size, dim=0)

    num_timesteps = 100
    ts = torch.linspace(1e-3, 1.0, num_timesteps).to("cuda")

    sampler = EulerMaruyama(score_model, cfg)
    #sampler = EulerMaruyamaExponential(score_model, cfg)

    print(pos_inp.shape, ts.shape)

    x = sampler.sample(pos_inp, ts) 
    if cfg.training.train_on == "darcy_flow":
        mean = 7.5 
        std = 9.0   
        x = x * std + mean

    fig, axes = plt.subplots(2,3, figsize=(14,7))

    for idx, ax in enumerate(axes.ravel()):
        if idx < x.shape[0]:
            im = ax.tripcolor(tri, x[idx].cpu().numpy().flatten(), cmap="jet",  shading='flat', edgecolors='k') # cmap='Blues',
            ax.axis('image')
            ax.set_aspect('equal', adjustable='box')
            ax.set_title("Sample")
            ax.axis("off")
            fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
        else:
            ax.axis("off")


    plt.savefig("generated_samples.png")
    plt.show()



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)