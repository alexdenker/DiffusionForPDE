
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation

import torch 
import numpy as np 
import time 

from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx.fem import functionspace

from utils import gen_conductivity
from diffusion import Diffusion
from simple_torch_NFFT import NFFT

device = "cuda"

omega, _, _ = gmshio.read_from_msh("disk_dense.msh", MPI.COMM_WORLD, gdim=2)
omega.topology.create_connectivity(1, 2)

xy = omega.geometry.x
cells = omega.geometry.dofmap.reshape((-1, omega.topology.dim + 1))

tri = Triangulation(xy[:, 0], xy[:, 1], cells)

V = functionspace(omega, ("DG", 0)) # piecewise constant
mesh_pos = np.array(V.tabulate_dof_coordinates()[:,:2])



def create_sample(mesh_pos):
    sigma_mesh = gen_conductivity(
        mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=1.0
    )
    return torch.from_numpy(sigma_mesh).float().unsqueeze(0)

def draw_batch(batch_size, mesh_pos):
    x = [] 
    for _ in range(batch_size):
        x.append(create_sample(mesh_pos))

    return torch.cat(x, dim=0)


batch_size = 1

pos = torch.from_numpy(mesh_pos).float().to("cuda").unsqueeze(0)
pos = torch.repeat_interleave(pos, repeats=batch_size, dim=0)

diffusion = Diffusion()


np.random.seed(123)
sigma = draw_batch(batch_size,mesh_pos).unsqueeze(1)
sigma = sigma.to("cuda")
print("sigma: ", sigma.shape)
print("pos: ", pos.shape)

N = (64,64) # frequency space 
nfft = NFFT(N, device=device)


# For the NFFT we need that the points are all scaled between [-0.5, 0.5]
# [min, max] - min 
# [0, max - min]

print("pos min max: ", pos[:,:,0].min(), pos[:,:,0].max())
print("pos min max: ", pos[:,:,1].min(), pos[:,:,1].max())

pos[:,:,0] = (pos[:,:,0] - torch.min(pos[:,:,0]))/(torch.max(pos[:,:,0]) - torch.min(pos[:,:,0])) - 0.5
pos[:,:,1] = (pos[:,:,1] - torch.min(pos[:,:,1]))/(torch.max(pos[:,:,1]) - torch.min(pos[:,:,1])) - 0.5

print("pos min max: ", pos[:,:,0].min(), pos[:,:,0].max())
print("pos min max: ", pos[:,:,1].min(), pos[:,:,1].max())

pos = pos.unsqueeze(1)

print("pos: ", pos.shape)

n_tries = 10
t1 = time.time()
for _ in range(n_tries):
    x_ft = nfft.adjoint(pos, sigma)
t2 = time.time()
print("Time for adjoint: ", (t2-t1)/n_tries, "s")
print(x_ft.shape)

x_ifft = nfft(pos, x_ft) 
x_ifft = x_ifft / x_ifft.size(-1) * 2


print(x_ifft.shape)


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))

im = ax1.tripcolor(tri, sigma[0].cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
ax1.axis('image')
ax1.set_aspect('equal', adjustable='box')
ax1.set_title("x")
ax1.axis("off")
fig.colorbar(im, ax=ax1,fraction=0.046, pad=0.04)

im = ax2.tripcolor(tri, x_ifft[0].real.cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
ax2.axis('image')
ax2.set_aspect('equal', adjustable='box')
ax2.set_title("F^-1(F(x))")
ax2.axis("off")
fig.colorbar(im, ax=ax2,fraction=0.046, pad=0.04)

plt.savefig("test_transform.png")
plt.close()




"""
def forward(x):
    x = x + 0j
    x_ft = nu_fft.forward(x)
    print(x_ft.shape)
    x_ft = x_ft.permute(0, 2, 1)
    x_ft = torch.reshape(x_ft, (batch_size, 1, 2*configs["model"]["modes"], 2*configs["model"]["modes"]-1))
    return torch.view_as_real(x_ft)

def inverse(x):
    x = torch.view_as_complex(x)[:,0,:,:]
    x_ft = torch.reshape(x, (batch_size, 1, 2*modes * (2*modes-1)))
    x_ft2 = x_ft[..., 2*modes:].flip(-1, -2).conj()
    x_ft = torch.cat([x_ft, x_ft2], dim=-1)

    x_ft = x_ft.permute(0, 2, 1)
    print("x comples: ", x.shape)
    print("x ft shape: ", x_ft.shape)
    x = nu_fft.inverse(x_ft).real # x [4, 20, 512, 512]
    print("x after nu_fft inverse ", x.shape)
    #x = x.permute(0, 2, 1)
    return x 


#import torchkbnufft as tkbn

#nufft_adj = tkbn.KbNufftAdjoint(im_size=(14,14)).to("cuda")



# Now take a regular FFT (uniform frequency grid)
#sigma_fft = torch.fft.fftshift(torch.fft.fft2(img_grid), dim=(-2, -1))
    
sigma_fft = forward(sigma)

sigma_ifft = inverse(sigma_fft)


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))

im = ax1.tripcolor(tri, sigma[0].cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
ax1.axis('image')
ax1.set_aspect('equal', adjustable='box')
ax1.set_title("x")
ax1.axis("off")
fig.colorbar(im, ax=ax1,fraction=0.046, pad=0.04)

im = ax2.tripcolor(tri, sigma_ifft[0].cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
ax2.axis('image')
ax2.set_aspect('equal', adjustable='box')
ax2.set_title("F^-1(F(x))")
ax2.axis("off")
fig.colorbar(im, ax=ax2,fraction=0.046, pad=0.04)

#im = ax3.tripcolor(tri, x0_pred[0].detach().cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
#ax3.axis('image')
#ax3.set_aspect('equal', adjustable='box')
#ax3.set_title("Denoised pred")
#ax3.axis("off")
#fig.colorbar(im, ax=ax3,fraction=0.046, pad=0.04)

plt.savefig("test_transform.png")
plt.close()



print("sigma: ", sigma.shape)
print("sigma_fft: ", sigma_fft.shape)

random_t = torch.randint(1, diffusion.num_diffusion_timesteps, (sigma.shape[0],), device=sigma.device)
z = torch.randn_like(sigma_fft)
#print("random_t.shape: ", random_t.shape)
alpha_t = diffusion.alpha(random_t).view(-1, 1,1, 1, 1)
print(alpha_t.shape, sigma_fft.shape)
# add noise to basis elements (choose basis)
perturbed_sigma_fft = alpha_t.sqrt() * sigma_fft + (1 - alpha_t).sqrt() * z 

perturbed_sigma = inverse(perturbed_sigma_fft)


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

#im = ax3.tripcolor(tri, x0_pred[0].detach().cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
#ax3.axis('image')
#ax3.set_aspect('equal', adjustable='box')
#ax3.set_title("Denoised pred")
#ax3.axis("off")
#fig.colorbar(im, ax=ax3,fraction=0.046, pad=0.04)

plt.savefig(f"test.png")
plt.close()
"""