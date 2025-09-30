"""
We are looking to solve a standard Poisson inverse problem
    - nabla( nabla u) = a(x)  in Omega
                   u  = 1  on Boundary

Calerdon 
- nabla(a(x) nabla u) = 0  in Omega
                   u  = 1  on Boundary


We assume that we have some internal measurements of the solution u. 

FunDPS for Poisson 

Discretise the PDE 
    M u = b(a)
    u = M^{-1} b(a)

d/da || y - M^{-1} b(a) ||

d/da M^{-1} f(a) = M^{-1} d/da f(a)

"""


from mpi4py import MPI
from dolfinx import mesh

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation

import ufl 
from dolfinx.io import gmshio
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.fem import (functionspace, form)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector

from scipy.sparse import csr_matrix


from neural_operator.nfft_neural_operator import NUFNO
from utils import gen_conductivity
from diffusion import Diffusion

import torch 
from tqdm import tqdm 

use_cuda = True

mesh_name = "disk_dense"

omega, _, _ = gmshio.read_from_msh(f"{mesh_name}.msh", MPI.COMM_WORLD, gdim=2)
omega.topology.create_connectivity(1, 2)


V = functionspace(omega, ("Lagrange", 1))
W = functionspace(omega, ("DG", 0))

tdim = omega.topology.dim
fdim = tdim - 1
omega.topology.create_connectivity(fdim, tdim)
print("fdim: ", fdim, " tdim: ", tdim)
boundary_facets = mesh.exterior_facet_indices(omega.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

uD = fem.Function(V)
uD.interpolate(lambda x: 0*x[0]+1)

bc = fem.dirichletbc(uD, boundary_dofs)

xy = omega.geometry.x
cells = omega.geometry.dofmap.reshape((-1, omega.topology.dim + 1))

tri = Triangulation(xy[:, 0], xy[:, 1], cells)

mesh_pos = np.array(W.tabulate_dof_coordinates()[:,:2])


ax = fem.Function(W)
np.random.seed(123)
ax.x.array[:] = gen_conductivity(mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=0.0).ravel()


u_ = fem.Function(V)

u = ufl.TrialFunction(V) 
v = ufl.TestFunction(V)

# first create the LHS matrix 

dofs = len(ax.x.array)
dofs_u = len(u_.x.array)

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

w = ufl.TrialFunction(W)  # lives in DG0

bilinear_form = fem.form(ufl.inner(v, w) * ufl.dx)

Q = fem.petsc.assemble_matrix(bilinear_form, [])
Q.assemble()

ai, aj, av = Q.getValuesCSR()

Q_matrix = csr_matrix((av, aj, ai)) 
Q.destroy() 

A = assemble_matrix(form(a), bcs=[bc])
A.assemble()

ai, aj, av = A.getValuesCSR()

M = csr_matrix((av, aj, ai)) # M will be matrix used to solve problem
M.resize(dofs_u,dofs_u) 

A.destroy() # dont need A anymore

Q = torch.tensor(Q_matrix.toarray(), dtype=torch.float32).to("cuda")  # or float64

print("Q: ", Q.shape)


# M is your csr_matrix
M_dense = torch.tensor(M.toarray(), dtype=torch.float32).to("cuda")  # or float64
a_true_torch = torch.tensor(ax.x.array[:],dtype=torch.float32).unsqueeze(-1).to("cuda")
rhs_torch = torch.matmul(Q, a_true_torch).to("cuda")

print("M_dense: ", M_dense.shape)
print("rhs_torch: ", rhs_torch.shape)

# Solve using torch.linalg.solve
sol = torch.linalg.solve(M_dense, rhs_torch).to("cuda")
 

#fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
#im = ax1.tripcolor(tri, sol.cpu().numpy().flatten(), cmap='jet', shading='gouraud', edgecolors='k')
#ax1.axis('image')
#ax1.set_aspect('equal', adjustable='box')
#ax1.set_title("Reconstruction")
#ax1.axis("off")
#fig.colorbar(im, ax=ax1)

#fig.colorbar(im, ax=ax2)
#plt.savefig("solution.png")

mask = np.random.choice(dofs_u, int(0.9*dofs_u), replace=False)

B = torch.eye(dofs_u)[mask].to("cuda")
print("B: ", B.shape)
print("sol: ", sol.shape)
y = torch.matmul(B, sol)

configs = {
    "mesh_name": "disk_dense",
    "save_dir": "exp/fno_dse",
    "model": {
    "modes": 14, 
    "width": 32 }
}

model = NUFNO(n_layers=4, 
              modes=configs["model"]["modes"], 
              width=configs["model"]["width"],
              in_channels=3,
              timestep_embedding_dim=33)
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
model.load_state_dict(torch.load("exp/fno_dse/circle/lno_model.pt"))
model.to("cuda")
model.eval()


def forward(a):
    Minv_a = torch.linalg.solve(M_dense, torch.matmul(Q, a))
    return torch.matmul(B, Minv_a)

batch_size = 1

pos = torch.from_numpy(mesh_pos).float().to("cuda").unsqueeze(0)
pos = torch.repeat_interleave(pos, repeats=batch_size, dim=0)
pos[:,:,0] = (pos[:,:,0] - torch.min(pos[:,:,0]))/(torch.max(pos[:,:,0]) - torch.min(pos[:,:,0])) - 0.5
pos[:,:,1] = (pos[:,:,1] - torch.min(pos[:,:,1]))/(torch.max(pos[:,:,1]) - torch.min(pos[:,:,1])) - 0.5

diffusion = Diffusion(beta_start=1e-4, beta_end=6e-3)

ts = torch.arange(0, diffusion.num_diffusion_timesteps).to("cuda")[::5]
x = torch.randn((batch_size, 1, pos.shape[1])).to("cuda")

def model_fn(x, t, pos):
    inp = torch.cat([pos.permute(0, 2, 1), x], dim=1)

    # output of models as (xt - sqrt(alpha_t) * model) / sqrt(1-alpha_t)
    sqrt_ab = diffusion.alpha(t).sqrt().view(-1, 1, 1)
    sqrt_omb = (1 - diffusion.alpha(t)).sqrt().view(-1, 1, 1)

    pred = model(inp, t, pos.unsqueeze(1))
    return (x - sqrt_ab * pred) / sqrt_omb


print("x: ", x.shape)

n = x.size(0)
ss = [-1] + list(ts[:-1])

xt = x
eta = 1.0

gamma = 30.0
for ti, si in tqdm(zip(reversed(ts), reversed(ss)), total=len(ts)):
    t = torch.ones(n).to(x.device).long() * ti
    s = torch.ones(n).to(x.device).long() * si

    xt = xt.clone().to('cuda').requires_grad_(True)

    alpha_t = diffusion.alpha(t).view(-1, 1, 1)
    alpha_s = diffusion.alpha(s).view(-1, 1, 1)
    c1 = (
        (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)
    ).sqrt() * eta
    c2 = ((1 - alpha_s) - c1**2).sqrt()
        
    et = model_fn(xt, t, pos)
    x0_pred = (xt - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

    Hx = forward(x0_pred.squeeze(1).unsqueeze(-1))

    mat_norm = ((y - Hx).reshape(n, -1) ** 2).sum(dim=1).sqrt().detach()
    mat = ((y - Hx).reshape(n, -1) ** 2).sum()
    grad_term = torch.autograd.grad(mat, xt, retain_graph=True)[0]
    
    print(mat_norm)
    grad_term = grad_term.detach()
    coeff = gamma / mat_norm.reshape(-1, 1, 1)
    #print(torch.linalg.norm(grad_term * coeff))
    xs = alpha_s.sqrt() * x0_pred + c1 * torch.randn_like(xt) + c2 * et.detach() # DDIM 
    xs = xs - grad_term * coeff # Data Consitency "getting closer to matching the observatoin"

    xt = xs.detach()


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))


im = ax1.tripcolor(tri, xt[0].cpu().numpy().flatten(), cmap='jet', shading='flat', edgecolors='k')
ax1.axis('image')
ax1.set_aspect('equal', adjustable='box')
ax1.set_title("Reconstruction")
ax1.axis("off")
fig.colorbar(im, ax=ax1)
im = ax2.tripcolor(tri, a_true_torch.cpu().numpy().flatten(), cmap='jet', shading='flat',  edgecolors='k')
ax2.axis('image')
ax2.set_aspect('equal', adjustable='box')
ax2.set_title("Ground truth")
ax2.axis("off")

fig.colorbar(im, ax=ax2)
plt.savefig("reconstruction.png")
plt.show()



