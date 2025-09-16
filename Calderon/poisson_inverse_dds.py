"""
We are looking to solve a standard Poisson inverse problem
    - nabla(a(x)  nabla u) = 0  in Omega
                        u  = 0  on Boundary

We assume that we have some internal measurements of the solution u. 
                        
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


from neural_operator.fourier_neural_operator_dse import FNO_dse
from utils import gen_conductivity
from diffusion import Diffusion

import torch 
from tqdm import tqdm 


def cg(op: callable, x, rhs, n_iter: int = 5):
    """
    Batched version of conjugate gradient (CG) descent, running for a fixed number of iterations. 
    The batching is done over the second dimension, i.e., we assume a dimension of x as (dimension, batch)
    
    Solves the (square) system of equations: B x = y 
    
    A note on the usage: 
        Most of the times we apply CG to the normal equations
            A^T A x = A^T y 
        or with Tikhonov
            (A^T A + gamma I) x = A^T y

    Arguments:
        op: implementation of the operator B 
        x: initialisation
        rhs: right hand side y
        n_iter: total number of iterations
    """
    # n x batch
    r = op(x)
    r = rhs - r
    p = r
    d = torch.zeros_like(x)
    
    sqnorm_r_old = torch.sum(r*r, dim=0)
    for _ in range(n_iter):
        
        d = op(p)
        
        inner_p_d = (p * d).sum(dim=0) 

        alpha = sqnorm_r_old / inner_p_d
        x = x + alpha[None,:]*p # x = x + alpha*p
        r = r - alpha[None,:]*d # r = r - alpha*d

        sqnorm_r_new = torch.sum(r * r, dim=0)

        beta = sqnorm_r_new / sqnorm_r_old
        sqnorm_r_old = sqnorm_r_new

        p = r + beta[None,:]*p # p = r + b * p

    return x 


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
ax.x.array[:] = gen_conductivity(mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=1.0).ravel()


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
 
mask = np.random.choice(dofs_u, int(0.9*dofs_u), replace=False)

B = torch.eye(dofs_u)[mask].to("cuda")
print("B: ", B.shape)
print("sol: ", sol.shape)
y = torch.matmul(B, sol)

configs = {
    "model": "fno_dse",
    "mesh_name": "disk_dense",
    "lr": 1e-3, 
    "save_dir": "exp/fno_dse",
    "model": {
    "modes": 14, 
    "width": 48 }
}

model = FNO_dse(modes=configs["model"]["modes"], width=configs["model"]["width"])
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
model.load_state_dict(torch.load("exp/fno_dse/circle/lno_model.pt"))
model.to("cuda")
model.eval()


gamma = 1e-6 #1e-5# 1e-4

def forward(a):
    Minv_a = torch.linalg.solve(M_dense, torch.matmul(Q, a))
    return torch.matmul(B, Minv_a)

def adjoint(u):
    BTu = torch.matmul(B.T, u)
    return torch.matmul(Q.T, torch.linalg.solve(M_dense, BTu))

def op(x):
    return adjoint(forward(x)) + gamma * x

batch_size = 1

pos = torch.from_numpy(mesh_pos).float().to("cuda").unsqueeze(0)
pos = torch.repeat_interleave(pos, repeats=batch_size, dim=0)


diffusion = Diffusion()

ts = torch.arange(0, diffusion.num_diffusion_timesteps).to("cuda")[::10]
x = torch.randn((batch_size, pos.shape[1], 1)).to("cuda")

print("x: ", x.shape)

n = x.size(0)
ss = [-1] + list(ts[:-1])
xt_s = [x.cpu()]
x0_s = []

xt = x
eta = 1.0

for ti, si in tqdm(zip(reversed(ts), reversed(ss)), total=len(ts)):
    t = torch.ones(n).to(x.device).long() * ti
    s = torch.ones(n).to(x.device).long() * si

    


    alpha_t = diffusion.alpha(t).view(-1, 1, 1)
    alpha_s = diffusion.alpha(s).view(-1, 1, 1)
    c1 = (
        (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)
    ).sqrt() * eta
    c2 = ((1 - alpha_s) - c1**2).sqrt()

    with torch.no_grad():
        inp = torch.cat([pos, xt], dim=-1)
            
        et = model(inp, t)
    
    x0_pred = (xt - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

    x0_pred = cg(op, x0_pred[:,:,0].T, rhs=adjoint(y), n_iter=10).T.unsqueeze(-1)

    xs = alpha_s.sqrt() * x0_pred + c1 * torch.randn_like(xt) + c2 * et.detach() 

    

    # xt_s.append(xs.cpu())
    # x0_s.append(x0_pred.cpu())
    xt = xs.detach()



print(xt.shape)


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))


im = ax1.tripcolor(tri, xt[0].cpu().numpy().flatten(), cmap='jet', shading='flat', vmin=0.01, vmax=4.0, edgecolors='k')
ax1.axis('image')
ax1.set_aspect('equal', adjustable='box')
ax1.set_title("Reconstruction")
ax1.axis("off")
fig.colorbar(im, ax=ax1)
im = ax2.tripcolor(tri, a_true_torch.cpu().numpy().flatten(), cmap='jet', shading='flat', vmin=0.01, vmax=4.0, edgecolors='k')
ax2.axis('image')
ax2.set_aspect('equal', adjustable='box')
ax2.set_title("Ground truth")
ax2.axis("off")

fig.colorbar(im, ax=ax2)
plt.savefig("reconstruction_cg.png")




