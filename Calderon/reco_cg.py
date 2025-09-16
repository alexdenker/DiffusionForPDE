


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

mesh_name = "L"

omega, _, _ = gmshio.read_from_msh(f"Poisson/data/mesh/{mesh_name}_dense.msh", MPI.COMM_WORLD, gdim=2)
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
uD.interpolate(lambda x: 0*x[0])

bc = fem.dirichletbc(uD, boundary_dofs)

### sample a 
Lmatrix = np.load(f"Poisson/L_{mesh_name}_dense_DG0.npy")
if use_cuda:
    Lmatrix = torch.from_numpy(Lmatrix).float().to("cuda")

np.random.seed(5)
torch.random.manual_seed(4)
if use_cuda:
    samples = torch.linalg.solve(Lmatrix, torch.randn(Lmatrix.shape[0], 1, device="cuda")).cpu().numpy()
else:
    samples = np.linalg.solve(Lmatrix, np.random.randn(Lmatrix.shape[0], 1))
        
samples[samples < 0] = 0

print("Samples shape: ", samples.shape)

ax = fem.Function(W)
ax.x.array[:] = samples[:,0]

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

Q = torch.tensor(Q_matrix.toarray(), dtype=torch.float32)  # or float64

print("Q: ", Q.shape)


# M is your csr_matrix
M_dense = torch.tensor(M.toarray(), dtype=torch.float32)  # or float64
a_true_torch = torch.tensor(samples[:,0],dtype=torch.float32).unsqueeze(-1)
rhs_torch = torch.matmul(Q, a_true_torch)

print("M_dense: ", M_dense.shape)
print("rhs_torch: ", rhs_torch.shape)

# Solve using torch.linalg.solve
sol = torch.linalg.solve(M_dense, rhs_torch)
 
mask = np.random.choice(dofs_u, int(0.6*dofs_u), replace=False)

B = torch.eye(dofs_u)[mask]
print("B: ", B.shape)
print("sol: ", sol.shape)
y = torch.matmul(B, sol)


xy = omega.geometry.x
cells = omega.geometry.dofmap.reshape((-1, omega.topology.dim + 1))
tri = Triangulation(xy[:, 0], xy[:, 1], cells)

fig, axes = plt.subplots(2, 2, figsize=(13, 6))

pred = np.array(ax.x.array[:]).flatten()
im = axes[0,0].tripcolor(tri, pred, cmap="jet", shading="flat")
axes[0,0].axis("image")
axes[0,0].set_aspect("equal", adjustable="box")
axes[0,0].set_title("Parameter a")
fig.colorbar(im, ax=axes[0,0], fraction=0.046, pad=0.04)
axes[0,0].axis("off")

pred = np.array(sol).flatten()
im = axes[0,1].tripcolor(tri, pred, cmap="Reds", shading="gouraud")
axes[0,1].axis("image")
axes[0,1].set_aspect("equal", adjustable="box")
axes[0,1].set_title("Solution u")
fig.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)
axes[0,1].axis("off")



plt.show()


gamma = 1e-6 #1e-5# 1e-4

def forward(a):
    Minv_a = torch.linalg.solve(M_dense, torch.matmul(Q, a))
    return torch.matmul(B, Minv_a)

def adjoint(u):
    BTu = torch.matmul(B.T, u)
    return torch.matmul(Q.T, torch.linalg.solve(M_dense, BTu))

def op(x):
    return adjoint(forward(x)) + gamma * x


x0 = torch.zeros((dofs,1))

forward_x0 = op(x0)

a_pred = cg(op, x0, rhs=adjoint(y), n_iter=10)

u_pred = torch.linalg.solve(M_dense, torch.matmul(Q,a_pred))



xy_2d = xy[:, 0:2]
mask_xy = xy[mask,:]

fig, axes = plt.subplots(2, 2, figsize=(13, 6))

pred = np.array(ax.x.array[:]).flatten()
im = axes[0,0].tripcolor(tri, pred, cmap="jet", shading="flat")
axes[0,0].axis("image")
axes[0,0].set_aspect("equal", adjustable="box")
axes[0,0].set_title("Parameter a")
fig.colorbar(im, ax=axes[0,0], fraction=0.046, pad=0.04)
axes[0,0].axis("off")

pred = np.array(sol).flatten()
im = axes[0,1].tripcolor(tri, pred, cmap="Reds", shading="gouraud")
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

pred = np.array(u_pred.cpu().numpy()).flatten()
im = axes[1,1].tripcolor(tri, pred, cmap="Reds", shading="gouraud")
axes[1,1].axis("image")
axes[1,1].set_aspect("equal", adjustable="box")
axes[1,1].set_title("Solution u_pred")
fig.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)
axes[1,1].axis("off")
axes[1,1].scatter(mask_xy[:,0], mask_xy[:,1],
               c="black", s=10, marker="o", label="Sensors")
axes[0,1].legend()

plt.show()