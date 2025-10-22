"""
Implementation of Darcy Flow operator using FEniCSx and PyTorch.
The PDE is given by
    - nabla( a(x) nabla u) = 0  in Omega
                        u  = g  on Boundary

We implement both a and u as piecewise linear functions.

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
from petsc4py import PETSc

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized


import torch 
from tqdm import tqdm 



class DarcyFlowOperator():
    def __init__(self, mesh_path = "data/disk/disk_dense.msh", 
                        device="cuda"):
        
        self.omega, _, _ = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=2)
        self.omega.topology.create_connectivity(1, 2)
        tdim = self.omega.topology.dim
        self.fdim = tdim - 1
        self.omega.topology.create_connectivity(self.fdim, tdim)

        self.V = functionspace(self.omega, ("Lagrange", 1)) # piecewise linear function space for solution u
        self.W = functionspace(self.omega, ("DG", 0)) # piecewise constant for parameter a

        # create Dirichlet BC
        boundary_facets = mesh.exterior_facet_indices(self.omega.topology)
        boundary_dofs = fem.locate_dofs_topological(self.V, self.fdim, boundary_facets)

        uD = fem.Function(self.V)
        uD.interpolate(lambda x: 0*x[0])

        bc = fem.dirichletbc(uD, boundary_dofs)


        # degrees of freedom for the two discrete spaces
        u_ = fem.Function(self.V)

        self.dofs = len(u_.x.array)

        self.device = device 
        
        self.solver = FEMSolver(self.V, self.W, bc)

        self.B = torch.eye(self.dofs, device=self.device)

    def set_observation_operator(self, B):
        self.B = B

    def forward(self, a):
        sol =  self.solver(a)
        return torch.matmul(self.B, sol).unsqueeze(-1)


class FEMSolver(torch.nn.Module):
    def __init__(self, V, W, bc):
        super().__init__()
        self.V = V
        self.W = W 
        self.bc = bc 

    def forward(self, x):
        return FEMSolverFunction.apply(x, self.V, self.W, self.bc)


class FEMSolverFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, V, W, bc):
        ctx.set_materialize_grads(False)

        # save python objects needed in backward
        ctx.V = V
        ctx.W = W
        ctx.bc = bc
        ctx.x = x

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        ax = fem.Function(W)

        ax.x.array[:] = x.detach().cpu().numpy().ravel()

        dofs = len(ax.x.array)
        ctx.dofs = dofs
        ctx.v = v

        a_form = ufl.dot(ax * ufl.grad(u), ufl.grad(v)) * ufl.dx

        A = assemble_matrix(form(a_form), bcs=[bc])
        A.assemble()

        # Set up PETSc LU solver
        ksp = PETSc.KSP().create(A.comm)
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.getPC().setType(PETSc.PC.Type.LU)
        ksp.setFromOptions()

        # RHS: zero forcing or source term
        L_form = v * ufl.dx
        b = fem.petsc.assemble_vector(fem.form(L_form))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Solve system
        u_fun = fem.Function(V)
        ksp.solve(b, u_fun.vector)

        # Save solution and solver context for backward
        ctx.u_fun = u_fun
        ctx.ksp = ksp
        ctx.ax = ax

        # Return solution as torch tensor
        u_tensor = torch.from_numpy(u_fun.x.array.copy().astype(np.float32)).to(x.device).unsqueeze(-1)
        return u_tensor

    @staticmethod
    def backward(ctx, grad_output):
        V = ctx.V
        W = ctx.W
        ksp = ctx.ksp
        u_fun = ctx.u_fun

        # Solve adjoint system: K^T p = grad_output
        p_fun = fem.Function(V)
        v = ufl.TestFunction(V)
        rhs_form = v * ufl.dx
        rhs = fem.petsc.create_vector(fem.form(rhs_form))
        rhs.array[:] = grad_output.detach().cpu().numpy().ravel()
        rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)


        ksp.solve(rhs, p_fun.vector)  # PETSc LU solves transpose automatically for symmetric K

        # Compute elementwise DG0 gradient: - int grad u . grad p dx
        w = ufl.TestFunction(W)
        grad_form = -ufl.inner(ufl.grad(u_fun), ufl.grad(p_fun)) * w * ufl.dx
        grad_vec = fem.petsc.assemble_vector(fem.form(grad_form))
        grad_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        grad_local = torch.from_numpy(np.array(grad_vec.array, copy=True)).to(grad_output.device).unsqueeze(-1)
        return grad_local, None, None, None



if __name__ == "__main__":
    # simple test of the DarcyFlowOperator
    device = "cuda"
    darcy = DarcyFlowOperator(device=device)


    xy = darcy.omega.geometry.x
    cells = darcy.omega.geometry.dofmap.reshape((-1, darcy.omega.topology.dim + 1))
    tri = Triangulation(xy[:, 0], xy[:, 1], cells)

    mesh_pos = np.array(darcy.W.tabulate_dof_coordinates()[:,:2])

    # create a random parameter field
    freq = 2.0 * torch.pi  
    a = 5.0 * (np.sin(freq * mesh_pos[:, 0]) * np.cos(freq * mesh_pos[:, 1])) + 6.0
    a = torch.from_numpy(a).float().unsqueeze(1).to(device)
    print("a shape: ", a.shape)
    # forward solve
    with torch.no_grad():
        u = darcy.forward(a)

    print("Solution u shape: ", u.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    im = ax1.tripcolor(tri, a.cpu().numpy().flatten(), cmap="jet", shading="flat")
    ax1.axis("image")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Parameter a")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.axis("off")

    im = ax2.tripcolor(tri, u.cpu().numpy().flatten(), cmap="Reds", shading="gouraud")
    ax2.axis("image")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("Solution u")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis("off")
    

    plt.show()
