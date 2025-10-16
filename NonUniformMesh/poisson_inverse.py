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
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.fem import functionspace, form

from scipy.sparse import csr_matrix
import os 
import yaml 

from physics.utils import gen_conductivity
from neural_operator.nfft_neural_operator_version2 import NUFNO
from neural_operator.sde import OU
from neural_operator.noise_sampler import RBFKernel

import torch 
from tqdm import tqdm 

device = "cuda"
###
load_path = "exp/NonUniformFNO/20251015_131900"
###

with open(os.path.join(load_path,"config.yaml"), "r") as f:
    configs = yaml.safe_load(f)

mesh_name = configs["mesh_name"]


omega, _, _ = gmshio.read_from_msh(f"data/disk/{mesh_name}.msh", MPI.COMM_WORLD, gdim=2)
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

Q = torch.tensor(Q_matrix.toarray(), dtype=torch.float32).to(device)  # or float64

print("Q: ", Q.shape)


# M is your csr_matrix
M_dense = torch.tensor(M.toarray(), dtype=torch.float32).to(device)  # or float64
a_true_torch = torch.tensor(ax.x.array[:],dtype=torch.float32).unsqueeze(-1).to(device)
rhs_torch = torch.matmul(Q, a_true_torch).to(device)

print("M_dense: ", M_dense.shape)
print("rhs_torch: ", rhs_torch.shape)

# Solve using torch.linalg.solve
sol = torch.linalg.solve(M_dense, rhs_torch).to(device)

mask = np.random.choice(dofs_u, int(0.7*dofs_u), replace=False)

B = torch.eye(dofs_u)[mask].to(device)
print("B: ", B.shape)
print("sol: ", sol.shape)
y = torch.matmul(B, sol)

model = NUFNO(n_layers=configs["model"]["n_layers"], 
            modes=configs["model"]["modes"], 
            width=configs["model"]["width"],
            in_channels=3,
            timestep_embedding_dim=33,
            max_period=configs["model"]["max_period"])
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
model.load_state_dict(torch.load(os.path.join(load_path, "fno_model.pt")))
model.to(device)
model.eval()


def forward(a):
    Minv_a = torch.linalg.solve(M_dense, torch.matmul(Q, a))
    return torch.matmul(B, Minv_a)

batch_size = 1

pos = torch.from_numpy(mesh_pos).float().to(device).unsqueeze(0)
pos = torch.repeat_interleave(pos, repeats=batch_size, dim=0)
pos[:,:,0] = (pos[:,:,0] - torch.min(pos[:,:,0]))/(torch.max(pos[:,:,0]) - torch.min(pos[:,:,0])) - 0.5
pos[:,:,1] = (pos[:,:,1] - torch.min(pos[:,:,1]))/(torch.max(pos[:,:,1]) - torch.min(pos[:,:,1])) - 0.5


scale = 0.6
eps = 0.01 
noise_sampler = RBFKernel(torch.from_numpy(mesh_pos).float().to(device), scale=scale, eps=eps, device=device)

sde = OU(beta_min=configs["beta_min"], beta_max=configs["beta_max"])

def model_fn(x, t, pos):
    inp = torch.cat([pos.permute(0, 2, 1), x], dim=1)

    var_factor = sde.cov_t_scaling(t, x)

    pred = model(inp, t, pos.unsqueeze(1))
    return pred / var_factor


num_timesteps = 100
ts = torch.linspace(1e-3, 1, num_timesteps).to("cuda")

delta_t = ts[1] - ts[0]
print("delta_t: ", delta_t)

gamma_list = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
mse_list = []
for gamma in gamma_list:

    torch.manual_seed(123)
    xt = noise_sampler.sample(batch_size).unsqueeze(1) # N(0,C)

    for ti in tqdm(reversed(ts), total=len(ts)):
        #print(ti)
        t = torch.ones(batch_size).to(xt.device)* ti
        cov_t = sde.cov_t_scaling(t, xt)
        mean_t_scaling = sde.mean_t_scaling(t, xt)

        xt.requires_grad_()
        #with torch.no_grad():
        score = model_fn(xt, t, pos)

        x0_pred = (xt + score *cov_t**2) / mean_t_scaling

        beta_t = sde.beta_t(t).view(-1, 1, 1)
        noise = noise_sampler.sample(batch_size).unsqueeze(1) # N(0,C)
        
        Hx = forward(x0_pred.squeeze(1).unsqueeze(-1))

        mat = ((Hx - y).reshape(batch_size, -1) ** 2).sum()
        mat_norm = ((y - Hx).reshape(batch_size, -1) ** 2).sum(dim=1).sqrt().detach()
        grad_term = torch.autograd.grad(mat, xt, retain_graph=True)[0]
        coeff = gamma / mat_norm.reshape(-1, 1, 1)
        xt = xt + beta_t/2.0 * delta_t*xt + beta_t* delta_t * score + beta_t.sqrt()*delta_t.sqrt() * noise 
        xt = xt - grad_term * coeff
        #print((coeff*grad_term).norm().item(), score.norm().item(), mat.item())
        xt = xt.detach()


    mse = torch.mean((xt - a_true_torch)**2)
        
    print("MSE: ", mse.item())

    mse_list.append(mse.item())


plt.figure()
plt.plot(gamma_list, mse_list)
plt.show()


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))


im = ax1.tripcolor(tri, xt[0].cpu().numpy().flatten(), shading='flat', edgecolors='k')
ax1.axis('image')
ax1.set_aspect('equal', adjustable='box')
ax1.set_title("Reconstruction")
ax1.axis("off")
fig.colorbar(im, ax=ax1)
im = ax2.tripcolor(tri, a_true_torch.cpu().numpy().flatten(), shading='flat',  edgecolors='k')
ax2.axis('image')
ax2.set_aspect('equal', adjustable='box')
ax2.set_title("Ground truth")
ax2.axis("off")

fig.colorbar(im, ax=ax2)
plt.savefig("reconstruction.png")
plt.show()



