
import torch 
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation
import numpy as np 

import sys, os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from physics.utils import gen_conductivity

use_dolfin = True 
try: 
    from dolfinx.io import gmshio
    from mpi4py import MPI
    from dolfinx.fem import functionspace
except ImportError as e:
    print("dolfin not installed, fall back to saved numpy files")
    use_dolfin = False 

def create_sample(mesh_pos):
    sigma_mesh = gen_conductivity(
        mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=0.0
    )
    return torch.from_numpy(sigma_mesh).float().unsqueeze(0)

device = "cuda"

# class for fully nonequispaced 2d points
class VFT:
    def __init__(self, x_positions, y_positions, modes):
        # it is important that positions are scaled between 0 and 2*pi
        x_positions -= torch.min(x_positions)
        self.x_positions = x_positions * 6.28 / (torch.max(x_positions))
        y_positions -= torch.min(y_positions)
        self.y_positions = y_positions * 6.28 / (torch.max(y_positions))
        self.number_points = x_positions.shape[1]
        self.batch_size = x_positions.shape[0]
        self.modes = modes

        self.X_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().cuda()
        self.Y_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes-1), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().cuda()


        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        m = (self.modes*2)*(self.modes*2-1)
        X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, (self.modes*2-1), 1)
        Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]).repeat(1, 1, self.modes*2).reshape(self.batch_size,m,self.number_points))
        forward_mat = torch.exp(-1j* (X_mat+Y_mat)) 

        inverse_mat = torch.conj(forward_mat.clone()).permute(0,2,1)

        return forward_mat, inverse_mat

    def forward(self, data):
        """
        data: (batch_size, num_mesh_points, num_features) 
        """

        data_fwd = torch.bmm(self.V_fwd, data)
        return data_fwd

    def inverse(self, data):
        """
        data: (batch_size, num_modes, num_features) 
        """

        data_inv = torch.bmm(self.V_inv, data)
        
        return data_inv
    

if use_dolfin:
    omega, _, _ = gmshio.read_from_msh("data/disk/disk_dense.msh", MPI.COMM_WORLD, gdim=2)
    omega.topology.create_connectivity(1, 2)

    xy = omega.geometry.x
    cells = omega.geometry.dofmap.reshape((-1, omega.topology.dim + 1))

    V = functionspace(omega, ("DG", 0)) # piecewise constant
    mesh_pos = np.array(V.tabulate_dof_coordinates()[:,:2])
else:
    xy = np.load("data/disk/disk_dense_xy.npy")
    cells = np.load("data/disk/disk_dense_cells.npy")
    mesh_pos = np.load("data/disk/disk_dense_mesh_pos.npy")


tri = Triangulation(xy[:, 0], xy[:, 1], cells)

torch.manual_seed(123)
np.random.seed(123)
sigma = create_sample(mesh_pos).to(device).unsqueeze(0)
mesh_pos = torch.from_numpy(mesh_pos).to(device).float()
nuft = VFT(mesh_pos[:,0].unsqueeze(0), mesh_pos[:,1].unsqueeze(0), 12)
print(nuft.V_fwd.shape, nuft.V_inv.shape)


print("sigma.shape: ", sigma.shape)
print("mesh_pos.shape: ", mesh_pos.shape)

sigma_ft = nuft.forward(sigma.permute(0, 2, 1).cfloat())


print("sigma_ft.shape: ", sigma_ft.shape)

sigma_ift = nuft.inverse(sigma_ft).real 
sigma_ift = sigma_ift / sigma_ift.size(-2) 
print("sigma_ift.shape: ", sigma_ift.shape)

fig, (ax1, ax2) = plt.subplots(1,2)
im = ax1.tripcolor(tri, sigma.cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
ax1.axis('image')
ax1.set_aspect('equal', adjustable='box')
ax1.axis("off")
fig.colorbar(im, ax=ax1,fraction=0.046, pad=0.04)

im = ax2.tripcolor(tri, sigma_ift.cpu().numpy().flatten(), cmap='jet', shading='flat',edgecolors='k')
ax2.axis('image')
ax2.set_aspect('equal', adjustable='box')
ax2.axis("off")
fig.colorbar(im, ax=ax2,fraction=0.046, pad=0.04)


plt.show()
