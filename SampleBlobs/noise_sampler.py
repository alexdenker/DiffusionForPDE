import torch 
import numpy as np 

class NoiseSampler(object):
    def sample(self, N):
        raise NotImplementedError

"""
Adapted from: https://github.com/neuraloperator/FunDPS/blob/main/training/noise_samplers.py

"""

class RBFKernel(NoiseSampler):
    """
    This sampler generates noise for a fixed discretization grid using an RBF kernel.
    
    """
    @torch.no_grad()
    def __init__(self, mesh_points, scale=1, eps=0.01, device=None):
        """
        mesh_points: (s, 2) tensor of mesh coordinates with s being the number of spatial points
        
        """

        self.num_points = mesh_points.shape[0]
        self.device = device
        self.scale = scale

        # (s^2, 2)
        # (s^2, s^2)
        C = torch.exp(-torch.cdist(mesh_points, mesh_points) / (2 * scale**2))
        I = torch.eye(C.size(-1)).to(device)

        I.mul_(eps**2)  # inplace multiply by eps**2
        C.add_(I)  # inplace add by I
        del I  # don't need it anymore

        self.L = torch.linalg.cholesky(C)

        del C  # don't need it anymore

    @torch.no_grad()
    def sample(self, N):

        samples = torch.zeros((N, self.num_points)).to(self.device)
        for ix in range(N):
            # (s^2, s^2) * (s^2, 2) -> (s^2, 2)
            z = torch.randn(self.num_points, 1).to(self.device)
            samples[ix] = torch.matmul(self.L, z)[:,0]

        return samples

if __name__ == "__main__":
    # create random mesh points in a [0,1]^2 domain
    mesh_points = torch.rand(2000, 2)

    sampler2 = RBFKernel(mesh_points, scale=0.1, eps=0.01, device='cpu')
    samples2 = sampler2.sample(4)
    print(samples2.shape)  # should be (4, 2000)      
    assert samples2.shape == (4, 2000), "Shape mismatch for RBF kernel noise sampler"


    