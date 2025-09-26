import torch 
import numpy as np 

class NoiseSampler(object):
    def sample(self, N):
        raise NotImplementedError

class SpectralNoiseSampler(NoiseSampler):
    def __init__(self, H, W, power=2.0, cutoff=None, device='cuda'):
        self.device = device   
        self.H = H 
        self.W = W

        self.power = power 
        self.cutoff = cutoff
        self.spectral_filter = self.make_spectral_filter(H, W, power=power, cutoff=cutoff, device=device)

    def make_spectral_filter(self, height, width, power=2.0, cutoff=None, device=None):
        """Return real spectral filter of shape (1,1,H,W) to multiply FFT(z).
        power controls decay ~ (1+|k|^2)^{-power/2}. cutoff (int) optionally zeroes high freq beyond radius.
        """
        if device is None:
            device = self.device
        
        ky = torch.fft.fftfreq(height, d=1.0/height, device=device)  
        kx = torch.fft.fftfreq(width, d=1.0/width, device=device)
        KX, KY = torch.meshgrid(kx, ky, indexing='xy')
        K2 = KX**2 + KY**2
        
        filt = (1.0 + K2)**(-power/2.0)  # shape (W, H) because of indexing='xy'

        # put in (1,1,H,W)
        filt = torch.tensor(filt, device=device).unsqueeze(0).unsqueeze(0)  # transpose to (H,W)
        if cutoff is not None:
            filt = torch.fft.fftshift(filt, dim=(-2, -1))  # shift zero freq to center

            # zero out frequencies with radius > cutoff
            center_y = height // 2
            center_x = width // 2
            Y, X = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='xy')
            R = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
            mask = (R <= cutoff).float()
            filt = filt * mask.unsqueeze(0).unsqueeze(0)
            filt = torch.fft.ifftshift(filt, dim=(-2, -1))  # shift back

        filt = filt / torch.norm(filt) * np.sqrt(height * width)  
        return filt

    @torch.no_grad()
    def sample(self, N):
        z = torch.randn(N, 1, self.H, self.W, device=self.device)
        z_f = torch.fft.fft2(z)
        z_f = z_f * self.spectral_filter
        z = torch.fft.ifft2(z_f).real
        return z
    

"""
Adapted from: https://github.com/neuraloperator/FunDPS/blob/main/training/noise_samplers.py

"""

def get_fixed_coords(H, W):
    xs = torch.linspace(0, 1, steps=H + 1)[0:-1]
    ys = torch.linspace(0, 1, steps=W + 1)[0:-1]
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    coords = torch.cat([yy.reshape(-1, 1), xx.reshape(-1, 1)], dim=-1)
    return coords

class RBFKernel(NoiseSampler):
    @torch.no_grad()
    def __init__(self, H, W, scale=1, eps=0.01, device=None):
        self.H = H
        self.W = W
        self.device = device
        self.scale = scale

        # (s^2, 2)
        meshgrid = get_fixed_coords(self.H, self.W).to(device)
        # (s^2, s^2)
        C = torch.exp(-torch.cdist(meshgrid, meshgrid) / (2 * scale**2))
        I = torch.eye(C.size(-1)).to(device)

        I.mul_(eps**2)  # inplace multiply by eps**2
        C.add_(I)  # inplace add by I
        del I  # don't need it anymore

        self.L = torch.linalg.cholesky(C)

        del C  # don't need it anymore

    @torch.no_grad()
    def sample(self, N):

        samples = torch.zeros((N, self.H * self.W, 1)).to(self.device)
        for ix in range(N):
            # (s^2, s^2) * (s^2, 2) -> (s^2, 2)
            z = torch.randn(self.H *self.W, 1).to(self.device)
            samples[ix] = torch.matmul(self.L, z)

        # reshape into (N, s, s, n_in)
        sample_rshp = samples.reshape(-1, self.H, self.W, 1)

        # reshape into (N, n_in, s, s)
        sample_rshp = sample_rshp.transpose(-1, -2).transpose(-2, -3)

        return sample_rshp

if __name__ == "__main__":
    sampler = SpectralNoiseSampler(64, 64, power=2.0, cutoff=32, device='cpu')
    samples = sampler.sample(4)
    print(samples.shape)  # should be (4, 1, 64, 64)
    assert samples.shape == (4, 1, 64, 64), "Shape mismatch for spectral noise sampler"

    sampler2 = RBFKernel(64, 64, scale=0.1, eps=0.01, device='cpu')
    samples2 = sampler2.sample(4)
    print(samples2.shape)  # should be (4, 1, 64, 64)      
    assert samples2.shape == (4, 1, 64, 64), "Shape mismatch for RBF kernel noise sampler"