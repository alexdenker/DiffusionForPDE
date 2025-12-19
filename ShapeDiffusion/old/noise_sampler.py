import torch 
import numpy as np 

from utils import fourier_coefficients, inverse_fourier

class NoiseSampler(object):
    def sample(self, N):
        raise NotImplementedError


"""
k: frequency
e_k: fourier basis element 
z_k: Gaussian noise N(0,1)

k = 0 do nothing 
noise = sum_k k^{-alpha} e_k z_k

"""

class ShapeNoise(NoiseSampler):
    def __init__(self, num_landmarks, alpha=1, device=None):

        self.dimension = 2 # 2D shapes
        self.num_landmarks = num_landmarks
        self.device = device
        self.alpha = alpha

        self.spectral_filter = self.make_spectral_filter(num_landmarks, alpha, device=device).unsqueeze(-1).unsqueeze(0)  # shape (1, s, 1)
        print("Spectral filter shape: ", self.spectral_filter.shape)

    def make_spectral_filter(self, num_landmarks, alpha, device):

        k = torch.arange(0, num_landmarks//2 + 1, dtype=torch.float32, device=device) #  range: 0..num_landmarks/2+1
        filt = torch.zeros(num_landmarks//2 + 1, dtype=torch.float32, device=device)
        nonzero = k != 0
        filt[nonzero] = torch.abs(k[nonzero]) ** (-alpha/2.0)  # k^{-alpha} amplitude

        return filt.to(device) 

    @torch.no_grad()
    def sample(self, N):
        
        z_ts = torch.randn(N, self.num_landmarks, self.dimension).to(self.device) # 2d points

        z_k = fourier_coefficients(z_ts, num_bases=self.num_landmarks)
        z_k = z_k * self.spectral_filter.unsqueeze(0)  # (N, num_landmarks)

        z = inverse_fourier(z_k, num_pts=self.num_landmarks)
        
        return z

    

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    num_landmarks = 64

    for alpha in [0.5, 1.0, 1.5, 2.0, 4.0]:

        sampler = ShapeNoise(num_landmarks=num_landmarks, alpha=alpha, device='cuda')

        x = sampler.sample(N=10)  # (num_landmarks, 2)
        assert x.shape == (10, num_landmarks, 2)
        
        fig, axes = plt.subplots(2,5, figsize=(12,6))
        for idx, ax in enumerate(axes.flatten()):
            pts = x[idx].cpu().numpy()
            pts = np.vstack([pts, pts[0]])  # append first point
            ax.plot(pts[:,0], pts[:,1])
            ax.axis('equal')
            ax.set_title(f'Sample {idx+1}')
        fig.suptitle(f'Shape Noise Samples with alpha={alpha}')
        plt.show()
