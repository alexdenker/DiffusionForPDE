import torch 
import numpy as np 

class NoiseSampler(object):
    def sample(self, N):
        raise NotImplementedError


class ShapeNoise(NoiseSampler):
    def __init__(self, num_landmarks, alpha=1, device=None):

        self.dimension = 2 # 2D shapes
        self.num_landmarks = num_landmarks
        self.device = device
        self.alpha = alpha

        self.spectral_filter = self.make_spectral_filter(num_landmarks, alpha, device=device)  # shape (s, 1)

    def make_spectral_filter(self, num_landmarks, alpha, device):

        k = torch.fft.fftfreq(num_landmarks) * num_landmarks  # symmetric range: 0..num_landmarks/2-1, -num_landmarks/2..-1
        filt = torch.zeros(num_landmarks)
        nonzero = k != 0
        filt[nonzero] = torch.abs(k[nonzero]) ** (-alpha)  # k^{-alpha} amplitude

        return filt.to(device) 

    @torch.no_grad()
    def sample(self, N):
        
        pts = torch.randn(N, self.num_landmarks, self.dimension).to(self.device) 
        z_ts = pts[:, :,0] + 1j*pts[:, :,1]

        z_k = torch.fft.fft(z_ts, dim=1)  # (N, num_landmarks)

        z_k = z_k * self.spectral_filter.unsqueeze(0)  # (N, num_landmarks)

        z = torch.fft.ifft(z_k)
        pts = torch.stack([z.real, z.imag], dim=-1)
        return pts


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    num_landmarks = 64
    alpha = 2.0 

    sampler = ShapeNoise(num_landmarks=num_landmarks, alpha=alpha, device='cpu')

    x = sampler.sample(N=10)  # (num_landmarks, 2)
    assert x.shape == (10, num_landmarks, 2)
    
    fig, axes = plt.subplots(2,5, figsize=(12,6))
    for idx, ax in enumerate(axes.flatten()):
        pts = x[idx].cpu().numpy()
        pts = np.vstack([pts, pts[0]])  # append first point
        ax.plot(pts[:,0], pts[:,1])
        ax.axis('equal')
        ax.set_title(f'Sample {idx+1}')

    plt.show()
