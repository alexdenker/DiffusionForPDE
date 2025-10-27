# Retry without scipy: use numpy.interp for resampling
import numpy as np
import matplotlib.pyplot as plt
import math

from dataset import MNISTShapesDataset

def pts_to_complex(pts):
    return pts[:,0] + 1j*pts[:,1]

def complex_to_pts(z):
    return np.vstack([z.real, z.imag]).T

def keep_low_freq(descr, K):
    out = np.zeros_like(descr)
    out[:K+1] = descr[:K+1]
    out[-K:] = descr[-K:]
    return out


dataset = MNISTShapesDataset()

pts = dataset[0]

pts_complex = pts_to_complex(pts)

pts_fft = np.fft.fft(pts_complex)
pts_fft[0] = 0 # remove translation 

ks = [1, 2, 4, 8, 12, 16, 24, 32]

fig, axes = plt.subplots(2, len(ks) // 2, figsize=(16,8))

ax = axes.ravel()

for idx, k in enumerate(ks):
    pts_fft_low = keep_low_freq(pts_fft, K=k)

    pts_recon = np.fft.ifft(pts_fft_low)
    pts_recon = complex_to_pts(pts_recon)

    ax[idx].plot(pts[:,0], pts[:,1], label='original (resampled)')
    ax[idx].scatter(pts[:,0], pts[:,1], s=2, color='k')
    ax[idx].plot(pts_recon[:,0], pts_recon[:,1], label='fft recon')
    ax[idx].set_title(f"k={k}")
    ax[idx].axis('equal')


fig.suptitle('Fourier shapes: Varying the number of components k')

plt.legend()
plt.show()

