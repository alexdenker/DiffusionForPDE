
import numpy as np
import matplotlib.pyplot as plt


def pts_to_complex(pts):
    return pts[:,0] + 1j*pts[:,1]

def complex_to_pts(z):
    return np.vstack([z.real, z.imag]).T

def keep_low_freq(descr, K):
    out = np.zeros_like(descr)
    out[:K+1] = descr[:K+1]
    out[-K:] = descr[-K:]
    return out


def random_shape_fft(N=256, alpha=1.5, scale=1.0):

    pts = np.random.randn(N, 2) 
    
    z_k = np.fft.fft(pts_to_complex(pts)) 

    k = np.fft.fftfreq(N) * N  # symmetric range: 0..N/2-1, -N/2..-1
    amp = np.zeros(N)
    nonzero = k != 0
    amp[nonzero] = np.abs(k[nonzero]) ** (-alpha)  # k^{-alpha} amplitude

    #z_k = (np.random.randn(N)+ 1j*np.random.randn(N)) / np.sqrt(2.0)
    z_k *= amp
    z = np.fft.ifft(z_k) * scale

    return z

N = 256

for alpha in [0.1, 0.5, 1.0, 1.5, 2.0]:

    fig, axes = plt.subplots(2,3, figsize=(12,8))

    for idx, ax in enumerate(axes.flatten()):
            

        z = random_shape_fft(N, alpha=alpha)


        pts = complex_to_pts(z)
        pts = np.vstack([pts, pts[0]])  # append first point
        ax.plot(pts[:,0], pts[:,1])
        ax.axis('equal')

    fig.suptitle(f'Random shapes with alpha={alpha}')
    plt.show()