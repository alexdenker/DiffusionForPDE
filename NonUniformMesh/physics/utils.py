import torch 
import numpy as np
from numpy.typing import NDArray


def create_sample(mesh_pos):
    sigma_mesh = gen_conductivity(
        mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=0.0
    )
    return torch.from_numpy(sigma_mesh).float().unsqueeze(0)


def draw_batch(batch_size, mesh_pos):
    x = [create_sample(mesh_pos) for _ in range(batch_size)]
    return torch.cat(x, dim=0)


def cart_ellipse_coords(x, y, h, k, a, b, alpha):
    """Compute normalized coordinates in rotated ellipse frame."""
    x_rot = (x - h) * np.cos(alpha) + (y - k) * np.sin(alpha)
    y_rot = -(x - h) * np.sin(alpha) + (y - k) * np.cos(alpha)
    return x_rot / a, y_rot / b


def sample_ellipse(test_step: int = 200, tolerance: int = 50):
    h = np.random.rand() * 1.6 - 0.8
    k = np.random.rand() * 1.6 - 0.8
    alpha = np.random.rand() * 2 * np.pi
    a = np.random.rand() * 0.2 + 0.2
    b = a * (0.8 + 0.2 * np.random.rand())
    #b = np.random.rand() * (a - 0.1) + 0.2

    theta = np.linspace(0, 2 * np.pi, test_step)
    x = h + a * np.cos(alpha) * np.cos(theta) - b * np.sin(alpha) * np.sin(theta)
    y = k + a * np.sin(alpha) * np.cos(theta) + b * np.cos(alpha) * np.sin(theta)

    i = 0
    while np.any(x**2 + y**2 > 0.8):
        b = np.random.rand() * (a - 0.1) + 0.2
        x = h + a * np.cos(alpha) * np.cos(theta) - b * np.sin(alpha) * np.sin(theta)
        y = k + a * np.sin(alpha) * np.cos(theta) + b * np.cos(alpha) * np.sin(theta)
        if i == tolerance:
            return sample_ellipse(test_step)
        i += 1

    return h, k, a, b, alpha


def sample_inclusions(numInc: int, test_step: int = 200, tolerance: int = 30):
    h, k, a, b, alpha = (np.zeros(numInc) for _ in range(5))
    h[0], k[0], a[0], b[0], alpha[0] = sample_ellipse(test_step)

    for i in range(1, numInc):
        overlap = True
        tol = 0
        while overlap and tol < tolerance:
            overlap = False
            h[i], k[i], a[i], b[i], alpha[i] = sample_ellipse(test_step)
            for j in range(i):
                if np.linalg.norm([h[j] - h[i], k[j] - k[i]]) < a[j] + a[i] + 0.1:
                    overlap = True
                    break
            tol += 1
        if tol == tolerance:
            return sample_inclusions(numInc, test_step, tolerance)
    return h, k, a, b, alpha


def gen_conductivity(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    max_numInc: int,
    backCond: float = 1.0,
) -> NDArray[np.float64]:
    numInc = np.random.randint(1, max_numInc + 1)
    condOut = np.ones(x1.shape) * backCond
    h, k, a, b, alpha = sample_inclusions(numInc)

    for i in range(numInc):
        amp = np.random.uniform(2.0, 4.0)  # peak intensity
        x_norm, y_norm = cart_ellipse_coords(x1, x2, h[i], k[i], a[i], b[i], alpha[i])
        
        # Gaussian blob (elliptical)
        blob = np.exp(-0.5 * (x_norm**2 + y_norm**2))
        
        # Blend with background: take the higher value (no sharp edges)
        condOut = np.maximum(condOut, backCond + (amp - backCond) * blob)

    return condOut
