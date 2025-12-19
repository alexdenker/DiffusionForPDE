"""
Testing how well the architecture works for just matching a single shape  
    min_\theta || f_\theta(z) - x ||^2

This is only a test to see if the architecture is capable of fitting a single shape from noise.

"""

import torch

import matplotlib.pyplot as plt
from tqdm import tqdm 

import sys, os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import MNISTShapesDataset
from noise_sampler import ShapeNoise
from simple_network import ScoreNet
from utils import fourier_coefficients, inverse_fourier



device = "cuda"

num_landmarks = 64

dataset = MNISTShapesDataset(class_label=3, num_landmarks=num_landmarks)

pts = dataset[0]

noise_sampler = ShapeNoise(num_landmarks=pts.shape[0], alpha=4.0, device="cuda")

torch.manual_seed(0)
z = noise_sampler.sample(N=1)[0].to(device).unsqueeze(0)

num_modes = 16
model = ScoreNet(input_dim=num_modes*4, output_dim=num_modes*4, hidden_dim=512, time_embed_dim=32, depth=8, max_period=2.0)
model.to(device)
model.train()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_iters = 250


t = torch.ones((z.shape[0],), device=device) * 0.5  # example time input


print("z: ", z.shape)

pts = pts.to(device).unsqueeze(0)
for i in tqdm(range(num_iters)):
    optimizer.zero_grad()


    xt_inp = fourier_coefficients(z, num_bases=num_modes)
    xt_inp = xt_inp.view(xt_inp.shape[0], -1)
    print("x_inp: ", xt_inp.shape)
    pred = model(xt_inp, t) 
    pred = pred.view(pred.shape[0], 2, num_modes, 2)
    pred = inverse_fourier(pred, num_pts=num_landmarks)

    loss = torch.sum((pred - pts)**2)
    loss.backward()

    optimizer.step()

    print("Loss:", loss.item())
    if i % 10 == 0:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

        ax1.plot(pts[0,:,0].cpu().numpy(), pts[0,:,1].cpu().numpy(), '-o')
        ax1.set_title('Original shape (MNIST digit 4)')

        ax2.plot(pred[0,:,0].detach().cpu().numpy(), pred[0,:,1].detach().cpu().numpy(), '-o')
        ax2.set_title(f'Reconstructed shape at iter {i}')

        plt.show()