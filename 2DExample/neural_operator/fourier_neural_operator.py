"""
Adapted from: https://github.com/camlab-ethz/DSE-for-NeuralOperators/tree/main

"""



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math 
import matplotlib.pyplot as plt

### Time embedding 
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

################################################################
# Fourier layer (using FFT)
################################################################


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # number of low-freq modes to keep in x-dir
        self.modes2 = modes2 # number of low-freq modes to keep in y-dir


        # Complex weights (real + imaginary parts)
        self.scale = 1 / (in_channels * out_channels)
        self.weights_pos = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights_neg = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2))

        # time modulation scaling 
        self.scale_A = 1 / emb_dim

        self.A_real_pos = nn.Parameter(
            self.scale_A * torch.rand(1, self.modes1, self.modes2, emb_dim, dtype=torch.float))
        self.A_imag_pos = nn.Parameter(
            self.scale_A * torch.rand(1, self.modes1, self.modes2, emb_dim, dtype=torch.float))        

        self.A_real_neg = nn.Parameter(
            self.scale_A * torch.rand(1, self.modes1, self.modes2, emb_dim, dtype=torch.float))
        self.A_imag_neg = nn.Parameter(
            self.scale_A * torch.rand(1, self.modes1, self.modes2, emb_dim, dtype=torch.float))     

    def compl_mul2d(self, input, weights):
        # input: (batch, in_channel, x, y), Fourier coeffs (complex)
        # weights: (in_channel, out_channel, x, y)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input, cweights)

    def forward(self, x, emb):
        """
        x: (B, C_in, H, W)
        emb: (B, emb_dim)
        
        Returns: (B, C_out, H, W)
        """

        batchsize = x.shape[0]

        x_ft = torch.fft.rfft2(x, norm="forward") # we assume only real valued inputs 

        # build time modulation 
        emb = emb[:,None,None,:] # [B,1,1,emb_dim]
        phi_real_pos = torch.einsum('bmwc,bmwc->bmw', self.A_real_pos, emb)
        phi_imag_pos = torch.einsum('bmwc,bmwc->bmw', self.A_imag_pos, emb)
        phi_complex_pos = phi_real_pos + 1j * phi_imag_pos  # (b, modes, modes)

        phi_real_neg = torch.einsum('bmwc,bmwc->bmw', self.A_real_neg, emb)
        phi_imag_neg = torch.einsum('bmwc,bmwc->bmw', self.A_imag_neg, emb)
        phi_complex_neg = phi_real_neg + 1j * phi_imag_neg  # (b, modes, modes)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
            )

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights_pos
            )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights_pos
            )

        # modulate with time embedding
        out_ft[:, :, :self.modes1, :self.modes2] *= phi_complex_pos.unsqueeze(1)
        out_ft[:, :, -self.modes1:, :self.modes2] *= phi_complex_neg.unsqueeze(1)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="forward")

        return x

class TimeMLP(nn.Module):
    def __init__(self, emb_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.SiLU(),                  # or ReLU if you like pain
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, t):
        return self.net(t)   # [batch, out_dim]

class Conv1DTime(nn.Module):
    def __init__(self, width, emb_dim):
        super().__init__()

        self.time_mlp = TimeMLP(emb_dim=emb_dim, out_dim=2*width)
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, x, emb):
        """
        x: [batch_size, channels, H, W]
        emb: [batch_size, c] c is  time step embedding dimension
        
        return: [batch_size, channels, H, W]
        
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W)  # [B, C, H*W]



        gamma, beta = self.time_mlp(emb).chunk(2, dim=-1)  # [batch, width] each
        x = self.w(x_flat) * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        x = x.view(B, C, H, W)
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2,  width, emb_dim):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.emb_dim = emb_dim

        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.emb_dim)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.emb_dim)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.emb_dim)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.emb_dim)
        self.w0 = Conv1DTime(self.width, self.emb_dim)
        self.w1 = Conv1DTime(self.width, self.emb_dim)
        self.w2 = Conv1DTime(self.width, self.emb_dim)
        self.w3 = Conv1DTime(self.width, self.emb_dim)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = nn.GELU()

    def forward(self, x, emb):
        """
        x: [batch_size, channels, H, W] 
        emb: [batch_size, c] c is  time step embedding dimension
        """

        x = x.permute(0, 2,3, 1)  # (batch, H, W, channels)
        x = self.fc0(x)  # (batch, H, W, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, H, W)
        
        x1 = self.conv0(x, emb)
        x2 = self.w0(x, emb)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv1(x, emb)
        x2 = self.w1(x, emb)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv2(x, emb)
        x2 = self.w2(x, emb)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv3(x, emb)
        x2 = self.w3(x, emb)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)  # (batch, H, W, width)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x.permute(0, 3, 1, 2)  # (batch, 1, H, W)

class FNO(nn.Module):
    def __init__(self, modes, width, timestep_embedding_dim=33):
        super(FNO, self).__init__()
        self.timestep_embedding_dim = timestep_embedding_dim
        self.conv1 = SimpleBlock2d(modes, modes,  width, timestep_embedding_dim)

    def forward(self, x, t):
        emb = timestep_embedding(t, self.timestep_embedding_dim)
        x = self.conv1(x, emb)
        return x


if __name__ == "__main__":
    # test spectral conv 
    
    model = FNO(modes=16, width=32, timestep_embedding_dim=33)
    x = torch.randn(4, 3, 64, 64)
    t = torch.randint(0,1000,(4,))
    y = model(x,t)
    print(y.shape)

    