import torch
import torch.nn as nn 
import torch.fft as fft
import torch.nn.functional as F
import math 


def shape_to_fourier(xy, num_modes):
    """
    xy: (N,2) tensor of 2D points
    num_modes: number of low-frequency modes to keep (excluding DC)
    returns: truncated Fourier coefficients including positive and negative modes
    """
    z = xy[...,0] + 1j*xy[...,1]
    Z = fft.fft(z, dim=-1)

    # Keep first num_modes positive and negative frequencies
    Z_pos = Z[:, 1:num_modes+1]         # positive frequencies
    Z_neg = Z[:, -num_modes:]           # negative frequencies
    # do not use the first DC component (Z[0]), this should always be zero for centered shapes

    Z_trunc = torch.cat([Z_pos, Z_neg], dim=1)
    return Z_trunc

def fourier_to_shape(Z_trunc, num_points, num_modes):
    """
    Reconstruct shape from truncated Fourier coefficients
    Z_trunc: (batch_size, 2*num_modes,) complex tensor (pos + neg)
    num_points: original number of points
    num_modes: number of low-frequency modes originally kept
    """
    # Zero-pad the rest of the spectrum
    Z_full = torch.zeros(Z_trunc.shape[0], num_points, dtype=torch.complex64, device=Z_trunc.device)
    # do not use the first DC component (Z_full[0]), this should always be zero for centered shapes


    Z_full[:, 1:num_modes+1] = Z_trunc[:, :num_modes]    # positive
    Z_full[:, -num_modes:] = Z_trunc[:, num_modes:]       # negative
    
    z_rec = fft.ifft(Z_full)
    xy_rec = torch.stack((z_rec.real, z_rec.imag), dim=-1)
    return xy_rec

def complex_to_real(Z):
    """Convert complex tensor to real tensor (num_modes*2C)"""
    return torch.stack((Z.real, Z.imag), dim=-1)

def real_to_complex(R, num_modes):
    """Convert flattened real tensor back to complex (pos + neg)"""
    R = R.view(R.shape[0],2*num_modes, 2)
    return R[...,0] + 1j*R[...,1]

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


class SimpleDiffusion2D(nn.Module):
    def __init__(self, n, hidden=128, time_emb_dim=32, max_period=0.1):
        super().__init__()

        self.time_emb_dim = time_emb_dim
        self.max_period = max_period
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )   

        self.net = nn.Sequential(
            nn.Linear(n + time_emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n)
        )

    def forward(self, x, t):
        """
        x: [batch_size, n] input vector
        t: [batch_size,] time
        returns: [batch_size, n] transformed vector
        """
        t_emb = timestep_embedding(t, self.time_emb_dim, self.max_period)
        t_emb = self.time_mlp(t_emb)
        xt = torch.cat([x, t_emb], dim=-1)
        return self.net(xt)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, time_embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2*hidden_dim)
        )
        #self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, t_emb):
        h = self.fc1(F.silu(x))
        scale, mean = self.time_mlp(t_emb).chunk(2, dim=-1)  # [batch, width] each
        h = h * (1 + scale) + mean
        #h = h + self.time_mlp(F.silu(t_emb))
        h = self.fc2(F.silu(h))
        return x + h #self.norm(x + h)


class ScoreNet(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, time_embed_dim=32, num_blocks=6, max_period = 2.0):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.time_embed_dim = time_embed_dim
        self.max_period = max_period

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        # t: (batch,)
        t_emb = timestep_embedding(t, dim=self.time_embed_dim, max_period=self.max_period)
        t_emb = self.time_mlp(t_emb)

        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h, t_emb)

        out = self.output_layer(F.silu(h))
        return out



if __name__ == "__main__":
    num_points = 128
    num_modes = 32

    # Circle shape
    theta = torch.linspace(0, 2*torch.pi, num_points)
    xy = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

    # Fourier transform
    Z_trunc = shape_to_fourier(xy, num_modes)
    Z_real = complex_to_real(Z_trunc).unsqueeze(0)

    print("Z_real shape:", Z_real.shape)  # should be (num_modes*2 + 2,)

    # ---- Network mock ----
    model = SimpleDiffusion2D(n=num_modes*4)

    t = torch.ones((Z_real.shape[0],)) * 0.5  # example time input
    Z_real_transformed = model(Z_real, t)

    print("Z_real_transformed shape:", Z_real_transformed.shape)

    # Convert back to complex
    Z_trunc_trans = real_to_complex(Z_real_transformed, num_modes)

    print("Z_trunc: ", Z_trunc_trans.shape)

    # Reconstruct shape
    xy_rec = fourier_to_shape(Z_trunc_trans, num_points, num_modes)
    print(xy_rec.shape)  # (128,2)
