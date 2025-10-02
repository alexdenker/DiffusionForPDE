import torch
import torch.nn as nn
import math 

from simple_torch_NFFT import NFFT


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


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, emb_dim):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does NFFT, linear transform, and Adjoint NFFT.    
        """

        self.nfft = NFFT((modes1, modes2))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        
        # time modulation scaling 
        self.emd_modulation = nn.Parameter(
            self.scale * torch.rand(1, self.modes1, self.modes2, emb_dim, dtype=torch.cfloat))


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, pos, emb):
        """
        Inputs:
        x (torch.Tensor): features with shape [batch_size, in_channels, num_points]
        pos (torch.Tensor): grid points [batch_size, 1, num_points, 2] (have to be scaled in [-0.5, 0.5] for the nfft to work) 
        emb (torch.Tensor): time embedding vector [batch_size, emb_dim]

        Output:
        transformed features, shape [batch_size, out_channels, num_points]
        """

        x_nfft = self.nfft.adjoint(pos, x)

        emb = emb[:,None,None,:] + 0j 
        phi = torch.einsum('bmwc,bmwc->bmw', self.emd_modulation, emb)

        x_nfft = self.compl_mul2d(x_nfft, self.weights)
        x_nfft = torch.einsum('bij,bfij->bfij', phi, x_nfft)

        x_ifft = self.nfft(pos, x_nfft) 
        x_ifft = x_ifft / x_ifft.size(-1) * 2
        
        return x_ifft.real



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
    
        gamma, beta = self.time_mlp(emb).chunk(2, dim=-1)  # [batch, width] each
        x = self.w(x) * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        return x

class NUFNOBlock(nn.Module):
    def __init__(self, modes1, modes2,  width, emb_dim):
        super(NUFNOBlock, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.emb_dim = emb_dim

        self.spectral_conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.emb_dim)
        self.conv1d = Conv1DTime(self.width, self.emb_dim)

        self.act = nn.SiLU()

    def forward(self, x, pos, emb):
        """
        x: [batch_size, num_points, dim] 
        emb: [batch_size, c] c is  time step embedding dimension
        pos: [batch_size, 1, num_points, 2] grid points
        """

        x1 = self.spectral_conv(x, pos, emb)
        x2 = self.conv1d(x, emb)
        x = x1 + x2
        x = self.act(x)

        return x



class NUFNO(nn.Module):
    def __init__(self, n_layers, modes, width, in_channels=1, timestep_embedding_dim=33, max_period=10000):
        super(NUFNO, self).__init__()
        self.timestep_embedding_dim = timestep_embedding_dim
        self.max_period = max_period

        self.fc0 = nn.Linear(in_channels, width)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = nn.GELU()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(NUFNOBlock(modes, modes, width, timestep_embedding_dim))

    def forward(self, x, t, pos):
        """
        x: [batch_size, num_points, dim] 
        t: [batch_size,] 
        pos: [batch_size, 1, num_points, 2] grid points
        """

        emb = timestep_embedding(t, self.timestep_embedding_dim, max_period=self.max_period)
        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1).contiguous()
        for layer in self.layers:
            x = layer(x, pos, emb)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x


if __name__ == "__main__":
    
    batch_size = 8
    num_points = 1600




    model = NUFNO(n_layers=4, modes=12, width=32, timestep_embedding_dim=33)
    model.to("cuda")
    pos = torch.randn(batch_size, 1, num_points, 2).to("cuda")
    t = torch.rand(batch_size).to("cuda")
    x = torch.randn(batch_size, 1, num_points).to("cuda")
    print("in: ", x.shape)
    out = model(x, t, pos)

    print("out: ", out.shape)
    print(out)