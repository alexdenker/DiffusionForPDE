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



# class for fully nonequispaced 2d points
class VFT:
    def __init__(self, x_positions, y_positions, modes):
        # it is important that positions are scaled between 0 and 2*pi
        x_positions -= torch.min(x_positions)
        self.x_positions = x_positions * 6.28 / (torch.max(x_positions))
        y_positions -= torch.min(y_positions)
        self.y_positions = y_positions * 6.28 / (torch.max(y_positions))
        self.number_points = x_positions.shape[1]
        self.batch_size = x_positions.shape[0]
        self.modes = modes

        self.X_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().cuda()
        self.Y_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes-1), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().cuda()


        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        m = (self.modes*2)*(self.modes*2-1)
        X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, (self.modes*2-1), 1)
        Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]).repeat(1, 1, self.modes*2).reshape(self.batch_size,m,self.number_points))
        forward_mat = torch.exp(-1j* (X_mat+Y_mat)) 

        inverse_mat = torch.conj(forward_mat.clone()).permute(0,2,1)

        return forward_mat, inverse_mat

    def forward(self, data):
        """
        data: (batch_size, num_mesh_points, num_features) 
        """

        data_fwd = torch.bmm(self.V_fwd, data)
        return data_fwd

    def inverse(self, data):
        """
        data: (batch_size, num_modes, num_features) 
        """

        data_inv = torch.bmm(self.V_inv, data)
        
        return data_inv
    

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, emb_dim):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does NFFT, linear transform, and Adjoint NFFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, self.modes, dtype=torch.cfloat))
        
        # time modulation scaling 
        self.emd_modulation1 = nn.Parameter(
            self.scale * torch.rand(1, self.modes, self.modes, emb_dim, dtype=torch.cfloat))
        self.emd_modulation2 = nn.Parameter(
                    self.scale * torch.rand(1, self.modes, self.modes, emb_dim, dtype=torch.cfloat))


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, nuft, emb):
        """
        Inputs:
        x (torch.Tensor): features with shape [batch_size, in_channels, num_points]
        nuft: class VFT
        emb (torch.Tensor): time embedding vector [batch_size, emb_dim]

        Output:
        transformed features, shape [batch_size, out_channels, num_points]
        """
        batchsize = x.shape[0]

        x_nfft = nuft.forward(x.permute(0, 2,1).cfloat())
        x_ft = x_nfft.permute(0, 2, 1)
        x_ft = torch.reshape(x_ft, (x.shape[0], self.out_channels, 2*self.modes, 2*self.modes-1))


        emb = emb[:,None,None,:] + 0j 
        phi1 = torch.einsum('bmwc,bmwc->bmw', self.emd_modulation1, emb)
        phi2 = torch.einsum('bmwc,bmwc->bmw', self.emd_modulation2, emb)

        # Spectral convolution
        out_ft = torch.zeros(batchsize, self.out_channels, 2*self.modes, self.modes, dtype=torch.cfloat, device=x.device)
        out_ft1 = self.compl_mul2d(x_ft[:, :, :self.modes, :self.modes], self.weights1)
        out_ft2 = self.compl_mul2d(x_ft[:, :, -self.modes:, :self.modes], self.weights2)
        out_ft[:, :, :self.modes, :self.modes] = out_ft1
        out_ft[:, :, -self.modes:, :self.modes] = out_ft2

        # Modulation
        modified1 = torch.einsum('bij,bfij->bfij', phi1, out_ft[:, :, :self.modes, :self.modes])
        modified2 = torch.einsum('bij,bfij->bfij', phi2, out_ft[:, :, -self.modes:, :self.modes])

        # Build output tensor safely
        out_ft_mod = out_ft.clone()
        out_ft_mod[:, :, :self.modes, :self.modes] = modified1
        out_ft_mod[:, :, -self.modes:, :self.modes] = modified2


        x_ft = torch.reshape(out_ft_mod, (batchsize, self.out_channels, 2*self.modes**2))
        x_ft2 = x_ft[..., 2*self.modes:].flip(-1, -2).conj()
        x_ft = torch.cat([x_ft, x_ft2], dim=-1)

        x_ft = x_ft.permute(0, 2, 1)
        x = nuft.inverse(x_ft) 
        x = x.permute(0, 2, 1)
        x = x / x.size(-1) * 2


        #x_nfft = self.compl_mul2d(x_nfft, self.weights)
        #x_nfft = torch.einsum('bij,bfij->bfij', phi, x_nfft)

        #x_ifft = self.nfft(pos, x_nfft) 
        #x_ifft = x_ifft / x_ifft.size(-1) * 2
        
        return x.real



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

        self.spectral_conv = SpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes1, emb_dim=self.emb_dim)
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
        self.modes = modes 

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
        nuft = VFT(pos[:,0,:,0], pos[:,0,:,1], modes=self.modes)


        emb = timestep_embedding(t, self.timestep_embedding_dim, max_period=self.max_period)
        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1).contiguous()
        for layer in self.layers:
            x = layer(x, nuft, emb)

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