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
# Fourier layer
################################################################
class VFT:
    def __init__(self, x_positions, y_positions, modes):
        # it is important that positions are scaled between 0 and 2*pi
        #x_positions -= torch.min(x_positions)
        #x_positions = x_positions.clone()
        
        #self.x_positions = x_positions * 6.28 / (torch.max(x_positions))
        #y_positions = y_positions - torch.min(y_positions)
        #y_positions -= torch.min(y_positions)
        
        #self.y_positions = y_positions * 6.28 / (torch.max(y_positions))
        self.x_positions = x_positions * 6.28 / (torch.max(x_positions))
        self.y_positions = y_positions * 6.28 / (torch.max(y_positions))

        self.number_points = x_positions.shape[1]
        self.batch_size = x_positions.shape[0]
        self.modes = modes

        self.X_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float()#.cuda()
        self.Y_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes-1), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float()#.cuda()
        self.X_ = self.X_.to(x_positions.device)
        self.Y_ = self.Y_.to(x_positions.device)
        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        m = (self.modes*2)*(self.modes*2-1)
        X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, (self.modes*2-1), 1)
        Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]).repeat(1, 1, self.modes*2).reshape(self.batch_size,m,self.number_points))
        forward_mat = torch.exp(-1j* (X_mat+Y_mat)) 

        inverse_mat = torch.conj(forward_mat.clone()).permute(0,2,1)

        return forward_mat, inverse_mat

    def forward(self, data):
        data_fwd = torch.bmm(self.V_fwd, data)
        return data_fwd

    def inverse(self, data):
        data_inv = torch.bmm(self.V_inv, data)
        
        return data_inv

class SpectralConv2d_dse (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, emb_dim):
        super(SpectralConv2d_dse, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        # time modulation scaling 
        self.A_real = nn.Parameter(
            self.scale * torch.rand(1, self.modes1, self.modes2,emb_dim, dtype=torch.float))
        self.A_imag = nn.Parameter(
            self.scale * torch.rand(1, self.modes1, self.modes2,emb_dim, dtype=torch.float))

    # Complex multiplication and complex batched multiplications
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, transformer, emb):
        batchsize = x.shape[0]
        num_pts = x.shape[-1]

        x = x.permute(0, 2, 1)
        x = x + 0j
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        #x_ft = transformer.forward(x.cfloat()) #[4, 20, 32, 16]
        x_ft = transformer.forward(x) #[4, 20, 32, 16] 

        x_ft = x_ft.permute(0, 2, 1)
        # out_ft = self.compl_mul1d(x_ft, self.weights3)
        x_ft = torch.reshape(x_ft, (batchsize, self.out_channels, 2*self.modes1, 2*self.modes1-1))

        emb = emb[:,None,None,:]
        phi_real = torch.einsum('bmwc,bmwc->bmw', self.A_real, emb)
        phi_imag = torch.einsum('bmwc,bmwc->bmw', self.A_imag, emb)
        phi_t =  torch.complex(phi_real, phi_imag)

        phi_t_exp = phi_t[:, None, :, :]  
        # # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, 2*self.modes1, self.modes1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes1] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes1], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes1] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes1], self.weights2)

        out_ft[:, :, :self.modes1, :self.modes1] *= phi_t_exp
        out_ft[:, :, -self.modes1:, :self.modes1] *= phi_t_exp

        # #Return to physical space
        x_ft = torch.reshape(out_ft, (batchsize, self.out_channels, 2*self.modes1**2))
        x_ft2 = x_ft[..., 2*self.modes1:].flip(-1, -2).conj()
        x_ft = torch.cat([x_ft, x_ft2], dim=-1)

        x_ft = x_ft.permute(0, 2, 1)
        x = transformer.inverse(x_ft) # x [4, 20, 512, 512]
        x = x.permute(0, 2, 1)
        x = x / x.size(-1) * 2

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

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2,  width, emb_dim):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.emb_dim = emb_dim

        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2, self.emb_dim)
        self.conv1 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2, self.emb_dim)
        self.conv2 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2, self.emb_dim)
        self.conv3 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2, self.emb_dim)
        self.w0 = Conv1DTime(self.width, self.emb_dim)
        self.w1 = Conv1DTime(self.width, self.emb_dim)
        self.w2 = Conv1DTime(self.width, self.emb_dim)
        self.w3 = Conv1DTime(self.width, self.emb_dim)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = nn.GELU()

    def forward(self, x, emb):
        """
        x: [batch_size, num_points, dim]
        emb: [batch_size, c] c is  time step embedding dimension
        """
        x1 = x[:,:,0] - torch.min(x[:,:,0])
        x2 = x[:,:,1] - torch.min(x[:,:,1])
      
        transform = VFT(x1, x2, self.modes1) 

        x = torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1), x[:,:,2:]], dim=-1)
        
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x1 = self.conv0(x, transform, emb)
        x2 = self.w0(x, emb)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv1(x, transform, emb)
        x2 = self.w1(x, emb)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv2(x, transform, emb)
        x2 = self.w2(x, emb)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv3(x, transform, emb)
        x2 = self.w3(x, emb)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x

class FNO_dse(nn.Module):
    def __init__(self, modes, width, timestep_embedding_dim=33):
        super(FNO_dse, self).__init__()
        self.timestep_embedding_dim = timestep_embedding_dim
        self.conv1 = SimpleBlock2d(modes, modes,  width, timestep_embedding_dim)

    def forward(self, x, t):
        emb = timestep_embedding(t, self.timestep_embedding_dim)
        x = self.conv1(x, emb)
        return x


if __name__ == "__main__":

    batch_size = 8
    num_points = 1600

    x = torch.randn(batch_size, num_points, 3)

    model = FNO_dse(modes=16, width=8)
    #model = model.to("cuda")
    #x = x.to("cuda")

    out = model(x)

    print(out.shape)
