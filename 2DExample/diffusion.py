import numpy as np
import torch


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


class Diffusion:
    def __init__(self, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2, num_diffusion_timesteps=1000, given_betas=None):
        if given_betas is None:
            if beta_schedule == "quad":
                betas = (
                    np.linspace(
                        beta_start**0.5,
                        beta_end**0.5,
                        num_diffusion_timesteps,
                        dtype=np.float64,
                    )
                    ** 2
                )
            elif beta_schedule == "linear":
                betas = np.linspace(
                    beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
                )
            elif beta_schedule == "const":
                betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(
                    num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
                )
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, num_diffusion_timesteps)
                betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
            else:
                raise NotImplementedError(beta_schedule)
            assert betas.shape == (num_diffusion_timesteps,)
            betas = torch.from_numpy(betas)
        else:
            betas = given_betas
        self.betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0).cuda().float()
        self.alphas = (1 - self.betas).cumprod(dim=0).cuda().float()
        self.num_diffusion_timesteps = num_diffusion_timesteps
    
    def alpha(self, t):
        return self.alphas.index_select(0, t+1)

    def beta(self, t):
        return self.betas.index_select(0, t + 1)
    

    def make_spectral_filter(self, height, width, power=2.0, cutoff=None, device=None):
        """Return real spectral filter of shape (1,1,H,W) to multiply FFT(z).
        power controls decay ~ (1+|k|^2)^{-power/2}. cutoff (int) optionally zeroes high freq beyond radius.
        """
        if device is None:
            device = self.device
        
        ky = torch.fft.fftfreq(height, d=1.0/height, device=device)  
        kx = torch.fft.fftfreq(width, d=1.0/width, device=device)
        KX, KY = torch.meshgrid(kx, ky, indexing='xy')
        K2 = KX**2 + KY**2
        
        filt = (1.0 + K2)**(-power/2.0)  # shape (W, H) because of indexing='xy'

        # put in (1,1,H,W)
        filt = torch.tensor(filt.T, device=device).unsqueeze(0).unsqueeze(0)  # transpose to (H,W)
        if cutoff is not None:
            filt = torch.fft.fftshift(filt, dim=(-2, -1))  # shift zero freq to center

            # zero out frequencies with radius > cutoff
            center_y = height // 2
            center_x = width // 2
            Y, X = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='xy')
            R = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
            mask = (R <= cutoff).float()
            filt = filt * mask.unsqueeze(0).unsqueeze(0)
            filt = torch.fft.ifftshift(filt, dim=(-2, -1))  # shift back

        filt = filt / torch.norm(filt) * np.sqrt(height * width)  
        return filt

    def colored_noise(self, shape, spectral_filter):
        """Generate real-valued noise field with given spectral_filter.
        shape: (B, C, H, W)
        spectral_filter: (1,1,H,W) real weights in frequency domain to multiply FFT(z)
        """
        # z real gaussian in pixel domain (keeps Hermitian symmetry after FFT)
        device = spectral_filter.device
        z = torch.randn(shape, device=device)
        Z = torch.fft.fft2(z, dim=(-2, -1))
        # multiply by filter (broadcast)
        Z_filtered = Z * spectral_filter
        xi = torch.fft.ifft2(Z_filtered, dim=(-2, -1)).real
        return xi

    def q_sample(self, x0, t_idx, spectral_filter=None):
        """Return x_t sampled from q(x_t | x_0) at time indices t_idx.
        t_idx: LongTensor of shape (B,) with indices in 0..T-1
        spectral_filter: (1,1,H,W) or None (None -> white noise)
        """
        assert x0.dim() == 4, "x0 should be (B,C,H,W)"
        B, C, H, W = x0.shape
        device = x0.device
        sqrt_ab = self.alpha(t_idx.to(device)).sqrt().view(B, 1, 1, 1)
        sqrt_omb = (1 - self.alpha(t_idx.to(device))).sqrt().view(B, 1, 1, 1)

        if spectral_filter is None:
            # white noise in pixel domain
            noise = torch.randn_like(x0, device=device)
        else:
            # ensure spectral_filter is on same device and shape
            sf = spectral_filter.to(device)
            noise = self.colored_noise(x0.shape, sf)

        x_t = sqrt_ab * x0 + sqrt_omb * noise
        return x_t, noise

    def start_sample(self, shape, spectral_filter=None):
        """Sample x_T ~ p(x_T) = N(0, I) or colored noise if spectral_filter is given.
        shape: (B,C,H,W)
        spectral_filter: (1,1,H,W) or None
        """
        device = spectral_filter.device if spectral_filter is not None else 'cpu'
        if spectral_filter is None:
            x_T = torch.randn(shape, device=device)
        else:
            x_T = self.colored_noise(shape, spectral_filter)
        return x_T