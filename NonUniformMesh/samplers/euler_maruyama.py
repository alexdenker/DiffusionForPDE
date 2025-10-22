import torch
from omegaconf import DictConfig

from tqdm import tqdm 

class EulerMaruyama():
    def __init__(self, model, cfg: DictConfig):
        self.model = model 
        self.sde = model.sde
        self.noise_sampler = model.noise_sampler 
        
        self.cfg = cfg 

    def sample(self, pos, ts, **kwargs):
        """
        pos: positions, [B, N, D]        
        ts: timesteps, [T] (forward in time)
        """

        batch_size = pos.shape[0]

        delta_t = ts[1] - ts[0]
        xt = self.noise_sampler.sample(batch_size).unsqueeze(1) # N(0,C)

        for ti in tqdm(reversed(ts), total=len(ts)):
            t = torch.ones(batch_size).to(xt.device)* ti

            with torch.no_grad():
                score, _ = self.model(xt, t, pos)

            beta_t = self.sde.beta_t(t).view(-1, 1, 1)
            noise = self.noise_sampler.sample(batch_size).unsqueeze(1) # N(0,C)
            
            xt = xt + beta_t/2.0 * delta_t*xt + beta_t* delta_t * score + beta_t.sqrt()*delta_t.sqrt() * noise 

        return xt 