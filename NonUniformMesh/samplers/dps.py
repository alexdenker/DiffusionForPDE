import torch
from omegaconf import DictConfig

from tqdm import tqdm 

class DPS():
    def __init__(self, model, forward_operator, cfg: DictConfig):
        self.model = model 
        self.sde = model.sde # TODO: Write a model wrapper which has sde as an attribute
        self.noise_sampler = model.noise_sampler # TODO: Write a model wrapper which has noise_sampler as an attribute
        self.forward_operator = forward_operator
        
        self.cfg = cfg 

    def sample(self, y, pos, ts, **kwargs):
        """
        y: observations, [B, M]
        pos: positions, [B, N, D]        
        ts: timesteps, [T] (forward in time)
        """

        batch_size = y.shape[0]

        delta_t = ts[1] - ts[0]
        xt = self.noise_sampler.sample(batch_size).unsqueeze(1) # N(0,C)

        for ti in tqdm(reversed(ts), total=len(ts)):
            #print(ti)
            t = torch.ones(batch_size).to(xt.device)* ti

            xt.requires_grad_()
            score, x0_pred = self.model(xt, t, pos)

            beta_t = self.sde.beta_t(t).view(-1, 1, 1)
            noise = self.noise_sampler.sample(batch_size).unsqueeze(1) # N(0,C)
            
            Hx = self.forward_operator.forward(x0_pred.squeeze(1).unsqueeze(-1)).squeeze(-1)
            #print("Hx: ", Hx.shape, " y: ", y.shape)
            mat = ((Hx - y).reshape(batch_size, -1) ** 2).sum()

            mat_norm = ((y - Hx).reshape(batch_size, -1) ** 2).sum(dim=1).sqrt().detach()
            grad_term = torch.autograd.grad(mat, xt, retain_graph=True)[0]
            coeff = self.cfg.gamma / mat_norm.reshape(-1, 1, 1)
            xt = xt + beta_t/2.0 * delta_t*xt + beta_t* delta_t * score + beta_t.sqrt()*delta_t.sqrt() * noise 
            xt = xt - grad_term * coeff
            xt = xt.detach()


        return xt 