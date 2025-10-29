import torch
from omegaconf import DictConfig

from tqdm import tqdm 
from physics.conjugate_gradient import cg 

class DDS():
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

        gamma = 1e-5 #1e-5# 1e-4

        def op(x):
            return self.forward_operator.adjoint(self.forward_operator.forward(x)) + gamma * x

        batch_size = y.shape[0]

        delta_t = ts[1] - ts[0]
        xt = self.noise_sampler.sample(batch_size).unsqueeze(1) # N(0,C)

        for ti in tqdm(reversed(ts), total=len(ts)):
            t = torch.ones(batch_size).to(xt.device)* ti

            with torch.no_grad():
                score, x0_pred = self.model(xt, t, pos)

                x0_pred = torch.clamp(x0_pred, 0, 5).squeeze().unsqueeze(-1)
                x0_y = cg(op, x0_pred, rhs=self.forward_operator.adjoint(y.T), n_iter=4).squeeze()
                x0_y = x0_y.unsqueeze(0).unsqueeze(0)

            m_t = self.sde.mean_t_scaling(t, xt)
            s_t = self.sde.std_t_scaling(t, xt)

            score_y = (m_t * x0_y - xt) / s_t**2

            beta_t = self.sde.beta_t(t).view(-1, 1, 1)
            noise = self.noise_sampler.sample(batch_size).unsqueeze(1) # N(0,C)
            
            xt = xt + beta_t/2.0 * delta_t*xt + beta_t* delta_t * score_y + beta_t.sqrt()*delta_t.sqrt() * noise 

        return xt