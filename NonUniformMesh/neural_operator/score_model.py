"""
Wrapper for the time-dependent neural operator.

"""


import torch
import torch.nn as nn 
from omegaconf import DictConfig

from .sde import OU

class ScoreModel():
    def __init__(self, model: nn.Module, sde: OU, noise_sampler, cfg: DictConfig):
        self.model = model
        self.sde = sde
        self.noise_sampler = noise_sampler
        self.cfg = cfg

    def __call__(self, xt, t, pos):
        # Returns both the score function and the predicted x0.
        
        inp = torch.cat([pos.permute(0, 2, 1), xt], dim=1)

        if self.cfg.model.precond_last_layer:
            var_factor = self.sde.cov_t_scaling(t, xt)
            pred = self.model(inp, t, pos.unsqueeze(1)) / var_factor
        
        if self.cfg.model.model_type == "raw":  # the network output is nabla log p
            score = self.noise_sampler.apply_C(pred.squeeze()).unsqueeze(1)
        elif self.cfg.model.model_type == "C_sqrt": # the network output is C^{1/2} nabla log p 
            score = self.noise_sampler.apply_Csqrt(pred.squeeze()).unsqueeze(1)
        elif self.cfg.model.model_type == "C": # the network output is C nabla log p
            score = pred
        else:
            raise NotImplementedError

        mean_t_scaling = self.sde.mean_t_scaling(t, xt)
        cov_t = self.sde.cov_t_scaling(t, xt)
        x0_pred = (xt + score * cov_t**2) / mean_t_scaling

        return score, x0_pred
