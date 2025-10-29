from dataclasses import dataclass, field
from omegaconf import OmegaConf
from enum import Enum

# Define possible model types
class ModelType(str, Enum):
    RAW = "raw" # the network output is nabla log p
    C_SQRT = "C_sqrt" # the network output is C^{1/2} nabla log p 
    C = "C" # the network output is C nabla log p

@dataclass
class ModelConfig:
    # Whether to apply preconditioning to the last layer
    precond_last_layer: bool = True

    # Defines what the model output represents
    model_type: ModelType = ModelType.RAW

    n_layers: int = 6
    modes: int = 16
    width: int = 32
    timestep_embedding_dim: int = 33
    max_period: int = 10

    mesh_name: str = "disk_dense"

@dataclass
class SDEConfig:
    beta_min: float = 0.001
    beta_max: float = 15.0

@dataclass
class NoiseConfig:
    scale: float = 0.6 # scale parameter for RBF kernel
    eps: float = 0.01 # regularisation parameter for inversion (C + eps I)^{-1}

@dataclass
class TrainingConfig:
    num_epochs: int = 1000
    lr: float = 1e-3
    loss_scaling: str = "positive"  # scaling the loss by sigma(t)
    save_dir: str = "exp/NonUniformFNO"
    log_wandb: bool = True
    batch_size: int = 16
    train_on: str = "gaussian_bumps"  # dataset to train on
    
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
