from dataclasses import dataclass, field

from spacetime.modules.quantizers import QuantizerType


@dataclass
class ModelArchConfig:
    """
    Transformer and patch embedding architecture.
    """
    num_heads: int = 8
    d_model: int = 512
    num_layers: int = 8
    d_linear: int = 1536
    num_linear_layers: int = 2
    num_groups: int = 8
    dropout: float = 0.1
    patch_size: int = 8
    frame_height: int = 64
    frame_width: int = 64
    num_frames: int = 16


@dataclass
class QuantizerConfig:
    """
    Vector quantization settings.
    """
    type: QuantizerType = QuantizerType.EMA
    codebook_size: int = 1024
    codebook_dim: int = 32
    decay: float = 0.985
    eps: float = 1e-5
    dead_code_threshold: float = 1e-4
    dead_code_noise: float = 1e-4


@dataclass
class BetaScheduleConfig:
    """
    Commitment loss beta schedule (warmup → peak → decay).
    """
    start: float = 0.05
    end: float = 0.25
    final: float = 0.01
    warmup_steps: int = 10_000
    decay_steps: int = 10_000


@dataclass
class OptimizerConfig:
    """
    AdamW optimizer and LR scheduler settings.
    """
    lr: float = 3e-4
    lr_quant: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.9)
    weight_decay: float = 1e-4
    warmup_steps: int = 10_000


@dataclass
class LossConfig:
    """
    Auxiliary loss weights.
    """
    entropy_weight: float = 0.0
    lpips_weight: float = 0.0


@dataclass
class Hyperparameters:
    """
    All model hyperparameters, hierarchically organized.
    
    Groups:
        - model: Transformer architecture
        - quantizer: Vector quantization settings
        - beta: Commitment loss schedule
        - optimizer: AdamW and scheduler settings
        - loss: Auxiliary loss weights
    """
    model: ModelArchConfig = field(default_factory=ModelArchConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    beta: BetaScheduleConfig = field(default_factory=BetaScheduleConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
