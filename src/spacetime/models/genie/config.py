from dataclasses import dataclass, field
from pathlib import Path

from spacetime.modules.quantizers import QuantizerType


@dataclass
class LamConfig:
    """
    Latent action model (LAM) architecture and loss settings.
    """

    num_heads: int = 8
    d_model: int = 512
    num_layers: int = 8
    d_linear: int = 2048
    num_discrete_actions: int = 6
    codebook_dim: int = 32
    patch_size: int = 4
    frame_height: int = 64
    frame_width: int = 64
    num_frames: int = 16
    num_linear_layers: int = 2
    num_groups: int = 8
    dropout: float = 0.1
    gradient_checkpointing: bool = False
    beta: float = 0.25
    quantizer_type: QuantizerType = QuantizerType.VANILLA
    quantizer_decay: float = 0.985
    quantizer_eps: float = 1e-5
    dead_code_threshold: float = 1e-4
    dead_code_noise: float = 1e-4


@dataclass
class DynamicsConfig:
    """
    Dynamics model (MaskGIT) architecture settings.
    """

    n_heads: int = 8
    d_model: int = 512
    n_layers: int = 12
    d_linear: int = 2048
    n_linear_layers: int = 2
    n_groups: int = 8
    dropout: float = 0.1
    gradient_checkpointing: bool = False
    p_sample: float = 0.2
    sampling_temperature: float = 1.0
    maskgit_steps: int = 25


@dataclass
class TrainingConfig:
    """
    Training loop and data settings.
    """

    shard_dir: Path = Path(__file__).resolve().parents[4] / "data/procgen_heist/shards"
    train_ratio: float = 0.8
    batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    max_epochs: int = 10
    precision: str = "bf16-mixed"
    matmul_precision: str = "high"
    compile_model: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = "default"


@dataclass
class OptimizerConfig:
    """
    AdamW optimizer and scheduler settings.
    """

    max_lr: float = 3e-5
    min_lr: float = 3e-6
    betas: tuple[float, float] = (0.9, 0.9)
    weight_decay: float = 1e-4
    warmup_steps: int = 5_000


@dataclass
class Hyperparameters:
    """
    Model hyperparameters, grouped by subsystem.
    """

    lam: LamConfig = field(default_factory=LamConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lambda_reconstruction: float = 0.01


@dataclass
class Config:
    """
    Top-level experiment config.
    """

    tokenizer_checkpoint: Path
    tokenizer_wandb_path: Path
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hparams: Hyperparameters = field(default_factory=Hyperparameters)
