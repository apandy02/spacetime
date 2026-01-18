from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LamConfig:
    """
    Latent action model (LAM) architecture and loss settings.
    """
    num_heads: int = 4
    d_model: int = 384
    num_layers: int = 2
    d_linear: int = 1536
    num_discrete_actions: int = 1024
    codebook_dim: int = 128
    patch_size: int = 8
    frame_height: int = 224
    frame_width: int = 224
    num_frames: int = 8
    num_linear_layers: int = 2
    num_groups: int = 8
    dropout: float = 0.1
    beta: float = 0.25


@dataclass
class DynamicsConfig:
    """
    Dynamics model config (model params only for now).
    """
    n_heads: int = 4
    d_model: int = 384
    n_layers: int = 2
    d_linear: int = 1536
    action_codebook_size: int = 1024
    action_dim: int = 128
    token_codebook_size: int = 1024
    token_dim: int = 32
    n_tokens: int = 256
    p_sample: float = 0.2
    n_linear_layers: int = 2
    n_groups: int = 8
    dropout: float = 0.1
    n_frames: int = 16


@dataclass
class TrainingConfig:
    """
    Training loop and data settings.
    """
    data_root: Path = Path(__file__).resolve().parents[4] / "data/UCF-101-downsized"
    annotation_path: Path = (
        Path(__file__).resolve().parents[4] / "data/ucfTrainTestlist"
    )
    frames_per_clip: int = 8
    step_between_clips: int = 8
    train_batch_size: int = 16
    val_batch_size: int = 4
    num_workers: int = 8
    pin_memory: bool = True
    max_epochs: int = 10
    precision: int = 16


@dataclass
class Hyperparameters:
    """
    Model hyperparameters, grouped by subsystem.
    """
    lam: LamConfig = field(default_factory=LamConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    lambda_reconstruction: float = 1.0


@dataclass
class Config:
    """
    Top-level experiment config.
    """
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hparams: Hyperparameters = field(default_factory=Hyperparameters)
