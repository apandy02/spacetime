from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
import tyro
import wandb
from torch.utils.data import DataLoader, random_split

from spacetime.models.tokenizer.model import QuantizerType
from spacetime.models.tokenizer.training_module import STVQVaeModule
from spacetime.utils.data import ProcgenShardDataset
from spacetime.utils import (
    get_logger,
    is_rank_zero,
    maybe_disable_wandb_for_non_zero_ranks,
    maybe_set_wandb_sandbox_key,
)


logger = get_logger("spacetime.tokenizer")


@dataclass
class Hyperparameters:
    num_heads: int = 8
    d_model: int = 512
    num_layers: int = 4
    d_linear: int = 1536
    codebook_size: int = 1024
    codebook_dim: int = 32
    patch_size: int = 8
    frame_height: int = 64
    frame_width: int = 64
    num_frames: int = 16
    num_linear_layers: int = 2
    num_groups: int = 8
    dropout: float = 0.1
    quantizer_type: str = "ema"
    beta: float = 0.1
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.9)
    weight_decay: float = 1e-4
    warmup_steps: int = 250


@dataclass
class Config:
    shard_dir: Path = Path(__file__).resolve().parents[4] / "data/procgen_heist/shards"
    train_ratio: float = 0.8
    batch_size: int = 48
    num_workers: int = 8
    pin_memory: bool = True
    max_epochs: int = 10
    precision: int = 16
    hparams: Hyperparameters = Hyperparameters()


def run(cfg: Config) -> None:
    maybe_set_wandb_sandbox_key()
    maybe_disable_wandb_for_non_zero_ranks()
    logger.info("Starting tokenizer training")

    shard_dir = cfg.shard_dir
    shard_dir.mkdir(parents=True, exist_ok=True)
    logger.info("loading sharded procgen dataset")
    shard_dataset = ProcgenShardDataset(shard_dir, normalize=True)
    logger.info("total clips loaded: %s", len(shard_dataset))

    train_size = int(cfg.train_ratio * len(shard_dataset))
    val_size = len(shard_dataset) - train_size
    train_dataset, val_dataset = random_split(
        shard_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    if is_rank_zero():
        wandb.init(
            project="genie",
            name=(
                f"tokenizer_ddp_{cfg.hparams.quantizer_type}_"
                f"{cfg.hparams.num_layers}_heads{cfg.hparams.num_heads}"
            ),
            config=cfg.hparams,
        )

    lightning_module = STVQVaeModule(
        num_heads=cfg.hparams.num_heads,
        d_model=cfg.hparams.d_model,
        num_layers=cfg.hparams.num_layers,
        d_linear=cfg.hparams.d_linear,
        codebook_size=cfg.hparams.codebook_size,
        codebook_dim=cfg.hparams.codebook_dim,
        patch_size=cfg.hparams.patch_size,
        frame_height=cfg.hparams.frame_height,
        frame_width=cfg.hparams.frame_width,
        num_frames=cfg.hparams.num_frames,
        num_linear_layers=cfg.hparams.num_linear_layers,
        num_groups=cfg.hparams.num_groups,
        dropout=cfg.hparams.dropout,
        quantizer_type=QuantizerType(cfg.hparams.quantizer_type),
        beta=cfg.hparams.beta,
        lr=cfg.hparams.lr,
        betas=tuple(cfg.hparams.betas),
        weight_decay=cfg.hparams.weight_decay,
        warmup_steps=cfg.hparams.warmup_steps,
    )

    logger.info("Initializing trainer (max_epochs=%s, precision=%s)", cfg.max_epochs, cfg.precision)
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        strategy="ddp_find_unused_parameters_true",
    )
    logger.info("Starting fit loop")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    wandb.finish()
    logger.info("Training complete")


def main() -> None:
    cfg = tyro.cli(Config)
    run(cfg)


if __name__ == "__main__":
    main()
