from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightning as L
import torch
import tyro
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, random_split

from spacetime.models.tokenizer.config import Hyperparameters
from spacetime.models.tokenizer.training_module import STVQVaeModule
from spacetime.utils import (
    get_logger,
    is_rank_zero,
    maybe_disable_wandb_for_non_zero_ranks,
    maybe_set_wandb_sandbox_key,
)
from spacetime.utils.data import ProcgenShardDataset

logger = get_logger("spacetime.tokenizer")


@dataclass
class Config:
    shard_dir: Path = Path(__file__).resolve().parents[4] / "data/procgen_heist/shards"
    train_ratio: float = 0.8
    batch_size: int = 48
    num_workers: int = 8
    pin_memory: bool = True
    max_epochs: int = 100
    precision: str = "bf16-mixed"
    ckpt_path: Path | None = None
    hparams: Hyperparameters = field(default_factory=Hyperparameters)


def run(cfg: Config) -> None:
    """
    Run the tokenizer training pipeline.
    
    Sets up the Procgen shard dataset, splits into train/val sets, initializes
    the STVQVae model with configured hyperparameters, and trains using PyTorch
    Lightning with Weights & Biases logging. Supports DDP training and automatic
    checkpoint resumption.
    
    Args:
        cfg: Training configuration including hyperparameters, data paths, and
            training settings.
    """
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

    hp = cfg.hparams
    is_ema = hp.quantizer.type.value == "ema"
    quant_suffix = f"_ema_d{hp.quantizer.decay}" if is_ema else "_vanilla"
    lr_suffix = (
        f"_lr{hp.optimizer.lr}" if is_ema else f"_lr{hp.optimizer.lr}_lq{hp.optimizer.lr_quant}"
    )
    name = (
        f"tokenizer{quant_suffix}_L{hp.model.num_layers}_H{hp.model.num_heads}"
        f"{lr_suffix}_b{hp.beta.start}to{hp.beta.end}"
    )
    if is_rank_zero():
        csv_logger = CSVLogger(save_dir="lightning_logs", name=name)
        wandb_logger = WandbLogger(project="genie", name=name, config=asdict(hp))
        loggers = [csv_logger, wandb_logger]
    else:
        loggers = False

    total_steps = cfg.max_epochs * len(train_dataloader)
    lightning_module = STVQVaeModule(cfg=hp, total_steps=total_steps)

    logger.info(
        "Initializing trainer (max_epochs=%s, precision=%s)",
        cfg.max_epochs,
        cfg.precision,
    )
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        strategy="ddp_find_unused_parameters_true",
        gradient_clip_val=1.0,
        logger=loggers,
    )
    logger.info("Starting fit loop")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=str(cfg.ckpt_path) if cfg.ckpt_path is not None else None,
    )

    logger.info("Training complete")


def main() -> None:
    cfg = tyro.cli(Config)
    run(cfg)


if __name__ == "__main__":
    main()
