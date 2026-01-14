from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightning as L
import torch
import tyro
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, random_split

from spacetime.models.tokenizer.config import Hyperparameters, TrainingConfig
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
    num_workers: int = 8
    pin_memory: bool = True
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hparams: Hyperparameters = field(default_factory=Hyperparameters)


def create_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """
    Load dataset, split into train/val, and create dataloaders.

    Args:
        cfg: Training configuration with shard_dir, train_ratio, and dataloader settings.

    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
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
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return train_dataloader, val_dataloader


def setup_loggers(cfg: Config) -> list | bool:
    """
    Configure WandB and CSV loggers for rank-zero process.
    """
    hp = cfg.hparams
    tc = cfg.training
    if is_rank_zero():
        wandb_config = {
            "effective_batch_size": tc.batch_size * tc.accumulate_grad_batches,
            **asdict(tc),
            **asdict(hp),
        }
        wandb_logger = WandbLogger(project="genie", config=wandb_config)
        run_id = wandb_logger.experiment.id
        csv_logger = CSVLogger(save_dir="lightning_logs", name=run_id, flush_logs_every_n_steps=1)
        return [csv_logger, wandb_logger]
    return False


def run(cfg: Config) -> None:
    """
    Run the tokenizer training pipeline.

    Orchestrates data loading, logger setup, model initialization, and training
    using PyTorch Lightning with Weights & Biases logging.

    Args:
        cfg: Training configuration including hyperparameters, data paths, and
            training settings.
    """
    maybe_set_wandb_sandbox_key()
    maybe_disable_wandb_for_non_zero_ranks()
    logger.info("Starting tokenizer training")

    train_dataloader, val_dataloader = create_dataloaders(cfg)
    loggers = setup_loggers(cfg)

    tc = cfg.training
    total_steps = tc.max_epochs * len(train_dataloader)
    lightning_module = STVQVaeModule(cfg=cfg.hparams, total_steps=total_steps)
    if tc.compile_model:
        lightning_module.model = torch.compile(lightning_module.model)

    logger.info(
        "Initializing trainer (max_epochs=%s, precision=%s)",
        tc.max_epochs,
        tc.precision,
    )
    trainer = L.Trainer(
        max_epochs=tc.max_epochs,
        precision=tc.precision,
        strategy="auto",
        gradient_clip_val=1.0,
        accumulate_grad_batches=tc.accumulate_grad_batches,
        logger=loggers,
    )
    logger.info("Starting fit loop")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=str(tc.ckpt_path) if tc.ckpt_path is not None else None,
    )

    logger.info("Training complete")


def main() -> None:
    cfg = tyro.cli(Config)
    run(cfg)


if __name__ == "__main__":
    main()
