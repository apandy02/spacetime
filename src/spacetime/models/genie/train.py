from dataclasses import asdict

import lightning as L
import torch
import tyro
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, random_split

from spacetime.models.genie.config import Config
from spacetime.models.genie.training_module import GenieTrainingModule
from spacetime.utils import (get_logger, is_rank_zero,
                             maybe_disable_wandb_for_non_zero_ranks,
                             maybe_set_wandb_sandbox_key)
from spacetime.utils.data import ProcgenShardDataset

logger = get_logger("spacetime.genie")


def run(cfg: Config) -> None:
    maybe_set_wandb_sandbox_key()
    maybe_disable_wandb_for_non_zero_ranks()
    logger.info("Starting latent action training")
    tc = cfg.training
    lam_cfg = cfg.hparams.lam
    logger.info("Shard dir: %s", tc.shard_dir)
    logger.info("Train/val batch size: %s", tc.batch_size)

    tc.shard_dir.mkdir(parents=True, exist_ok=True)
    shard_dataset = ProcgenShardDataset(tc.shard_dir, normalize=True)
    train_size = int(tc.train_ratio * len(shard_dataset))
    val_size = len(shard_dataset) - train_size
    train_dataset, val_dataset = random_split(
        shard_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=tc.batch_size,
        shuffle=True,
        num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=tc.batch_size,
        shuffle=False,
        num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
    )

    if is_rank_zero():
        # Let wandb generate a random name, then use its run ID for CSV logger
        wandb_logger = WandbLogger(project="spacetime", config=asdict(lam_cfg))
        run_id = wandb_logger.experiment.id
        csv_logger = CSVLogger(save_dir="lightning_logs", name=run_id)
        loggers = [csv_logger, wandb_logger]
    else:
        loggers = False

    lightning_module = GenieTrainingModule(cfg)

    logger.info("Initializing trainer (max_epochs=%s, precision=%s)", tc.max_epochs, tc.precision)
    trainer = L.Trainer(
        max_epochs=tc.max_epochs,
        precision=tc.precision,
        strategy="ddp_find_unused_parameters_true",
        logger=loggers,
    )
    logger.info("Starting fit loop")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    logger.info("Training complete")


def main() -> None:
    cfg = tyro.cli(Config)
    run(cfg)


if __name__ == "__main__":
    main()
