"""
trainer module for Genie: Generative Interactive Environments
"""
import lightning as L
import torch
import tyro
from torch.utils.data import DataLoader, random_split

from spacetime.models.genie.config import Config
from spacetime.models.genie.training_module import GenieTrainingModule
from spacetime.utils import (get_logger,
                             maybe_disable_wandb_for_non_zero_ranks,
                             maybe_set_wandb_sandbox_key,
                             setup_wandb_csv_loggers)
from spacetime.utils.data import ProcgenShardDataset

logger = get_logger("spacetime.genie")


def setup_loggers(cfg: Config) -> list | bool:
    """
    Configure WandB and CSV loggers for rank-zero process.
    """
    hp = cfg.hparams
    tc = cfg.training
    return setup_wandb_csv_loggers(
        project="spacetime",
        config_parts=[tc, hp],
        effective_batch_size=tc.batch_size,
    )


def run(cfg: Config) -> None:
    maybe_set_wandb_sandbox_key()
    maybe_disable_wandb_for_non_zero_ranks()
    logger.info("Starting latent action training")
    tc = cfg.training
    if tc.matmul_precision:
        torch.set_float32_matmul_precision(tc.matmul_precision)
    logger.info("Shard dir: %s", tc.shard_dir)
    logger.info("Train/val batch size: %s", tc.batch_size)

    tc.shard_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading ProcgenShardDataset from %s", tc.shard_dir)
    shard_dataset = ProcgenShardDataset(tc.shard_dir, normalize=True)
    logger.info("Loaded dataset with %d samples", len(shard_dataset))

    train_size = int(tc.train_ratio * len(shard_dataset))
    val_size = len(shard_dataset) - train_size
    logger.info("Splitting dataset: %d train, %d val", train_size, val_size)
    train_dataset, val_dataset = random_split(
        shard_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info("Creating train and validation dataloaders")
    dataloader_kwargs = {
        "batch_size": tc.batch_size,
        "num_workers": tc.num_workers,
        "pin_memory": tc.pin_memory,
    }
    if tc.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = tc.persistent_workers
        dataloader_kwargs["prefetch_factor"] = tc.prefetch_factor

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )

    logger.info("Dataloaders created: %d training batches, %d validation batches",
                len(train_dataloader), len(val_dataloader))

    loggers = setup_loggers(cfg)

    lightning_module = GenieTrainingModule(cfg)
    if tc.compile_model:
        lightning_module.genie_model = torch.compile(
            lightning_module.genie_model,
            backend=tc.compile_backend,
            mode=tc.compile_mode,
        )

    logger.info(
        "Initializing trainer (max_epochs=%s, precision=%s)", tc.max_epochs, tc.precision
    )
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
