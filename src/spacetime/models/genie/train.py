"""
trainer module for Genie: Generative Interactive Environments
"""

import lightning as L
import torch
import tyro
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from spacetime.models.genie.config import Config
from spacetime.models.genie.training_module import GenieTrainingModule
from spacetime.utils import (
    get_logger,
    maybe_disable_wandb_for_non_zero_ranks,
    maybe_set_wandb_sandbox_key,
    setup_wandb_csv_loggers,
)
from spacetime.utils.data import ProcgenShardDataset

logger = get_logger("spacetime.genie")


def setup_loggers(cfg: Config) -> list | bool:
    """
    Configure WandB and CSV loggers for rank-zero process.
    """
    hp = cfg.hparams
    tc = cfg.training
    return setup_wandb_csv_loggers(
        project="genie",
        config_parts=[tc, hp],
        effective_batch_size=tc.batch_size,
    )


def build_checkpoint_callbacks(cfg: Config) -> list[ModelCheckpoint]:
    """
    Build checkpoint callbacks for step-based persistence and resume safety.
    """
    tc = cfg.training
    step_interval = tc.checkpoint_every_n_train_steps
    if step_interval <= 0:
        logger.info("Step-based checkpointing disabled (every_n_train_steps=%s)", step_interval)
        return []

    monitor_metric: str | None = None
    mode = "min"
    if tc.checkpoint_save_top_k > 1:
        monitor_metric = "step"
        mode = "max"

    checkpoint_callback = ModelCheckpoint(
        filename="epoch={epoch}-step={step}",
        monitor=monitor_metric,
        mode=mode,
        every_n_train_steps=step_interval,
        every_n_epochs=None,
        save_on_train_epoch_end=False,
        save_top_k=tc.checkpoint_save_top_k,
        save_last=tc.checkpoint_save_last,
        save_on_exception=tc.checkpoint_save_on_exception,
    )
    logger.info(
        "Step checkpointing enabled: every %s steps, save_top_k=%s, monitor=%s, save_last=%s, save_on_exception=%s",
        step_interval,
        tc.checkpoint_save_top_k,
        monitor_metric if monitor_metric is not None else "none",
        tc.checkpoint_save_last,
        tc.checkpoint_save_on_exception,
    )
    return [checkpoint_callback]


def run(cfg: Config) -> None:
    maybe_set_wandb_sandbox_key()
    maybe_disable_wandb_for_non_zero_ranks()
    tc = cfg.training
    if tc.matmul_precision:
        torch.set_float32_matmul_precision(tc.matmul_precision)

    logger.info("Starting Genie training (shard_dir=%s, batch_size=%s)", tc.shard_dir, tc.batch_size)

    train_dataloader, val_dataloader = create_dataloaders(cfg)
    lightning_module = build_model(cfg)
    trainer = build_trainer(cfg)

    if tc.ckpt_path is not None:
        logger.info("Resuming from checkpoint: %s", tc.ckpt_path)
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=str(tc.ckpt_path) if tc.ckpt_path is not None else None,
    )
    logger.info("Training complete")


def create_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    tc = cfg.training
    tc.shard_dir.mkdir(parents=True, exist_ok=True)

    shard_dataset = ProcgenShardDataset(tc.shard_dir, normalize=True)
    logger.info("Loaded dataset with %d samples from %s", len(shard_dataset), tc.shard_dir)

    train_size = int(tc.train_ratio * len(shard_dataset))
    val_size = len(shard_dataset) - train_size
    logger.info("Splitting dataset: %d train, %d val", train_size, val_size)
    train_dataset, val_dataset = random_split(
        shard_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    dataloader_kwargs = {
        "batch_size": tc.batch_size,
        "num_workers": tc.num_workers,
        "pin_memory": tc.pin_memory,
    }
    if tc.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = tc.persistent_workers
        dataloader_kwargs["prefetch_factor"] = tc.prefetch_factor

    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)

    logger.info(
        "Dataloaders ready: %d training batches, %d validation batches",
        len(train_dataloader),
        len(val_dataloader),
    )
    return train_dataloader, val_dataloader


def build_model(cfg: Config) -> GenieTrainingModule:
    tc = cfg.training
    lightning_module = GenieTrainingModule(cfg)
    if tc.compile_model:
        lightning_module.genie_model = torch.compile(
            lightning_module.genie_model,
            backend=tc.compile_backend,
            mode=tc.compile_mode,
        )
    return lightning_module


def build_trainer(cfg: Config) -> L.Trainer:
    tc = cfg.training
    strategy = _select_training_strategy()
    logger.info("Initializing trainer (max_epochs=%s, precision=%s)", tc.max_epochs, tc.precision)
    logger.info("Trainer strategy: %s", strategy)
    return L.Trainer(
        max_epochs=tc.max_epochs,
        precision=tc.precision,
        strategy=strategy,
        logger=setup_loggers(cfg),
        callbacks=build_checkpoint_callbacks(cfg),
    )


def _select_training_strategy() -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return "ddp_find_unused_parameters_true"
    return "auto"


def main() -> None:
    cfg = tyro.cli(Config)
    run(cfg)


if __name__ == "__main__":
    main()
