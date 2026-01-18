from dataclasses import asdict

import lightning as L
import torch
import torch.nn.functional as F
import tyro
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
from torchvision.datasets import UCF101

from spacetime.models.genie.config import Config
from spacetime.models.genie.training_module import GenieTrainingModule
from spacetime.utils import (
    get_logger,
    is_rank_zero,
    maybe_disable_wandb_for_non_zero_ranks,
    maybe_set_wandb_sandbox_key,
)

logger = get_logger("spacetime.genie")


def collate_ucf101(batch):
    xs, ys = [], []
    for v, _, l in batch:
        v = v.permute(0, 3, 1, 2)
        v = v.float() / 255.0
        v = F.interpolate(v, size=(224, 224), mode="bilinear", align_corners=False)
        v = v.permute(1, 0, 2, 3).contiguous()
        xs.append(v.clone())
        ys.append(int(l))
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)


def run(cfg: Config) -> None:
    maybe_set_wandb_sandbox_key()
    maybe_disable_wandb_for_non_zero_ranks()
    logger.info("Starting latent action training")
    tc = cfg.training
    lam_cfg = cfg.hparams.lam
    logger.info("Data root: %s", tc.data_root)
    logger.info("Annotation path: %s", tc.annotation_path)
    logger.info(
        "Train/val batch sizes: %s/%s", tc.train_batch_size, tc.val_batch_size
    )

    train_dataset = UCF101(
        root=str(tc.data_root),
        annotation_path=str(tc.annotation_path),
        frames_per_clip=tc.frames_per_clip,
        step_between_clips=tc.step_between_clips,
        train=True,
    )
    val_dataset = UCF101(
        root=str(tc.data_root),
        annotation_path=str(tc.annotation_path),
        frames_per_clip=tc.frames_per_clip,
        step_between_clips=tc.step_between_clips,
        train=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=tc.train_batch_size,
        shuffle=True,
        num_workers=tc.num_workers,
        collate_fn=collate_ucf101,
        pin_memory=tc.pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=tc.val_batch_size,
        shuffle=False,
        num_workers=tc.num_workers,
        collate_fn=collate_ucf101,
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
    logger.info("Train clips: %s | Val clips: %s", len(train_dataset), len(val_dataset))
