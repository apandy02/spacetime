from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
import tyro
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
from torchvision.datasets import UCF101

from spacetime.models.latent_actions.training_module import LatentActionTrainingModule
from spacetime.utils import (
    get_logger,
    is_rank_zero,
    maybe_disable_wandb_for_non_zero_ranks,
    maybe_set_wandb_sandbox_key,
)

logger = get_logger("spacetime.latent_actions")


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


@dataclass
class Hyperparameters:
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
class Config:
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
    hparams: Hyperparameters = Hyperparameters()


def run(cfg: Config) -> None:
    maybe_set_wandb_sandbox_key()
    maybe_disable_wandb_for_non_zero_ranks()
    logger.info("Starting latent action training")
    logger.info("Data root: %s", cfg.data_root)
    logger.info("Annotation path: %s", cfg.annotation_path)
    logger.info(
        "Train/val batch sizes: %s/%s", cfg.train_batch_size, cfg.val_batch_size
    )

    train_dataset = UCF101(
        root=str(cfg.data_root),
        annotation_path=str(cfg.annotation_path),
        frames_per_clip=cfg.frames_per_clip,
        step_between_clips=cfg.step_between_clips,
        train=True,
    )
    val_dataset = UCF101(
        root=str(cfg.data_root),
        annotation_path=str(cfg.annotation_path),
        frames_per_clip=cfg.frames_per_clip,
        step_between_clips=cfg.step_between_clips,
        train=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_ucf101,
        pin_memory=cfg.pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_ucf101,
        pin_memory=cfg.pin_memory,
    )

    name = (
        f"latent_actions_layers{cfg.hparams.num_layers}_codebook_dim{cfg.hparams.codebook_dim}_"
        f"actions{cfg.hparams.num_discrete_actions}_heads{cfg.hparams.num_heads}"
    )
    if is_rank_zero():
        csv_logger = CSVLogger(save_dir="lightning_logs", name=name)
        wandb_logger = WandbLogger(project="spacetime", name=name, config=asdict(cfg.hparams))
        loggers = [csv_logger, wandb_logger]
    else:
        loggers = False

    lightning_module = LatentActionTrainingModule(
        num_heads=cfg.hparams.num_heads,
        d_model=cfg.hparams.d_model,
        num_layers=cfg.hparams.num_layers,
        d_linear=cfg.hparams.d_linear,
        num_discrete_actions=cfg.hparams.num_discrete_actions,
        codebook_dim=cfg.hparams.codebook_dim,
        patch_size=cfg.hparams.patch_size,
        frame_height=cfg.hparams.frame_height,
        frame_width=cfg.hparams.frame_width,
        num_frames=cfg.hparams.num_frames,
        num_linear_layers=cfg.hparams.num_linear_layers,
        num_groups=cfg.hparams.num_groups,
        dropout=cfg.hparams.dropout,
        beta=cfg.hparams.beta,
    )

    logger.info(
        "Initializing trainer (max_epochs=%s, precision=%s)",
        cfg.max_epochs,
        cfg.precision,
    )
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
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
