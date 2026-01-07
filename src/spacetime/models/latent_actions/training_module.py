import lightning as L
import lpips
import torch

import wandb
from lightning.pytorch.loggers import WandbLogger

from spacetime.models.latent_actions.model import LatentActionModel


class LatentActionTrainingModule(L.LightningModule):
    def __init__(
        self,
        num_heads,
        d_model,
        num_layers,
        d_linear,
        num_discrete_actions,
        codebook_dim,
        patch_size,
        frame_height,
        frame_width,
        num_frames,
        num_linear_layers=2,
        num_groups=8,
        dropout=0.1,
        beta=0.25,
    ):
        super().__init__()
        self.model = LatentActionModel(
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            d_linear=d_linear,
            num_discrete_actions=num_discrete_actions,
            codebook_dim=codebook_dim,
            patch_size=patch_size,
            frame_height=frame_height,
            frame_width=frame_width,
            num_frames=num_frames,
            num_linear_layers=num_linear_layers,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.beta = beta
        self.example_clip = None
        self.example_recon = None

        self.lpips_metric = lpips.LPIPS(net="vgg")
        self.lpips_metric.eval()
        for p in self.lpips_metric.parameters():
            p.requires_grad = False

    def forward(self, inputs):
        return self.model(inputs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=3e-4)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_pred, z_e, z_quantized = self(x)
        recon_loss = torch.nn.functional.mse_loss(x_pred, x)
        codebook_loss = torch.nn.functional.mse_loss(z_quantized, z_e.detach())
        commit_loss = torch.nn.functional.mse_loss(z_e, z_quantized.detach())
        loss = recon_loss + codebook_loss + (self.beta * commit_loss)

        self._log_losses(loss, recon_loss, codebook_loss, commit_loss, is_training=True)

        if batch_idx == 0:
            self.example_clip = x[:1].detach().cpu()
            self.example_recon = x_pred[:1].detach().cpu()
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_pred, z_e, z_quantized = self(x)
        recon_loss = torch.nn.functional.mse_loss(x_pred, x)
        codebook_loss = torch.nn.functional.mse_loss(z_quantized, z_e.detach())
        commit_loss = torch.nn.functional.mse_loss(z_e, z_quantized.detach())
        loss = recon_loss + codebook_loss + (self.beta * commit_loss)

        self._log_losses(
            loss, recon_loss, codebook_loss, commit_loss, is_training=False
        )

        with torch.no_grad():
            # LPIPS expects inputs in [-1,1]; convert if you're in [0,1].
            B, C, F, H, W = x.shape
            to_lpips = lambda t: ((t * 2.0) - 1.0).reshape(B * F, C, H, W)
            lpips_val = self.lpips_metric(to_lpips(x_pred), to_lpips(x)).mean()
        self.log("val_lpips", lpips_val, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if self.example_clip is None:
            return
        if not self.trainer.is_global_zero:
            return

        clip = (self.example_clip.clamp(0, 1) * 255).to(torch.uint8)
        recon = (self.example_recon.clamp(0, 1) * 255).to(torch.uint8)
        video = torch.cat([clip, recon], dim=4)
        video = video.squeeze(0).permute(1, 0, 2, 3)  # (F, C, H, W)

        # Log video to wandb if available
        wandb_logger = self._get_wandb_logger()
        if wandb_logger is not None:
            wandb_logger.experiment.log(
                {"recon_video": wandb.Video(video.squeeze(0), fps=4, format="mp4")},
                step=self.global_step,
            )
        self.example_clip = None
        self.example_recon = None

    def _get_wandb_logger(self) -> WandbLogger | None:
        """Get WandbLogger from trainer's loggers if available."""
        if self.trainer.logger is None:
            return None
        if isinstance(self.trainer.logger, WandbLogger):
            return self.trainer.logger
        if hasattr(self.trainer.logger, "experiment"):
            for logger in self.trainer.loggers:
                if isinstance(logger, WandbLogger):
                    return logger
        return None

    def _log_losses(
        self, loss, recon_loss, codebook_loss, commit_loss, is_training=True
    ):
        prefix = "train" if is_training else "val"
        log_on_step = True if is_training else False
        log_on_epoch = True

        self.log(
            f"{prefix}_loss",
            loss,
            on_step=log_on_step,
            on_epoch=log_on_epoch,
            prog_bar=True,
            logger=True,
            sync_dist=not is_training,
        )
        self.log(
            f"{prefix}_recon_loss",
            recon_loss,
            on_step=log_on_step,
            on_epoch=log_on_epoch,
            prog_bar=False,
            logger=True,
            sync_dist=not is_training,
        )
        self.log(
            f"{prefix}_codebook_loss",
            codebook_loss,
            on_step=log_on_step,
            on_epoch=log_on_epoch,
            prog_bar=False,
            logger=True,
            sync_dist=not is_training,
        )
        self.log(
            f"{prefix}_commit_loss",
            commit_loss,
            on_step=log_on_step,
            on_epoch=log_on_epoch,
            prog_bar=False,
            logger=True,
            sync_dist=not is_training,
        )
