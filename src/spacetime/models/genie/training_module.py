import lightning as L
import lpips
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from spacetime.models.dynamics_model.dynamics_model import DynamicsModel
from spacetime.models.genie.config import Config
from spacetime.models.latent_actions.model import LatentActionModel
from spacetime.models.tokenizer.load import \
    load_pretrained_tokenizer_from_checkpoint
from spacetime.modules.quantizers import QuantizerType
from spacetime.utils.vq_losses import compute_lpips_loss, compute_vq_losses


class GenieTrainingModule(L.LightningModule):
    """
    PyTorch Lightning module for end to end training of the Genie model.
    Args:
        cfg: Configuration for the Genie model.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        lam_cfg = cfg.hparams.lam
        self.latent_action_model = LatentActionModel(
            num_heads=lam_cfg.num_heads,
            d_model=lam_cfg.d_model,
            num_layers=lam_cfg.num_layers,
            d_linear=lam_cfg.d_linear,
            num_discrete_actions=lam_cfg.num_discrete_actions,
            codebook_dim=lam_cfg.codebook_dim,
            patch_size=lam_cfg.patch_size,
            frame_height=lam_cfg.frame_height,
            frame_width=lam_cfg.frame_width,
            num_frames=lam_cfg.num_frames,
            num_linear_layers=lam_cfg.num_linear_layers,
            num_groups=lam_cfg.num_groups,
            dropout=lam_cfg.dropout,
        )
        self.lam_beta = lam_cfg.beta
        self.example_clip = None
        self.example_recon = None

        self.lpips_metric = lpips.LPIPS(net="vgg")
        self.lpips_metric.eval()
        for p in self.lpips_metric.parameters():
            p.requires_grad = False

        self.tokenizer, self.tokenizer_cfg = load_pretrained_tokenizer_from_checkpoint(
            self.cfg.tokenizer_checkpoint,
            self.cfg.tokenizer_wandb_path,
        )
        
        self.dynamics_model = DynamicsModel(
            dynamics_cfg=self.cfg.hparams.dynamics,
            lam_cfg=self.cfg.hparams.lam,
            tokenizer_cfg=self.tokenizer_cfg,
        )

    def forward(self, inputs):
        return self.latent_action_model(inputs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.latent_action_model.parameters(), lr=3e-4)

    def training_step(self, batch, batch_idx):
        """
        Notes: 
        
        at each step we will: run the forward pass (this is contained in the genie model wrapper)

        this will return the maskgit output + the lam vq vae decoder and encoder outputs 

        we have a dynamics model loss + the vq vae losses (reconstruction and commitment
        -- abstracted away by the compute_vq_losses function) 

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        x, _ = batch
        x_pred, z_e, z_quantized = self(x)
        loss, losses = compute_vq_losses(
            x_pred,
            x,
            z_e,
            z_quantized,
            indices=None,
            beta=self.lam_beta,
            quantizer_type=QuantizerType.VANILLA,
            codebook_size=self.latent_action_model.codebook.shape[0],
            entropy_weight=0.0,
            lpips_weight=0.0,
            lpips_metric=self.lpips_metric,
        )

        self._log_losses(
            loss,
            losses["recon_loss"],
            losses["codebook_loss"],
            losses["commit_loss"],
            is_training=True,
        )

        if batch_idx == 0:
            self.example_clip = x[:1].detach().cpu()
            self.example_recon = x_pred[:1].detach().cpu()
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_pred, z_e, z_quantized = self(x)
        loss, losses = compute_vq_losses(
            x_pred,
            x,
            z_e,
            z_quantized,
            indices=None,
            beta=self.lam_beta,
            quantizer_type=QuantizerType.VANILLA,
            codebook_size=self.latent_action_model.codebook.shape[0],
            entropy_weight=0.0,
            lpips_weight=0.0,
            lpips_metric=self.lpips_metric,
        )

        self._log_losses(
            loss,
            losses["recon_loss"],
            losses["codebook_loss"],
            losses["commit_loss"],
            is_training=False,
        )

        with torch.no_grad():
            lpips_val = compute_lpips_loss(self.lpips_metric, x_pred, x)
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
