from typing import Any, Tuple

import lightning as L
import lpips
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger

import wandb
from spacetime.models.genie.config import Config
from spacetime.models.genie.model import GenieModel
from spacetime.models.tokenizer.load import load_pretrained_tokenizer_from_checkpoint
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
        self.lam_beta = lam_cfg.beta
        self.lambda_reconstruction = cfg.hparams.lambda_reconstruction
        self.example_clip = None
        self.example_recon = None
        self.example_dynamics_recon = None

        self.lpips_metric = lpips.LPIPS(net="vgg")
        self.lpips_metric.eval()
        for p in self.lpips_metric.parameters():
            p.requires_grad = False

        self.tokenizer, self.tokenizer_cfg = load_pretrained_tokenizer_from_checkpoint(
            self.cfg.tokenizer_checkpoint,
            self.cfg.tokenizer_wandb_path,
        )

        self.genie_model = GenieModel(
            lam_cfg=lam_cfg,
            dyn_cfg=self.cfg.hparams.dynamics,
            tokenizer_cfg=self.tokenizer_cfg,
            pretrained_tokenizer=self.tokenizer,
        )

    def forward(self, inputs):
        return self.genie_model(inputs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.genie_model.parameters(), lr=3e-4)

    def training_step(self, batch: Tuple[torch.Tensor, Any], batch_idx: int) -> torch.Tensor:
        """
        Notes:

        at each step we will: run the forward pass (this is contained in the genie model wrapper)

        this will return the maskgit output + the lam vq vae decoder and encoder outputs

        we have a dynamics model loss (cross entropy on token indices) + the vq vae losses 
        (reconstruction and commitment -- abstracted away by the compute_vq_losses function)

        Args:
            batch: Tuple[torch.Tensor, Any]
            batch_idx: int - Index of the current batch within the epoch.

        Returns:
            torch.Tensor: The computed total loss for the current training step.
        """
        x, _ = batch
        genie_output = self(x)
        loss, losses = compute_vq_losses(
            x_pred=genie_output.lam_reconstruction,
            x=x,
            z_e=genie_output.z_e,
            z_quantized=genie_output.actions,
            indices=None,
            beta=self.lam_beta,
            quantizer_type=self.cfg.hparams.lam.quantizer_type,
            codebook_size=self.genie_model.lam.vector_quantizer.codebook_size,
            entropy_weight=0.0,
            lpips_weight=0.0,
            lpips_metric=self.lpips_metric,
        )
        dynamics_loss = self._compute_dynamics_loss(
            genie_output.output_tokens, genie_output.token_indices
        )
        total_loss = (self.lambda_reconstruction * loss) + dynamics_loss

        self._log_losses(
            total_loss,
            losses["recon_loss"],
            losses["codebook_loss"],
            losses["commit_loss"],
            dynamics_loss,
            is_training=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Run a validation forward pass and log reconstruction, codebook,
        commitment, and dynamics losses, plus LPIPS and example reconstructions.
        """
        x, _ = batch
        genie_output = self(x)
        loss, losses = compute_vq_losses(
            x_pred=genie_output.lam_reconstruction,
            x=x,
            z_e=genie_output.z_e,
            z_quantized=genie_output.actions,
            indices=None,
            beta=self.lam_beta,
            quantizer_type=self.cfg.hparams.lam.quantizer_type,
            codebook_size=self.genie_model.lam.vector_quantizer.codebook_size,
            entropy_weight=0.0,
            lpips_weight=0.0,
            lpips_metric=self.lpips_metric,
        )
        dynamics_loss = self._compute_dynamics_loss(
            genie_output.output_tokens, genie_output.token_indices
        )
        total_loss = (self.lambda_reconstruction * loss) + dynamics_loss

        self._log_losses(
            total_loss,
            losses["recon_loss"],
            losses["codebook_loss"],
            losses["commit_loss"],
            dynamics_loss,
            is_training=False,
        )

        with torch.no_grad():
            lpips_val = compute_lpips_loss(self.lpips_metric, genie_output.lam_reconstruction, x)
            if batch_idx == 0:
                self.example_clip = x[:1].detach().cpu()
                self.example_recon = genie_output.lam_reconstruction[:1].detach().cpu()
                pred_indices = genie_output.output_tokens.argmax(dim=-1)
                self.example_dynamics_recon = (
                    self.tokenizer.decode_indices(pred_indices[:1]).detach().cpu()
                )
        self.log("val_lpips", lpips_val, prog_bar=False, logger=True, sync_dist=True)
        return total_loss

    def on_validation_epoch_end(self):
        """
        Log example reconstruction and dynamics videos to W&B after validation.
        """
        if self.example_clip is None:
            return
        if not self.trainer.is_global_zero:
            return

        clip = (self.example_clip.clamp(0, 1) * 255).to(torch.uint8)
        # Log video to wandb if available
        wandb_logger = self._get_wandb_logger()
        if wandb_logger is not None and self.example_recon is not None:
            lam_recon = (self.example_recon.clamp(0, 1) * 255).to(torch.uint8)
            lam_video = torch.cat([clip, lam_recon], dim=4)
            lam_video = lam_video.squeeze(0).permute(1, 0, 2, 3)
            wandb_logger.experiment.log(
                {"lam_video": wandb.Video(lam_video, fps=4, format="mp4")},
                step=self.global_step,
            )
        if wandb_logger is not None and self.example_dynamics_recon is not None:
            dyn_recon = (self.example_dynamics_recon.clamp(0, 1) * 255).to(torch.uint8)
            dyn_video = torch.cat([clip, dyn_recon], dim=4)
            dyn_video = dyn_video.squeeze(0).permute(1, 0, 2, 3)
            wandb_logger.experiment.log(
                {"dynamics_video": wandb.Video(dyn_video, fps=4, format="mp4")},
                step=self.global_step,
            )
        self.example_clip = None
        self.example_recon = None
        self.example_dynamics_recon = None

    def _get_wandb_logger(self) -> WandbLogger | None:
        if self.trainer.logger is None:
            return None
        if isinstance(self.trainer.logger, WandbLogger):
            return self.trainer.logger
        if hasattr(self.trainer.logger, "experiment"):
            for logger in self.trainer.loggers:
                if isinstance(logger, WandbLogger):
                    return logger
        return None
    
    def _compute_dynamics_loss(
        self, output_logits: torch.Tensor, target_indices: torch.Tensor
    ) -> torch.Tensor:
        logits = output_logits.reshape(-1, output_logits.shape[-1])
        targets = target_indices.reshape(-1)
        return F.cross_entropy(logits, targets)

    def _log_losses(
        self,
        loss,
        recon_loss,
        codebook_loss,
        commit_loss,
        dynamics_loss,
        is_training=True,
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
        if recon_loss is not None:
            self.log(
                f"{prefix}_recon_loss",
                recon_loss,
                on_step=log_on_step,
                on_epoch=log_on_epoch,
                prog_bar=False,
                logger=True,
                sync_dist=not is_training,
            )
        if codebook_loss is not None:
            self.log(
                f"{prefix}_codebook_loss",
                codebook_loss,
                on_step=log_on_step,
                on_epoch=log_on_epoch,
                prog_bar=False,
                logger=True,
                sync_dist=not is_training,
            )
        if commit_loss is not None:
            self.log(
                f"{prefix}_commit_loss",
                commit_loss,
                on_step=log_on_step,
                on_epoch=log_on_epoch,
                prog_bar=False,
                logger=True,
                sync_dist=not is_training,
            )
        if dynamics_loss is not None:
            self.log(
                f"{prefix}_dynamics_loss",
                dynamics_loss,
                on_step=log_on_step,
                on_epoch=log_on_epoch,
                prog_bar=False,
                logger=True,
                sync_dist=not is_training,
            )
