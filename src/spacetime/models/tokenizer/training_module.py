import math

import lightning as L
import lpips
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from spacetime.models.tokenizer.config import Hyperparameters
from spacetime.models.tokenizer.model import VQTokenizer
from spacetime.modules.quantizers import QuantizerType
from spacetime.utils.optimizers import ParamGroupConfig, build_optimizer_with_schedule
from spacetime.utils.vq_losses import (compute_entropy_loss,
                                       compute_lpips_loss, compute_vq_losses)


class VQTokenizerModule(L.LightningModule):
    """
    PyTorch Lightning module for training the spacetime transformer-based VQ-VAE tokenizer.

    Args:
        cfg: Hierarchical hyperparameters configuration.
        total_steps: Total training steps for LR scheduler. If None, uses warmup_steps.
    """

    def __init__(self, cfg: Hyperparameters, total_steps: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.total_steps = total_steps

        self.model = VQTokenizer(cfg)

        self.example_clip = None
        self.example_recon = None

        self.lpips_metric = lpips.LPIPS(net="vgg")
        self.lpips_metric.eval()
        for p in self.lpips_metric.parameters():
            p.requires_grad = False

    def forward(self, inputs):
        return self.model(inputs)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.lpips_metric.to(self.device)

    def configure_optimizers(self):
        main_params = []
        quant_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "vector_quantizer" in name:
                quant_params.append(param)
            else:
                main_params.append(param)

        opt_cfg = self.cfg.optimizer
        param_groups = [
            ParamGroupConfig(
                params=main_params,
                lr=opt_cfg.lr,
                warmup_steps=opt_cfg.warmup_steps,
                min_lr_ratio=0.0,  # Decay to 0
            ),
        ]
        if quant_params:
            param_groups.append(
                ParamGroupConfig(
                    params=quant_params,
                    lr=opt_cfg.lr_quant,
                    warmup_steps=opt_cfg.warmup_steps,
                    min_lr_ratio=0.0,  # Decay to 0
                )
            )

        total_steps = self.total_steps if self.total_steps is not None else opt_cfg.warmup_steps

        return build_optimizer_with_schedule(
            param_groups=param_groups,
            total_steps=total_steps,
            betas=opt_cfg.betas,
            weight_decay=opt_cfg.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        """
        Execute a single training step.

        Performs forward pass, computes all losses (reconstruction, commitment,
        codebook, entropy, LPIPS), logs metrics to W&B and Lightning, and stores
        example frames for visualization.

        Args:
            batch: Tuple of (frames, labels) where frames is the input video clip.
            batch_idx: Index of the current batch in the epoch.

        Returns:
            Total weighted loss tensor for backpropagation.
        """
        x, _ = batch
        x_pred, z_e, z_quantized, indices = self.model(x)

        loss, losses = compute_vq_losses(
            x_pred,
            x,
            z_e,
            z_quantized,
            indices,
            beta=self._current_beta(),
            quantizer_type=self.cfg.quantizer.type,
            codebook_size=self.model.vector_quantizer.codebook_size,
            entropy_weight=self.cfg.loss.entropy_weight,
            lpips_weight=self.cfg.loss.lpips_weight,
            lpips_metric=self.lpips_metric,
        )

        self._log_losses(losses, is_training=True)
        beta = self._current_beta()
        self.log("train_beta", beta, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self._log_codebook_usage(indices, is_training=True)
        self._log_lrs()

        if batch_idx == 0:
            self.example_clip = x[:1].detach().cpu()
            self.example_recon = x_pred[:1].detach().cpu()
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_pred, z_e, z_quantized, indices = self.model(x)

        recon_loss = torch.nn.functional.mse_loss(x_pred, x)
        commit_loss = torch.nn.functional.mse_loss(z_e, z_quantized.detach())
        is_vanilla = self.cfg.quantizer.type == QuantizerType.VANILLA
        codebook_loss = (
            torch.nn.functional.mse_loss(z_quantized, z_e.detach()) if is_vanilla else 0.0
        )
        loss_cfg = self.cfg.loss
        entropy_loss = (
            compute_entropy_loss(indices, self.model.vector_quantizer.codebook_size)
            if loss_cfg.entropy_weight
            else None
        )

        beta = self._current_beta()
        loss = recon_loss + (beta * commit_loss) + codebook_loss
        if entropy_loss is not None:
            loss = loss + loss_cfg.entropy_weight * entropy_loss

        losses = {
            "loss": loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "codebook_loss": codebook_loss if is_vanilla else None,
            "entropy_loss": entropy_loss,
        }
        self._log_losses(losses, is_training=False)
        self._log_codebook_usage(indices, is_training=False)

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
        # Single logger case
        if isinstance(self.trainer.logger, WandbLogger):
            return self.trainer.logger
        # Multiple loggers case
        if hasattr(self.trainer.logger, "experiment"):
            # LoggerCollection or similar
            for logger in self.trainer.loggers:
                if isinstance(logger, WandbLogger):
                    return logger
        return None

    def _log_losses(self, losses: dict[str, torch.Tensor], is_training: bool = True):
        """Log a dictionary of losses to all configured loggers."""
        prefix = "train" if is_training else "val"
        log_on_step = is_training

        for name, value in losses.items():
            if value is None:
                continue
            self.log(
                f"{prefix}_{name}",
                value,
                on_step=log_on_step,
                on_epoch=True,
                prog_bar=(name == "loss"),
                logger=True,
                sync_dist=not is_training,
            )

    def _log_codebook_usage(self, indices, is_training: bool) -> None:
        if indices is None:
            return

        codebook_size = self.model.vector_quantizer.codebook_size
        flat_indices = indices.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=codebook_size).float()
        usage = (counts > 0).float().mean()
        probs = counts / (counts.sum() + 1e-8)
        perplexity = torch.exp(-(probs * (probs + 1e-8).log()).sum())

        prefix = "train" if is_training else "val"
        self.log(
            f"{prefix}_codebook_usage",
            usage,
            on_step=is_training,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=not is_training,
        )
        self.log(
            f"{prefix}_codebook_perplexity",
            perplexity,
            on_step=is_training,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=not is_training,
        )

    def _current_beta(self) -> float:
        """
        Beta schedule: warmup from beta.start to beta.end, then cosine decay to beta.final.
        """
        beta_cfg = self.cfg.beta
        step = self.global_step + 1

        # Warmup (ramp up)
        if step <= beta_cfg.warmup_steps:
            if beta_cfg.warmup_steps <= 0:
                return beta_cfg.end
            progress = step / beta_cfg.warmup_steps
            return beta_cfg.start + progress * (beta_cfg.end - beta_cfg.start)

        # Decay (cosine decay from beta.end to beta.final)
        if beta_cfg.decay_steps <= 0:
            return beta_cfg.end

        decay_step = step - beta_cfg.warmup_steps
        progress = min(1.0, decay_step / beta_cfg.decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return beta_cfg.final + cosine_decay * (beta_cfg.end - beta_cfg.final)

    def _log_lrs(self) -> None:
        if self.global_step % 50 != 0:
            return
        if not self.trainer or not self.trainer.optimizers:
            return
        optimizer = self.trainer.optimizers[0]
        if not optimizer.param_groups:
            return
        lr_main = optimizer.param_groups[0].get("lr", None)
        lr_quant = (
            optimizer.param_groups[1].get("lr", None) if len(optimizer.param_groups) > 1 else None
        )
        if lr_main is not None:
            self.log(
                "train_lr",
                lr_main,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )
        if lr_quant is not None:
            self.log(
                "train_lr_quant",
                lr_quant,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )
