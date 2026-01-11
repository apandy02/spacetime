import math

import lightning as L
import lpips
import torch

import wandb
from lightning.pytorch.loggers import WandbLogger

from spacetime.models.tokenizer.config import Hyperparameters
from spacetime.models.tokenizer.model import STVQVae
from spacetime.modules.quantizers import QuantizerType


class STVQVaeModule(L.LightningModule):
    """
    PyTorch Lightning module for training the STVQVae tokenizer.
    
    Args:
        cfg: Hierarchical hyperparameters configuration.
        total_steps: Total training steps for LR scheduler. If None, uses warmup_steps.
    """

    def __init__(self, cfg: Hyperparameters, total_steps: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.total_steps = total_steps
        
        self.model = STVQVae(
            num_heads=cfg.model.num_heads,
            d_model=cfg.model.d_model,
            num_layers=cfg.model.num_layers,
            d_linear=cfg.model.d_linear,
            codebook_size=cfg.quantizer.codebook_size,
            codebook_dim=cfg.quantizer.codebook_dim,
            patch_size=cfg.model.patch_size,
            frame_height=cfg.model.frame_height,
            frame_width=cfg.model.frame_width,
            num_frames=cfg.model.num_frames,
            num_linear_layers=cfg.model.num_linear_layers,
            num_groups=cfg.model.num_groups,
            dropout=cfg.model.dropout,
            quantizer_type=cfg.quantizer.type,
            quantizer_decay=cfg.quantizer.decay,
            quantizer_eps=cfg.quantizer.eps,
            dead_code_threshold=cfg.quantizer.dead_code_threshold,
            dead_code_noise=cfg.quantizer.dead_code_noise,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
        )
        
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
            {"params": main_params, "lr": opt_cfg.lr},
        ]
        if quant_params:
            param_groups.append({"params": quant_params, "lr": opt_cfg.lr_quant})

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=opt_cfg.betas,
            weight_decay=opt_cfg.weight_decay,
        )

        total_steps = self.total_steps
        if total_steps is None:
            total_steps = opt_cfg.warmup_steps

        def warmup_cosine_lambda(step):
            if opt_cfg.warmup_steps > 0 and step < opt_cfg.warmup_steps:
                return min(1.0, (step + 1) / opt_cfg.warmup_steps)
            if total_steps <= opt_cfg.warmup_steps:
                return 1.0
            progress = (step - opt_cfg.warmup_steps) / (total_steps - opt_cfg.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_cosine_lambda
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

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

        loss, losses = self._compute_losses(x_pred, x, z_e, z_quantized, indices)
        
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
            torch.nn.functional.mse_loss(z_quantized, z_e.detach())
            if is_vanilla
            else 0.0
        )
        loss_cfg = self.cfg.loss
        entropy_loss = self._compute_entropy_loss(indices) if loss_cfg.entropy_weight else None

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
            # LPIPS expects inputs in [-1,1]; convert if in [0,1].
            B, C, F, H, W = x.shape
            to_lpips = lambda t: ((t * 2.0) - 1.0).reshape(B * F, C, H, W)
            lpips_val = self.lpips_metric(to_lpips(x_pred), to_lpips(x)).mean()
        self.log("val_lpips", lpips_val, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def _compute_losses(
        self,
        x_pred: torch.Tensor,
        x: torch.Tensor,
        z_e: torch.Tensor,
        z_quantized: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple:
        """
        Compute all training losses for the VQ-VAE model.
        
        Calculates reconstruction loss (MSE), commitment loss, codebook loss
        (for vanilla quantizer), entropy regularization loss, and optional
        perceptual loss (LPIPS). Combines losses with current beta schedule
        weight and configured loss weights.
        
        Args:
            x_pred: Reconstructed video frames from the decoder.
            x: Ground truth input video frames.
            z_e: Encoder output embeddings before quantization.
            z_quantized: Quantized embeddings from the codebook.
            indices: Codebook indices selected during quantization.
            
        Returns:
            Tuple of (total_loss, losses_dict) where losses_dict contains
            individual loss components for logging.
        """
        recon_loss = torch.nn.functional.mse_loss(x_pred, x)
        commit_loss = torch.nn.functional.mse_loss(z_e, z_quantized.detach())
        
        is_vanilla = self.cfg.quantizer.type == QuantizerType.VANILLA
        codebook_loss = (
            torch.nn.functional.mse_loss(z_quantized, z_e.detach())
            if is_vanilla
            else 0.0
        )
        loss_cfg = self.cfg.loss
        entropy_loss = self._compute_entropy_loss(indices) if loss_cfg.entropy_weight else None
        lpips_loss = self._compute_lpips_loss(x_pred, x) if loss_cfg.lpips_weight else None

        beta = self._current_beta()
        loss = recon_loss + (beta * commit_loss) + codebook_loss
        
        if entropy_loss is not None:
            loss = loss + loss_cfg.entropy_weight * entropy_loss
        if lpips_loss is not None:
            loss = loss + loss_cfg.lpips_weight * lpips_loss

        losses = {
            "loss": loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "codebook_loss": codebook_loss if is_vanilla else None,
            "entropy_loss": entropy_loss,
            "lpips_loss": lpips_loss,
        }
        
        return loss, losses

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

    def _compute_entropy_loss(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized entropy loss to encourage diverse codebook usage.

        Returns a value in [0, 1] where 0 means perfect uniform usage
        and 1 means complete collapse to a single code.
        """
        codebook_size = self.model.vector_quantizer.codebook_size
        
        flat_indices = indices.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=codebook_size).float()
        
        probs = counts / (counts.sum() + 1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        max_entropy = math.log(codebook_size)
        
        return (max_entropy - entropy) / max_entropy

    def _compute_lpips_loss(self, x_pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS perceptual loss between prediction and target.
        """
        B, C, F, H, W = x.shape
        to_lpips = lambda t: ((t * 2.0) - 1.0).reshape(B * F, C, H, W)
        return self.lpips_metric(to_lpips(x_pred), to_lpips(x)).mean()

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
            optimizer.param_groups[1].get("lr", None)
            if len(optimizer.param_groups) > 1
            else None
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
