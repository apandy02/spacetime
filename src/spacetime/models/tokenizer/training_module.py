import math

import lightning as L
import lpips
import torch

import wandb
from spacetime.models.tokenizer.model import QuantizerType, STVQVae


class STVQVaeModule(L.LightningModule):
    def __init__(
        self,
        num_heads,
        d_model,
        num_layers,
        d_linear,
        codebook_size,
        codebook_dim,
        patch_size,
        frame_height,
        frame_width,
        num_frames,
        quantizer_type: QuantizerType = QuantizerType.EMA,
        num_linear_layers=2,
        num_groups=8,
        dropout=0.1,
        beta_start=0.05,
        beta_end=0.35,
        beta_warmup_steps=10_000,
        beta_decay_steps=10_000,
        beta_final=0.01,
        lr=3e-4,
        lr_quant=1e-4,
        betas=(0.9, 0.9),
        weight_decay=1e-4,
        warmup_steps=1_000,
        total_steps: int | None = None,
        quantizer_decay=0.985,
        quantizer_eps=1e-5,
        dead_code_threshold=0.01,
        dead_code_noise=1e-4,
        entropy_weight=0.1,
        lpips_weight=0.0,
    ):
        super().__init__()
        self.model = STVQVae(
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            d_linear=d_linear,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            patch_size=patch_size,
            frame_height=frame_height,
            frame_width=frame_width,
            num_frames=num_frames,
            num_linear_layers=num_linear_layers,
            num_groups=num_groups,
            dropout=dropout,
            quantizer_type=quantizer_type,
            quantizer_decay=quantizer_decay,
            quantizer_eps=quantizer_eps,
            dead_code_threshold=dead_code_threshold,
            dead_code_noise=dead_code_noise,
        )
        self.quantizer_type = quantizer_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_warmup_steps = beta_warmup_steps
        self.beta_decay_steps = beta_decay_steps
        self.beta_final = beta_final
        self.example_clip = None
        self.example_recon = None
        self.lr = lr
        self.lr_quant = lr_quant
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.entropy_weight = entropy_weight
        self.lpips_weight = lpips_weight

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

        param_groups = [
            {"params": main_params, "lr": self.lr},
        ]
        if quant_params:
            param_groups.append({"params": quant_params, "lr": self.lr_quant})

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        total_steps = self.total_steps
        if total_steps is None:
            total_steps = self.warmup_steps

        def warmup_cosine_lambda(step):
            if self.warmup_steps > 0 and step < self.warmup_steps:
                return min(1.0, (step + 1) / self.warmup_steps)
            if total_steps <= self.warmup_steps:
                return 1.0
            progress = (step - self.warmup_steps) / (total_steps - self.warmup_steps)
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
        x, _ = batch
        x_pred, z_e, z_quantized, indices = self.model(x)

        recon_loss = torch.nn.functional.mse_loss(x_pred, x)
        commit_loss = torch.nn.functional.mse_loss(z_e, z_quantized.detach())
        codebook_loss = (
            torch.nn.functional.mse_loss(z_quantized, z_e.detach())
            if self.quantizer_type == QuantizerType.VANILLA
            else 0.0
        )
        entropy_loss = self._compute_entropy_loss(indices) if self.entropy_weight else None
        lpips_loss = self._compute_lpips_loss(x_pred, x) if self.lpips_weight else None

        beta = self._current_beta()
        loss = recon_loss + (beta * commit_loss) + codebook_loss
        if entropy_loss is not None:
            loss = loss + self.entropy_weight * entropy_loss
        if lpips_loss is not None:
            loss = loss + self.lpips_weight * lpips_loss

        losses = {
            "loss": loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "codebook_loss": codebook_loss if self.quantizer_type == QuantizerType.VANILLA else None,
            "entropy_loss": entropy_loss,
            "lpips_loss": lpips_loss,
        }
        self._log_losses(losses, is_training=True)
        if wandb.run is not None and self.trainer.is_global_zero:
            wandb.log({"train_beta": beta}, step=self.global_step)
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
        codebook_loss = (
            torch.nn.functional.mse_loss(z_quantized, z_e.detach())
            if self.quantizer_type == QuantizerType.VANILLA
            else 0.0
        )
        entropy_loss = self._compute_entropy_loss(indices) if self.entropy_weight else None

        beta = self._current_beta()
        loss = recon_loss + (beta * commit_loss) + codebook_loss
        if entropy_loss is not None:
            loss = loss + self.entropy_weight * entropy_loss

        losses = {
            "loss": loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "codebook_loss": codebook_loss if self.quantizer_type == QuantizerType.VANILLA else None,
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
        if wandb.run is not None and self.trainer.is_global_zero:
            wandb.log({"val_lpips": lpips_val.item()}, step=self.global_step)
        return loss

    def on_validation_epoch_end(self):
        if self.example_clip is None or wandb.run is None:
            return
        clip = (self.example_clip.clamp(0, 1) * 255).to(torch.uint8)
        recon = (self.example_recon.clamp(0, 1) * 255).to(torch.uint8)
        video = torch.cat([clip, recon], dim=4)
        video = video.squeeze(0).permute(1, 0, 2, 3)  # (F, C, H, W)
        if wandb.run is not None and self.trainer.is_global_zero:
            wandb.log(
                {"recon_video": wandb.Video(video.squeeze(0), fps=4, format="mp4")},
                step=self.global_step,
            )
        self.example_clip = None
        self.example_recon = None

    def _log_losses(self, losses: dict[str, torch.Tensor], is_training: bool = True):
        """
        Log a dictionary of losses to lightning logger and wandb.
        """
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

        if wandb.run is not None and self.trainer.is_global_zero:
            log_dict = {
                f"{prefix}_{k}": v.item() for k, v in losses.items() if v is not None
            }
            wandb.log(log_dict, step=self.global_step)

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

        if wandb.run is not None and self.trainer.is_global_zero:
            wandb.log(
                {
                    f"{prefix}_codebook_usage": usage.item(),
                    f"{prefix}_codebook_perplexity": perplexity.item(),
                },
                step=self.global_step,
            )

    def _current_beta(self) -> float:
        """
        Beta schedule: warmup from beta_start to beta_end, then cosine decay to beta_final.
        """
        step = self.global_step + 1

        # Phase 1: Warmup (ramp up)
        if step <= self.beta_warmup_steps:
            if self.beta_warmup_steps <= 0:
                return self.beta_end
            progress = step / self.beta_warmup_steps
            return self.beta_start + progress * (self.beta_end - self.beta_start)

        # Phase 2: Decay (cosine decay from beta_end to beta_final)
        if self.beta_decay_steps <= 0:
            return self.beta_end
        decay_step = step - self.beta_warmup_steps
        progress = min(1.0, decay_step / self.beta_decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.beta_final + cosine_decay * (self.beta_end - self.beta_final)

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
        if wandb.run is not None and self.trainer.is_global_zero:
            log_dict = {}
            if lr_main is not None:
                log_dict["train_lr"] = lr_main
            if lr_quant is not None:
                log_dict["train_lr_quant"] = lr_quant
            if log_dict:
                wandb.log(log_dict, step=self.global_step)
