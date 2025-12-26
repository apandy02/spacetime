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
        lr=3e-4,
        lr_quant=1e-4,
        betas=(0.9, 0.9),
        weight_decay=1e-4,
        warmup_steps=1_000,
        quantizer_decay=0.985,
        quantizer_eps=1e-5,
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
        )
        self.quantizer_type = quantizer_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_warmup_steps = beta_warmup_steps
        self.example_clip = None
        self.example_recon = None
        self.lr = lr
        self.lr_quant = lr_quant
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

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

        def warmup_lambda(step):
            return min(1.0, (step + 1) / self.warmup_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
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
        outputs = self.model(x, return_indices=True)
        if len(outputs) == 4:
            x_pred, z_e, z_quantized, indices = outputs
        else:
            x_pred, z_e, z_quantized = outputs
            indices = None
        recon_loss = torch.nn.functional.mse_loss(x_pred, x)
        commit_loss = torch.nn.functional.mse_loss(z_e, z_quantized.detach())
        codebook_loss = (
            torch.nn.functional.mse_loss(z_quantized, z_e.detach())
            if self.quantizer_type == QuantizerType.VANILLA
            else 0.0
        )
        beta = self._current_beta()
        loss = recon_loss + (beta * commit_loss) + codebook_loss

        self._log_losses(loss, recon_loss, commit_loss, codebook_loss, is_training=True)
        if wandb.run is not None and self.trainer.is_global_zero:
            wandb.log({"train_beta": beta}, step=self.global_step)
        self._log_codebook_usage(indices, is_training=True)

        if batch_idx == 0:
            self.example_clip = x[:1].detach().cpu()
            self.example_recon = x_pred[:1].detach().cpu()
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        outputs = self.model(x, return_indices=True)
        if len(outputs) == 4:
            x_pred, z_e, z_quantized, indices = outputs
        else:
            x_pred, z_e, z_quantized = outputs
            indices = None
        recon_loss = torch.nn.functional.mse_loss(x_pred, x)
        commit_loss = torch.nn.functional.mse_loss(z_e, z_quantized.detach())
        codebook_loss = (
            torch.nn.functional.mse_loss(z_quantized, z_e.detach())
            if self.quantizer_type == QuantizerType.VANILLA
            else 0.0
        )
        beta = self._current_beta()
        loss = recon_loss + (beta * commit_loss) + codebook_loss

        self._log_losses(loss, recon_loss, commit_loss, codebook_loss, is_training=False)
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

    def _log_losses(self, loss, recon_loss, commit_loss, codebook_loss, is_training=True):
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
            f"{prefix}_commit_loss",
            commit_loss,
            on_step=log_on_step,
            on_epoch=log_on_epoch,
            prog_bar=False,
            logger=True,
            sync_dist=not is_training,
        )
        if self.quantizer_type == QuantizerType.VANILLA:
            self.log(
                f"{prefix}_codebook_loss",
                codebook_loss,
                on_step=log_on_step,
                on_epoch=log_on_epoch,
                prog_bar=False,
                logger=True,
                sync_dist=not is_training,
            )

        if wandb.run is not None and self.trainer.is_global_zero:
            log_dict = {
                f"{prefix}_loss": loss.item(),
                f"{prefix}_recon_loss": recon_loss.item(),
                f"{prefix}_commit_loss": commit_loss.item(),
            }
            if self.quantizer_type == QuantizerType.VANILLA:
                log_dict[f"{prefix}_codebook_loss"] = codebook_loss.item()
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
        if self.beta_warmup_steps <= 0:
            return self.beta_end
        progress = min(1.0, (self.global_step + 1) / self.beta_warmup_steps)
        return self.beta_start + progress * (self.beta_end - self.beta_start)
