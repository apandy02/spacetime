import lightning as L
import lpips
import torch
import wandb

from spacetime.models.tokenizer.model import STVQVae, QuantizerType


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
        beta=0.1,
        lr=3e-4,
        betas=(0.9, 0.9),
        weight_decay=1e-4,
        warmup_steps=1_000,
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
        )
        self.quantizer_type = quantizer_type
        self.beta = beta
        self.example_clip = None
        self.example_recon = None
        self.lr = lr
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
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
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
        x_pred, z_e, z_quantized = self(x)
        recon_loss = torch.nn.functional.mse_loss(x_pred, x)
        commit_loss = torch.nn.functional.mse_loss(z_e, z_quantized.detach())
        codebook_loss = (
            torch.nn.functional.mse_loss(z_quantized, z_e.detach())
            if self.quantizer_type == QuantizerType.VANILLA
            else 0.0
        )
        loss = recon_loss + (self.beta * commit_loss) + codebook_loss

        self._log_losses(loss, recon_loss, commit_loss, codebook_loss, is_training=True)

        if batch_idx == 0:
            self.example_clip = x[:1].detach().cpu()
            self.example_recon = x_pred[:1].detach().cpu()
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_pred, z_e, z_quantized = self(x)
        recon_loss = torch.nn.functional.mse_loss(x_pred, x)
        commit_loss = torch.nn.functional.mse_loss(z_e, z_quantized.detach())
        codebook_loss = (
            torch.nn.functional.mse_loss(z_quantized, z_e.detach())
            if self.quantizer_type == QuantizerType.VANILLA
            else 0.0
        )
        loss = recon_loss + (self.beta * commit_loss) + codebook_loss

        self._log_losses(loss, recon_loss, commit_loss, codebook_loss, is_training=False)

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
