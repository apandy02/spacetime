import torch
import torch.nn as nn
from spacetime.modules.transformer import STEncoder


class STVQVae(nn.Module):
    """
    space time vq vae tokenizer
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        num_layers: int,
        d_linear: int,
        codebook_size: int,
        latent_dim: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super(STVQVae, self).__init__()
        self.pos_embed_space = nn.Parameter(torch.zeros(1, 1, self.n_patches + 1, d_model))
        self.pos_embed_time = nn.Parameter(torch.zeros(1, self.num_frames, 1, d_model))
        
        self.encoder = VQVAEVideoEncoder(
            num_heads,
            d_model,
            num_layers,
            d_linear,
            num_linear_layers,
            num_groups,
            dropout,
        )
        self.codebook = nn.Parameter(
            torch.randn(codebook_size, latent_dim) * 0.02, requires_grad=True
        )
        self.decoder = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the STVQVAE.
        """
        x = self._patchify(x)
        z_e = self.encoder(x + self.pos_embed_space + self.pos_embed_time)
        z_q = self._quantize_encoder_output(z_e)
        return z_q

    def _quantize_encoder_output(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Quantizing the encoded tensor by snapping its elements to the closest codebook
        entry. TODO: update to make spacetime compliant

        Args:
            z_e: encoded representation

        Returns:
            z_q: quantized representation
        """
        batch_size, latent_dim, h, w = z_e.shape
        encoded = z_e.permute(0, 2, 3, 1).reshape(batch_size * h * w, latent_dim)
        quantized = self.codebook[
            torch.argmin(torch.cdist(encoded, self.codebook), dim=1)
        ]
        z_q = quantized.reshape(batch_size, h, w, latent_dim).permute(0, 3, 1, 2)
        return z_q

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits a batch of videos into non-overlapping patches.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, F, H, W]

        Returns:
            torch.Tensor: Patchified tensor of shape [B, F, NUM_P, DIM_P],
                        where DIM_P = channels * patch_size * patch_size.
        """
        batch_size, channels, frames, _, _ = x.shape
        n_patch_side = self.image_size // self.patch_size
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(
            batch_size,
            frames,
            channels,
            n_patch_side,
            self.patch_size,
            n_patch_side,
            self.patch_size,
        )
        x = x.permute(0, 1, 3, 5, 2, 4, 6)
        return x.reshape(
            batch_size, frames, -1, channels * self.patch_size * self.patch_size
        )


class VQVAEVideoEncoder(nn.Module):
    """
    vq vae video encoder
    """

    def __init__(
        self,
        num_heads,
        d_model: int,
        num_layers: int,
        d_linear: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super(VQVAEVideoEncoder, self).__init__()
        self.causal_st_encoder = STEncoder(
            num_heads,
            d_model,
            num_layers,
            d_linear,
            num_linear_layers,
            num_groups,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the VQVAEVideoEncoder.
        """
        return self.causal_st_encoder(x)
